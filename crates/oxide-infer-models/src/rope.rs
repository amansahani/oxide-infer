use anyhow::Result;
use oxide_infer_core::{DeviceBuffer, GpuContext};
use oxide_infer_kernels::PtxModule;
use cuda_core::launch_kernel_on_stream;

const ROPE_KERNEL_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

// Applies partial RoPE:
// x has shape [num_heads, head_dim]. head_dim = 128.
// rotate_half: x1 = x[0..64], x2 = x[64..128]. rotated = [-x2, x1].
// out = x * cos + rotated * sin
.visible .entry apply_rope_fp16(
    .param .u64 x_io,
    .param .u64 cos_table,
    .param .u64 sin_table,
    .param .u32 pos_id,
    .param .u32 num_heads,
    .param .u32 head_dim
) {
    .reg .pred %p0;
    .reg .b32 %head_idx, %dim_idx, %nh, %hd, %pid, %half_hd, %other_dim_idx;
    .reg .b64 %x_ptr, %cos_ptr, %sin_ptr, %pos_offset, %head_offset, %dim_offset, %other_offset;
    .reg .b16 %hx, %h_other, %hcos, %hsin, %hout;
    .reg .f32 %fx, %f_other, %fcos, %fsin, %fres;

    ld.param.u32 %pid, [pos_id];
    ld.param.u32 %nh, [num_heads];
    ld.param.u32 %hd, [head_dim];

    mov.u32 %head_idx, %ctaid.x;
    setp.ge.u32 %p0, %head_idx, %nh;
    @%p0 bra EXIT;

    mov.u32 %dim_idx, %tid.x;
    setp.ge.u32 %p0, %dim_idx, %hd;
    @%p0 bra EXIT;

    ld.param.u64 %x_ptr, [x_io];
    ld.param.u64 %cos_ptr, [cos_table];
    ld.param.u64 %sin_ptr, [sin_table];

    // Load cos[pos_id, dim_idx] and sin[pos_id, dim_idx]
    mul.wide.u32 %pos_offset, %pid, %hd;
    cvt.u64.u32 %dim_offset, %dim_idx;
    add.u64 %pos_offset, %pos_offset, %dim_offset;
    shl.b64 %pos_offset, %pos_offset, 1;

    add.u64 %cos_ptr, %cos_ptr, %pos_offset;
    ld.global.u16 %hcos, [%cos_ptr];

    add.u64 %sin_ptr, %sin_ptr, %pos_offset;
    ld.global.u16 %hsin, [%sin_ptr];

    // Load x[head_idx, dim_idx]
    mul.wide.u32 %head_offset, %head_idx, %hd;
    add.u64 %head_offset, %head_offset, %dim_offset;
    shl.b64 %head_offset, %head_offset, 1;
    add.u64 %head_offset, %x_ptr, %head_offset;
    ld.global.u16 %hx, [%head_offset];

    // rotate_half logic:
    // if dim_idx < 64: other is (dim_idx + 64), rotate_half value is -x[other]
    // if dim_idx >= 64: other is (dim_idx - 64), rotate_half value is +x[other]
    shr.u32 %half_hd, %hd, 1;
    setp.lt.u32 %p0, %dim_idx, %half_hd;

    @%p0 add.u32 %other_dim_idx, %dim_idx, %half_hd;
    @!%p0 sub.u32 %other_dim_idx, %dim_idx, %half_hd;

    mul.wide.u32 %other_offset, %head_idx, %hd;
    cvt.u64.u32 %dim_offset, %other_dim_idx;
    add.u64 %other_offset, %other_offset, %dim_offset;
    shl.b64 %other_offset, %other_offset, 1;
    add.u64 %other_offset, %x_ptr, %other_offset;
    ld.global.u16 %h_other, [%other_offset];

    cvt.f32.f16 %fx, %hx;
    cvt.f32.f16 %f_other, %h_other;
    cvt.f32.f16 %fcos, %hcos;
    cvt.f32.f16 %fsin, %hsin;

    // if dim_idx < 64: f_other = -f_other
    @%p0 neg.f32 %f_other, %f_other;

    // out = fx * fcos + f_other * fsin
    mul.f32 %fres, %fx, %fcos;
    fma.rn.f32 %fres, %f_other, %fsin, %fres;
    cvt.rn.f16.f32 %hout, %fres;

    st.global.u16 [%head_offset], %hout;

EXIT:
    ret;
}
"#;

/// Process-wide cache of RoPE tables.
///
/// A table depends only on `(theta, partial factor, window, max_seq_len)` and
/// is read-only at inference, so layers that share those parameters can share
/// one allocation. LFM3 has only two distinct configurations (full attention:
/// theta 1e6, partial 0.5; sliding: theta 1e4, partial 1.0), so 11 tables
/// collapse to 2. At a 262144-token limit that is 1.48 GiB -> 0.27 GiB.
static ROPE_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<(u32, u32, u32, usize), std::sync::Arc<RotaryEmbeddingTable>>>,
> = std::sync::OnceLock::new();

/// Get the shared table for these parameters, creating it on first use.
pub fn shared_rope_table(
    gpu: &GpuContext,
    max_seq_len: usize,
    head_dim: usize,
    rope_theta: f32,
    partial_rotary_factor: f32,
    window: u32,
) -> Result<std::sync::Arc<RotaryEmbeddingTable>> {
    let key = (
        rope_theta.to_bits(),
        partial_rotary_factor.to_bits(),
        window,
        max_seq_len,
    );
    let mut cache = ROPE_CACHE
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
        .lock()
        .expect("rope cache lock");
    if let Some(t) = cache.get(&key) {
        return Ok(t.clone());
    }
    let table = std::sync::Arc::new(RotaryEmbeddingTable::new(
        gpu,
        max_seq_len,
        head_dim,
        rope_theta,
        partial_rotary_factor,
        window,
    )?);
    cache.insert(key, table.clone());
    Ok(table)
}

pub struct RotaryEmbeddingTable {
    pub cos_table: DeviceBuffer<half::f16>,
    pub sin_table: DeviceBuffer<half::f16>,
    pub ptx: PtxModule,
    pub max_seq_len: usize,
    pub head_dim: usize,
    /// Sliding-attention window in tokens, or `0` for full attention.
    /// LFM3 mixes both layer types; consumers pass this to the attention
    /// kernel so the two kinds of layer see the reference mask.
    pub window: u32,
}

impl RotaryEmbeddingTable {
    pub fn new(
        gpu: &GpuContext,
        max_seq_len: usize,
        head_dim: usize,
        rope_theta: f32,
        partial_rotary_factor: f32,
        window: u32,
    ) -> Result<Self> {
        let rope_angles = (partial_rotary_factor * head_dim as f32) as usize / 2;
        let mut inv_freq = vec![0.0f32; head_dim / 2];
        for i in 0..rope_angles {
            inv_freq[i] = 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32);
        }

        let mut host_cos = Vec::with_capacity(max_seq_len * head_dim);
        let mut host_sin = Vec::with_capacity(max_seq_len * head_dim);

        for pos in 0..max_seq_len {
            let mut freqs = Vec::with_capacity(head_dim);
            for &f in &inv_freq {
                freqs.push(pos as f32 * f);
            }
            for &f in &inv_freq {
                freqs.push(pos as f32 * f);
            }
            for &angle in &freqs {
                host_cos.push(half::f16::from_f32(angle.cos()));
                host_sin.push(half::f16::from_f32(angle.sin()));
            }
        }

        let cos_table = DeviceBuffer::from_slice(gpu, &host_cos)?;
        let sin_table = DeviceBuffer::from_slice(gpu, &host_sin)?;
        let ptx = PtxModule::load_from_ptx(gpu, ROPE_KERNEL_PTX)?;

        Ok(Self {
            cos_table,
            sin_table,
            ptx,
            max_seq_len,
            head_dim,
            window,
        })
    }

    pub fn apply(
        &self,
        gpu: &GpuContext,
        x_io: &mut DeviceBuffer<half::f16>,
        pos_id: usize,
        num_heads: usize,
    ) -> Result<()> {
        let func = self.ptx.get_function("apply_rope_fp16")?;
        let stream = gpu.compute_stream();

        let mut d_x = x_io.inner_mut().cu_deviceptr();
        let mut d_cos = self.cos_table.inner().cu_deviceptr();
        let mut d_sin = self.sin_table.inner().cu_deviceptr();
        let mut pid_u32 = pos_id as u32;
        let mut nh_u32 = num_heads as u32;
        let mut hd_u32 = self.head_dim as u32;

        let mut args: [*mut std::ffi::c_void; 6] = [
            &mut d_x as *mut _ as *mut std::ffi::c_void,
            &mut d_cos as *mut _ as *mut std::ffi::c_void,
            &mut d_sin as *mut _ as *mut std::ffi::c_void,
            &mut pid_u32 as *mut _ as *mut std::ffi::c_void,
            &mut nh_u32 as *mut _ as *mut std::ffi::c_void,
            &mut hd_u32 as *mut _ as *mut std::ffi::c_void,
        ];

        unsafe {
            launch_kernel_on_stream(
                &func,
                (nh_u32, 1, 1),
                (hd_u32, 1, 1),
                0,
                stream,
                &mut args,
            )?;
        }
        Ok(())
    }
}
