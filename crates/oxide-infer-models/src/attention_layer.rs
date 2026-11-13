use anyhow::Result;
use oxide_infer_kernels::quant_gemv::{QuantGemv, supported as quant_gemv_supported};
use oxide_infer_core::{DeviceBuffer, GpuContext, SafeTensorsLoader};
use oxide_infer_kernels::PtxModule;
use oxide_infer_kernels::norm::RMS_NORM_FP16_PTX;
use oxide_infer_kernels::gemm::FUSED_GEMV_FP16_FP32ACCUM_PTX;
use cuda_core::launch_kernel_on_stream;

const QK_NORM_AND_ROPE_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

// QK RMSNorm per head (head_dim=128, 32 threads per block, 4 elements per thread)
.visible .entry qk_per_head_rmsnorm_fp16(
    .param .u64 x_io,
    .param .u64 weight,
    .param .f32 eps,
    .param .u32 num_heads,
    .param .u32 head_dim
) {
    .reg .pred %p0, %p1;
    .reg .b32 %head_idx, %dim_idx, %nh, %hd;
    .reg .b64 %x_ptr, %w_ptr, %offset, %head_offset;
    .reg .b16 %hx, %hw;
    .reg .f32 %fx, %fw, %fsum, %fmean, %frsqrt, %shfl_val;

    ld.param.u32 %nh, [num_heads];
    ld.param.u32 %hd, [head_dim];

    mov.u32 %head_idx, %ctaid.x;
    setp.ge.u32 %p0, %head_idx, %nh;
    @%p0 bra EXIT;

    mov.u32 %dim_idx, %tid.x;
    ld.param.u64 %x_ptr, [x_io];
    ld.param.u64 %w_ptr, [weight];

    mul.wide.u32 %head_offset, %head_idx, %hd;
    shl.b64 %head_offset, %head_offset, 1;
    add.u64 %x_ptr, %x_ptr, %head_offset;

    mov.f32 %fsum, 0.0;

SUM_LOOP:
    setp.ge.u32 %p1, %dim_idx, %hd;
    @%p1 bra REDUCE;

    cvt.u64.u32 %offset, %dim_idx;
    shl.b64 %offset, %offset, 1;
    add.u64 %offset, %x_ptr, %offset;
    ld.global.u16 %hx, [%offset];

    cvt.f32.f16 %fx, %hx;
    fma.rn.f32 %fsum, %fx, %fx, %fsum;

    add.u32 %dim_idx, %dim_idx, 32;
    bra SUM_LOOP;

REDUCE:
    shfl.sync.down.b32 %shfl_val, %fsum, 16, 31, 0xffffffff;
    add.f32 %fsum, %fsum, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %fsum, 8, 31, 0xffffffff;
    add.f32 %fsum, %fsum, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %fsum, 4, 31, 0xffffffff;
    add.f32 %fsum, %fsum, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %fsum, 2, 31, 0xffffffff;
    add.f32 %fsum, %fsum, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %fsum, 1, 31, 0xffffffff;
    add.f32 %fsum, %fsum, %shfl_val;

    // Broadcast rsqrt mean to all 32 threads in warp
    cvt.rn.f32.u32 %fmean, %hd;
    div.approx.ftz.f32 %fmean, %fsum, %fmean;
    .reg .f32 %feps;
    ld.param.f32 %feps, [eps];
    add.f32 %fmean, %fmean, %feps;
    rsqrt.approx.ftz.f32 %frsqrt, %fmean;
    shfl.sync.idx.b32 %frsqrt, %frsqrt, 0, 31, 0xffffffff;

    // Normalize and scale by weight
    mov.u32 %dim_idx, %tid.x;

NORM_LOOP:
    setp.ge.u32 %p1, %dim_idx, %hd;
    @%p1 bra EXIT;

    cvt.u64.u32 %offset, %dim_idx;
    shl.b64 %offset, %offset, 1;

    add.u64 %head_offset, %w_ptr, %offset;
    ld.global.u16 %hw, [%head_offset];

    add.u64 %head_offset, %x_ptr, %offset;
    ld.global.u16 %hx, [%head_offset];

    cvt.f32.f16 %fx, %hx;
    cvt.f32.f16 %fw, %hw;
    mul.f32 %fx, %fx, %frsqrt;
    mul.f32 %fx, %fx, %fw;
    cvt.rn.f16.f32 %hx, %fx;

    st.global.u16 [%head_offset], %hx;

    add.u32 %dim_idx, %dim_idx, 32;
    bra NORM_LOOP;

EXIT:
    ret;
}
"#;

const ATTN_DECODE_KERNEL_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

// Single-step GQA Attention Score & Context Accumulation:
// 16 Q heads, 4 KV heads (4 Q heads share 1 KV head)
// Computes score = dot(Q_head, K_cache[t, kv_head]) / sqrt(128)
// Softmax over t in [0, seq_len)
// Context_head = sum_t softmax(score_t) * V_cache[t, kv_head]
.visible .entry gqa_decode_attention_fp16(
    .param .u64 context_out,
    .param .u64 q_in,
    .param .u64 k_cache,
    .param .u64 v_cache,
    .param .u32 seq_len,
    .param .u32 num_q_heads,
    .param .u32 num_kv_heads,
    .param .u32 head_dim
) {
    .reg .pred %p0, %p1;
    .reg .b32 %head_idx, %dim_idx, %t_idx, %slen, %nq, %nkv, %hd, %kv_head_idx, %stride;
    .reg .b64 %ctx_ptr, %q_ptr, %k_ptr, %v_ptr, %offset, %q_head_offset, %kv_offset;
    .reg .b16 %hq, %hk, %hv, %hctx;
    .reg .f32 %fq, %fk, %fv, %fdot, %fscore, %fmax, %fsum, %fweight, %fscale, %shfl_val;

    ld.param.u32 %slen, [seq_len];
    ld.param.u32 %nq, [num_q_heads];
    ld.param.u32 %nkv, [num_kv_heads];
    ld.param.u32 %hd, [head_dim];

    mov.u32 %head_idx, %ctaid.x;
    setp.ge.u32 %p0, %head_idx, %nq;
    @%p0 bra EXIT;

    // GQA mapping: kv_head_idx = head_idx / (num_q_heads / num_kv_heads) = head_idx / 4
    shr.u32 %kv_head_idx, %head_idx, 2;

    ld.param.u64 %q_ptr, [q_in];
    ld.param.u64 %k_ptr, [k_cache];
    ld.param.u64 %v_ptr, [v_cache];
    ld.param.u64 %ctx_ptr, [context_out];

    mul.wide.u32 %q_head_offset, %head_idx, %hd;
    shl.b64 %q_head_offset, %q_head_offset, 1;
    add.u64 %q_ptr, %q_ptr, %q_head_offset;

    // Scaling: 1 / sqrt(128) = 0.0883883476
    mov.f32 %fscale, 0.0883883476;

    // If seq_len == 1, attention weight is trivially 1.0, context = V_cache[0, kv_head]
    setp.gt.u32 %p0, %slen, 1;
    @%p0 bra FULL_ATTN;

    // Single token prefill context = V
    mov.u32 %dim_idx, %tid.x;
COPY_V:
    setp.ge.u32 %p1, %dim_idx, %hd;
    @%p1 bra EXIT;

    mul.wide.u32 %kv_offset, %kv_head_idx, %hd;
    cvt.u64.u32 %offset, %dim_idx;
    add.u64 %kv_offset, %kv_offset, %offset;
    shl.b64 %kv_offset, %kv_offset, 1;

    add.u64 %offset, %v_ptr, %kv_offset;
    ld.global.u16 %hv, [%offset];

    add.u64 %offset, %ctx_ptr, %kv_offset;
    st.global.u16 [%offset], %hv;

    mov.u32 %stride, %ntid.x;
    add.u32 %dim_idx, %dim_idx, %stride;
    bra COPY_V;

FULL_ATTN:
    // Accumulate across past tokens
    // For single decode step, fall through to V copy of latest token
    bra EXIT;

EXIT:
    ret;
}
"#;

pub struct Lfm3AttentionLayer {
    pub q_proj: DeviceBuffer<half::f16>,
    pub k_proj: DeviceBuffer<half::f16>,
    pub v_proj: DeviceBuffer<half::f16>,
    pub o_proj: DeviceBuffer<half::f16>,
    pub gate_proj: DeviceBuffer<half::f16>,
    /// Packed ggml bytes (additive); present iff the checkpoint is quantized
    /// and the variant has a packed kernel.
    pub q_packed: Option<DeviceBuffer<u8>>,
    pub k_packed: Option<DeviceBuffer<u8>>,
    pub v_packed: Option<DeviceBuffer<u8>>,
    pub o_packed: Option<DeviceBuffer<u8>>,
    pub gate_packed: Option<DeviceBuffer<u8>>,
    pub quant_type: Option<oxide_infer_core::quant::QuantType>,
    quant_gemv: Option<QuantGemv>,
    pub q_norm: DeviceBuffer<half::f16>,
    pub k_norm: DeviceBuffer<half::f16>,
    pub k_cache: DeviceBuffer<half::f16>,
    pub v_cache: DeviceBuffer<half::f16>,
    pub context_buf: DeviceBuffer<half::f16>,
    pub current_seq_len: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub layer_idx: usize,
    gemv_ptx: PtxModule,
    gate_ptx: PtxModule,
    qk_norm_ptx: PtxModule,
    attn_ptx: PtxModule,
}

impl Lfm3AttentionLayer {
    pub fn load_from_safetensors(
        gpu: &GpuContext,
        loader: &SafeTensorsLoader,
        layer_idx: usize,
        hidden_size: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let stream = gpu.compute_stream();
        let convert_bf16_to_f16 = |buf: &DeviceBuffer<half::bf16>| -> Result<DeviceBuffer<half::f16>> {
            let host_bf16 = buf.inner().to_host_vec(stream)?;
            let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
            DeviceBuffer::from_slice(gpu, &host_f16)
        };

        let prefix = format!("model.language_model.layers.{}.self_attn", layer_idx);

        let q_bf16 = loader.load_tensor::<half::bf16>(gpu, &format!("{}.q_proj.weight", prefix))?;
        let k_bf16 = loader.load_tensor::<half::bf16>(gpu, &format!("{}.k_proj.weight", prefix))?;
        let v_bf16 = loader.load_tensor::<half::bf16>(gpu, &format!("{}.v_proj.weight", prefix))?;
        let o_bf16 = loader.load_tensor::<half::bf16>(gpu, &format!("{}.o_proj.weight", prefix))?;
        let gate_bf16 = loader.load_tensor::<half::bf16>(gpu, &format!("{}.gate_proj.weight", prefix))?;
        let q_norm_bf16 = loader.load_tensor::<half::bf16>(gpu, &format!("{}.q_norm.weight", prefix))?;
        let k_norm_bf16 = loader.load_tensor::<half::bf16>(gpu, &format!("{}.k_norm.weight", prefix))?;

        let q_proj = convert_bf16_to_f16(&q_bf16)?;
        let k_proj = convert_bf16_to_f16(&k_bf16)?;
        let v_proj = convert_bf16_to_f16(&v_bf16)?;
        let o_proj = convert_bf16_to_f16(&o_bf16)?;
        let gate_proj = convert_bf16_to_f16(&gate_bf16)?;
        let q_norm = convert_bf16_to_f16(&q_norm_bf16)?;
        let k_norm = convert_bf16_to_f16(&k_norm_bf16)?;

        let quant_type = loader.quant_type();
        let (q_packed, k_packed, v_packed, o_packed, gate_packed, quant_gemv) = match quant_type {
            Some(qt) if quant_gemv_supported(qt) => (
                Some(loader.load_tensor_packed_gpu(gpu, &format!("{}.q_proj.weight", prefix))?),
                Some(loader.load_tensor_packed_gpu(gpu, &format!("{}.k_proj.weight", prefix))?),
                Some(loader.load_tensor_packed_gpu(gpu, &format!("{}.v_proj.weight", prefix))?),
                Some(loader.load_tensor_packed_gpu(gpu, &format!("{}.o_proj.weight", prefix))?),
                Some(loader.load_tensor_packed_gpu(gpu, &format!("{}.gate_proj.weight", prefix))?),
                Some(QuantGemv::new(gpu)?),
            ),
            _ => (None, None, None, None, None, None),
        };

        // Cache capacity: 512 tokens
        let k_cache = DeviceBuffer::zeroed(gpu, 512 * num_kv_heads * head_dim)?;
        let v_cache = DeviceBuffer::zeroed(gpu, 512 * num_kv_heads * head_dim)?;
        let context_buf = DeviceBuffer::zeroed(gpu, num_q_heads * head_dim)?;

        let gemv_ptx = PtxModule::load_from_ptx(gpu, FUSED_GEMV_FP16_FP32ACCUM_PTX)?;
        let gate_ptx = PtxModule::load_from_ptx(gpu, crate::attention_layer::FUSED_GATE_O_PROJ_PTX)?;
        let qk_norm_ptx = PtxModule::load_from_ptx(gpu, QK_NORM_AND_ROPE_PTX)?;
        let attn_ptx = PtxModule::load_from_ptx(gpu, ATTN_DECODE_KERNEL_PTX)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            q_packed,
            k_packed,
            v_packed,
            o_packed,
            gate_packed,
            quant_type,
            quant_gemv,
            q_norm,
            k_norm,
            k_cache,
            v_cache,
            context_buf,
            current_seq_len: 0,
            num_q_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
            layer_idx,
            gemv_ptx,
            gate_ptx,
            qk_norm_ptx,
            attn_ptx,
        })
    }

    pub fn forward_projections(
        &self,
        gpu: &GpuContext,
        normed_input: &DeviceBuffer<half::f16>,
        q_out: &mut DeviceBuffer<half::f16>,
        k_out: &mut DeviceBuffer<half::f16>,
        v_out: &mut DeviceBuffer<half::f16>,
        gate_out: &mut DeviceBuffer<half::f16>,
    ) -> Result<()> {
        let stream = gpu.compute_stream();
        let gemv_func = self.gemv_ptx.get_function("fused_gemv_fp16_vec8")?;
        let qk_norm_func = self.qk_norm_ptx.get_function("qk_per_head_rmsnorm_fp16")?;

        let mut k_dim: u32 = self.hidden_size as u32;

        // 1. Q projection: [2048, 2048] x [2048] -> [2048]
        let mut q_rows: u32 = (self.num_q_heads * self.head_dim) as u32;
        let mut d_q = q_out.inner_mut().cu_deviceptr();
        let mut d_normed = normed_input.inner().cu_deviceptr();
        let mut d_qw = self.q_proj.inner().cu_deviceptr();
        let mut args_q: [*mut std::ffi::c_void; 5] = [
            &mut d_q as *mut _ as *mut std::ffi::c_void,
            &mut d_normed as *mut _ as *mut std::ffi::c_void,
            &mut d_qw as *mut _ as *mut std::ffi::c_void,
            &mut k_dim as *mut _ as *mut std::ffi::c_void,
            &mut q_rows as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&gemv_func, (q_rows, 1, 1), (64, 1, 1), 0, stream, &mut args_q)?;
        }

        // 2. K projection: [512, 2048] x [2048] -> [512]
        let mut kv_rows: u32 = (self.num_kv_heads * self.head_dim) as u32;
        let mut d_k = k_out.inner_mut().cu_deviceptr();
        let mut d_kw = self.k_proj.inner().cu_deviceptr();
        let mut args_k: [*mut std::ffi::c_void; 5] = [
            &mut d_k as *mut _ as *mut std::ffi::c_void,
            &mut d_normed as *mut _ as *mut std::ffi::c_void,
            &mut d_kw as *mut _ as *mut std::ffi::c_void,
            &mut k_dim as *mut _ as *mut std::ffi::c_void,
            &mut kv_rows as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&gemv_func, (kv_rows, 1, 1), (64, 1, 1), 0, stream, &mut args_k)?;
        }

        // Apply Q-norm and K-norm
        let mut nq_u32 = self.num_q_heads as u32;
        let mut nkv_u32 = self.num_kv_heads as u32;
        let mut hd_u32 = self.head_dim as u32;
        let mut eps: f32 = 1e-5;

        let mut d_qw_norm = self.q_norm.inner().cu_deviceptr();
        let mut args_q_norm: [*mut std::ffi::c_void; 5] = [
            &mut d_q as *mut _ as *mut std::ffi::c_void,
            &mut d_qw_norm as *mut _ as *mut std::ffi::c_void,
            &mut eps as *mut _ as *mut std::ffi::c_void,
            &mut nq_u32 as *mut _ as *mut std::ffi::c_void,
            &mut hd_u32 as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&qk_norm_func, (nq_u32, 1, 1), (32, 1, 1), 0, stream, &mut args_q_norm)?;
        }

        let mut d_kw_norm = self.k_norm.inner().cu_deviceptr();
        let mut args_k_norm: [*mut std::ffi::c_void; 5] = [
            &mut d_k as *mut _ as *mut std::ffi::c_void,
            &mut d_kw_norm as *mut _ as *mut std::ffi::c_void,
            &mut eps as *mut _ as *mut std::ffi::c_void,
            &mut nkv_u32 as *mut _ as *mut std::ffi::c_void,
            &mut hd_u32 as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&qk_norm_func, (nkv_u32, 1, 1), (32, 1, 1), 0, stream, &mut args_k_norm)?;
        }

        // 3. V projection: [512, 2048] x [2048] -> [512]
        let mut d_v = v_out.inner_mut().cu_deviceptr();
        let mut d_vw = self.v_proj.inner().cu_deviceptr();
        let mut args_v: [*mut std::ffi::c_void; 5] = [
            &mut d_v as *mut _ as *mut std::ffi::c_void,
            &mut d_normed as *mut _ as *mut std::ffi::c_void,
            &mut d_vw as *mut _ as *mut std::ffi::c_void,
            &mut k_dim as *mut _ as *mut std::ffi::c_void,
            &mut kv_rows as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&gemv_func, (kv_rows, 1, 1), (64, 1, 1), 0, stream, &mut args_v)?;
        }

        // 4. Gate projection: [16, 2048] x [2048] -> [16]
        let mut gate_rows: u32 = self.num_q_heads as u32;
        let mut d_gate = gate_out.inner_mut().cu_deviceptr();
        let mut d_gw = self.gate_proj.inner().cu_deviceptr();
        let mut args_g: [*mut std::ffi::c_void; 5] = [
            &mut d_gate as *mut _ as *mut std::ffi::c_void,
            &mut d_normed as *mut _ as *mut std::ffi::c_void,
            &mut d_gw as *mut _ as *mut std::ffi::c_void,
            &mut k_dim as *mut _ as *mut std::ffi::c_void,
            &mut gate_rows as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&gemv_func, (gate_rows, 1, 1), (64, 1, 1), 0, stream, &mut args_g)?;
        }

        Ok(())
    }

    pub fn forward_gating_and_o_proj(
        &mut self,
        gpu: &GpuContext,
        gate: &DeviceBuffer<half::f16>,
        out: &mut DeviceBuffer<half::f16>,
    ) -> Result<()> {
        let stream = gpu.compute_stream();

        // 1. Fused Sigmoid Gating + O-Projection GEMV with 128-bit Vectorized Loads
        // Reads context_buf, scales in registers by sigmoid(gate[head_idx]), and multiplies by o_weight
        // completely in registers without intermediate DRAM round-trip!
        let fused_o_func = self.gate_ptx.get_function("fused_sigmoid_gate_o_proj_vec8_fp16")?;
        let mut o_rows: u32 = self.hidden_size as u32;
        let mut k_dim: u32 = (self.num_q_heads * self.head_dim) as u32;
        let mut nq: u32 = self.num_q_heads as u32;
        let mut hd: u32 = self.head_dim as u32;

        let mut d_final = out.inner_mut().cu_deviceptr();
        let mut d_ctx = self.context_buf.inner().cu_deviceptr();
        let mut d_gate = gate.inner().cu_deviceptr();
        let mut d_ow = self.o_proj.inner().cu_deviceptr();

        let mut args_fused_o: [*mut std::ffi::c_void; 7] = [
            &mut d_final as *mut _ as *mut std::ffi::c_void,
            &mut d_ctx as *mut _ as *mut std::ffi::c_void,
            &mut d_gate as *mut _ as *mut std::ffi::c_void,
            &mut d_ow as *mut _ as *mut std::ffi::c_void,
            &mut k_dim as *mut _ as *mut std::ffi::c_void,
            &mut nq as *mut _ as *mut std::ffi::c_void,
            &mut hd as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&fused_o_func, (o_rows, 1, 1), (64, 1, 1), 0, stream, &mut args_fused_o)?;
        }

        Ok(())
    }

    pub fn forward_output(
        &mut self,
        gpu: &GpuContext,
        q: &DeviceBuffer<half::f16>,
        k: &DeviceBuffer<half::f16>,
        v: &DeviceBuffer<half::f16>,
        gate: &DeviceBuffer<half::f16>,
        out: &mut DeviceBuffer<half::f16>,
    ) -> Result<()> {
        let stream = gpu.compute_stream();

        // 1. Attention decode over KV cache
        let attn_func = self.attn_ptx.get_function("gqa_decode_attention_fp16")?;
        let mut d_ctx = self.context_buf.inner_mut().cu_deviceptr();
        let mut d_q = q.inner().cu_deviceptr();
        let mut d_k = k.inner().cu_deviceptr();
        let mut d_v = v.inner().cu_deviceptr();

        let mut slen: u32 = 1;
        let mut nq: u32 = self.num_q_heads as u32;
        let mut nkv: u32 = self.num_kv_heads as u32;
        let mut hd: u32 = self.head_dim as u32;

        let mut args_attn: [*mut std::ffi::c_void; 8] = [
            &mut d_ctx as *mut _ as *mut std::ffi::c_void,
            &mut d_q as *mut _ as *mut std::ffi::c_void,
            &mut d_k as *mut _ as *mut std::ffi::c_void,
            &mut d_v as *mut _ as *mut std::ffi::c_void,
            &mut slen as *mut _ as *mut std::ffi::c_void,
            &mut nq as *mut _ as *mut std::ffi::c_void,
            &mut nkv as *mut _ as *mut std::ffi::c_void,
            &mut hd as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&attn_func, (nq, 1, 1), (hd, 1, 1), 0, stream, &mut args_attn)?;
        }

        // 2. Sigmoid gating per head: context_buf *= sigmoid(gate)
        let gate_func = self.gate_ptx.get_function("apply_per_head_sigmoid_gate_fp16")?;
        let mut d_gate = gate.inner().cu_deviceptr();
        let mut args_gate: [*mut std::ffi::c_void; 4] = [
            &mut d_ctx as *mut _ as *mut std::ffi::c_void,
            &mut d_gate as *mut _ as *mut std::ffi::c_void,
            &mut nq as *mut _ as *mut std::ffi::c_void,
            &mut hd as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&gate_func, (nq, 1, 1), (hd, 1, 1), 0, stream, &mut args_gate)?;
        }

        // 3. o_proj GEMV: [2048, 2048] x [2048] -> [2048]
        let gemv_func = self.gemv_ptx.get_function("fused_gemv_fp16_fp32accum")?;
        let mut o_rows: u32 = self.hidden_size as u32;
        let mut k_dim: u32 = (self.num_q_heads * self.head_dim) as u32;
        let mut d_final = out.inner_mut().cu_deviceptr();
        let mut d_ow = self.o_proj.inner().cu_deviceptr();

        let mut args_o: [*mut std::ffi::c_void; 5] = [
            &mut d_final as *mut _ as *mut std::ffi::c_void,
            &mut d_ctx as *mut _ as *mut std::ffi::c_void,
            &mut d_ow as *mut _ as *mut std::ffi::c_void,
            &mut k_dim as *mut _ as *mut std::ffi::c_void,
            &mut o_rows as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&gemv_func, (o_rows, 1, 1), (64, 1, 1), 4, stream, &mut args_o)?;
        }

        Ok(())
    }
}

pub const FUSED_GATE_O_PROJ_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry apply_per_head_sigmoid_gate_fp16(
    .param .u64 attn_output,
    .param .u64 gate,
    .param .u32 num_heads,
    .param .u32 head_dim
) {
    ret;
}

.visible .entry fused_sigmoid_gate_o_proj_vec8_fp16(
    .param .u64 out_final,
    .param .u64 context_in,
    .param .u64 gate_in,
    .param .u64 o_weight,
    .param .u32 k_dim,
    .param .u32 num_heads,
    .param .u32 head_dim
) {
    .reg .pred %p0, %p1;
    .reg .b32 %row_idx, %v_idx, %kdim, %nh, %hd, %lane_id, %warp_id, %num_warps, %tid_val, %ntid_val;
    .reg .b32 %h_vec_idx, %h_vec_stride, %h_num_vecs, %s_addr;
    .reg .b64 %out_ptr, %ctx_ptr, %gate_ptr, %w_ptr, %row_offset, %elem_offset, %byte_offset;
    .reg .b32 %xv0, %xv1, %xv2, %xv3;
    .reg .b32 %wv0, %wv1, %wv2, %wv3;
    .reg .b32 %head_idx0;
    .reg .b64 %g_ptr;
    .reg .b16 %hg;
    .reg .f32 %fg, %f_exp, %sig0;
    .reg .f32 %f_val, %f_w, %f_sum, %shfl_val;

    ld.param.u32 %kdim, [k_dim];
    ld.param.u32 %nh, [num_heads];
    ld.param.u32 %hd, [head_dim];

    mov.u32 %row_idx, %ctaid.x;
    setp.ge.u32 %p0, %row_idx, %kdim;
    @%p0 bra EXIT;

    mov.u32 %tid_val, %tid.x;
    mov.u32 %ntid_val, %ntid.x;
    ld.param.u64 %ctx_ptr, [context_in];
    ld.param.u64 %gate_ptr, [gate_in];
    ld.param.u64 %w_ptr, [o_weight];

    mov.f32 %f_sum, 0.0;
    mul.wide.u32 %row_offset, %row_idx, %kdim;

    shr.u32 %h_num_vecs, %kdim, 3;
    mov.u32 %h_vec_stride, %ntid_val;
    mov.u32 %h_vec_idx, %tid_val;

VEC_LOOP:
    setp.ge.u32 %p1, %h_vec_idx, %h_num_vecs;
    @%p1 bra REDUCE;

    cvt.u64.u32 %byte_offset, %h_vec_idx;
    shl.b64 %byte_offset, %byte_offset, 4;

    add.u64 %elem_offset, %ctx_ptr, %byte_offset;
    ld.global.v4.u32 {%xv0, %xv1, %xv2, %xv3}, [%elem_offset];

    shl.b64 %elem_offset, %row_offset, 1;
    add.u64 %elem_offset, %elem_offset, %w_ptr;
    add.u64 %elem_offset, %elem_offset, %byte_offset;
    ld.global.v4.u32 {%wv0, %wv1, %wv2, %wv3}, [%elem_offset];

    // Compute head_idx: (h_vec_idx * 8) / 128 = h_vec_idx >> 4
    shr.u32 %head_idx0, %h_vec_idx, 4;

    // Read gate[head_idx]
    cvt.u64.u32 %elem_offset, %head_idx0;
    shl.b64 %elem_offset, %elem_offset, 1;
    add.u64 %g_ptr, %gate_ptr, %elem_offset;
    ld.global.u16 %hg, [%g_ptr];
    cvt.f32.f16 %fg, %hg;

    // Sigmoid(gate)
    neg.f32 %fg, %fg;
    mul.f32 %fg, %fg, 1.44269504;
    ex2.approx.ftz.f32 %f_exp, %fg;
    add.f32 %f_exp, %f_exp, 1.0;
    rcp.approx.ftz.f32 %sig0, %f_exp;

    // Pair 0 & 1
    cvt.f32.f16 %f_val, %xv0;
    mul.f32 %f_val, %f_val, %sig0;
    cvt.f32.f16 %f_w, %wv0;
    fma.rn.f32 %f_sum, %f_val, %f_w, %f_sum;

    shr.u32 %v_idx, %xv0, 16;
    cvt.f32.f16 %f_val, %v_idx;
    mul.f32 %f_val, %f_val, %sig0;
    shr.u32 %v_idx, %wv0, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %f_sum, %f_val, %f_w, %f_sum;

    // Pair 2 & 3
    cvt.f32.f16 %f_val, %xv1;
    mul.f32 %f_val, %f_val, %sig0;
    cvt.f32.f16 %f_w, %wv1;
    fma.rn.f32 %f_sum, %f_val, %f_w, %f_sum;

    shr.u32 %v_idx, %xv1, 16;
    cvt.f32.f16 %f_val, %v_idx;
    mul.f32 %f_val, %f_val, %sig0;
    shr.u32 %v_idx, %wv1, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %f_sum, %f_val, %f_w, %f_sum;

    // Pair 4 & 5
    cvt.f32.f16 %f_val, %xv2;
    mul.f32 %f_val, %f_val, %sig0;
    cvt.f32.f16 %f_w, %wv2;
    fma.rn.f32 %f_sum, %f_val, %f_w, %f_sum;

    shr.u32 %v_idx, %xv2, 16;
    cvt.f32.f16 %f_val, %v_idx;
    mul.f32 %f_val, %f_val, %sig0;
    shr.u32 %v_idx, %wv2, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %f_sum, %f_val, %f_w, %f_sum;

    // Pair 6 & 7
    cvt.f32.f16 %f_val, %xv3;
    mul.f32 %f_val, %f_val, %sig0;
    cvt.f32.f16 %f_w, %wv3;
    fma.rn.f32 %f_sum, %f_val, %f_w, %f_sum;

    shr.u32 %v_idx, %xv3, 16;
    cvt.f32.f16 %f_val, %v_idx;
    mul.f32 %f_val, %f_val, %sig0;
    shr.u32 %v_idx, %wv3, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %f_sum, %f_val, %f_w, %f_sum;

    add.u32 %h_vec_idx, %h_vec_idx, %h_vec_stride;
    bra VEC_LOOP;

REDUCE:
    shfl.sync.down.b32 %shfl_val, %f_sum, 16, 31, 0xffffffff;
    add.f32 %f_sum, %f_sum, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %f_sum, 8, 31, 0xffffffff;
    add.f32 %f_sum, %f_sum, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %f_sum, 4, 31, 0xffffffff;
    add.f32 %f_sum, %f_sum, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %f_sum, 2, 31, 0xffffffff;
    add.f32 %f_sum, %f_sum, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %f_sum, 1, 31, 0xffffffff;
    add.f32 %f_sum, %f_sum, %shfl_val;

    .shared .f32 s_sum[32];
    and.b32 %lane_id, %tid_val, 31;
    shr.u32 %warp_id, %tid_val, 5;
    setp.eq.u32 %p0, %lane_id, 0;
    mov.u32 %s_addr, s_sum;
    shl.b32 %h_vec_stride, %warp_id, 2;
    add.u32 %s_addr, %s_addr, %h_vec_stride;
    @%p0 st.shared.f32 [%s_addr], %f_sum;

    bar.sync 0;

    shr.u32 %num_warps, %ntid_val, 5;
    setp.lt.u32 %p0, %tid_val, %num_warps;
    mov.u32 %s_addr, s_sum;
    shl.b32 %h_vec_stride, %tid_val, 2;
    add.u32 %s_addr, %s_addr, %h_vec_stride;
    @%p0 ld.shared.f32 %f_sum, [%s_addr];
    @!%p0 mov.f32 %f_sum, 0.0;

    shfl.sync.down.b32 %shfl_val, %f_sum, 16, 31, 0xffffffff;
    add.f32 %f_sum, %f_sum, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %f_sum, 8, 31, 0xffffffff;
    add.f32 %f_sum, %f_sum, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %f_sum, 4, 31, 0xffffffff;
    add.f32 %f_sum, %f_sum, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %f_sum, 2, 31, 0xffffffff;
    add.f32 %f_sum, %f_sum, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %f_sum, 1, 31, 0xffffffff;
    add.f32 %f_sum, %f_sum, %shfl_val;

    setp.eq.u32 %p0, %tid_val, 0;
    @!%p0 bra EXIT;

    ld.param.u64 %out_ptr, [out_final];
    cvt.u64.u32 %row_offset, %row_idx;
    shl.b64 %row_offset, %row_offset, 1;
    add.u64 %out_ptr, %out_ptr, %row_offset;
    cvt.rn.f16.f32 %hg, %f_sum;
    st.global.u16 [%out_ptr], %hg;

EXIT:
    ret;
}
"#;

pub const SIGMOID_GATE_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry apply_per_head_sigmoid_gate_fp16(
    .param .u64 attn_output,
    .param .u64 gate,
    .param .u32 num_heads,
    .param .u32 head_dim
) {
    .reg .pred %p0;
    .reg .b32 %head_idx, %dim_idx, %nh, %hd, %hstride;
    .reg .b64 %out_ptr, %gate_ptr, %gate_offset, %elem_offset;
    .reg .b16 %h_gate, %h_val;
    .reg .f32 %f_gate, %f_val, %f_sig, %f_exp;

    ld.param.u32 %nh, [num_heads];
    ld.param.u32 %hd, [head_dim];

    mov.u32 %head_idx, %ctaid.x;
    mov.u32 %dim_idx, %tid.x;

    setp.ge.u32 %p0, %head_idx, %nh;
    @%p0 bra EXIT;

    ld.param.u64 %gate_ptr, [gate];
    ld.param.u64 %out_ptr, [attn_output];

    cvt.u64.u32 %gate_offset, %head_idx;
    shl.b64 %gate_offset, %gate_offset, 1;
    add.u64 %gate_ptr, %gate_ptr, %gate_offset;
    ld.global.u16 %h_gate, [%gate_ptr];

    cvt.f32.f16 %f_gate, %h_gate;
    neg.f32 %f_gate, %f_gate;
    mul.f32 %f_gate, %f_gate, 1.44269504;
    ex2.approx.ftz.f32 %f_exp, %f_gate;
    add.f32 %f_exp, %f_exp, 1.0;
    rcp.approx.ftz.f32 %f_sig, %f_exp;

DIM_LOOP:
    setp.ge.u32 %p0, %dim_idx, %hd;
    @%p0 bra EXIT;

    mul.wide.u32 %elem_offset, %head_idx, %hd;
    cvt.u64.u32 %gate_offset, %dim_idx;
    add.u64 %elem_offset, %elem_offset, %gate_offset;
    shl.b64 %elem_offset, %elem_offset, 1;

    add.u64 %elem_offset, %out_ptr, %elem_offset;
    ld.global.u16 %h_val, [%elem_offset];

    cvt.f32.f16 %f_val, %h_val;
    mul.f32 %f_val, %f_val, %f_sig;
    cvt.rn.f16.f32 %h_val, %f_val;

    st.global.u16 [%elem_offset], %h_val;

    mov.u32 %hstride, %ntid.x;
    add.u32 %dim_idx, %dim_idx, %hstride;
    bra DIM_LOOP;

EXIT:
    ret;
}
"#;
