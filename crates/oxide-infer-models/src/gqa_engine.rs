use anyhow::Result;
use oxide_infer_core::{DeviceBuffer, GpuContext};
use oxide_infer_kernels::PtxModule;
use cuda_core::launch_kernel_on_stream;

const ATTN_SCORES_AND_V_ACCUM_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry gqa_softmax_context_fp16(
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
    .reg .b32 %head_idx, %dim_idx, %t_idx, %slen, %nq, %nkv, %hd, %kv_head_idx, %hstride, %kv_idx_32, %tid_val;
    .reg .b32 %s_offset, %s_addr, %lane_id_r, %warp_id_r, %wctr, %nw;
    .reg .b64 %ctx_ptr, %q_ptr, %k_ptr, %v_ptr, %offset, %q_offset;
    .reg .b16 %hq, %hk, %hv, %hctx;
    .reg .f32 %fq, %fk, %fv, %fdot, %fscore, %fmax, %fsum, %fweight, %fscale, %shfl_val, %faccum;

    ld.param.u32 %slen, [seq_len];
    ld.param.u32 %nq, [num_q_heads];
    ld.param.u32 %nkv, [num_kv_heads];
    ld.param.u32 %hd, [head_dim];

    mov.u32 %head_idx, %ctaid.x;
    setp.ge.u32 %p0, %head_idx, %nq;
    @%p0 bra EXIT;

    shr.u32 %kv_head_idx, %head_idx, 2;

    ld.param.u64 %q_ptr, [q_in];
    ld.param.u64 %k_ptr, [k_cache];
    ld.param.u64 %v_ptr, [v_cache];
    ld.param.u64 %ctx_ptr, [context_out];

    mul.wide.u32 %q_offset, %head_idx, %hd;
    shl.b64 %q_offset, %q_offset, 1;
    add.u64 %q_ptr, %q_ptr, %q_offset;

    mov.f32 %fscale, 0.0883883476;

    .shared .f32 s_scores[512];
    .shared .f32 s_max;
    .shared .f32 s_denom;
    .shared .f32 s_warp_dot[8];

    mov.u32 %t_idx, 0;
TOKEN_LOOP:
    setp.ge.u32 %p0, %t_idx, %slen;
    @%p0 bra COMPUTE_SOFTMAX;

    mov.u32 %dim_idx, %tid.x;
    mov.f32 %fdot, 0.0;

DOT_LOOP:
    setp.ge.u32 %p1, %dim_idx, %hd;
    @%p1 bra DOT_REDUCE;

    cvt.u64.u32 %offset, %dim_idx;
    shl.b64 %offset, %offset, 1;
    add.u64 %offset, %q_ptr, %offset;
    ld.global.u16 %hq, [%offset];

    mul.lo.u32 %kv_idx_32, %t_idx, %nkv;
    add.u32 %kv_idx_32, %kv_idx_32, %kv_head_idx;
    mul.lo.u32 %kv_idx_32, %kv_idx_32, %hd;
    add.u32 %kv_idx_32, %kv_idx_32, %dim_idx;
    cvt.u64.u32 %offset, %kv_idx_32;
    shl.b64 %offset, %offset, 1;
    add.u64 %offset, %k_ptr, %offset;
    ld.global.u16 %hk, [%offset];

    cvt.f32.f16 %fq, %hq;
    cvt.f32.f16 %fk, %hk;
    fma.rn.f32 %fdot, %fq, %fk, %fdot;

    mov.u32 %hstride, %ntid.x;
    add.u32 %dim_idx, %dim_idx, %hstride;
    bra DOT_LOOP;

DOT_REDUCE:
    shfl.sync.down.b32 %shfl_val, %fdot, 16, 31, 0xffffffff;
    add.f32 %fdot, %fdot, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %fdot, 8, 31, 0xffffffff;
    add.f32 %fdot, %fdot, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %fdot, 4, 31, 0xffffffff;
    add.f32 %fdot, %fdot, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %fdot, 2, 31, 0xffffffff;
    add.f32 %fdot, %fdot, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %fdot, 1, 31, 0xffffffff;
    add.f32 %fdot, %fdot, %shfl_val;

    // Cross-warp reduction: lane 0 of each warp publishes its partial, then
    // thread 0 sums all warp partials.
    mov.u32 %tid_val, %tid.x;
    and.b32 %lane_id_r, %tid_val, 31;
    shr.u32 %warp_id_r, %tid_val, 5;
    setp.eq.u32 %p1, %lane_id_r, 0;
    mov.u32 %s_addr, s_warp_dot;
    shl.b32 %s_offset, %warp_id_r, 2;
    add.u32 %s_addr, %s_addr, %s_offset;
    @%p1 st.shared.f32 [%s_addr], %fdot;
    bar.sync 0;

    mov.u32 %tid_val, %tid.x;
    setp.eq.u32 %p1, %tid_val, 0;
    @!%p1 bra DOT_DONE;
    mov.f32 %fdot, 0.0;
    mov.u32 %nw, %ntid.x;
    shr.u32 %nw, %nw, 5;
    mov.u32 %wctr, 0;
DOT_WARP_SUM:
    setp.ge.u32 %p1, %wctr, %nw;
    @%p1 bra DOT_HAVE;
    mov.u32 %s_addr, s_warp_dot;
    shl.b32 %s_offset, %wctr, 2;
    add.u32 %s_addr, %s_addr, %s_offset;
    ld.shared.f32 %shfl_val, [%s_addr];
    add.f32 %fdot, %fdot, %shfl_val;
    add.u32 %wctr, %wctr, 1;
    bra DOT_WARP_SUM;
DOT_HAVE:
    mul.f32 %fscore, %fdot, %fscale;
    shl.b32 %s_offset, %t_idx, 2;
    mov.u32 %s_addr, s_scores;
    add.u32 %s_addr, %s_addr, %s_offset;
    st.shared.f32 [%s_addr], %fscore;
DOT_DONE:
    bar.sync 0;

    add.u32 %t_idx, %t_idx, 1;
    bra TOKEN_LOOP;

COMPUTE_SOFTMAX:
    bar.sync 0;

    mov.u32 %tid_val, %tid.x;
    setp.eq.u32 %p0, %tid_val, 0;
    @!%p0 bra WAIT_SOFTMAX;

    mov.u32 %s_addr, s_scores;
    ld.shared.f32 %fmax, [%s_addr];
    mov.u32 %t_idx, 1;
MAX_LOOP:
    setp.ge.u32 %p1, %t_idx, %slen;
    @%p1 bra CALC_EXP_SUM;

    shl.b32 %s_offset, %t_idx, 2;
    mov.u32 %s_addr, s_scores;
    add.u32 %s_addr, %s_addr, %s_offset;
    ld.shared.f32 %fscore, [%s_addr];
    max.f32 %fmax, %fmax, %fscore;
    add.u32 %t_idx, %t_idx, 1;
    bra MAX_LOOP;

CALC_EXP_SUM:
    mov.u32 %s_addr, s_max;
    st.shared.f32 [%s_addr], %fmax;
    mov.f32 %fsum, 0.0;
    mov.u32 %t_idx, 0;
EXP_LOOP:
    setp.ge.u32 %p1, %t_idx, %slen;
    @%p1 bra NORM_SOFTMAX;

    shl.b32 %s_offset, %t_idx, 2;
    mov.u32 %s_addr, s_scores;
    add.u32 %s_addr, %s_addr, %s_offset;
    ld.shared.f32 %fscore, [%s_addr];
    sub.f32 %fscore, %fscore, %fmax;
    mul.f32 %fscore, %fscore, 1.44269504;
    ex2.approx.ftz.f32 %fweight, %fscore;
    st.shared.f32 [%s_addr], %fweight;
    add.f32 %fsum, %fsum, %fweight;

    add.u32 %t_idx, %t_idx, 1;
    bra EXP_LOOP;

NORM_SOFTMAX:
    rcp.approx.ftz.f32 %fsum, %fsum;
    mov.u32 %s_addr, s_denom;
    st.shared.f32 [%s_addr], %fsum;

WAIT_SOFTMAX:
    bar.sync 0;

    mov.u32 %dim_idx, %tid.x;
V_DIM_LOOP:
    setp.ge.u32 %p0, %dim_idx, %hd;
    @%p0 bra EXIT;

    mov.f32 %faccum, 0.0;
    mov.u32 %s_addr, s_denom;
    ld.shared.f32 %fsum, [%s_addr];

    mov.u32 %t_idx, 0;
V_TOK_LOOP:
    setp.ge.u32 %p1, %t_idx, %slen;
    @%p1 bra WRITE_CTX;

    shl.b32 %s_offset, %t_idx, 2;
    mov.u32 %s_addr, s_scores;
    add.u32 %s_addr, %s_addr, %s_offset;
    ld.shared.f32 %fweight, [%s_addr];
    mul.f32 %fweight, %fweight, %fsum;

    mul.lo.u32 %kv_idx_32, %t_idx, %nkv;
    add.u32 %kv_idx_32, %kv_idx_32, %kv_head_idx;
    mul.lo.u32 %kv_idx_32, %kv_idx_32, %hd;
    add.u32 %kv_idx_32, %kv_idx_32, %dim_idx;
    cvt.u64.u32 %offset, %kv_idx_32;
    shl.b64 %offset, %offset, 1;
    add.u64 %offset, %v_ptr, %offset;
    ld.global.u16 %hv, [%offset];

    cvt.f32.f16 %fv, %hv;
    fma.rn.f32 %faccum, %fweight, %fv, %faccum;

    add.u32 %t_idx, %t_idx, 1;
    bra V_TOK_LOOP;

WRITE_CTX:
    mul.wide.u32 %offset, %head_idx, %hd;
    cvt.u64.u32 %q_offset, %dim_idx;
    add.u64 %offset, %offset, %q_offset;
    shl.b64 %offset, %offset, 1;
    add.u64 %offset, %ctx_ptr, %offset;

    cvt.rn.f16.f32 %hctx, %faccum;
    st.global.u16 [%offset], %hctx;

    mov.u32 %hstride, %ntid.x;
    add.u32 %dim_idx, %dim_idx, %hstride;
    bra V_DIM_LOOP;

EXIT:
    ret;
}
"#;

pub struct GqaAttentionEngine {
    ptx: PtxModule,
}

impl GqaAttentionEngine {
    pub fn new(gpu: &GpuContext) -> Result<Self> {
        let ptx = PtxModule::load_from_ptx(gpu, ATTN_SCORES_AND_V_ACCUM_PTX)?;
        Ok(Self { ptx })
    }

    pub fn compute_attention(
        &self,
        gpu: &GpuContext,
        context_out: &mut DeviceBuffer<half::f16>,
        q_in: &DeviceBuffer<half::f16>,
        k_cache: &DeviceBuffer<half::f16>,
        v_cache: &DeviceBuffer<half::f16>,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        let func = self.ptx.get_function("gqa_softmax_context_fp16")?;
        let stream = gpu.compute_stream();

        let mut d_ctx = context_out.inner_mut().cu_deviceptr();
        let mut d_q = q_in.inner().cu_deviceptr();
        let mut d_k = k_cache.inner().cu_deviceptr();
        let mut d_v = v_cache.inner().cu_deviceptr();

        let mut slen = seq_len as u32;
        let mut nq = num_q_heads as u32;
        let mut nkv = num_kv_heads as u32;
        let mut hd = head_dim as u32;

        let mut args: [*mut std::ffi::c_void; 8] = [
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
            launch_kernel_on_stream(
                &func,
                (nq, 1, 1),
                (128, 1, 1),
                0,
                stream,
                &mut args,
            )?;
        }
        Ok(())
    }
}
