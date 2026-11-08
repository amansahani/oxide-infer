//! Batched kernels for the attention decode / prefill path.
//!
//! All kernels use `ctaid.y` as the batch index and assume the vector data is
//! laid out as `[batch, ...]`. Per-sequence KV caches are addressed through
//! device-side pointer tables so the caches can stay independently allocated.

/// Per-head RMSNorm applied to a `[batch, num_heads, head_dim]` tensor.
pub const BATCHED_QK_RMSNORM_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry batched_qk_rmsnorm_fp16(
    .param .u64 x_io,
    .param .u64 weight,
    .param .f32 eps,
    .param .u32 num_heads,
    .param .u32 head_dim,
    .param .u32 batch_size
) {
    .reg .pred %p0, %p1;
    .reg .b32 %head_idx, %dim_idx, %nh, %hd, %b_idx, %b_size, %tmp32;
    .reg .b64 %x_ptr, %w_ptr, %offset, %head_offset, %batch_offset;
    .reg .b16 %hx, %hw;
    .reg .f32 %fx, %fw, %fsum, %fmean, %frsqrt, %shfl_val;

    ld.param.u32 %nh, [num_heads];
    ld.param.u32 %hd, [head_dim];
    ld.param.u32 %b_size, [batch_size];

    mov.u32 %head_idx, %ctaid.x;
    setp.ge.u32 %p0, %head_idx, %nh;
    @%p0 bra EXIT;

    mov.u32 %b_idx, %ctaid.y;
    setp.ge.u32 %p0, %b_idx, %b_size;
    @%p0 bra EXIT;

    mov.u32 %dim_idx, %tid.x;
    ld.param.u64 %x_ptr, [x_io];
    ld.param.u64 %w_ptr, [weight];

    mul.lo.u32 %tmp32, %b_idx, %nh;
    mul.lo.u32 %tmp32, %tmp32, %hd;
    cvt.u64.u32 %batch_offset, %tmp32;
    shl.b64 %batch_offset, %batch_offset, 1;
    add.u64 %x_ptr, %x_ptr, %batch_offset;

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

    cvt.rn.f32.u32 %fmean, %hd;
    div.approx.ftz.f32 %fmean, %fsum, %fmean;
    .reg .f32 %feps;
    ld.param.f32 %feps, [eps];
    add.f32 %fmean, %fmean, %feps;
    rsqrt.approx.ftz.f32 %frsqrt, %fmean;
    shfl.sync.idx.b32 %frsqrt, %frsqrt, 0, 31, 0xffffffff;

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

/// Partial RoPE applied to a `[batch, num_heads, head_dim]` tensor with a
/// per-batch position id.
pub const BATCHED_ROPE_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry batched_rope_fp16(
    .param .u64 x_io,
    .param .u64 cos_table,
    .param .u64 sin_table,
    .param .u64 pos_ids,
    .param .u32 num_heads,
    .param .u32 head_dim,
    .param .u32 batch_size
) {
    .reg .pred %p0;
    .reg .b32 %head_idx, %dim_idx, %nh, %hd, %b_idx, %b_size, %pid, %half_hd, %other_dim_idx, %tmp32;
    .reg .b64 %x_ptr, %cos_ptr, %sin_ptr, %pos_offset, %head_offset, %dim_offset, %other_offset, %batch_offset, %offset, %pos_ptr;
    .reg .b16 %hx, %h_other, %hcos, %hsin, %hout;
    .reg .f32 %fx, %f_other, %fcos, %fsin, %fres;

    ld.param.u32 %nh, [num_heads];
    ld.param.u32 %hd, [head_dim];
    ld.param.u32 %b_size, [batch_size];

    mov.u32 %head_idx, %ctaid.x;
    setp.ge.u32 %p0, %head_idx, %nh;
    @%p0 bra EXIT;

    mov.u32 %b_idx, %ctaid.y;
    setp.ge.u32 %p0, %b_idx, %b_size;
    @%p0 bra EXIT;

    mov.u32 %dim_idx, %tid.x;
    setp.ge.u32 %p0, %dim_idx, %hd;
    @%p0 bra EXIT;

    ld.param.u64 %x_ptr, [x_io];
    ld.param.u64 %cos_ptr, [cos_table];
    ld.param.u64 %sin_ptr, [sin_table];
    ld.param.u64 %pos_ptr, [pos_ids];

    mul.lo.u32 %tmp32, %b_idx, %nh;
    mul.lo.u32 %tmp32, %tmp32, %hd;
    cvt.u64.u32 %batch_offset, %tmp32;
    shl.b64 %batch_offset, %batch_offset, 1;
    add.u64 %x_ptr, %x_ptr, %batch_offset;

    mul.wide.u32 %offset, %b_idx, 4;
    add.u64 %offset, %pos_ptr, %offset;
    ld.global.u32 %pid, [%offset];

    mul.wide.u32 %pos_offset, %pid, %hd;
    cvt.u64.u32 %dim_offset, %dim_idx;
    add.u64 %pos_offset, %pos_offset, %dim_offset;
    shl.b64 %pos_offset, %pos_offset, 1;

    add.u64 %cos_ptr, %cos_ptr, %pos_offset;
    ld.global.u16 %hcos, [%cos_ptr];

    add.u64 %sin_ptr, %sin_ptr, %pos_offset;
    ld.global.u16 %hsin, [%sin_ptr];

    mul.wide.u32 %head_offset, %head_idx, %hd;
    add.u64 %head_offset, %head_offset, %dim_offset;
    shl.b64 %head_offset, %head_offset, 1;
    add.u64 %head_offset, %x_ptr, %head_offset;
    ld.global.u16 %hx, [%head_offset];

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

    @%p0 neg.f32 %f_other, %f_other;

    mul.f32 %fres, %fx, %fcos;
    fma.rn.f32 %fres, %f_other, %fsin, %fres;
    cvt.rn.f16.f32 %hout, %fres;

    st.global.u16 [%head_offset], %hout;

EXIT:
    ret;
}
"#;

/// Copy the freshly projected K or V slice of every batch element into its
/// per-sequence KV cache.
///
/// * `dst_ptrs` is a device array of `batch_size` per-sequence cache pointers.
/// * `src` is a `[batch, kv_dim]` tensor.
/// * `pos_ids` is a device array of `batch_size` positions.
pub const BATCHED_KV_COPY_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry batched_kv_copy_fp16(
    .param .u64 dst_ptrs,
    .param .u64 src,
    .param .u64 pos_ids,
    .param .u32 kv_dim,
    .param .u32 batch_size
) {
    .reg .pred %p0;
    .reg .b32 %b_idx, %b_size, %kv_dim, %idx, %pid, %stride;
    .reg .b64 %dst_ptr, %src_ptr, %offset, %pos_offset, %dst_tbl, %src_base, %pos_ptr;
    .reg .b32 %v0, %v1, %v2, %v3;

    ld.param.u32 %kv_dim, [kv_dim];
    ld.param.u32 %b_size, [batch_size];

    mov.u32 %b_idx, %ctaid.x;
    setp.ge.u32 %p0, %b_idx, %b_size;
    @%p0 bra EXIT;

    mov.u32 %idx, %tid.x;
    mul.lo.u32 %idx, %idx, 8;
    setp.ge.u32 %p0, %idx, %kv_dim;
    @%p0 bra EXIT;

    ld.param.u64 %dst_tbl, [dst_ptrs];
    ld.param.u64 %src_base, [src];
    ld.param.u64 %pos_ptr, [pos_ids];

    mul.wide.u32 %offset, %b_idx, 8;
    add.u64 %offset, %dst_tbl, %offset;
    ld.global.u64 %dst_ptr, [%offset];

    mul.wide.u32 %offset, %b_idx, %kv_dim;
    shl.b64 %offset, %offset, 1;
    add.u64 %src_ptr, %src_base, %offset;

    mul.wide.u32 %offset, %b_idx, 4;
    add.u64 %offset, %pos_ptr, %offset;
    ld.global.u32 %pid, [%offset];

    mul.wide.u32 %pos_offset, %pid, %kv_dim;
    shl.b64 %pos_offset, %pos_offset, 1;
    add.u64 %dst_ptr, %dst_ptr, %pos_offset;

    cvt.u64.u32 %offset, %idx;
    shl.b64 %offset, %offset, 1;
    add.u64 %src_ptr, %src_ptr, %offset;
    add.u64 %dst_ptr, %dst_ptr, %offset;

    ld.global.v4.u32 {%v0, %v1, %v2, %v3}, [%src_ptr];
    st.global.v4.u32 [%dst_ptr], {%v0, %v1, %v2, %v3};

EXIT:
    ret;
}
"#;

/// Batched GQA decode attention: one block per (query head, batch element).
/// The K/V caches are addressed through per-batch pointer tables and the
/// sequence length is read from a per-batch device array.
pub const BATCHED_GQA_ATTENTION_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry batched_gqa_attention_fp16(
    .param .u64 context_out,
    .param .u64 q_in,
    .param .u64 k_ptrs,
    .param .u64 v_ptrs,
    .param .u64 seq_lens,
    .param .u32 num_q_heads,
    .param .u32 num_kv_heads,
    .param .u32 head_dim,
    .param .u32 batch_size
) {
    .reg .pred %p0, %p1;
    .reg .b32 %head_idx, %dim_idx, %t_idx, %slen, %nq, %nkv, %hd, %b_idx, %b_size, %kv_head_idx, %hstride, %kv_idx_32, %tid_val, %tmp32;
    .reg .b32 %s_offset2, %s_addrr, %lane_id_r, %warp_id_r, %wctr, %nw;
    .reg .b32 %s_offset, %s_addr;
    .reg .b64 %ctx_ptr, %q_ptr, %k_ptr, %v_ptr, %offset, %q_offset, %batch_offset, %k_tbl, %v_tbl, %sl_tbl;
    .reg .b16 %hq, %hk, %hv, %hctx;
    .reg .f32 %fq, %fk, %fv, %fdot, %fscore, %fmax, %fsum, %fweight, %fscale, %shfl_val, %faccum;

    ld.param.u32 %nq, [num_q_heads];
    ld.param.u32 %nkv, [num_kv_heads];
    ld.param.u32 %hd, [head_dim];
    ld.param.u32 %b_size, [batch_size];

    mov.u32 %head_idx, %ctaid.x;
    setp.ge.u32 %p0, %head_idx, %nq;
    @%p0 bra EXIT;

    mov.u32 %b_idx, %ctaid.y;
    setp.ge.u32 %p0, %b_idx, %b_size;
    @%p0 bra EXIT;

    shr.u32 %kv_head_idx, %head_idx, 2;

    ld.param.u64 %q_ptr, [q_in];
    ld.param.u64 %ctx_ptr, [context_out];
    ld.param.u64 %k_tbl, [k_ptrs];
    ld.param.u64 %v_tbl, [v_ptrs];

    mul.lo.u32 %tmp32, %b_idx, %nq;
    mul.lo.u32 %tmp32, %tmp32, %hd;
    cvt.u64.u32 %batch_offset, %tmp32;
    shl.b64 %batch_offset, %batch_offset, 1;
    add.u64 %q_ptr, %q_ptr, %batch_offset;
    add.u64 %ctx_ptr, %ctx_ptr, %batch_offset;

    mul.wide.u32 %q_offset, %head_idx, %hd;
    shl.b64 %q_offset, %q_offset, 1;
    add.u64 %q_ptr, %q_ptr, %q_offset;

    mul.wide.u32 %offset, %b_idx, 8;
    add.u64 %offset, %k_tbl, %offset;
    ld.global.u64 %k_ptr, [%offset];

    mul.wide.u32 %offset, %b_idx, 8;
    add.u64 %offset, %v_tbl, %offset;
    ld.global.u64 %v_ptr, [%offset];

    ld.param.u64 %sl_tbl, [seq_lens];
    mul.wide.u32 %offset, %b_idx, 4;
    add.u64 %offset, %sl_tbl, %offset;
    ld.global.u32 %slen, [%offset];

    mov.f32 %fscale, 0.0883883476;

    .shared .f32 s_scores[1024];
    .shared .f32 s_warp_dot[8];
    .shared .f32 s_max;
    .shared .f32 s_denom;

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
    // thread 0 sums all warp partials before storing the score.
    mov.u32 %tid_val, %tid.x;
    and.b32 %lane_id_r, %tid_val, 31;
    shr.u32 %warp_id_r, %tid_val, 5;
    setp.eq.u32 %p1, %lane_id_r, 0;
    mov.u32 %s_addr, s_warp_dot;
    shl.b32 %s_offset, %warp_id_r, 2;
    add.u32 %s_addr, %s_addr, %s_offset;
    @%p1 st.shared.f32 [%s_addr], %fdot;
    bar.sync 0;

    setp.eq.u32 %p1, %tid_val, 0;
    @!%p1 bra DOT_DONE;
    mov.f32 %fdot, 0.0;
    mov.u32 %nw, %ntid.x;
    shr.u32 %nw, %nw, 5;
    mov.u32 %wctr, 0;
WB_LOOP:
    setp.ge.u32 %p1, %wctr, %nw;
    @%p1 bra W_HAVE;
    mov.u32 %s_addr, s_warp_dot;
    shl.b32 %s_offset, %wctr, 2;
    add.u32 %s_addr, %s_addr, %s_offset;
    ld.shared.f32 %shfl_val, [%s_addr];
    add.f32 %fdot, %fdot, %shfl_val;
    add.u32 %wctr, %wctr, 1;
    bra WB_LOOP;
W_HAVE:
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

/// Apply a per-head sigmoid gate to a `[batch, num_heads, head_dim]` tensor.
pub const BATCHED_SIGMOID_GATE_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry batched_sigmoid_gate_fp16(
    .param .u64 ctx_io,
    .param .u64 gate_in,
    .param .u32 num_heads,
    .param .u32 head_dim,
    .param .u32 batch_size
) {
    .reg .pred %p0;
    .reg .b32 %head_idx, %dim_idx, %nh, %hd, %b_idx, %b_size, %tmp32, %stride;
    .reg .b64 %ctx_ptr, %gate_ptr, %offset, %batch_offset;
    .reg .b16 %hctx, %hgate;
    .reg .f32 %fctx, %fgate, %fneg, %fexp, %fsig;

    ld.param.u32 %nh, [num_heads];
    ld.param.u32 %hd, [head_dim];
    ld.param.u32 %b_size, [batch_size];

    mov.u32 %head_idx, %ctaid.x;
    setp.ge.u32 %p0, %head_idx, %nh;
    @%p0 bra EXIT;

    mov.u32 %b_idx, %ctaid.y;
    setp.ge.u32 %p0, %b_idx, %b_size;
    @%p0 bra EXIT;

    ld.param.u64 %ctx_ptr, [ctx_io];
    ld.param.u64 %gate_ptr, [gate_in];

    mul.lo.u32 %tmp32, %b_idx, %nh;
    mul.lo.u32 %tmp32, %tmp32, %hd;
    cvt.u64.u32 %batch_offset, %tmp32;
    shl.b64 %batch_offset, %batch_offset, 1;
    add.u64 %ctx_ptr, %ctx_ptr, %batch_offset;

    mul.lo.u32 %tmp32, %b_idx, %nh;
    add.u32 %tmp32, %tmp32, %head_idx;
    mul.wide.u32 %offset, %tmp32, 2;
    add.u64 %gate_ptr, %gate_ptr, %offset;

    ld.global.u16 %hgate, [%gate_ptr];
    cvt.f32.f16 %fgate, %hgate;
    neg.f32 %fneg, %fgate;
    mul.f32 %fneg, %fneg, 1.44269504;
    ex2.approx.ftz.f32 %fexp, %fneg;
    add.f32 %fexp, %fexp, 1.0;
    rcp.approx.ftz.f32 %fsig, %fexp;

    mul.wide.u32 %offset, %head_idx, %hd;
    shl.b64 %offset, %offset, 1;
    add.u64 %ctx_ptr, %ctx_ptr, %offset;

    mov.u32 %dim_idx, %tid.x;
    mov.u32 %stride, %ntid.x;

GATE_LOOP:
    setp.ge.u32 %p0, %dim_idx, %hd;
    @%p0 bra EXIT;

    cvt.u64.u32 %offset, %dim_idx;
    shl.b64 %offset, %offset, 1;
    add.u64 %offset, %ctx_ptr, %offset;
    ld.global.u16 %hctx, [%offset];
    cvt.f32.f16 %fctx, %hctx;
    mul.f32 %fctx, %fctx, %fsig;
    cvt.rn.f16.f32 %hctx, %fctx;
    st.global.u16 [%offset], %hctx;

    add.u32 %dim_idx, %dim_idx, %stride;
    bra GATE_LOOP;

EXIT:
    ret;
}
"#;

/// Batched token embedding lookup: `out[b] = table[token_ids[b]]`.
pub const BATCHED_EMBED_LOOKUP_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry batched_embed_lookup_fp16(
    .param .u64 out,
    .param .u64 table,
    .param .u64 token_ids,
    .param .u32 hidden_size,
    .param .u32 batch_size
) {
    .reg .pred %p0;
    .reg .b32 %b_idx, %b_size, %h_size, %idx, %tok;
    .reg .b64 %out_ptr, %table_ptr, %offset, %batch_offset, %tok_tbl;
    .reg .b32 %v0, %v1, %v2, %v3;

    ld.param.u32 %h_size, [hidden_size];
    ld.param.u32 %b_size, [batch_size];

    mov.u32 %b_idx, %ctaid.x;
    setp.ge.u32 %p0, %b_idx, %b_size;
    @%p0 bra EXIT;

    mov.u32 %idx, %tid.x;
    mul.lo.u32 %idx, %idx, 8;
    setp.ge.u32 %p0, %idx, %h_size;
    @%p0 bra EXIT;

    ld.param.u64 %out_ptr, [out];
    ld.param.u64 %table_ptr, [table];
    ld.param.u64 %tok_tbl, [token_ids];

    mul.wide.u32 %offset, %b_idx, 4;
    add.u64 %offset, %tok_tbl, %offset;
    ld.global.u32 %tok, [%offset];

    mul.wide.u32 %offset, %tok, %h_size;
    shl.b64 %offset, %offset, 1;
    add.u64 %table_ptr, %table_ptr, %offset;

    mul.wide.u32 %batch_offset, %b_idx, %h_size;
    shl.b64 %batch_offset, %batch_offset, 1;
    add.u64 %out_ptr, %out_ptr, %batch_offset;

    cvt.u64.u32 %offset, %idx;
    shl.b64 %offset, %offset, 1;
    add.u64 %table_ptr, %table_ptr, %offset;
    add.u64 %out_ptr, %out_ptr, %offset;

    ld.global.v4.u32 {%v0, %v1, %v2, %v3}, [%table_ptr];
    st.global.v4.u32 [%out_ptr], {%v0, %v1, %v2, %v3};

EXIT:
    ret;
}
"#;

/// Batched causal Conv1D + gating using a per-sequence state pointer table.
///
/// Identical math to `batched_fused_causal_conv1d_gating_fp16`, but the state
/// buffer of every request is addressed independently, so no state packing or
/// unpacking is required.
pub const BATCHED_CONV_PTR_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry batched_fused_causal_conv1d_gating_ptr_fp16(
    .param .u64 y_out,
    .param .u64 state_ptrs,
    .param .u64 bcx_in,
    .param .u64 conv_weight,
    .param .u32 hidden_size,
    .param .u32 k_len,
    .param .u32 batch_size
) {
    .reg .pred %p0, %p1, %p_loop;
    .reg .b32 %idx, %hsize, %klen, %b_idx, %stride, %c_offset, %x_offset, %batch;
    .reg .b32 %k, %k_minus_1, %row_offset_32;
    .reg .b64 %y_ptr, %cache_ptr, %bcx_ptr, %w_ptr;
    .reg .b64 %c_ptr, %x_ptr, %b_ptr, %offset, %c_state_ptr, %c_w_ptr, %batch_off_bcx, %batch_off_y, %state_tbl;
    .reg .b16 %hb, %hx, %hc, %hin, %hw, %hs;
    .reg .f32 %fb, %fx, %fin, %fc, %fw, %fs, %fsum;

    ld.param.u32 %hsize, [hidden_size];
    ld.param.u32 %klen, [k_len];
    ld.param.u32 %batch, [batch_size];

    mov.u32 %b_idx, %ctaid.y;
    setp.ge.u32 %p0, %b_idx, %batch;
    @%p0 bra EXIT;

    mov.u32 %idx, %tid.x;
    .reg .b32 %bx;
    mov.u32 %bx, %ctaid.x;
    mov.u32 %stride, %ntid.x;
    mad.lo.u32 %idx, %bx, %stride, %idx;

    setp.ge.u32 %p0, %idx, %hsize;
    @%p0 bra EXIT;

    ld.param.u64 %bcx_ptr, [bcx_in];
    ld.param.u64 %w_ptr, [conv_weight];
    ld.param.u64 %y_ptr, [y_out];

    mul.wide.u32 %batch_off_bcx, %b_idx, %hsize;
    mul.lo.u64 %batch_off_bcx, %batch_off_bcx, 3;
    shl.b64 %batch_off_bcx, %batch_off_bcx, 1;
    add.u64 %bcx_ptr, %bcx_ptr, %batch_off_bcx;

    mul.wide.u32 %batch_off_y, %b_idx, %hsize;
    shl.b64 %batch_off_y, %batch_off_y, 1;
    add.u64 %y_ptr, %y_ptr, %batch_off_y;

    ld.param.u64 %state_tbl, [state_ptrs];
    mul.wide.u32 %offset, %b_idx, 8;
    add.u64 %offset, %state_tbl, %offset;
    ld.global.u64 %cache_ptr, [%offset];

    cvt.u64.u32 %offset, %idx;
    shl.b64 %offset, %offset, 1;
    add.u64 %b_ptr, %bcx_ptr, %offset;
    ld.global.u16 %hb, [%b_ptr];

    shl.b32 %x_offset, %hsize, 1;
    add.u32 %x_offset, %x_offset, %idx;
    cvt.u64.u32 %offset, %x_offset;
    shl.b64 %offset, %offset, 1;
    add.u64 %x_ptr, %bcx_ptr, %offset;
    ld.global.u16 %hx, [%x_ptr];

    cvt.f32.f16 %fb, %hb;
    cvt.f32.f16 %fx, %hx;
    mul.f32 %fin, %fb, %fx;
    cvt.rn.f16.f32 %hin, %fin;

    sub.u32 %k_minus_1, %klen, 1;

    mul.lo.u32 %row_offset_32, %idx, %k_minus_1;
    cvt.u64.u32 %offset, %row_offset_32;
    shl.b64 %offset, %offset, 1;
    add.u64 %c_state_ptr, %cache_ptr, %offset;

    mul.lo.u32 %row_offset_32, %idx, %klen;
    cvt.u64.u32 %offset, %row_offset_32;
    shl.b64 %offset, %offset, 1;
    add.u64 %c_w_ptr, %w_ptr, %offset;

    mov.f32 %fsum, 0.0;
    mov.u32 %k, 0;

LOOP_K:
    setp.ge.u32 %p_loop, %k, %k_minus_1;
    @%p_loop bra AFTER_LOOP;

    cvt.u64.u32 %offset, %k;
    shl.b64 %offset, %offset, 1;
    add.u64 %b_ptr, %c_state_ptr, %offset;
    ld.global.u16 %hs, [%b_ptr];
    cvt.f32.f16 %fs, %hs;

    add.u64 %x_ptr, %c_w_ptr, %offset;
    ld.global.u16 %hw, [%x_ptr];
    cvt.f32.f16 %fw, %hw;

    fma.rn.f32 %fsum, %fs, %fw, %fsum;

    add.u32 %k, %k, 1;
    bra LOOP_K;

AFTER_LOOP:
    cvt.u64.u32 %offset, %k_minus_1;
    shl.b64 %offset, %offset, 1;
    add.u64 %x_ptr, %c_w_ptr, %offset;
    ld.global.u16 %hw, [%x_ptr];
    cvt.f32.f16 %fw, %hw;
    fma.rn.f32 %fsum, %fin, %fw, %fsum;

    mov.u32 %k, 0;
SHIFT_LOOP:
    add.u32 %stride, %k, 1;
    setp.ge.u32 %p_loop, %stride, %k_minus_1;
    @%p_loop bra SHIFT_LAST;

    cvt.u64.u32 %offset, %stride;
    shl.b64 %offset, %offset, 1;
    add.u64 %x_ptr, %c_state_ptr, %offset;
    ld.global.u16 %hs, [%x_ptr];

    cvt.u64.u32 %offset, %k;
    shl.b64 %offset, %offset, 1;
    add.u64 %b_ptr, %c_state_ptr, %offset;
    st.global.u16 [%b_ptr], %hs;

    add.u32 %k, %k, 1;
    bra SHIFT_LOOP;

SHIFT_LAST:
    sub.u32 %stride, %k_minus_1, 1;
    cvt.u64.u32 %offset, %stride;
    shl.b64 %offset, %offset, 1;
    add.u64 %b_ptr, %c_state_ptr, %offset;
    st.global.u16 [%b_ptr], %hin;

    add.u32 %c_offset, %hsize, %idx;
    cvt.u64.u32 %offset, %c_offset;
    shl.b64 %offset, %offset, 1;
    add.u64 %c_ptr, %bcx_ptr, %offset;
    ld.global.u16 %hc, [%c_ptr];
    cvt.f32.f16 %fc, %hc;

    mul.f32 %fsum, %fsum, %fc;
    cvt.rn.f16.f32 %hin, %fsum;

    cvt.u64.u32 %offset, %idx;
    shl.b64 %offset, %offset, 1;
    add.u64 %b_ptr, %y_ptr, %offset;
    st.global.u16 [%b_ptr], %hin;

EXIT:
    ret;
}
"#;

/// Batched argmax over the vocab dimension: one block per batch element.
pub const BATCHED_ARGMAX_DIRECT_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry batched_argmax_direct_f16(
    .param .u64 tokens_out,
    .param .u64 logits_in,
    .param .u32 vocab_size
) {
    .reg .pred %p_loop, %p_swap, %p_lane0;
    .reg .b32 %batch_idx, %t_id, %nt_id, %vsize, %idx;
    .reg .b64 %out_ptr, %in_ptr, %offset, %byte_off;
    .reg .b16 %h_val;
    .reg .f32 %f_val, %max_val, %shfl_val;
    .reg .b32 %best_idx, %shfl_idx, %lane_id, %warp_id, %s_addr, %stride;
    .shared .f32 s_warp_max[8];
    .shared .b32 s_warp_idx[8];

    ld.param.u32 %vsize, [vocab_size];
    mov.u32 %batch_idx, %ctaid.x;
    mov.u32 %t_id, %tid.x;
    mov.u32 %nt_id, %ntid.x;

    mov.f32 %max_val, -1.0e30;
    mov.u32 %best_idx, 0;

    ld.param.u64 %in_ptr, [logits_in];
    mul.wide.u32 %offset, %batch_idx, %vsize;
    shl.b64 %offset, %offset, 1;
    add.u64 %in_ptr, %in_ptr, %offset;

    mov.u32 %idx, %t_id;

LOOP:
    setp.ge.u32 %p_loop, %idx, %vsize;
    @%p_loop bra REDUCE;

    cvt.u64.u32 %byte_off, %idx;
    shl.b64 %byte_off, %byte_off, 1;
    add.u64 %offset, %in_ptr, %byte_off;
    ld.global.u16 %h_val, [%offset];
    cvt.f32.f16 %f_val, %h_val;

    setp.gt.f32 %p_swap, %f_val, %max_val;
    @%p_swap mov.f32 %max_val, %f_val;
    @%p_swap mov.u32 %best_idx, %idx;

    add.u32 %idx, %idx, %nt_id;
    bra LOOP;

REDUCE:
    shfl.sync.down.b32 %shfl_val, %max_val, 16, 31, 0xffffffff;
    shfl.sync.down.b32 %shfl_idx, %best_idx, 16, 31, 0xffffffff;
    setp.gt.f32 %p_swap, %shfl_val, %max_val;
    @%p_swap mov.f32 %max_val, %shfl_val;
    @%p_swap mov.u32 %best_idx, %shfl_idx;

    shfl.sync.down.b32 %shfl_val, %max_val, 8, 31, 0xffffffff;
    shfl.sync.down.b32 %shfl_idx, %best_idx, 8, 31, 0xffffffff;
    setp.gt.f32 %p_swap, %shfl_val, %max_val;
    @%p_swap mov.f32 %max_val, %shfl_val;
    @%p_swap mov.u32 %best_idx, %shfl_idx;

    shfl.sync.down.b32 %shfl_val, %max_val, 4, 31, 0xffffffff;
    shfl.sync.down.b32 %shfl_idx, %best_idx, 4, 31, 0xffffffff;
    setp.gt.f32 %p_swap, %shfl_val, %max_val;
    @%p_swap mov.f32 %max_val, %shfl_val;
    @%p_swap mov.u32 %best_idx, %shfl_idx;

    shfl.sync.down.b32 %shfl_val, %max_val, 2, 31, 0xffffffff;
    shfl.sync.down.b32 %shfl_idx, %best_idx, 2, 31, 0xffffffff;
    setp.gt.f32 %p_swap, %shfl_val, %max_val;
    @%p_swap mov.f32 %max_val, %shfl_val;
    @%p_swap mov.u32 %best_idx, %shfl_idx;

    shfl.sync.down.b32 %shfl_val, %max_val, 1, 31, 0xffffffff;
    shfl.sync.down.b32 %shfl_idx, %best_idx, 1, 31, 0xffffffff;
    setp.gt.f32 %p_swap, %shfl_val, %max_val;
    @%p_swap mov.f32 %max_val, %shfl_val;
    @%p_swap mov.u32 %best_idx, %shfl_idx;

    and.b32 %lane_id, %t_id, 31;
    shr.u32 %warp_id, %t_id, 5;
    setp.eq.u32 %p_lane0, %lane_id, 0;

    mov.u32 %s_addr, s_warp_max;
    shl.b32 %stride, %warp_id, 2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p_lane0 st.shared.f32 [%s_addr], %max_val;

    mov.u32 %s_addr, s_warp_idx;
    add.u32 %s_addr, %s_addr, %stride;
    @%p_lane0 st.shared.b32 [%s_addr], %best_idx;

    bar.sync 0;

    setp.lt.u32 %p_lane0, %t_id, 8;
    @!%p_lane0 bra FINISH;

    mov.u32 %s_addr, s_warp_max;
    shl.b32 %stride, %t_id, 2;
    add.u32 %s_addr, %s_addr, %stride;
    ld.shared.f32 %max_val, [%s_addr];

    mov.u32 %s_addr, s_warp_idx;
    add.u32 %s_addr, %s_addr, %stride;
    ld.shared.b32 %best_idx, [%s_addr];

    shfl.sync.down.b32 %shfl_val, %max_val, 4, 31, 0xffffffff;
    shfl.sync.down.b32 %shfl_idx, %best_idx, 4, 31, 0xffffffff;
    setp.gt.f32 %p_swap, %shfl_val, %max_val;
    @%p_swap mov.f32 %max_val, %shfl_val;
    @%p_swap mov.u32 %best_idx, %shfl_idx;

    shfl.sync.down.b32 %shfl_val, %max_val, 2, 31, 0xffffffff;
    shfl.sync.down.b32 %shfl_idx, %best_idx, 2, 31, 0xffffffff;
    setp.gt.f32 %p_swap, %shfl_val, %max_val;
    @%p_swap mov.f32 %max_val, %shfl_val;
    @%p_swap mov.u32 %best_idx, %shfl_idx;

    shfl.sync.down.b32 %shfl_val, %max_val, 1, 31, 0xffffffff;
    shfl.sync.down.b32 %shfl_idx, %best_idx, 1, 31, 0xffffffff;
    setp.gt.f32 %p_swap, %shfl_val, %max_val;
    @%p_swap mov.f32 %max_val, %shfl_val;
    @%p_swap mov.u32 %best_idx, %shfl_idx;

    setp.eq.u32 %p_lane0, %t_id, 0;
    @!%p_lane0 bra FINISH;

    ld.param.u64 %out_ptr, [tokens_out];
    cvt.u64.u32 %offset, %batch_idx;
    shl.b64 %offset, %offset, 2;
    add.u64 %out_ptr, %out_ptr, %offset;
    st.global.u32 [%out_ptr], %best_idx;

FINISH:
    ret;
}
"#;


/// Fill per-token conv-state rows for token-batched prefill.
///
/// For every row (token) of a prefill wave this writes the 6-slot ring buffer
/// consumed by `batched_fused_causal_conv1d_gating_ptr_fp16`:
/// `state[c][k] = B[c] * x[c]` of the token at window offset `local_pos - 6 + k`.
/// Window positions before the chunk start (`local_pos - 6 + k < 0`) come from
/// the request's carry state (its persistent per-layer conv buffer at chunk
/// start). After the convolution, the chunk's last row holds the decode-ready
/// state for that request.
///
/// Layouts (all f16):
///   bcx:        [rows, 3*hidden] in_proj output of this wave
///   pool:       per-layer slice, `row_stride` elements per row,
///               channel-major [hidden, 6] inside a row
///   carry_tbl:  [rows] u64 pointers (request conv buffer for this layer)
///   origin_tbl: [rows] u32 wave row index where the request's chunk starts
///   pos_tbl:    [rows] u32 row position within its chunk
pub const CONV_STATE_PREFILL_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry conv_state_prefill_fp16(
    .param .u64 pool,
    .param .u64 carry_tbl,
    .param .u64 origin_tbl,
    .param .u64 pos_tbl,
    .param .u64 bcx,
    .param .u32 hidden,
    .param .u32 k_len,
    .param .u32 rows,
    .param .u32 row_stride,
    .param .u32 bcx_row_bytes,
    .param .u32 x_off_bytes
) {
    .reg .pred %p, %pz;
    .reg .b32 %idx, %row, %hsize, %klen, %nrows, %rstride;
    .reg .b32 %bcxb, %xoff, %k, %km1, %six;
    .reg .s32 %j, %slot, %lpos, %origin, %srow, %soff;
    .reg .b64 %pool_p, %dst_p, %carry_p, %bcx_p, %off, %roff, %bp, %xp;
    .reg .b16 %hb, %hx, %v;
    .reg .f32 %fb, %fx;

    ld.param.u64 %pool_p, [pool];
    ld.param.u64 %bcx_p, [bcx];
    ld.param.u32 %hsize, [hidden];
    ld.param.u32 %klen, [k_len];
    ld.param.u32 %nrows, [rows];
    ld.param.u32 %rstride, [row_stride];
    ld.param.u32 %bcxb, [bcx_row_bytes];
    ld.param.u32 %xoff, [x_off_bytes];

    mov.u32 %row, %ctaid.y;
    setp.ge.u32 %p, %row, %nrows;
    @%p bra EXIT;

    mov.u32 %idx, %tid.x;
    mov.u32 %k, %ctaid.x;
    mov.u32 %soff, %ntid.x;
    mad.lo.u32 %idx, %k, %soff, %idx;
    setp.ge.u32 %p, %idx, %hsize;
    @%p bra EXIT;

    // dst = pool + row * row_stride * 2
    mul.lo.u32 %rstride, %row, %rstride;
    cvt.u64.u32 %off, %rstride;
    shl.b64 %off, %off, 1;
    add.u64 %dst_p, %pool_p, %off;

    // metadata
    cvt.u64.u32 %off, %row;
    shl.b64 %off, %off, 2;
    ld.param.u64 %bp, [origin_tbl];
    add.u64 %bp, %bp, %off;
    ld.global.s32 %origin, [%bp];

    ld.param.u64 %bp, [pos_tbl];
    add.u64 %bp, %bp, %off;
    ld.global.s32 %lpos, [%bp];

    cvt.u64.u32 %off, %row;
    shl.b64 %off, %off, 3;
    ld.param.u64 %bp, [carry_tbl];
    add.u64 %bp, %bp, %off;
    ld.global.u64 %carry_p, [%bp];

    sub.u32 %km1, %klen, 1;
    mov.u32 %six, 6;

    mov.u32 %k, 0;
LOOP_K:
    setp.ge.u32 %p, %k, %km1;
    @%p bra DONE;

    add.s32 %j, %lpos, %k;
    sub.s32 %j, %j, 6;
    setp.lt.s32 %pz, %j, 0;
    @%pz bra USE_CARRY;

    // window token inside this wave: value = B[c] * x[c]
    add.s32 %srow, %origin, %j;
    mul.lo.s32 %soff, %srow, %bcxb;
    cvt.u64.s32 %roff, %soff;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 1;              // idx * 2
    add.u64 %bp, %bcx_p, %roff;
    add.u64 %bp, %bp, %off;
    ld.global.u16 %hb, [%bp];

    cvt.u64.u32 %off, %xoff;
    add.u64 %xp, %bcx_p, %roff;
    add.u64 %xp, %xp, %off;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 1;
    add.u64 %xp, %xp, %off;
    ld.global.u16 %hx, [%xp];

    cvt.f32.f16 %fb, %hb;
    cvt.f32.f16 %fx, %hx;
    mul.f32 %fb, %fb, %fx;
    cvt.rn.f16.f32 %v, %fb;
    bra STORE;

USE_CARRY:
    // carry[idx * 6 + (6 + j)]
    add.s32 %slot, %j, 6;
    mul.lo.s32 %soff, %idx, 6;
    add.s32 %soff, %soff, %slot;
    cvt.u64.s32 %off, %soff;
    shl.b64 %off, %off, 1;
    add.u64 %bp, %carry_p, %off;
    ld.global.u16 %v, [%bp];

STORE:
    // dst[idx * 6 + k]
    mul.lo.s32 %soff, %idx, 6;
    add.s32 %soff, %soff, %k;
    cvt.u64.s32 %off, %soff;
    shl.b64 %off, %off, 1;
    add.u64 %off, %dst_p, %off;
    st.global.u16 [%off], %v;

    add.u32 %k, %k, 1;
    bra LOOP_K;

DONE:
EXIT:
    ret;
}
"#;
