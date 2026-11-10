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
    .param .u32 batch_size,
    .param .u32 window
) {
    .reg .pred %p0, %p1;
    .reg .b32 %head_idx, %dim_idx, %t_idx, %slen, %nq, %nkv, %hd, %b_idx, %b_size, %kv_head_idx, %hstride, %kv_idx_32, %tid_val, %tmp32;
    .reg .b32 %window, %t_start, %cnt, %s_idx;
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

    // Sliding-window mask: layer `window` (0 = full attention) keeps only the
    // newest `window` keys, i.e. key `k` is masked when `p - k >= window`.
    // The query is at `slen - 1`, so start the scan at `slen - window`.
    ld.param.u32 %window, [window];
    mov.u32 %t_start, 0;
    setp.eq.u32 %p1, %window, 0;
    @%p1 bra SWA_DONE;
    setp.le.u32 %p1, %slen, %window;
    @%p1 bra SWA_DONE;
    sub.u32 %t_start, %slen, %window;
SWA_DONE:
    sub.u32 %cnt, %slen, %t_start;

    mov.f32 %fscale, 0.0883883476;

    .shared .f32 s_scores[8192];
    .shared .f32 s_warp_dot[8];
    .shared .f32 s_max;
    .shared .f32 s_denom;

    mov.u32 %t_idx, %t_start;
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
    sub.u32 %s_idx, %t_idx, %t_start;
    shl.b32 %s_offset, %s_idx, 2;
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
    mov.u32 %s_idx, 1;
MAX_LOOP:
    setp.ge.u32 %p1, %s_idx, %cnt;
    @%p1 bra CALC_EXP_SUM;

    shl.b32 %s_offset, %s_idx, 2;
    mov.u32 %s_addr, s_scores;
    add.u32 %s_addr, %s_addr, %s_offset;
    ld.shared.f32 %fscore, [%s_addr];
    max.f32 %fmax, %fmax, %fscore;
    add.u32 %s_idx, %s_idx, 1;
    bra MAX_LOOP;

CALC_EXP_SUM:
    mov.u32 %s_addr, s_max;
    st.shared.f32 [%s_addr], %fmax;
    mov.f32 %fsum, 0.0;
    mov.u32 %s_idx, 0;
EXP_LOOP:
    setp.ge.u32 %p1, %s_idx, %cnt;
    @%p1 bra NORM_SOFTMAX;

    shl.b32 %s_offset, %s_idx, 2;
    mov.u32 %s_addr, s_scores;
    add.u32 %s_addr, %s_addr, %s_offset;
    ld.shared.f32 %fscore, [%s_addr];
    sub.f32 %fscore, %fscore, %fmax;
    mul.f32 %fscore, %fscore, 1.44269504;
    ex2.approx.ftz.f32 %fweight, %fscore;
    st.shared.f32 [%s_addr], %fweight;
    add.f32 %fsum, %fsum, %fweight;

    add.u32 %s_idx, %s_idx, 1;
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

    mov.u32 %s_idx, 0;
V_TOK_LOOP:
    setp.ge.u32 %p1, %s_idx, %cnt;
    @%p1 bra WRITE_CTX;

    shl.b32 %s_offset, %s_idx, 2;
    mov.u32 %s_addr, s_scores;
    add.u32 %s_addr, %s_addr, %s_offset;
    ld.shared.f32 %fweight, [%s_addr];
    mul.f32 %fweight, %fweight, %fsum;

    add.u32 %t_idx, %s_idx, %t_start;
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

    add.u32 %s_idx, %s_idx, 1;
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

/// Tiled online-softmax GQA kernel.
///
/// One warp per query head, four query heads per block (one per KV head).
/// Lane `l` owns head dims `4l .. 4l+3`, so K/V/Q are touched with a single
/// coalesced 8-byte `v4.u16` transaction per lane and the softmax normaliser
/// stays in registers: no shared score tile, no barrier inside the key loop.
/// Head dim is fixed at 128 and `num_q_heads == 4 * num_kv_heads` (LFM3).
///
/// `window` > 0 keeps only the newest `window` keys (`p_key - p_query >=
/// window` is masked), matching llama.cpp's `LLAMA_SWA_TYPE_STANDARD`.
pub const BATCHED_GQA_ATTENTION_ONLINE_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry batched_gqa_attention_online_fp16(
    .param .u64 context_out,
    .param .u64 q_in,
    .param .u64 k_ptrs,
    .param .u64 v_ptrs,
    .param .u64 seq_lens,
    .param .u32 num_q_heads,
    .param .u32 num_kv_heads,
    .param .u32 head_dim,
    .param .u32 batch_size,
    .param .u32 window
) {
    .reg .pred %p0, %p1, %p2;
    .reg .b32 %tidv, %lane, %warp, %head, %kvh, %b_idx, %slen, %nq, %nkv, %hd, %t, %t_start, %window, %kvbase, %tmp;
    .reg .b64 %q_ptr, %ctx_ptr, %k_ptr, %v_ptr, %k_tbl, %v_tbl, %sl_tbl, %off, %addr, %laneoff;
    .reg .f32 %q0, %q1, %q2, %q3, %a0, %a1, %a2, %a3, %m, %s, %dot, %sc, %w, %corr, %tmpf, %sh;
    .reg .f32 %k0, %k1, %k2, %k3, %v0, %v1, %v2, %v3;
    .reg .b16 %h0, %h1, %h2, %h3, %g0, %g1, %g2, %g3;

    ld.param.u32 %nq, [num_q_heads];
    ld.param.u32 %nkv, [num_kv_heads];
    ld.param.u32 %hd, [head_dim];

    mov.u32 %b_idx, %ctaid.y;
    ld.param.u32 %tmp, [batch_size];
    setp.ge.u32 %p0, %b_idx, %tmp;
    @%p0 bra EXIT;

    mov.u32 %kvh, %ctaid.x;
    setp.ge.u32 %p0, %kvh, %nkv;
    @%p0 bra EXIT;

    mov.u32 %tidv, %tid.x;
    shr.u32 %warp, %tidv, 5;
    and.b32 %lane, %tidv, 31;
    // Four query heads share each KV head: head = kv_head * 4 + warp.
    shl.b32 %head, %kvh, 2;
    add.u32 %head, %head, %warp;

    cvt.u64.u32 %laneoff, %lane;
    shl.b64 %laneoff, %laneoff, 3;      // 4 halves per lane

    ld.param.u64 %q_ptr, [q_in];
    ld.param.u64 %ctx_ptr, [context_out];

    mul.lo.u32 %tmp, %b_idx, %nq;
    add.u32 %tmp, %tmp, %head;
    mul.lo.u32 %tmp, %tmp, %hd;
    cvt.u64.u32 %off, %tmp;
    shl.b64 %off, %off, 1;
    add.u64 %q_ptr, %q_ptr, %off;
    add.u64 %ctx_ptr, %ctx_ptr, %off;
    add.u64 %addr, %q_ptr, %laneoff;
    ld.global.v4.u16 {%h0, %h1, %h2, %h3}, [%addr];
    cvt.f32.f16 %q0, %h0;
    cvt.f32.f16 %q1, %h1;
    cvt.f32.f16 %q2, %h2;
    cvt.f32.f16 %q3, %h3;

    ld.param.u64 %k_tbl, [k_ptrs];
    ld.param.u64 %v_tbl, [v_ptrs];
    mul.wide.u32 %off, %b_idx, 8;
    add.u64 %off, %k_tbl, %off;
    ld.global.u64 %k_ptr, [%off];
    mul.wide.u32 %off, %b_idx, 8;
    add.u64 %off, %v_tbl, %off;
    ld.global.u64 %v_ptr, [%off];

    ld.param.u64 %sl_tbl, [seq_lens];
    mul.wide.u32 %off, %b_idx, 4;
    add.u64 %off, %sl_tbl, %off;
    ld.global.u32 %slen, [%off];

    ld.param.u32 %window, [window];
    mov.u32 %t_start, 0;
    setp.eq.u32 %p1, %window, 0;
    @%p1 bra W_DONE;
    setp.le.u32 %p1, %slen, %window;
    @%p1 bra W_DONE;
    sub.u32 %t_start, %slen, %window;
W_DONE:

    mov.f32 %m, -1.0e30;
    mov.f32 %s, 0.0;
    mov.f32 %a0, 0.0;
    mov.f32 %a1, 0.0;
    mov.f32 %a2, 0.0;
    mov.f32 %a3, 0.0;
    mov.u32 %t, %t_start;

KEY_LOOP:
    setp.ge.u32 %p0, %t, %slen;
    @%p0 bra FINISH;

    mul.lo.u32 %kvbase, %t, %nkv;
    add.u32 %kvbase, %kvbase, %kvh;
    mul.lo.u32 %kvbase, %kvbase, %hd;
    cvt.u64.u32 %off, %kvbase;
    shl.b64 %off, %off, 1;
    add.u64 %off, %off, %laneoff;

    // K and V loads are independent, so both are in flight together.
    add.u64 %addr, %k_ptr, %off;
    ld.global.v4.u16 {%h0, %h1, %h2, %h3}, [%addr];
    add.u64 %addr, %v_ptr, %off;
    ld.global.v4.u16 {%g0, %g1, %g2, %g3}, [%addr];

    cvt.f32.f16 %k0, %h0;
    cvt.f32.f16 %k1, %h1;
    cvt.f32.f16 %k2, %h2;
    cvt.f32.f16 %k3, %h3;

    mul.f32 %dot, %q0, %k0;
    fma.rn.f32 %dot, %q1, %k1, %dot;
    fma.rn.f32 %dot, %q2, %k2, %dot;
    fma.rn.f32 %dot, %q3, %k3, %dot;
    shfl.sync.down.b32 %sh, %dot, 16, 31, 0xffffffff;
    add.f32 %dot, %dot, %sh;
    shfl.sync.down.b32 %sh, %dot, 8, 31, 0xffffffff;
    add.f32 %dot, %dot, %sh;
    shfl.sync.down.b32 %sh, %dot, 4, 31, 0xffffffff;
    add.f32 %dot, %dot, %sh;
    shfl.sync.down.b32 %sh, %dot, 2, 31, 0xffffffff;
    add.f32 %dot, %dot, %sh;
    shfl.sync.down.b32 %sh, %dot, 1, 31, 0xffffffff;
    add.f32 %dot, %dot, %sh;
    shfl.sync.idx.b32 %dot, %dot, 0, 31, 0xffffffff;
    mul.f32 %sc, %dot, 0.0883883476;

    // Online softmax. The rescale branch is warp-uniform because %sc is
    // broadcast, and it is taken only when the running max grows.
    setp.gt.f32 %p2, %sc, %m;
    @!%p2 bra NO_RESCALE;
    sub.f32 %tmpf, %m, %sc;
    mul.f32 %tmpf, %tmpf, 1.44269504;
    ex2.approx.ftz.f32 %corr, %tmpf;
    mul.f32 %a0, %a0, %corr;
    mul.f32 %a1, %a1, %corr;
    mul.f32 %a2, %a2, %corr;
    mul.f32 %a3, %a3, %corr;
    mul.f32 %s, %s, %corr;
    mov.f32 %m, %sc;
NO_RESCALE:
    sub.f32 %tmpf, %sc, %m;
    mul.f32 %tmpf, %tmpf, 1.44269504;
    ex2.approx.ftz.f32 %w, %tmpf;
    add.f32 %s, %s, %w;

    cvt.f32.f16 %v0, %g0;
    cvt.f32.f16 %v1, %g1;
    cvt.f32.f16 %v2, %g2;
    cvt.f32.f16 %v3, %g3;
    fma.rn.f32 %a0, %w, %v0, %a0;
    fma.rn.f32 %a1, %w, %v1, %a1;
    fma.rn.f32 %a2, %w, %v2, %a2;
    fma.rn.f32 %a3, %w, %v3, %a3;

    add.u32 %t, %t, 1;
    bra KEY_LOOP;

FINISH:
    rcp.approx.ftz.f32 %tmpf, %s;
    mul.f32 %a0, %a0, %tmpf;
    mul.f32 %a1, %a1, %tmpf;
    mul.f32 %a2, %a2, %tmpf;
    mul.f32 %a3, %a3, %tmpf;

    add.u64 %addr, %ctx_ptr, %laneoff;
    cvt.rn.f16.f32 %h0, %a0;
    cvt.rn.f16.f32 %h1, %a1;
    cvt.rn.f16.f32 %h2, %a2;
    cvt.rn.f16.f32 %h3, %a3;
    st.global.v4.u16 [%addr], {%h0, %h1, %h2, %h3};

EXIT:
    ret;
}
"#;

/// Split-K attention: pass 1 of 2.
///
/// Same per-warp maths as `batched_gqa_attention_online_fp16`, but each block
/// owns one key range `[ctaid.z * chunk_keys, +chunk_keys)` and writes its
/// unnormalised partial (running max, normaliser, 128 weighted values) to
/// `partial_acc` / `partial_stats` instead of the final context.
///
/// A single-block-per-row decode touches the GPU with only `num_kv_heads`
/// blocks, so the key loop is latency bound (~0.94 ms/layer at 2 000 keys).
/// Splitting the keys across blocks fills the SMs, and
/// `batched_gqa_attn_combine_fp16` merges the partials exactly.
pub const BATCHED_GQA_ATTENTION_SPLIT_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry batched_gqa_attention_split_fp16(
    .param .u64 q_in,
    .param .u64 k_ptrs,
    .param .u64 v_ptrs,
    .param .u64 seq_lens,
    .param .u64 partial_acc,
    .param .u64 partial_stats,
    .param .u32 num_q_heads,
    .param .u32 num_kv_heads,
    .param .u32 head_dim,
    .param .u32 batch_size,
    .param .u32 window,
    .param .u32 chunk_keys,
    .param .u32 nchunks
) {
    .reg .pred %p0, %p1, %p2;
    .reg .b32 %tidv, %lane, %warp, %head, %kvh, %b_idx, %c, %slen, %nq, %nkv, %hd;
    .reg .b32 %t, %t_start, %t_end, %wstart, %window, %chunk, %nchunks, %kvbase, %tmp, %pidx;
    .reg .b64 %q_ptr, %k_ptr, %v_ptr, %k_tbl, %v_tbl, %sl_tbl, %off, %addr, %laneoff8, %laneoff16, %base;
    .reg .f32 %q0, %q1, %q2, %q3, %a0, %a1, %a2, %a3, %m, %s, %dot, %sc, %w, %corr, %tmpf, %sh;
    .reg .f32 %k0, %k1, %k2, %k3, %v0, %v1, %v2, %v3;
    .reg .b16 %h0, %h1, %h2, %h3, %g0, %g1, %g2, %g3;

    ld.param.u32 %nq, [num_q_heads];
    ld.param.u32 %nkv, [num_kv_heads];
    ld.param.u32 %hd, [head_dim];

    mov.u32 %b_idx, %ctaid.y;
    ld.param.u32 %tmp, [batch_size];
    setp.ge.u32 %p0, %b_idx, %tmp;
    @%p0 bra EXIT;

    mov.u32 %kvh, %ctaid.x;
    setp.ge.u32 %p0, %kvh, %nkv;
    @%p0 bra EXIT;

    mov.u32 %tidv, %tid.x;
    shr.u32 %warp, %tidv, 5;
    and.b32 %lane, %tidv, 31;
    shl.b32 %head, %kvh, 2;
    add.u32 %head, %head, %warp;

    cvt.u64.u32 %laneoff8, %lane;
    shl.b64 %laneoff8, %laneoff8, 3;
    cvt.u64.u32 %laneoff16, %lane;
    shl.b64 %laneoff16, %laneoff16, 4;

    ld.param.u64 %q_ptr, [q_in];
    mul.lo.u32 %tmp, %b_idx, %nq;
    add.u32 %tmp, %tmp, %head;
    mul.lo.u32 %tmp, %tmp, %hd;
    cvt.u64.u32 %off, %tmp;
    shl.b64 %off, %off, 1;
    add.u64 %q_ptr, %q_ptr, %off;
    add.u64 %addr, %q_ptr, %laneoff8;
    ld.global.v4.u16 {%h0, %h1, %h2, %h3}, [%addr];
    cvt.f32.f16 %q0, %h0;
    cvt.f32.f16 %q1, %h1;
    cvt.f32.f16 %q2, %h2;
    cvt.f32.f16 %q3, %h3;

    ld.param.u64 %k_tbl, [k_ptrs];
    ld.param.u64 %v_tbl, [v_ptrs];
    mul.wide.u32 %off, %b_idx, 8;
    add.u64 %off, %k_tbl, %off;
    ld.global.u64 %k_ptr, [%off];
    mul.wide.u32 %off, %b_idx, 8;
    add.u64 %off, %v_tbl, %off;
    ld.global.u64 %v_ptr, [%off];

    ld.param.u64 %sl_tbl, [seq_lens];
    mul.wide.u32 %off, %b_idx, 4;
    add.u64 %off, %sl_tbl, %off;
    ld.global.u32 %slen, [%off];

    // This block's key range: the chunk intersected with the layer window.
    ld.param.u32 %chunk, [chunk_keys];
    ld.param.u32 %nchunks, [nchunks];
    mov.u32 %c, %ctaid.z;
    ld.param.u32 %window, [window];

    mul.lo.u32 %t_start, %c, %chunk;
    add.u32 %t_end, %t_start, %chunk;
    setp.gt.u32 %p0, %t_end, %slen;
    @%p0 mov.u32 %t_end, %slen;

    setp.eq.u32 %p1, %window, 0;
    @%p1 bra W_DONE;
    mov.u32 %wstart, 0;
    setp.gt.u32 %p1, %slen, %window;
    @!%p1 bra W_CHECK;
    sub.u32 %wstart, %slen, %window;
W_CHECK:
    setp.lt.u32 %p1, %t_start, %wstart;
    @!%p1 bra W_DONE;
    mov.u32 %t_start, %wstart;
W_DONE:

    mov.f32 %m, -1.0e30;
    mov.f32 %s, 0.0;
    mov.f32 %a0, 0.0;
    mov.f32 %a1, 0.0;
    mov.f32 %a2, 0.0;
    mov.f32 %a3, 0.0;
    mov.u32 %t, %t_start;

KEY_LOOP:
    setp.ge.u32 %p0, %t, %t_end;
    @%p0 bra FINISH;

    mul.lo.u32 %kvbase, %t, %nkv;
    add.u32 %kvbase, %kvbase, %kvh;
    mul.lo.u32 %kvbase, %kvbase, %hd;
    cvt.u64.u32 %off, %kvbase;
    shl.b64 %off, %off, 1;
    add.u64 %off, %off, %laneoff8;

    add.u64 %addr, %k_ptr, %off;
    ld.global.v4.u16 {%h0, %h1, %h2, %h3}, [%addr];
    add.u64 %addr, %v_ptr, %off;
    ld.global.v4.u16 {%g0, %g1, %g2, %g3}, [%addr];

    cvt.f32.f16 %k0, %h0;
    cvt.f32.f16 %k1, %h1;
    cvt.f32.f16 %k2, %h2;
    cvt.f32.f16 %k3, %h3;

    mul.f32 %dot, %q0, %k0;
    fma.rn.f32 %dot, %q1, %k1, %dot;
    fma.rn.f32 %dot, %q2, %k2, %dot;
    fma.rn.f32 %dot, %q3, %k3, %dot;
    shfl.sync.down.b32 %sh, %dot, 16, 31, 0xffffffff;
    add.f32 %dot, %dot, %sh;
    shfl.sync.down.b32 %sh, %dot, 8, 31, 0xffffffff;
    add.f32 %dot, %dot, %sh;
    shfl.sync.down.b32 %sh, %dot, 4, 31, 0xffffffff;
    add.f32 %dot, %dot, %sh;
    shfl.sync.down.b32 %sh, %dot, 2, 31, 0xffffffff;
    add.f32 %dot, %dot, %sh;
    shfl.sync.down.b32 %sh, %dot, 1, 31, 0xffffffff;
    add.f32 %dot, %dot, %sh;
    shfl.sync.idx.b32 %dot, %dot, 0, 31, 0xffffffff;
    mul.f32 %sc, %dot, 0.0883883476;

    setp.gt.f32 %p2, %sc, %m;
    @!%p2 bra NO_RESCALE;
    sub.f32 %tmpf, %m, %sc;
    mul.f32 %tmpf, %tmpf, 1.44269504;
    ex2.approx.ftz.f32 %corr, %tmpf;
    mul.f32 %a0, %a0, %corr;
    mul.f32 %a1, %a1, %corr;
    mul.f32 %a2, %a2, %corr;
    mul.f32 %a3, %a3, %corr;
    mul.f32 %s, %s, %corr;
    mov.f32 %m, %sc;
NO_RESCALE:
    sub.f32 %tmpf, %sc, %m;
    mul.f32 %tmpf, %tmpf, 1.44269504;
    ex2.approx.ftz.f32 %w, %tmpf;
    add.f32 %s, %s, %w;

    cvt.f32.f16 %v0, %g0;
    cvt.f32.f16 %v1, %g1;
    cvt.f32.f16 %v2, %g2;
    cvt.f32.f16 %v3, %g3;
    fma.rn.f32 %a0, %w, %v0, %a0;
    fma.rn.f32 %a1, %w, %v1, %a1;
    fma.rn.f32 %a2, %w, %v2, %a2;
    fma.rn.f32 %a3, %w, %v3, %a3;

    add.u32 %t, %t, 1;
    bra KEY_LOOP;

FINISH:
    // Partial slot: ((row * nchunks + chunk) * nq + head).
    mul.lo.u32 %pidx, %b_idx, %nchunks;
    add.u32 %pidx, %pidx, %c;
    mul.lo.u32 %pidx, %pidx, %nq;
    add.u32 %pidx, %pidx, %head;

    ld.param.u64 %base, [partial_stats];
    cvt.u64.u32 %off, %pidx;
    shl.b64 %off, %off, 3;
    add.u64 %base, %base, %off;
    setp.eq.u32 %p1, %lane, 0;
    @!%p1 bra NO_STATS;
    st.global.f32 [%base], %m;
    st.global.f32 [%base+4], %s;
NO_STATS:

    ld.param.u64 %base, [partial_acc];
    mul.lo.u32 %tmp, %pidx, %hd;
    cvt.u64.u32 %off, %tmp;
    shl.b64 %off, %off, 2;
    add.u64 %base, %base, %off;
    add.u64 %base, %base, %laneoff16;
    st.global.v4.f32 [%base], {%a0, %a1, %a2, %a3};

EXIT:
    ret;
}

/// Split-K attention: pass 2 of 2.
///
/// One warp per (row, query head), 32 threads per block. Lane `l` owns dims
/// `4l..4l+3`. Chunk partials are merged with the same online rescaling used
/// while scanning keys, so the result is the exact softmax over the full key
/// range intersected with the per-layer window.
.visible .entry batched_gqa_attn_combine_fp16(
    .param .u64 context_out,
    .param .u64 partial_acc,
    .param .u64 partial_stats,
    .param .u32 num_q_heads,
    .param .u32 head_dim,
    .param .u32 batch_size,
    .param .u32 nchunks
) {
    .reg .pred %p0, %p1;
    .reg .b32 %tidv, %lane, %head, %b_idx, %nq, %hd, %nchunks, %c, %pidx, %tmp;
    .reg .b64 %ctx_ptr, %acc_ptr, %st_ptr, %off, %addr, %laneoff8, %laneoff16;
    .reg .f32 %a0, %a1, %a2, %a3, %m, %s, %mc, %sc, %w, %corr, %tmpf, %denom;
    .reg .f32 %p0v, %p1v, %p2v, %p3v;
    .reg .b16 %h0, %h1, %h2, %h3;

    ld.param.u32 %nq, [num_q_heads];
    ld.param.u32 %hd, [head_dim];
    ld.param.u32 %nchunks, [nchunks];

    mov.u32 %b_idx, %ctaid.y;
    ld.param.u32 %tmp, [batch_size];
    setp.ge.u32 %p0, %b_idx, %tmp;
    @%p0 bra EXIT;

    mov.u32 %head, %ctaid.x;
    setp.ge.u32 %p0, %head, %nq;
    @%p0 bra EXIT;

    mov.u32 %tidv, %tid.x;
    and.b32 %lane, %tidv, 31;
    cvt.u64.u32 %laneoff8, %lane;
    shl.b64 %laneoff8, %laneoff8, 3;
    cvt.u64.u32 %laneoff16, %lane;
    shl.b64 %laneoff16, %laneoff16, 4;

    mov.f32 %m, -1.0e30;
    mov.f32 %s, 0.0;
    mov.f32 %a0, 0.0;
    mov.f32 %a1, 0.0;
    mov.f32 %a2, 0.0;
    mov.f32 %a3, 0.0;
    mov.u32 %c, 0;

CHUNK_LOOP:
    setp.ge.u32 %p0, %c, %nchunks;
    @%p0 bra FINISH;

    mul.lo.u32 %pidx, %b_idx, %nchunks;
    add.u32 %pidx, %pidx, %c;
    mul.lo.u32 %pidx, %pidx, %nq;
    add.u32 %pidx, %pidx, %head;

    ld.param.u64 %st_ptr, [partial_stats];
    cvt.u64.u32 %off, %pidx;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %st_ptr, %off;
    ld.global.f32 %mc, [%addr];
    ld.global.f32 %sc, [%addr+4];

    // Fold this chunk's partial into the running (max, normaliser, acc).
    setp.gt.f32 %p1, %mc, %m;
    @!%p1 bra NO_RESCALE;
    sub.f32 %tmpf, %m, %mc;
    mul.f32 %tmpf, %tmpf, 1.44269504;
    ex2.approx.ftz.f32 %corr, %tmpf;
    mul.f32 %a0, %a0, %corr;
    mul.f32 %a1, %a1, %corr;
    mul.f32 %a2, %a2, %corr;
    mul.f32 %a3, %a3, %corr;
    mul.f32 %s, %s, %corr;
    mov.f32 %m, %mc;
NO_RESCALE:
    sub.f32 %tmpf, %mc, %m;
    mul.f32 %tmpf, %tmpf, 1.44269504;
    ex2.approx.ftz.f32 %w, %tmpf;
    // The partial accumulator is already normalised by its chunk's own
    // normaliser, so it only takes the max correction; the denominator takes
    // the correction times the chunk normaliser.
    mul.f32 %tmpf, %w, %sc;
    add.f32 %s, %s, %tmpf;

    ld.param.u64 %acc_ptr, [partial_acc];
    mul.lo.u32 %tmp, %pidx, %hd;
    cvt.u64.u32 %off, %tmp;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %acc_ptr, %off;
    add.u64 %addr, %addr, %laneoff16;
    ld.global.v4.f32 {%p0v, %p1v, %p2v, %p3v}, [%addr];
    fma.rn.f32 %a0, %w, %p0v, %a0;
    fma.rn.f32 %a1, %w, %p1v, %a1;
    fma.rn.f32 %a2, %w, %p2v, %a2;
    fma.rn.f32 %a3, %w, %p3v, %a3;

    add.u32 %c, %c, 1;
    bra CHUNK_LOOP;

FINISH:
    rcp.approx.ftz.f32 %denom, %s;
    mul.f32 %a0, %a0, %denom;
    mul.f32 %a1, %a1, %denom;
    mul.f32 %a2, %a2, %denom;
    mul.f32 %a3, %a3, %denom;

    ld.param.u64 %ctx_ptr, [context_out];
    mul.lo.u32 %tmp, %b_idx, %nq;
    add.u32 %tmp, %tmp, %head;
    mul.lo.u32 %tmp, %tmp, %hd;
    cvt.u64.u32 %off, %tmp;
    shl.b64 %off, %off, 1;
    add.u64 %ctx_ptr, %ctx_ptr, %off;
    add.u64 %addr, %ctx_ptr, %laneoff8;
    cvt.rn.f16.f32 %h0, %a0;
    cvt.rn.f16.f32 %h1, %a1;
    cvt.rn.f16.f32 %h2, %a2;
    cvt.rn.f16.f32 %h3, %a3;
    st.global.v4.u16 [%addr], {%h0, %h1, %h2, %h3};

EXIT:
    ret;
}
"#;
