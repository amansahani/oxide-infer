use anyhow::Result;
use oxide_infer_core::{DeviceBuffer, GpuContext, SafeTensorsLoader};
use crate::{CausalConvLayer, Lfm3AttentionLayer, FfnLayer, RotaryEmbeddingTable, GqaAttentionEngine};
use oxide_infer_kernels::norm::RMS_NORM_FP16_PTX;
use oxide_infer_kernels::PtxModule;
use cuda_core::launch_kernel_on_stream;

pub const RESIDUAL_ADD_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry batched_rms_norm_vec8_fp16(
    .param .u64 out_ptr,
    .param .u64 in_ptr,
    .param .u64 weight_ptr,
    .param .f32 epsilon,
    .param .u32 hidden_size,
    .param .u32 batch_size
) {
    .reg .pred %p0, %p1, %p_lane0;
    .reg .b32 %lane_id, %warp_id, %num_warps, %tid_val, %ntid_val, %stride, %k_idx, %h_size, %b_idx, %b_size, %s_addr;
    .reg .b64 %y_ptr, %x_ptr, %w_ptr, %offset, %b_offset;
    .reg .b32 %v0, %v1, %v2, %v3, %wv0, %wv1, %wv2, %wv3;
    .reg .b32 %out0, %out1, %out2, %out3;
    .reg .b16 %h0, %h1, %hw0, %hw1, %hn0, %hn1;
    .reg .f32 %f0, %f1, %fw0, %fw1;
    .reg .f32 %f_norm0, %f_norm1, %f_norm2, %f_norm3, %f_norm4, %f_norm5, %f_norm6, %f_norm7;
    .reg .f32 %sum_sq, %shfl_val, %inv_rms, %h_size_f32, %eps, %mean_sq;
    .shared .f32 s_warp_sums[8];
    .shared .f32 s_inv_rms;

    ld.param.u32 %h_size, [hidden_size];
    ld.param.u32 %b_size, [batch_size];
    ld.param.f32 %eps, [epsilon];

    mov.u32 %b_idx, %ctaid.y;
    setp.ge.u32 %p0, %b_idx, %b_size;
    @%p0 bra EXIT_BNORM;

    ld.param.u64 %y_ptr, [out_ptr];
    ld.param.u64 %x_ptr, [in_ptr];
    ld.param.u64 %w_ptr, [weight_ptr];

    mul.wide.u32 %b_offset, %b_idx, %h_size;
    shl.b64 %b_offset, %b_offset, 1;
    add.u64 %y_ptr, %y_ptr, %b_offset;
    add.u64 %x_ptr, %x_ptr, %b_offset;

    mov.u32 %tid_val, %tid.x;
    and.b32 %lane_id, %tid_val, 31;
    shr.u32 %warp_id, %tid_val, 5;
    mov.u32 %ntid_val, %ntid.x;
    shr.u32 %num_warps, %ntid_val, 5;

    shl.b32 %k_idx, %tid_val, 3;
    cvt.u64.u32 %offset, %k_idx;
    shl.b64 %offset, %offset, 1;

    mov.f32 %sum_sq, 0.0;

    setp.ge.u32 %p0, %k_idx, %h_size;
    @%p0 bra REDUCE_BNORM;

    add.u64 %x_ptr, %x_ptr, %offset;
    ld.global.v4.u32 {%v0, %v1, %v2, %v3}, [%x_ptr];

    mov.b32 {%h0, %h1}, %v0;
    cvt.f32.f16 %f0, %h0;
    cvt.f32.f16 %f1, %h1;
    fma.rn.f32 %sum_sq, %f0, %f0, %sum_sq;
    fma.rn.f32 %sum_sq, %f1, %f1, %sum_sq;

    mov.b32 {%h0, %h1}, %v1;
    cvt.f32.f16 %f0, %h0;
    cvt.f32.f16 %f1, %h1;
    fma.rn.f32 %sum_sq, %f0, %f0, %sum_sq;
    fma.rn.f32 %sum_sq, %f1, %f1, %sum_sq;

    mov.b32 {%h0, %h1}, %v2;
    cvt.f32.f16 %f0, %h0;
    cvt.f32.f16 %f1, %h1;
    fma.rn.f32 %sum_sq, %f0, %f0, %sum_sq;
    fma.rn.f32 %sum_sq, %f1, %f1, %sum_sq;

    mov.b32 {%h0, %h1}, %v3;
    cvt.f32.f16 %f0, %h0;
    cvt.f32.f16 %f1, %h1;
    fma.rn.f32 %sum_sq, %f0, %f0, %sum_sq;
    fma.rn.f32 %sum_sq, %f1, %f1, %sum_sq;

REDUCE_BNORM:
    shfl.sync.down.b32 %shfl_val, %sum_sq, 16, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 8, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 4, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 2, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 1, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;

    setp.eq.u32 %p_lane0, %lane_id, 0;
    mov.u32 %s_addr, s_warp_sums;
    shl.b32 %stride, %warp_id, 2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p_lane0 st.shared.f32 [%s_addr], %sum_sq;
    bar.sync 0;

    setp.eq.u32 %p0, %warp_id, 0;
    @!%p0 bra NORM_WRITE_BNORM;

    setp.lt.u32 %p1, %tid_val, %num_warps;
    mov.u32 %s_addr, s_warp_sums;
    shl.b32 %stride, %tid_val, 2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p1 ld.shared.f32 %sum_sq, [%s_addr];
    @!%p1 mov.f32 %sum_sq, 0.0;

    shfl.sync.down.b32 %shfl_val, %sum_sq, 4, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 2, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 1, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;

    setp.eq.u32 %p_lane0, %tid_val, 0;
    @!%p_lane0 bra NORM_WRITE_BNORM;

    cvt.rn.f32.u32 %h_size_f32, %h_size;
    div.rn.f32 %mean_sq, %sum_sq, %h_size_f32;
    add.f32 %mean_sq, %mean_sq, %eps;
    rsqrt.approx.ftz.f32 %inv_rms, %mean_sq;
    st.shared.f32 [s_inv_rms], %inv_rms;

NORM_WRITE_BNORM:
    bar.sync 0;
    ld.shared.f32 %inv_rms, [s_inv_rms];

    setp.ge.u32 %p0, %k_idx, %h_size;
    @%p0 bra EXIT_BNORM;

    add.u64 %w_ptr, %w_ptr, %offset;
    ld.global.v4.u32 {%wv0, %wv1, %wv2, %wv3}, [%w_ptr];

    mov.b32 {%hw0, %hw1}, %wv0;
    cvt.f32.f16 %fw0, %hw0;
    cvt.f32.f16 %fw1, %hw1;
    mov.b32 {%h0, %h1}, %v0;
    cvt.f32.f16 %f0, %h0;
    cvt.f32.f16 %f1, %h1;
    mul.f32 %f_norm0, %f0, %inv_rms;
    mul.f32 %f_norm0, %f_norm0, %fw0;
    mul.f32 %f_norm1, %f1, %inv_rms;
    mul.f32 %f_norm1, %f_norm1, %fw1;
    cvt.rn.f16.f32 %hn0, %f_norm0;
    cvt.rn.f16.f32 %hn1, %f_norm1;
    mov.b32 %out0, {%hn0, %hn1};

    mov.b32 {%hw0, %hw1}, %wv1;
    cvt.f32.f16 %fw0, %hw0;
    cvt.f32.f16 %fw1, %hw1;
    mov.b32 {%h0, %h1}, %v1;
    cvt.f32.f16 %f0, %h0;
    cvt.f32.f16 %f1, %h1;
    mul.f32 %f_norm2, %f0, %inv_rms;
    mul.f32 %f_norm2, %f_norm2, %fw0;
    mul.f32 %f_norm3, %f1, %inv_rms;
    mul.f32 %f_norm3, %f_norm3, %fw1;
    cvt.rn.f16.f32 %hn0, %f_norm2;
    cvt.rn.f16.f32 %hn1, %f_norm3;
    mov.b32 %out1, {%hn0, %hn1};

    mov.b32 {%hw0, %hw1}, %wv2;
    cvt.f32.f16 %fw0, %hw0;
    cvt.f32.f16 %fw1, %hw1;
    mov.b32 {%h0, %h1}, %v2;
    cvt.f32.f16 %f0, %h0;
    cvt.f32.f16 %f1, %h1;
    mul.f32 %f_norm4, %f0, %inv_rms;
    mul.f32 %f_norm4, %f_norm4, %fw0;
    mul.f32 %f_norm5, %f1, %inv_rms;
    mul.f32 %f_norm5, %f_norm5, %fw1;
    cvt.rn.f16.f32 %hn0, %f_norm4;
    cvt.rn.f16.f32 %hn1, %f_norm5;
    mov.b32 %out2, {%hn0, %hn1};

    mov.b32 {%hw0, %hw1}, %wv3;
    cvt.f32.f16 %fw0, %hw0;
    cvt.f32.f16 %fw1, %hw1;
    mov.b32 {%h0, %h1}, %v3;
    cvt.f32.f16 %f0, %h0;
    cvt.f32.f16 %f1, %h1;
    mul.f32 %f_norm6, %f0, %inv_rms;
    mul.f32 %f_norm6, %f_norm6, %fw0;
    mul.f32 %f_norm7, %f1, %inv_rms;
    mul.f32 %f_norm7, %f_norm7, %fw1;
    cvt.rn.f16.f32 %hn0, %f_norm6;
    cvt.rn.f16.f32 %hn1, %f_norm7;
    mov.b32 %out3, {%hn0, %hn1};

    add.u64 %y_ptr, %y_ptr, %offset;
    st.global.v4.u32 [%y_ptr], {%out0, %out1, %out2, %out3};

EXIT_BNORM:
    ret;
}

.visible .entry batched_swiglu_separate_fp16(
    .param .u64 act_out,
    .param .u64 gate_in,
    .param .u64 up_in,
    .param .u32 intermediate_size,
    .param .u32 batch_size
) {
    .reg .pred %p0;
    .reg .b32 %idx, %b_idx, %b_size, %stride, %inter_size;
    .reg .b64 %out_ptr, %gate_ptr, %up_ptr, %elem_offset, %b_offset;
    .reg .b16 %h_gate, %h_up, %h_act;
    .reg .f32 %f_gate, %f_up, %f_neg_gate, %f_exp, %f_sig, %f_act;

    ld.param.u32 %inter_size, [intermediate_size];
    ld.param.u32 %b_size, [batch_size];

    mov.u32 %b_idx, %ctaid.y;
    setp.ge.u32 %p0, %b_idx, %b_size;
    @%p0 bra EXIT_SWIGLU_SEP;

    mov.u32 %idx, %tid.x;
    mov.u32 %stride, %ntid.x;
    .reg .b32 %bx;
    mov.u32 %bx, %ctaid.x;
    mad.lo.u32 %idx, %bx, %stride, %idx;

    setp.ge.u32 %p0, %idx, %inter_size;
    @%p0 bra EXIT_SWIGLU_SEP;

    ld.param.u64 %out_ptr, [act_out];
    ld.param.u64 %gate_ptr, [gate_in];
    ld.param.u64 %up_ptr, [up_in];

    mul.wide.u32 %b_offset, %b_idx, %inter_size;
    shl.b64 %b_offset, %b_offset, 1;
    add.u64 %out_ptr, %out_ptr, %b_offset;
    add.u64 %gate_ptr, %gate_ptr, %b_offset;
    add.u64 %up_ptr, %up_ptr, %b_offset;

    cvt.u64.u32 %elem_offset, %idx;
    shl.b64 %elem_offset, %elem_offset, 1;

    add.u64 %gate_ptr, %gate_ptr, %elem_offset;
    add.u64 %up_ptr, %up_ptr, %elem_offset;
    add.u64 %out_ptr, %out_ptr, %elem_offset;

    ld.global.u16 %h_gate, [%gate_ptr];
    ld.global.u16 %h_up, [%up_ptr];

    cvt.f32.f16 %f_gate, %h_gate;
    cvt.f32.f16 %f_up, %h_up;

    neg.f32 %f_neg_gate, %f_gate;
    mul.f32 %f_neg_gate, %f_neg_gate, 1.44269504;
    ex2.approx.ftz.f32 %f_exp, %f_neg_gate;
    add.f32 %f_exp, %f_exp, 1.0;
    rcp.approx.ftz.f32 %f_sig, %f_exp;

    mul.f32 %f_act, %f_gate, %f_sig;
    mul.f32 %f_act, %f_act, %f_up;

    cvt.rn.f16.f32 %h_act, %f_act;

    st.global.u16 [%out_ptr], %h_act;

EXIT_SWIGLU_SEP:
    ret;
}

.visible .entry batched_residual_add_fp16(
    .param .u64 residual_inout,
    .param .u64 update_in,
    .param .u32 hidden_size,
    .param .u32 batch_size
) {
    .reg .pred %p0;
    .reg .b32 %idx, %h_size, %b_idx, %b_size, %stride;
    .reg .b64 %res_ptr, %upd_ptr, %offset, %b_offset;
    .reg .b32 %rv0, %rv1, %rv2, %rv3, %uv0, %uv1, %uv2, %uv3, %out0, %out1, %out2, %out3;
    .reg .b16 %hr0, %hr1, %hu0, %hu1, %ho0, %ho1;
    .reg .f32 %fr0, %fr1, %fu0, %fu1, %fo0, %fo1;

    ld.param.u32 %h_size, [hidden_size];
    ld.param.u32 %b_size, [batch_size];

    mov.u32 %b_idx, %ctaid.y;
    setp.ge.u32 %p0, %b_idx, %b_size;
    @%p0 bra EXIT_RES;

    mov.u32 %idx, %tid.x;
    mov.u32 %stride, %ntid.x;
    .reg .b32 %bx;
    mov.u32 %bx, %ctaid.x;
    mad.lo.u32 %idx, %bx, %stride, %idx;
    shl.b32 %idx, %idx, 3;

    setp.ge.u32 %p0, %idx, %h_size;
    @%p0 bra EXIT_RES;

    ld.param.u64 %res_ptr, [residual_inout];
    ld.param.u64 %upd_ptr, [update_in];

    mul.wide.u32 %b_offset, %b_idx, %h_size;
    shl.b64 %b_offset, %b_offset, 1;
    add.u64 %res_ptr, %res_ptr, %b_offset;
    add.u64 %upd_ptr, %upd_ptr, %b_offset;

    cvt.u64.u32 %offset, %idx;
    shl.b64 %offset, %offset, 1;
    add.u64 %res_ptr, %res_ptr, %offset;
    add.u64 %upd_ptr, %upd_ptr, %offset;

    ld.global.v4.u32 {%rv0, %rv1, %rv2, %rv3}, [%res_ptr];
    ld.global.v4.u32 {%uv0, %uv1, %uv2, %uv3}, [%upd_ptr];

    mov.b32 {%hr0, %hr1}, %rv0;
    mov.b32 {%hu0, %hu1}, %uv0;
    cvt.f32.f16 %fr0, %hr0;
    cvt.f32.f16 %fr1, %hr1;
    cvt.f32.f16 %fu0, %hu0;
    cvt.f32.f16 %fu1, %hu1;
    add.f32 %fo0, %fr0, %fu0;
    add.f32 %fo1, %fr1, %fu1;
    cvt.rn.f16.f32 %ho0, %fo0;
    cvt.rn.f16.f32 %ho1, %fo1;
    mov.b32 %out0, {%ho0, %ho1};

    mov.b32 {%hr0, %hr1}, %rv1;
    mov.b32 {%hu0, %hu1}, %uv1;
    cvt.f32.f16 %fr0, %hr0;
    cvt.f32.f16 %fr1, %hr1;
    cvt.f32.f16 %fu0, %hu0;
    cvt.f32.f16 %fu1, %hu1;
    add.f32 %fo0, %fr0, %fu0;
    add.f32 %fo1, %fr1, %fu1;
    cvt.rn.f16.f32 %ho0, %fo0;
    cvt.rn.f16.f32 %ho1, %fo1;
    mov.b32 %out1, {%ho0, %ho1};

    mov.b32 {%hr0, %hr1}, %rv2;
    mov.b32 {%hu0, %hu1}, %uv2;
    cvt.f32.f16 %fr0, %hr0;
    cvt.f32.f16 %fr1, %hr1;
    cvt.f32.f16 %fu0, %hu0;
    cvt.f32.f16 %fu1, %hu1;
    add.f32 %fo0, %fr0, %fu0;
    add.f32 %fo1, %fr1, %fu1;
    cvt.rn.f16.f32 %ho0, %fo0;
    cvt.rn.f16.f32 %ho1, %fo1;
    mov.b32 %out2, {%ho0, %ho1};

    mov.b32 {%hr0, %hr1}, %rv3;
    mov.b32 {%hu0, %hu1}, %uv3;
    cvt.f32.f16 %fr0, %hr0;
    cvt.f32.f16 %fr1, %hr1;
    cvt.f32.f16 %fu0, %hu0;
    cvt.f32.f16 %fu1, %hu1;
    add.f32 %fo0, %fr0, %fu0;
    add.f32 %fo1, %fr1, %fu1;
    cvt.rn.f16.f32 %ho0, %fo0;
    cvt.rn.f16.f32 %ho1, %fo1;
    mov.b32 %out3, {%ho0, %ho1};

    st.global.v4.u32 [%res_ptr], {%out0, %out1, %out2, %out3};

EXIT_RES:
    ret;
}


.visible .entry copy_kv_slice_vec8_fp16(
    .param .u64 dst_cache,
    .param .u64 src_slice,
    .param .u32 num_elements
) {
    .reg .pred %p0;
    .reg .b32 %idx, %n, %b_idx, %stride, %v4_n;
    .reg .b64 %dst_ptr, %src_ptr, %offset;
    .reg .b32 %v0, %v1, %v2, %v3;

    ld.param.u32 %n, [num_elements];
    shr.u32 %v4_n, %n, 3; // 8 elements per thread (16 bytes)

    mov.u32 %idx, %tid.x;
    mov.u32 %b_idx, %ctaid.x;
    mov.u32 %stride, %ntid.x;
    mad.lo.u32 %idx, %b_idx, %stride, %idx;

    setp.ge.u32 %p0, %idx, %v4_n;
    @%p0 bra EXIT;

    ld.param.u64 %dst_ptr, [dst_cache];
    ld.param.u64 %src_ptr, [src_slice];

    cvt.u64.u32 %offset, %idx;
    shl.b64 %offset, %offset, 4; // 16 bytes per vector

    add.u64 %src_ptr, %src_ptr, %offset;
    ld.global.v4.u32 {%v0, %v1, %v2, %v3}, [%src_ptr];

    add.u64 %dst_ptr, %dst_ptr, %offset;
    st.global.v4.u32 [%dst_ptr], {%v0, %v1, %v2, %v3};

EXIT:
    ret;
}

.visible .entry fused_residual_rms_norm_vec8_fp16(
    .param .u64 residual_inout,
    .param .u64 update_in,
    .param .u64 normed_out,
    .param .u64 weight_in,
    .param .f32 epsilon,
    .param .u32 hidden_size
) {
    .reg .pred %p0, %p1, %p_lane0;
    .reg .b32 %lane_id, %warp_id, %num_warps, %tid_val, %ntid_val, %stride, %k_idx, %h_size, %s_addr;
    .reg .b64 %res_ptr, %upd_ptr, %norm_ptr, %w_ptr, %offset, %byte_off;
    .reg .b32 %rv0, %rv1, %rv2, %rv3, %uv0, %uv1, %uv2, %uv3, %wv0, %wv1, %wv2, %wv3;
    .reg .b32 %out0, %out1, %out2, %out3, %res_out0, %res_out1, %res_out2, %res_out3;
    .reg .b16 %h_r0, %h_r1, %h_u0, %h_u1, %h_w0, %h_w1, %h_sum0, %h_sum1, %h_n0, %h_n1;
    .reg .f32 %f_r0, %f_r1, %f_u0, %f_u1, %f_w0, %f_w1;
    .reg .f32 %f_res0, %f_res1, %f_res2, %f_res3, %f_res4, %f_res5, %f_res6, %f_res7;
    .reg .f32 %f_norm0, %f_norm1, %f_norm2, %f_norm3, %f_norm4, %f_norm5, %f_norm6, %f_norm7;
    .reg .f32 %sum_sq, %shfl_val, %inv_rms, %h_size_f32, %eps, %mean_sq;
    .shared .f32 s_warp_sums[8];
    .shared .f32 s_inv_rms;

    ld.param.u32 %h_size, [hidden_size];
    ld.param.f32 %eps, [epsilon];

    ld.param.u64 %res_ptr, [residual_inout];
    ld.param.u64 %upd_ptr, [update_in];
    ld.param.u64 %norm_ptr, [normed_out];
    ld.param.u64 %w_ptr, [weight_in];

    mov.u32 %tid_val, %tid.x;
    and.b32 %lane_id, %tid_val, 31;
    shr.u32 %warp_id, %tid_val, 5;
    mov.u32 %ntid_val, %ntid.x;
    shr.u32 %num_warps, %ntid_val, 5;

    // Each thread does 8 elements. Thread tid processes elements [tid*8 .. tid*8 + 7]
    shl.b32 %k_idx, %tid_val, 3;
    cvt.u64.u32 %offset, %k_idx;
    shl.b64 %byte_off, %offset, 1; // 2 bytes per half element

    mov.f32 %sum_sq, 0.0;

    setp.ge.u32 %p0, %k_idx, %h_size;
    @%p0 bra REDUCE;

    // Load residual and update vectors (16 bytes = 8 half elements)
    add.u64 %res_ptr, %res_ptr, %byte_off;
    add.u64 %upd_ptr, %upd_ptr, %byte_off;
    ld.global.v4.u32 {%rv0, %rv1, %rv2, %rv3}, [%res_ptr];
    ld.global.v4.u32 {%uv0, %uv1, %uv2, %uv3}, [%upd_ptr];

    // Vector 0: elements 0 & 1
    mov.b32 {%h_r0, %h_r1}, %rv0;
    mov.b32 {%h_u0, %h_u1}, %uv0;
    cvt.f32.f16 %f_r0, %h_r0;
    cvt.f32.f16 %f_u0, %h_u0;
    add.f32 %f_res0, %f_r0, %f_u0;
    fma.rn.f32 %sum_sq, %f_res0, %f_res0, %sum_sq;

    cvt.f32.f16 %f_r1, %h_r1;
    cvt.f32.f16 %f_u1, %h_u1;
    add.f32 %f_res1, %f_r1, %f_u1;
    fma.rn.f32 %sum_sq, %f_res1, %f_res1, %sum_sq;

    cvt.rn.f16.f32 %h_sum0, %f_res0;
    cvt.rn.f16.f32 %h_sum1, %f_res1;
    mov.b32 %res_out0, {%h_sum0, %h_sum1};

    // Vector 1: elements 2 & 3
    mov.b32 {%h_r0, %h_r1}, %rv1;
    mov.b32 {%h_u0, %h_u1}, %uv1;
    cvt.f32.f16 %f_r0, %h_r0;
    cvt.f32.f16 %f_u0, %h_u0;
    add.f32 %f_res2, %f_r0, %f_u0;
    fma.rn.f32 %sum_sq, %f_res2, %f_res2, %sum_sq;

    cvt.f32.f16 %f_r1, %h_r1;
    cvt.f32.f16 %f_u1, %h_u1;
    add.f32 %f_res3, %f_r1, %f_u1;
    fma.rn.f32 %sum_sq, %f_res3, %f_res3, %sum_sq;

    cvt.rn.f16.f32 %h_sum0, %f_res2;
    cvt.rn.f16.f32 %h_sum1, %f_res3;
    mov.b32 %res_out1, {%h_sum0, %h_sum1};

    // Vector 2: elements 4 & 5
    mov.b32 {%h_r0, %h_r1}, %rv2;
    mov.b32 {%h_u0, %h_u1}, %uv2;
    cvt.f32.f16 %f_r0, %h_r0;
    cvt.f32.f16 %f_u0, %h_u0;
    add.f32 %f_res4, %f_r0, %f_u0;
    fma.rn.f32 %sum_sq, %f_res4, %f_res4, %sum_sq;

    cvt.f32.f16 %f_r1, %h_r1;
    cvt.f32.f16 %f_u1, %h_u1;
    add.f32 %f_res5, %f_r1, %f_u1;
    fma.rn.f32 %sum_sq, %f_res5, %f_res5, %sum_sq;

    cvt.rn.f16.f32 %h_sum0, %f_res4;
    cvt.rn.f16.f32 %h_sum1, %f_res5;
    mov.b32 %res_out2, {%h_sum0, %h_sum1};

    // Vector 3: elements 6 & 7
    mov.b32 {%h_r0, %h_r1}, %rv3;
    mov.b32 {%h_u0, %h_u1}, %uv3;
    cvt.f32.f16 %f_r0, %h_r0;
    cvt.f32.f16 %f_u0, %h_u0;
    add.f32 %f_res6, %f_r0, %f_u0;
    fma.rn.f32 %sum_sq, %f_res6, %f_res6, %sum_sq;

    cvt.f32.f16 %f_r1, %h_r1;
    cvt.f32.f16 %f_u1, %h_u1;
    add.f32 %f_res7, %f_r1, %f_u1;
    fma.rn.f32 %sum_sq, %f_res7, %f_res7, %sum_sq;

    cvt.rn.f16.f32 %h_sum0, %f_res6;
    cvt.rn.f16.f32 %h_sum1, %f_res7;
    mov.b32 %res_out3, {%h_sum0, %h_sum1};

    // Write back updated residual to residual_inout!
    st.global.v4.u32 [%res_ptr], {%res_out0, %res_out1, %res_out2, %res_out3};

REDUCE:
    // Warp-level shuffle reduce
    shfl.sync.down.b32 %shfl_val, %sum_sq, 16, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 8, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 4, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 2, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 1, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;

    setp.eq.u32 %p_lane0, %lane_id, 0;
    mov.u32 %s_addr, s_warp_sums;
    shl.b32 %stride, %warp_id, 2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p_lane0 st.shared.f32 [%s_addr], %sum_sq;
    bar.sync 0;

    // Warp 0 aggregates across warps
    setp.eq.u32 %p0, %warp_id, 0;
    @!%p0 bra NORMALIZE;

    setp.lt.u32 %p1, %tid_val, %num_warps;
    mov.u32 %s_addr, s_warp_sums;
    shl.b32 %stride, %tid_val, 2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p1 ld.shared.f32 %sum_sq, [%s_addr];
    @!%p1 mov.f32 %sum_sq, 0.0;

    shfl.sync.down.b32 %shfl_val, %sum_sq, 4, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 2, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 1, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;

    setp.eq.u32 %p0, %tid_val, 0;
    @!%p0 bra NORMALIZE;

    // tid 0 computes rsqrt(sum_sq / hidden_size + eps)
    cvt.rn.f32.u32 %h_size_f32, %h_size;
    div.approx.ftz.f32 %mean_sq, %sum_sq, %h_size_f32;
    add.f32 %mean_sq, %mean_sq, %eps;
    rsqrt.approx.ftz.f32 %inv_rms, %mean_sq;
    st.shared.f32 [s_inv_rms], %inv_rms;

NORMALIZE:
    bar.sync 0;
    ld.shared.f32 %inv_rms, [s_inv_rms];

    setp.ge.u32 %p0, %k_idx, %h_size;
    @%p0 bra EXIT;

    // Load weight vector (16 bytes = 8 half elements)
    add.u64 %w_ptr, %w_ptr, %byte_off;
    add.u64 %norm_ptr, %norm_ptr, %byte_off;
    ld.global.v4.u32 {%wv0, %wv1, %wv2, %wv3}, [%w_ptr];

    // Scale and multiply by weight: normed = (res * inv_rms) * weight
    // Pair 0 & 1
    mov.b32 {%h_w0, %h_w1}, %wv0;
    cvt.f32.f16 %f_w0, %h_w0;
    cvt.f32.f16 %f_w1, %h_w1;
    mul.f32 %f_norm0, %f_res0, %inv_rms;
    mul.f32 %f_norm0, %f_norm0, %f_w0;
    mul.f32 %f_norm1, %f_res1, %inv_rms;
    mul.f32 %f_norm1, %f_norm1, %f_w1;
    cvt.rn.f16.f32 %h_n0, %f_norm0;
    cvt.rn.f16.f32 %h_n1, %f_norm1;
    mov.b32 %out0, {%h_n0, %h_n1};

    // Pair 2 & 3
    mov.b32 {%h_w0, %h_w1}, %wv1;
    cvt.f32.f16 %f_w0, %h_w0;
    cvt.f32.f16 %f_w1, %h_w1;
    mul.f32 %f_norm2, %f_res2, %inv_rms;
    mul.f32 %f_norm2, %f_norm2, %f_w0;
    mul.f32 %f_norm3, %f_res3, %inv_rms;
    mul.f32 %f_norm3, %f_norm3, %f_w1;
    cvt.rn.f16.f32 %h_n0, %f_norm2;
    cvt.rn.f16.f32 %h_n1, %f_norm3;
    mov.b32 %out1, {%h_n0, %h_n1};

    // Pair 4 & 5
    mov.b32 {%h_w0, %h_w1}, %wv2;
    cvt.f32.f16 %f_w0, %h_w0;
    cvt.f32.f16 %f_w1, %h_w1;
    mul.f32 %f_norm4, %f_res4, %inv_rms;
    mul.f32 %f_norm4, %f_norm4, %f_w0;
    mul.f32 %f_norm5, %f_res5, %inv_rms;
    mul.f32 %f_norm5, %f_norm5, %f_w1;
    cvt.rn.f16.f32 %h_n0, %f_norm4;
    cvt.rn.f16.f32 %h_n1, %f_norm5;
    mov.b32 %out2, {%h_n0, %h_n1};

    // Pair 6 & 7
    mov.b32 {%h_w0, %h_w1}, %wv3;
    cvt.f32.f16 %f_w0, %h_w0;
    cvt.f32.f16 %f_w1, %h_w1;
    mul.f32 %f_norm6, %f_res6, %inv_rms;
    mul.f32 %f_norm6, %f_norm6, %f_w0;
    mul.f32 %f_norm7, %f_res7, %inv_rms;
    mul.f32 %f_norm7, %f_norm7, %f_w1;
    cvt.rn.f16.f32 %h_n0, %f_norm6;
    cvt.rn.f16.f32 %h_n1, %f_norm7;
    mov.b32 %out3, {%h_n0, %h_n1};

    // Write out 128-bit vector
    st.global.v4.u32 [%norm_ptr], {%out0, %out1, %out2, %out3};

EXIT:
    ret;
}

// Fused in-place post-norm + residual-add:
// Computes: upd = rms_norm(upd) * post_w, then res += upd
.visible .entry fused_post_norm_residual_add_vec8_fp16(
    .param .u64 residual_inout,
    .param .u64 update_inout,
    .param .u64 weight_in,
    .param .f32 epsilon,
    .param .u32 hidden_size
) {
    .reg .pred %p0, %p1, %p_lane0;
    .reg .b32 %lane_id, %warp_id, %num_warps, %tid_val, %ntid_val, %stride, %k_idx, %h_size, %s_addr;
    .reg .b64 %res_ptr, %upd_ptr, %w_ptr, %offset, %byte_off;
    .reg .b32 %rv0, %rv1, %rv2, %rv3, %uv0, %uv1, %uv2, %uv3, %wv0, %wv1, %wv2, %wv3;
    .reg .b32 %res_out0, %res_out1, %res_out2, %res_out3;
    .reg .b16 %h_r0, %h_r1, %h_u0, %h_u1, %h_w0, %h_w1, %h_res0, %h_res1;
    .reg .f32 %f_r0, %f_r1, %f_u0, %f_u1, %f_w0, %f_w1;
    .reg .f32 %f_norm0, %f_norm1, %f_norm2, %f_norm3, %f_norm4, %f_norm5, %f_norm6, %f_norm7;
    .reg .f32 %f_sum0, %f_sum1, %f_sum2, %f_sum3, %f_sum4, %f_sum5, %f_sum6, %f_sum7;
    .reg .f32 %sum_sq, %shfl_val, %inv_rms, %h_size_f32, %eps, %mean_sq;
    .shared .f32 s_warp_sums[8];
    .shared .f32 s_inv_rms;

    ld.param.u32 %h_size, [hidden_size];
    ld.param.f32 %eps, [epsilon];

    ld.param.u64 %res_ptr, [residual_inout];
    ld.param.u64 %upd_ptr, [update_inout];
    ld.param.u64 %w_ptr, [weight_in];

    mov.u32 %tid_val, %tid.x;
    and.b32 %lane_id, %tid_val, 31;
    shr.u32 %warp_id, %tid_val, 5;
    mov.u32 %ntid_val, %ntid.x;
    shr.u32 %num_warps, %ntid_val, 5;

    // 8 elements per thread
    shl.b32 %k_idx, %tid_val, 3;
    cvt.u64.u32 %offset, %k_idx;
    shl.b64 %byte_off, %offset, 1;

    mov.f32 %sum_sq, 0.0;

    setp.ge.u32 %p0, %k_idx, %h_size;
    @%p0 bra REDUCE;

    // Load update vector (16 bytes = 8 elements)
    add.u64 %upd_ptr, %upd_ptr, %byte_off;
    ld.global.v4.u32 {%uv0, %uv1, %uv2, %uv3}, [%upd_ptr];

    // Compute sum of squares of update
    // Pair 0 & 1
    mov.b32 {%h_u0, %h_u1}, %uv0;
    cvt.f32.f16 %f_u0, %h_u0;
    cvt.f32.f16 %f_u1, %h_u1;
    fma.rn.f32 %sum_sq, %f_u0, %f_u0, %sum_sq;
    fma.rn.f32 %sum_sq, %f_u1, %f_u1, %sum_sq;

    // Pair 2 & 3
    mov.b32 {%h_u0, %h_u1}, %uv1;
    cvt.f32.f16 %f_u0, %h_u0;
    cvt.f32.f16 %f_u1, %h_u1;
    fma.rn.f32 %sum_sq, %f_u0, %f_u0, %sum_sq;
    fma.rn.f32 %sum_sq, %f_u1, %f_u1, %sum_sq;

    // Pair 4 & 5
    mov.b32 {%h_u0, %h_u1}, %uv2;
    cvt.f32.f16 %f_u0, %h_u0;
    cvt.f32.f16 %f_u1, %h_u1;
    fma.rn.f32 %sum_sq, %f_u0, %f_u0, %sum_sq;
    fma.rn.f32 %sum_sq, %f_u1, %f_u1, %sum_sq;

    // Pair 6 & 7
    mov.b32 {%h_u0, %h_u1}, %uv3;
    cvt.f32.f16 %f_u0, %h_u0;
    cvt.f32.f16 %f_u1, %h_u1;
    fma.rn.f32 %sum_sq, %f_u0, %f_u0, %sum_sq;
    fma.rn.f32 %sum_sq, %f_u1, %f_u1, %sum_sq;

REDUCE:
    shfl.sync.down.b32 %shfl_val, %sum_sq, 16, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 8, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 4, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 2, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 1, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;

    setp.eq.u32 %p_lane0, %lane_id, 0;
    mov.u32 %s_addr, s_warp_sums;
    shl.b32 %stride, %warp_id, 2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p_lane0 st.shared.f32 [%s_addr], %sum_sq;
    bar.sync 0;

    setp.eq.u32 %p0, %warp_id, 0;
    @!%p0 bra NORM_AND_ADD;

    setp.lt.u32 %p1, %tid_val, %num_warps;
    mov.u32 %s_addr, s_warp_sums;
    shl.b32 %stride, %tid_val, 2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p1 ld.shared.f32 %sum_sq, [%s_addr];
    @!%p1 mov.f32 %sum_sq, 0.0;

    shfl.sync.down.b32 %shfl_val, %sum_sq, 4, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 2, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 1, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;

    setp.eq.u32 %p0, %tid_val, 0;
    @!%p0 bra NORM_AND_ADD;

    cvt.rn.f32.u32 %h_size_f32, %h_size;
    div.approx.ftz.f32 %mean_sq, %sum_sq, %h_size_f32;
    add.f32 %mean_sq, %mean_sq, %eps;
    rsqrt.approx.ftz.f32 %inv_rms, %mean_sq;
    st.shared.f32 [s_inv_rms], %inv_rms;

NORM_AND_ADD:
    bar.sync 0;
    ld.shared.f32 %inv_rms, [s_inv_rms];

    setp.ge.u32 %p0, %k_idx, %h_size;
    @%p0 bra EXIT;

    // Load weight and residual vectors
    add.u64 %w_ptr, %w_ptr, %byte_off;
    add.u64 %res_ptr, %res_ptr, %byte_off;
    ld.global.v4.u32 {%wv0, %wv1, %wv2, %wv3}, [%w_ptr];
    ld.global.v4.u32 {%rv0, %rv1, %rv2, %rv3}, [%res_ptr];

    // Pair 0 & 1
    mov.b32 {%h_u0, %h_u1}, %uv0;
    mov.b32 {%h_w0, %h_w1}, %wv0;
    mov.b32 {%h_r0, %h_r1}, %rv0;
    cvt.f32.f16 %f_u0, %h_u0;
    cvt.f32.f16 %f_u1, %h_u1;
    cvt.f32.f16 %f_w0, %h_w0;
    cvt.f32.f16 %f_w1, %h_w1;
    cvt.f32.f16 %f_r0, %h_r0;
    cvt.f32.f16 %f_r1, %h_r1;

    mul.f32 %f_norm0, %f_u0, %inv_rms;
    fma.rn.f32 %f_sum0, %f_norm0, %f_w0, %f_r0;
    mul.f32 %f_norm1, %f_u1, %inv_rms;
    fma.rn.f32 %f_sum1, %f_norm1, %f_w1, %f_r1;
    cvt.rn.f16.f32 %h_res0, %f_sum0;
    cvt.rn.f16.f32 %h_res1, %f_sum1;
    mov.b32 %res_out0, {%h_res0, %h_res1};

    // Pair 2 & 3
    mov.b32 {%h_u0, %h_u1}, %uv1;
    mov.b32 {%h_w0, %h_w1}, %wv1;
    mov.b32 {%h_r0, %h_r1}, %rv1;
    cvt.f32.f16 %f_u0, %h_u0;
    cvt.f32.f16 %f_u1, %h_u1;
    cvt.f32.f16 %f_w0, %h_w0;
    cvt.f32.f16 %f_w1, %h_w1;
    cvt.f32.f16 %f_r0, %h_r0;
    cvt.f32.f16 %f_r1, %h_r1;

    mul.f32 %f_norm2, %f_u0, %inv_rms;
    fma.rn.f32 %f_sum2, %f_norm2, %f_w0, %f_r0;
    mul.f32 %f_norm3, %f_u1, %inv_rms;
    fma.rn.f32 %f_sum3, %f_norm3, %f_w1, %f_r1;
    cvt.rn.f16.f32 %h_res0, %f_sum2;
    cvt.rn.f16.f32 %h_res1, %f_sum3;
    mov.b32 %res_out1, {%h_res0, %h_res1};

    // Pair 4 & 5
    mov.b32 {%h_u0, %h_u1}, %uv2;
    mov.b32 {%h_w0, %h_w1}, %wv2;
    mov.b32 {%h_r0, %h_r1}, %rv2;
    cvt.f32.f16 %f_u0, %h_u0;
    cvt.f32.f16 %f_u1, %h_u1;
    cvt.f32.f16 %f_w0, %h_w0;
    cvt.f32.f16 %f_w1, %h_w1;
    cvt.f32.f16 %f_r0, %h_r0;
    cvt.f32.f16 %f_r1, %h_r1;

    mul.f32 %f_norm4, %f_u0, %inv_rms;
    fma.rn.f32 %f_sum4, %f_norm4, %f_w0, %f_r0;
    mul.f32 %f_norm5, %f_u1, %inv_rms;
    fma.rn.f32 %f_sum5, %f_norm5, %f_w1, %f_r1;
    cvt.rn.f16.f32 %h_res0, %f_sum4;
    cvt.rn.f16.f32 %h_res1, %f_sum5;
    mov.b32 %res_out2, {%h_res0, %h_res1};

    // Pair 6 & 7
    mov.b32 {%h_u0, %h_u1}, %uv3;
    mov.b32 {%h_w0, %h_w1}, %wv3;
    mov.b32 {%h_r0, %h_r1}, %rv3;
    cvt.f32.f16 %f_u0, %h_u0;
    cvt.f32.f16 %f_u1, %h_u1;
    cvt.f32.f16 %f_w0, %h_w0;
    cvt.f32.f16 %f_w1, %h_w1;
    cvt.f32.f16 %f_r0, %h_r0;
    cvt.f32.f16 %f_r1, %h_r1;

    mul.f32 %f_norm6, %f_u0, %inv_rms;
    fma.rn.f32 %f_sum6, %f_norm6, %f_w0, %f_r0;
    mul.f32 %f_norm7, %f_u1, %inv_rms;
    fma.rn.f32 %f_sum7, %f_norm7, %f_w1, %f_r1;
    cvt.rn.f16.f32 %h_res0, %f_sum6;
    cvt.rn.f16.f32 %h_res1, %f_sum7;
    mov.b32 %res_out3, {%h_res0, %h_res1};

    // Store updated residual directly to residual_inout!
    st.global.v4.u32 [%res_ptr], {%res_out0, %res_out1, %res_out2, %res_out3};

EXIT:
    ret;
}

// 128-bit Vectorized RMSNorm (256 threads x 8 elements = 2048 elements)
.visible .entry rms_norm_vec8_fp16(
    .param .u64 normed_out,
    .param .u64 input_in,
    .param .u64 weight_in,
    .param .f32 epsilon,
    .param .u32 hidden_size
) {
    .reg .pred %p0, %p1, %p_lane0;
    .reg .b32 %lane_id, %warp_id, %num_warps, %tid_val, %ntid_val, %stride, %k_idx, %h_size, %s_addr;
    .reg .b64 %in_ptr, %norm_ptr, %w_ptr, %offset, %byte_off;
    .reg .b32 %iv0, %iv1, %iv2, %iv3, %wv0, %wv1, %wv2, %wv3, %out0, %out1, %out2, %out3;
    .reg .b16 %h_i0, %h_i1, %h_w0, %h_w1, %h_n0, %h_n1;
    .reg .f32 %f_i0, %f_i1, %f_w0, %f_w1, %f_norm0, %f_norm1;
    .reg .f32 %f_in0, %f_in1, %f_in2, %f_in3, %f_in4, %f_in5, %f_in6, %f_in7;
    .reg .f32 %sum_sq, %shfl_val, %inv_rms, %h_size_f32, %eps, %mean_sq;
    .shared .f32 s_warp_sums[8];
    .shared .f32 s_inv_rms;

    ld.param.u32 %h_size, [hidden_size];
    ld.param.f32 %eps, [epsilon];

    ld.param.u64 %norm_ptr, [normed_out];
    ld.param.u64 %in_ptr, [input_in];
    ld.param.u64 %w_ptr, [weight_in];

    mov.u32 %tid_val, %tid.x;
    and.b32 %lane_id, %tid_val, 31;
    shr.u32 %warp_id, %tid_val, 5;
    mov.u32 %ntid_val, %ntid.x;
    shr.u32 %num_warps, %ntid_val, 5;

    // 8 elements per thread
    shl.b32 %k_idx, %tid_val, 3;
    cvt.u64.u32 %offset, %k_idx;
    shl.b64 %byte_off, %offset, 1;

    mov.f32 %sum_sq, 0.0;

    setp.ge.u32 %p0, %k_idx, %h_size;
    @%p0 bra REDUCE;

    // Load 128-bit vector
    add.u64 %in_ptr, %in_ptr, %byte_off;
    ld.global.v4.u32 {%iv0, %iv1, %iv2, %iv3}, [%in_ptr];

    // Pair 0 & 1
    mov.b32 {%h_i0, %h_i1}, %iv0;
    cvt.f32.f16 %f_in0, %h_i0;
    cvt.f32.f16 %f_in1, %h_i1;
    fma.rn.f32 %sum_sq, %f_in0, %f_in0, %sum_sq;
    fma.rn.f32 %sum_sq, %f_in1, %f_in1, %sum_sq;

    // Pair 2 & 3
    mov.b32 {%h_i0, %h_i1}, %iv1;
    cvt.f32.f16 %f_in2, %h_i0;
    cvt.f32.f16 %f_in3, %h_i1;
    fma.rn.f32 %sum_sq, %f_in2, %f_in2, %sum_sq;
    fma.rn.f32 %sum_sq, %f_in3, %f_in3, %sum_sq;

    // Pair 4 & 5
    mov.b32 {%h_i0, %h_i1}, %iv2;
    cvt.f32.f16 %f_in4, %h_i0;
    cvt.f32.f16 %f_in5, %h_i1;
    fma.rn.f32 %sum_sq, %f_in4, %f_in4, %sum_sq;
    fma.rn.f32 %sum_sq, %f_in5, %f_in5, %sum_sq;

    // Pair 6 & 7
    mov.b32 {%h_i0, %h_i1}, %iv3;
    cvt.f32.f16 %f_in6, %h_i0;
    cvt.f32.f16 %f_in7, %h_i1;
    fma.rn.f32 %sum_sq, %f_in6, %f_in6, %sum_sq;
    fma.rn.f32 %sum_sq, %f_in7, %f_in7, %sum_sq;

REDUCE:
    shfl.sync.down.b32 %shfl_val, %sum_sq, 16, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 8, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 4, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 2, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 1, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;

    setp.eq.u32 %p_lane0, %lane_id, 0;
    mov.u32 %s_addr, s_warp_sums;
    shl.b32 %stride, %warp_id, 2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p_lane0 st.shared.f32 [%s_addr], %sum_sq;
    bar.sync 0;

    setp.eq.u32 %p0, %warp_id, 0;
    @!%p0 bra NORMALIZE;

    setp.lt.u32 %p1, %tid_val, %num_warps;
    mov.u32 %s_addr, s_warp_sums;
    shl.b32 %stride, %tid_val, 2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p1 ld.shared.f32 %sum_sq, [%s_addr];
    @!%p1 mov.f32 %sum_sq, 0.0;

    shfl.sync.down.b32 %shfl_val, %sum_sq, 4, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 2, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum_sq, 1, 31, 0xffffffff;
    add.f32 %sum_sq, %sum_sq, %shfl_val;

    setp.eq.u32 %p0, %tid_val, 0;
    @!%p0 bra NORMALIZE;

    cvt.rn.f32.u32 %h_size_f32, %h_size;
    div.approx.ftz.f32 %mean_sq, %sum_sq, %h_size_f32;
    add.f32 %mean_sq, %mean_sq, %eps;
    rsqrt.approx.ftz.f32 %inv_rms, %mean_sq;
    st.shared.f32 [s_inv_rms], %inv_rms;

NORMALIZE:
    bar.sync 0;
    ld.shared.f32 %inv_rms, [s_inv_rms];

    setp.ge.u32 %p0, %k_idx, %h_size;
    @%p0 bra EXIT;

    // Load weight vector
    add.u64 %w_ptr, %w_ptr, %byte_off;
    add.u64 %norm_ptr, %norm_ptr, %byte_off;
    ld.global.v4.u32 {%wv0, %wv1, %wv2, %wv3}, [%w_ptr];

    // Scale and multiply by weight: normed = (in * inv_rms) * weight
    // Pair 0 & 1
    mov.b32 {%h_w0, %h_w1}, %wv0;
    cvt.f32.f16 %f_w0, %h_w0;
    cvt.f32.f16 %f_w1, %h_w1;
    mul.f32 %f_norm0, %f_in0, %inv_rms;
    mul.f32 %f_norm0, %f_norm0, %f_w0;
    mul.f32 %f_norm1, %f_in1, %inv_rms;
    mul.f32 %f_norm1, %f_norm1, %f_w1;
    cvt.rn.f16.f32 %h_n0, %f_norm0;
    cvt.rn.f16.f32 %h_n1, %f_norm1;
    mov.b32 %out0, {%h_n0, %h_n1};

    // Pair 2 & 3
    mov.b32 {%h_w0, %h_w1}, %wv1;
    cvt.f32.f16 %f_w0, %h_w0;
    cvt.f32.f16 %f_w1, %h_w1;
    mul.f32 %f_norm0, %f_in2, %inv_rms;
    mul.f32 %f_norm0, %f_norm0, %f_w0;
    mul.f32 %f_norm1, %f_in3, %inv_rms;
    mul.f32 %f_norm1, %f_norm1, %f_w1;
    cvt.rn.f16.f32 %h_n0, %f_norm0;
    cvt.rn.f16.f32 %h_n1, %f_norm1;
    mov.b32 %out1, {%h_n0, %h_n1};

    // Pair 4 & 5
    mov.b32 {%h_w0, %h_w1}, %wv2;
    cvt.f32.f16 %f_w0, %h_w0;
    cvt.f32.f16 %f_w1, %h_w1;
    mul.f32 %f_norm0, %f_in4, %inv_rms;
    mul.f32 %f_norm0, %f_norm0, %f_w0;
    mul.f32 %f_norm1, %f_in5, %inv_rms;
    mul.f32 %f_norm1, %f_norm1, %f_w1;
    cvt.rn.f16.f32 %h_n0, %f_norm0;
    cvt.rn.f16.f32 %h_n1, %f_norm1;
    mov.b32 %out2, {%h_n0, %h_n1};

    // Pair 6 & 7
    mov.b32 {%h_w0, %h_w1}, %wv3;
    cvt.f32.f16 %f_w0, %h_w0;
    cvt.f32.f16 %f_w1, %h_w1;
    mul.f32 %f_norm0, %f_in6, %inv_rms;
    mul.f32 %f_norm0, %f_norm0, %f_w0;
    mul.f32 %f_norm1, %f_in7, %inv_rms;
    mul.f32 %f_norm1, %f_norm1, %f_w1;
    cvt.rn.f16.f32 %h_n0, %f_norm0;
    cvt.rn.f16.f32 %h_n1, %f_norm1;
    mov.b32 %out3, {%h_n0, %h_n1};

    st.global.v4.u32 [%norm_ptr], {%out0, %out1, %out2, %out3};

EXIT:
    ret;
}

.visible .entry residual_add_fp16(
    .param .u64 a,
    .param .u64 b,
    .param .u32 num_elements
) {
    .reg .pred %p0;
    .reg .b32 %idx, %n, %b_idx, %stride;
    .reg .b64 %a_ptr, %b_ptr, %offset;
    .reg .b16 %ha, %hb;
    .reg .f32 %fa, %fb, %fres;

    ld.param.u32 %n, [num_elements];
    mov.u32 %idx, %tid.x;
    mov.u32 %b_idx, %ctaid.x;
    mov.u32 %stride, %ntid.x;
    mad.lo.u32 %idx, %b_idx, %stride, %idx;

    setp.ge.u32 %p0, %idx, %n;
    @%p0 bra EXIT;

    ld.param.u64 %a_ptr, [a];
    ld.param.u64 %b_ptr, [b];

    cvt.u64.u32 %offset, %idx;
    shl.b64 %offset, %offset, 1;

    add.u64 %b_ptr, %b_ptr, %offset;
    ld.global.u16 %hb, [%b_ptr];

    add.u64 %a_ptr, %a_ptr, %offset;
    ld.global.u16 %ha, [%a_ptr];

    cvt.f32.f16 %fa, %ha;
    cvt.f32.f16 %fb, %hb;
    add.f32 %fres, %fa, %fb;
    cvt.rn.f16.f32 %ha, %fres;

    st.global.u16 [%a_ptr], %ha;

EXIT:
    ret;
}
"#;

pub enum LayerType {
    Conv(CausalConvLayer),
    Attention(Lfm3AttentionLayer),
}

pub struct FullLfm3DecoderLayer {
    pub inner_layer: LayerType,
    pub ffn: FfnLayer,
    pub operator_norm: DeviceBuffer<half::f16>,
    pub ffn_norm: DeviceBuffer<half::f16>,
    pub post_operator_norm: DeviceBuffer<half::f16>,
    pub post_ffn_norm: DeviceBuffer<half::f16>,
    pub norm_ptx: PtxModule,
    pub add_ptx: PtxModule,
    pub hidden_size: usize,
    pub rope: Option<std::sync::Arc<RotaryEmbeddingTable>>,
    pub gqa: Option<GqaAttentionEngine>,
    pub k_cache: Option<DeviceBuffer<half::f16>>,
    pub v_cache: Option<DeviceBuffer<half::f16>>,
}

impl FullLfm3DecoderLayer {
    fn maybe_dump_qkv(
        gpu: &GpuContext,
        layer_idx: usize,
        pos_id: usize,
        q_out: &DeviceBuffer<half::f16>,
        k_out: &DeviceBuffer<half::f16>,
        v_out: &DeviceBuffer<half::f16>,
    ) {
        let dir = match std::env::var("OXIDE_DUMP_QKV_DIR") {
            Ok(d) => d,
            Err(_) => return,
        };
        let l_ok = std::env::var("OXIDE_DUMP_LAYER").ok().and_then(|s| s.parse::<usize>().ok()).map(|v| v == layer_idx).unwrap_or(true);
        let p_ok = std::env::var("OXIDE_DUMP_POS").ok().and_then(|s| s.parse::<usize>().ok()).map(|v| v == pos_id).unwrap_or(true);
        if !l_ok || !p_ok {
            return;
        }
        let stream = gpu.compute_stream();
        let _ = stream.synchronize();
        let write = |name: &str, buf: &DeviceBuffer<half::f16>| {
            if let Ok(host) = buf.inner().to_host_vec(stream) {
                let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 2) };
                let _ = std::fs::write(format!("{}/l{}_p{}_{}.bin", dir, layer_idx, pos_id, name), bytes);
            }
        };
        write("q", q_out);
        write("k", k_out);
        write("v", v_out);
        eprintln!("[qkv] dumped layer {} pos {}", layer_idx, pos_id);
    }

    pub fn load_from_safetensors(
        gpu: &GpuContext,
        loader: &SafeTensorsLoader,
        layer_idx: usize,
        is_attention: bool,
        is_sliding: bool,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        // RoPE tables are pure functions of (theta, partial factor, window)
        // and position, and are read-only at inference. Size them to the
        // server's sequence limit rather than a hard 131072: at the default
        // 8192 that is 11 layers x 2 tables x 64 MiB -> 8 MiB, a 0.68 GiB
        // saving. `OXIDE_MAX_SEQ_TOKENS` must match the server's admission
        // limit or a long position would index past the table.
        let max_seq_tokens = std::env::var("OXIDE_MAX_SEQ_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8192);
        let stream = gpu.compute_stream();
        let convert_bf16_to_f16 = |buf: &DeviceBuffer<half::bf16>| -> Result<DeviceBuffer<half::f16>> {
            let host_bf16 = buf.inner().to_host_vec(stream)?;
            let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
            DeviceBuffer::from_slice(gpu, &host_f16)
        };

        let (inner_layer, rope, gqa, k_cache, v_cache) = if is_attention {
            let attn = Lfm3AttentionLayer::load_from_safetensors(
                gpu, loader, layer_idx, hidden_size, 16, 4, 128,
            )?;
            let theta = if is_sliding { 10000.0 } else { 1000000.0 };
            let partial_factor = if is_sliding { 1.0 } else { 0.5 };
            // LFM3 mixes full and sliding layers: the sliding ones use a
            // 256-token window (config `sliding_window`), so the kernel must
            // mask keys whose distance reaches the window. `OXIDE_SWA`
            // overrides the window for A/B experiments (0 = full attention).
            let window = if is_sliding {
                std::env::var("OXIDE_SWA")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or(256)
            } else {
                0
            };
            let rope_table =
                crate::rope::shared_rope_table(
                    gpu, max_seq_tokens, 128, theta, partial_factor, window,
                )?;
            let gqa_engine = GqaAttentionEngine::new(gpu)?;
            // Per-layer KV caches exist only for the scalar forward path;
            // the batched engine keeps its sequences in the shared KV pool.
            // Skip them (11 layers x 8 MiB) when the batched engine is used.
            let scalar = std::env::var("OXIDE_SCALAR_BATCH").is_ok();
            let (kc, vc) = if scalar {
                (
                    Some(DeviceBuffer::<half::f16>::zeroed(gpu, 4096 * 512)?),
                    Some(DeviceBuffer::<half::f16>::zeroed(gpu, 4096 * 512)?),
                )
            } else {
                (None, None)
            };
            (LayerType::Attention(attn), Some(rope_table), Some(gqa_engine), kc, vc)
        } else {
            let conv = CausalConvLayer::load_from_safetensors(
                gpu, loader, layer_idx, hidden_size, 7,
            )?;
            (LayerType::Conv(conv), None, None, None, None)
        };

        let ffn = FfnLayer::load_from_safetensors(gpu, loader, layer_idx, hidden_size, intermediate_size)?;

        let op_norm_bf16 = loader.load_tensor::<half::bf16>(gpu, &format!("model.language_model.layers.{}.operator_norm.weight", layer_idx))?;
        let ffn_norm_bf16 = loader.load_tensor::<half::bf16>(gpu, &format!("model.language_model.layers.{}.ffn_norm.weight", layer_idx))?;
        let post_op_bf16 = loader.load_tensor::<half::bf16>(gpu, &format!("model.language_model.layers.{}.post_operator_norm.weight", layer_idx))?;
        let post_ffn_bf16 = loader.load_tensor::<half::bf16>(gpu, &format!("model.language_model.layers.{}.post_ffn_norm.weight", layer_idx))?;

        let operator_norm = convert_bf16_to_f16(&op_norm_bf16)?;
        let ffn_norm = convert_bf16_to_f16(&ffn_norm_bf16)?;
        let post_operator_norm = convert_bf16_to_f16(&post_op_bf16)?;
        let post_ffn_norm = convert_bf16_to_f16(&post_ffn_bf16)?;

        let norm_ptx = PtxModule::load_from_ptx(gpu, RMS_NORM_FP16_PTX)?;
        let add_ptx = PtxModule::load_from_ptx(gpu, RESIDUAL_ADD_PTX)?;

        Ok(Self {
            inner_layer,
            ffn,
            operator_norm,
            ffn_norm,
            post_operator_norm,
            post_ffn_norm,
            norm_ptx,
            add_ptx,
            hidden_size,
            rope,
            gqa,
            k_cache,
            v_cache,
        })
    }

    pub fn forward(
        &mut self,
        gpu: &GpuContext,
        hidden_states: &mut DeviceBuffer<half::f16>,
        scratch_norm: &mut DeviceBuffer<half::f16>,
        scratch_layer_out: &mut DeviceBuffer<half::f16>,
        scratch_ffn_out: &mut DeviceBuffer<half::f16>,
        // Attention scratch buffers
        q_out: &mut DeviceBuffer<half::f16>,
        k_out: &mut DeviceBuffer<half::f16>,
        v_out: &mut DeviceBuffer<half::f16>,
        gate_out: &mut DeviceBuffer<half::f16>,
        pos_id: usize,
    ) -> Result<()> {
        let stream = gpu.compute_stream();
        let rms_func = self.add_ptx.get_function("rms_norm_vec8_fp16")?;
        let add_func = self.add_ptx.get_function("residual_add_fp16")?;
        let mut epsilon: f32 = 1e-5;
        let mut h_size = self.hidden_size as u32;

        let block_dim: u32 = 128;
        let grid_dim: u32 = (h_size + block_dim - 1) / block_dim;

        // 1. operator_norm(hidden_states) -> scratch_norm
        let mut d_normed = scratch_norm.inner_mut().cu_deviceptr();
        let mut d_hidden = hidden_states.inner().cu_deviceptr();
        let mut d_op_w = self.operator_norm.inner().cu_deviceptr();

        let mut args_norm1: [*mut std::ffi::c_void; 5] = [
            &mut d_normed as *mut _ as *mut std::ffi::c_void,
            &mut d_hidden as *mut _ as *mut std::ffi::c_void,
            &mut d_op_w as *mut _ as *mut std::ffi::c_void,
            &mut epsilon as *mut _ as *mut std::ffi::c_void,
            &mut h_size as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&rms_func, (1, 1, 1), (256, 1, 1), 0, stream, &mut args_norm1)?;
        }

        // 2. Inner Layer forward -> scratch_layer_out
        match &mut self.inner_layer {
            LayerType::Conv(conv) => {
                conv.forward(gpu, scratch_norm, scratch_layer_out)?;
            }
            LayerType::Attention(attn) => {
                attn.forward_projections(gpu, scratch_norm, q_out, k_out, v_out, gate_out)?;

                // Apply RoPE to q_out (16 heads) and k_out (4 heads)
                if let Some(rope) = &self.rope {
                    rope.apply(gpu, q_out, pos_id, 16)?;
                    rope.apply(gpu, k_out, pos_id, 4)?;
                }

                Self::maybe_dump_qkv(gpu, attn.layer_idx, pos_id, q_out, k_out, v_out);

                // Copy current k_out and v_out into k_cache and v_cache at pos_id directly on device (D2D)!
                let kv_dim = 4 * 128; // 512 elements
                if let (Some(kc), Some(vc)) = (&mut self.k_cache, &mut self.v_cache) {
                    let offset = pos_id * kv_dim;
                    let mut kc_ptr = kc.inner_mut().cu_deviceptr() + (offset * 2) as u64;
                    let mut vc_ptr = vc.inner_mut().cu_deviceptr() + (offset * 2) as u64;
                    let mut d_k = k_out.inner().cu_deviceptr();
                    let mut d_v = v_out.inner().cu_deviceptr();
                    let mut kv_dim_u32 = kv_dim as u32;

                    let copy_func = self.add_ptx.get_function("copy_kv_slice_vec8_fp16")?;
                    let mut args_k: [*mut std::ffi::c_void; 3] = [
                        &mut kc_ptr as *mut _ as *mut std::ffi::c_void,
                        &mut d_k as *mut _ as *mut std::ffi::c_void,
                        &mut kv_dim_u32 as *mut _ as *mut std::ffi::c_void,
                    ];
                    let mut args_v: [*mut std::ffi::c_void; 3] = [
                        &mut vc_ptr as *mut _ as *mut std::ffi::c_void,
                        &mut d_v as *mut _ as *mut std::ffi::c_void,
                        &mut kv_dim_u32 as *mut _ as *mut std::ffi::c_void,
                    ];

                    // 512 elements / 8 = 64 threads (1 block of 64 threads)
                    unsafe {
                        launch_kernel_on_stream(&copy_func, (1, 1, 1), (64, 1, 1), 0, stream, &mut args_k)?;
                        launch_kernel_on_stream(&copy_func, (1, 1, 1), (64, 1, 1), 0, stream, &mut args_v)?;
                    }

                    // Run GQA attention engine over the accumulated KV cache [0..=pos_id]
                    if let Some(gqa) = &self.gqa {
                        gqa.compute_attention(
                            gpu,
                            &mut attn.context_buf,
                            q_out,
                            kc,
                            vc,
                            pos_id + 1,
                            16,
                            4,
                            128,
                        )?;
                        if std::env::var("OXIDE_DUMP_QKV_DIR").is_ok() {
                            let _ = stream.synchronize();
                            if let Ok(host) = attn.context_buf.inner().to_host_vec(stream) {
                                let bytes = unsafe {
                                    std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 2)
                                };
                                let dir = std::env::var("OXIDE_DUMP_QKV_DIR").unwrap_or_default();
                                let _ = std::fs::write(
                                    format!("{}/l{}_p{}_ctx.bin", dir, attn.layer_idx, pos_id),
                                    bytes,
                                );
                            }
                        }
                    }
                }

                // Apply per-head sigmoid gating and o_proj
                attn.forward_gating_and_o_proj(gpu, gate_out, scratch_layer_out)?;
            }
        }

        // 3. post_operator_norm on scratch_layer_out
        let mut d_layer_out = scratch_layer_out.inner_mut().cu_deviceptr();
        let mut d_post_op = self.post_operator_norm.inner().cu_deviceptr();
        let mut args_post_op: [*mut std::ffi::c_void; 5] = [
            &mut d_layer_out as *mut _ as *mut std::ffi::c_void,
            &mut d_layer_out as *mut _ as *mut std::ffi::c_void,
            &mut d_post_op as *mut _ as *mut std::ffi::c_void,
            &mut epsilon as *mut _ as *mut std::ffi::c_void,
            &mut h_size as *mut _ as *mut std::ffi::c_void,
        ];
        // 2048 elements / 8 elements per thread = 256 threads
        unsafe {
            launch_kernel_on_stream(&rms_func, (1, 1, 1), (256, 1, 1), 0, stream, &mut args_post_op)?;
        }

        // 4 & 5. Fused Residual 1 + ffn_norm:
        // hidden_states += scratch_layer_out, and scratch_norm = rms_norm(hidden_states) * ffn_norm
        let fused_res_norm_func = self.add_ptx.get_function("fused_residual_rms_norm_vec8_fp16")?;
        let mut d_hidden_mut = hidden_states.inner_mut().cu_deviceptr();
        let mut d_ffn_w = self.ffn_norm.inner().cu_deviceptr();
        let mut args_fused_res1: [*mut std::ffi::c_void; 6] = [
            &mut d_hidden_mut as *mut _ as *mut std::ffi::c_void,
            &mut d_layer_out as *mut _ as *mut std::ffi::c_void,
            &mut d_normed as *mut _ as *mut std::ffi::c_void,
            &mut d_ffn_w as *mut _ as *mut std::ffi::c_void,
            &mut epsilon as *mut _ as *mut std::ffi::c_void,
            &mut h_size as *mut _ as *mut std::ffi::c_void,
        ];
        // 2048 elements / 8 elements per thread = 256 threads (1 block of 256 threads)
        unsafe {
            launch_kernel_on_stream(&fused_res_norm_func, (1, 1, 1), (256, 1, 1), 0, stream, &mut args_fused_res1)?;
        }

        // 6. FFN forward: scratch_norm -> scratch_ffn_out
        self.ffn.forward(gpu, scratch_norm, scratch_ffn_out)?;

        // 7 & 8. Fused Post-FFN Norm + Residual 2:
        // Computes: scratch_ffn_out = rms_norm(scratch_ffn_out) * post_ffn_norm, then hidden_states += scratch_ffn_out in 1 kernel!
        let fused_post_add_func = self.add_ptx.get_function("fused_post_norm_residual_add_vec8_fp16")?;
        let mut d_ffn_out = scratch_ffn_out.inner_mut().cu_deviceptr();
        let mut d_post_ffn = self.post_ffn_norm.inner().cu_deviceptr();
        let mut args_fused_res2: [*mut std::ffi::c_void; 5] = [
            &mut d_hidden_mut as *mut _ as *mut std::ffi::c_void,
            &mut d_ffn_out as *mut _ as *mut std::ffi::c_void,
            &mut d_post_ffn as *mut _ as *mut std::ffi::c_void,
            &mut epsilon as *mut _ as *mut std::ffi::c_void,
            &mut h_size as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&fused_post_add_func, (1, 1, 1), (256, 1, 1), 0, stream, &mut args_fused_res2)?;
        }

        Ok(())
    }

    /// Part 1 of forward: Operator norm, inner layer (Conv or Attention), post-operator norm,
    /// and fused residual add + ffn_norm.
    /// Supports per-request state (k_cache, v_cache, conv_state) for multi-sequence concurrency.
    pub fn forward_pre_ffn_with_cache(
        &mut self,
        gpu: &GpuContext,
        hidden_states: &mut DeviceBuffer<half::f16>,
        scratch_norm: &mut DeviceBuffer<half::f16>,
        scratch_layer_out: &mut DeviceBuffer<half::f16>,
        q_out: &mut DeviceBuffer<half::f16>,
        k_out: &mut DeviceBuffer<half::f16>,
        v_out: &mut DeviceBuffer<half::f16>,
        gate_out: &mut DeviceBuffer<half::f16>,
        pos_id: usize,
        req_k_cache: Option<&mut DeviceBuffer<half::f16>>,
        req_v_cache: Option<&mut DeviceBuffer<half::f16>>,
        req_conv_state: Option<&mut DeviceBuffer<half::f16>>,
    ) -> Result<()> {
        let stream = gpu.compute_stream();
        let rms_func = self.add_ptx.get_function("rms_norm_vec8_fp16")?;
        let mut epsilon: f32 = 1e-5;
        let mut h_size = self.hidden_size as u32;

        // 1. operator_norm(hidden_states) -> scratch_norm
        let mut d_normed = scratch_norm.inner_mut().cu_deviceptr();
        let mut d_hidden = hidden_states.inner().cu_deviceptr();
        let mut d_op_w = self.operator_norm.inner().cu_deviceptr();

        let mut args_norm1: [*mut std::ffi::c_void; 5] = [
            &mut d_normed as *mut _ as *mut std::ffi::c_void,
            &mut d_hidden as *mut _ as *mut std::ffi::c_void,
            &mut d_op_w as *mut _ as *mut std::ffi::c_void,
            &mut epsilon as *mut _ as *mut std::ffi::c_void,
            &mut h_size as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&rms_func, (1, 1, 1), (256, 1, 1), 0, stream, &mut args_norm1)?;
        }

        // 2. Inner Layer forward -> scratch_layer_out
        match &mut self.inner_layer {
            LayerType::Conv(conv) => {
                if let Some(cs) = req_conv_state {
                    conv.forward_with_cache(gpu, scratch_norm, scratch_layer_out, cs)?;
                } else {
                    conv.forward(gpu, scratch_norm, scratch_layer_out)?;
                }
            }
            LayerType::Attention(attn) => {
                attn.forward_projections(gpu, scratch_norm, q_out, k_out, v_out, gate_out)?;

                if let Some(rope) = &self.rope {
                    rope.apply(gpu, q_out, pos_id, 16)?;
                    rope.apply(gpu, k_out, pos_id, 4)?;
                }

                Self::maybe_dump_qkv(gpu, attn.layer_idx, pos_id, q_out, k_out, v_out);

                let kv_dim = 4 * 128; // 512 elements
                let (kc, vc) = match (req_k_cache, req_v_cache) {
                    (Some(kc), Some(vc)) => (Some(kc), Some(vc)),
                    _ => (self.k_cache.as_mut(), self.v_cache.as_mut()),
                };

                if let (Some(kc_buf), Some(vc_buf)) = (kc, vc) {
                    let offset = pos_id * kv_dim;
                    let mut kc_ptr = kc_buf.inner_mut().cu_deviceptr() + (offset * 2) as u64;
                    let mut vc_ptr = vc_buf.inner_mut().cu_deviceptr() + (offset * 2) as u64;
                    let mut d_k = k_out.inner().cu_deviceptr();
                    let mut d_v = v_out.inner().cu_deviceptr();
                    let mut kv_dim_u32 = kv_dim as u32;

                    let copy_func = self.add_ptx.get_function("copy_kv_slice_vec8_fp16")?;
                    let mut args_k: [*mut std::ffi::c_void; 3] = [
                        &mut kc_ptr as *mut _ as *mut std::ffi::c_void,
                        &mut d_k as *mut _ as *mut std::ffi::c_void,
                        &mut kv_dim_u32 as *mut _ as *mut std::ffi::c_void,
                    ];
                    let mut args_v: [*mut std::ffi::c_void; 3] = [
                        &mut vc_ptr as *mut _ as *mut std::ffi::c_void,
                        &mut d_v as *mut _ as *mut std::ffi::c_void,
                        &mut kv_dim_u32 as *mut _ as *mut std::ffi::c_void,
                    ];

                    unsafe {
                        launch_kernel_on_stream(&copy_func, (1, 1, 1), (64, 1, 1), 0, stream, &mut args_k)?;
                        launch_kernel_on_stream(&copy_func, (1, 1, 1), (64, 1, 1), 0, stream, &mut args_v)?;
                    }

                    if let Some(gqa) = &self.gqa {
                        gqa.compute_attention(
                            gpu,
                            &mut attn.context_buf,
                            q_out,
                            kc_buf,
                            vc_buf,
                            pos_id + 1,
                            16,
                            4,
                            128,
                        )?;
                        if std::env::var("OXIDE_DUMP_QKV_DIR").is_ok() {
                            let _ = stream.synchronize();
                            for (name, buf) in [("kcache", kc_buf), ("vcache", vc_buf)] {
                                if let Ok(host) = buf.inner().to_host_vec(stream) {
                                    let bytes = unsafe {
                                        std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 2)
                                    };
                                    let dir = std::env::var("OXIDE_DUMP_QKV_DIR").unwrap_or_default();
                                    let _ = std::fs::write(
                                        format!("{}/l{}_p{}_{}.bin", dir, attn.layer_idx, pos_id, name),
                                        bytes,
                                    );
                                }
                            }
                        }
                        if std::env::var("OXIDE_DUMP_QKV_DIR").is_ok() {
                            let _ = stream.synchronize();
                            if let Ok(host) = attn.context_buf.inner().to_host_vec(stream) {
                                let bytes = unsafe {
                                    std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 2)
                                };
                                let dir = std::env::var("OXIDE_DUMP_QKV_DIR").unwrap_or_default();
                                let _ = std::fs::write(
                                    format!("{}/l{}_p{}_ctx.bin", dir, attn.layer_idx, pos_id),
                                    bytes,
                                );
                            }
                        }
                    }
                }

                attn.forward_gating_and_o_proj(gpu, gate_out, scratch_layer_out)?;
            }
        }

        // 3. post_operator_norm on scratch_layer_out
        let mut d_layer_out = scratch_layer_out.inner_mut().cu_deviceptr();
        let mut d_post_op = self.post_operator_norm.inner().cu_deviceptr();
        let mut args_post_op: [*mut std::ffi::c_void; 5] = [
            &mut d_layer_out as *mut _ as *mut std::ffi::c_void,
            &mut d_layer_out as *mut _ as *mut std::ffi::c_void,
            &mut d_post_op as *mut _ as *mut std::ffi::c_void,
            &mut epsilon as *mut _ as *mut std::ffi::c_void,
            &mut h_size as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&rms_func, (1, 1, 1), (256, 1, 1), 0, stream, &mut args_post_op)?;
        }

        // 4 & 5. Fused Residual 1 + ffn_norm:
        // hidden_states += scratch_layer_out, and scratch_norm = rms_norm(hidden_states) * ffn_norm
        let fused_res_norm_func = self.add_ptx.get_function("fused_residual_rms_norm_vec8_fp16")?;
        let mut d_hidden_mut = hidden_states.inner_mut().cu_deviceptr();
        let mut d_ffn_w = self.ffn_norm.inner().cu_deviceptr();
        let mut args_fused_res1: [*mut std::ffi::c_void; 6] = [
            &mut d_hidden_mut as *mut _ as *mut std::ffi::c_void,
            &mut d_layer_out as *mut _ as *mut std::ffi::c_void,
            &mut d_normed as *mut _ as *mut std::ffi::c_void,
            &mut d_ffn_w as *mut _ as *mut std::ffi::c_void,
            &mut epsilon as *mut _ as *mut std::ffi::c_void,
            &mut h_size as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&fused_res_norm_func, (1, 1, 1), (256, 1, 1), 0, stream, &mut args_fused_res1)?;
        }
        Ok(())
    }

    pub fn forward_pre_ffn(
        &mut self,
        gpu: &GpuContext,
        hidden_states: &mut DeviceBuffer<half::f16>,
        scratch_norm: &mut DeviceBuffer<half::f16>,
        scratch_layer_out: &mut DeviceBuffer<half::f16>,
        q_out: &mut DeviceBuffer<half::f16>,
        k_out: &mut DeviceBuffer<half::f16>,
        v_out: &mut DeviceBuffer<half::f16>,
        gate_out: &mut DeviceBuffer<half::f16>,
        pos_id: usize,
    ) -> Result<()> {
        self.forward_pre_ffn_with_cache(
            gpu,
            hidden_states,
            scratch_norm,
            scratch_layer_out,
            q_out,
            k_out,
            v_out,
            gate_out,
            pos_id,
            None,
            None,
            None,
        )
    }

    /// Part 2 of forward: Fused post-FFN norm + Residual 2
    pub fn forward_post_ffn(
        &mut self,
        gpu: &GpuContext,
        hidden_states: &mut DeviceBuffer<half::f16>,
        scratch_ffn_out: &mut DeviceBuffer<half::f16>,
    ) -> Result<()> {
        let stream = gpu.compute_stream();
        let fused_post_add_func = self.add_ptx.get_function("fused_post_norm_residual_add_vec8_fp16")?;
        let mut d_hidden_mut = hidden_states.inner_mut().cu_deviceptr();
        let mut d_ffn_out = scratch_ffn_out.inner_mut().cu_deviceptr();
        let mut d_post_ffn = self.post_ffn_norm.inner().cu_deviceptr();
        let mut epsilon: f32 = 1e-5;
        let mut h_size = self.hidden_size as u32;

        let mut args_fused_res2: [*mut std::ffi::c_void; 5] = [
            &mut d_hidden_mut as *mut _ as *mut std::ffi::c_void,
            &mut d_ffn_out as *mut _ as *mut std::ffi::c_void,
            &mut d_post_ffn as *mut _ as *mut std::ffi::c_void,
            &mut epsilon as *mut _ as *mut std::ffi::c_void,
            &mut h_size as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&fused_post_add_func, (1, 1, 1), (256, 1, 1), 0, stream, &mut args_fused_res2)?;
        }
        Ok(())
    }

}
