use anyhow::Result;
use oxide_infer_core::{DeviceBuffer, GpuContext, HostPinnedBuffer};
use oxide_infer_kernels::PtxModule;
use cuda_core::launch_kernel_on_stream;

const LM_HEAD_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry lm_head_tied_gemv_fp16_4rows(
    .param .u64 logits_out,
    .param .u64 hidden_in,
    .param .u64 embed_table,
    .param .u32 vocab_size,
    .param .u32 hidden_size
) {
    .reg .pred %p0, %p1, %p_v1, %p_v2, %p_v3;
    .reg .b32 %b_idx, %v0_idx, %v1_idx, %v2_idx, %v3_idx, %v_idx, %vsize, %hsize, %lane_id, %warp_id, %num_warps, %tid_val, %ntid_val;
    .reg .b32 %h_vec_idx, %h_vec_stride, %h_num_vecs, %s_addr;
    .reg .b64 %logits_ptr, %hidden_ptr, %embed_ptr, %r0_offset, %r1_offset, %r2_offset, %r3_offset, %elem_offset, %byte_offset;
    .reg .b32 %xv0, %xv1, %xv2, %xv3;
    .reg .b32 %w0_v0, %w0_v1, %w0_v2, %w0_v3;
    .reg .b32 %w1_v0, %w1_v1, %w1_v2, %w1_v3;
    .reg .b32 %w2_v0, %w2_v1, %w2_v2, %w2_v3;
    .reg .b32 %w3_v0, %w3_v1, %w3_v2, %w3_v3;
    .reg .f32 %f_val, %f_w, %sum0, %sum1, %sum2, %sum3, %shfl_val;

    ld.param.u32 %vsize, [vocab_size];
    ld.param.u32 %hsize, [hidden_size];

    mov.u32 %b_idx, %ctaid.x;
    shl.b32 %v0_idx, %b_idx, 2;
    add.u32 %v1_idx, %v0_idx, 1;
    add.u32 %v2_idx, %v0_idx, 2;
    add.u32 %v3_idx, %v0_idx, 3;

    setp.ge.u32 %p0, %v0_idx, %vsize;
    @%p0 bra EXIT;

    setp.lt.u32 %p_v1, %v1_idx, %vsize;
    setp.lt.u32 %p_v2, %v2_idx, %vsize;
    setp.lt.u32 %p_v3, %v3_idx, %vsize;

    mov.u32 %tid_val, %tid.x;
    mov.u32 %ntid_val, %ntid.x;
    ld.param.u64 %hidden_ptr, [hidden_in];
    ld.param.u64 %embed_ptr, [embed_table];

    mov.f32 %sum0, 0.0;
    mov.f32 %sum1, 0.0;
    mov.f32 %sum2, 0.0;
    mov.f32 %sum3, 0.0;

    mul.wide.u32 %r0_offset, %v0_idx, %hsize;
    mul.wide.u32 %r1_offset, %v1_idx, %hsize;
    mul.wide.u32 %r2_offset, %v2_idx, %hsize;
    mul.wide.u32 %r3_offset, %v3_idx, %hsize;

    shr.u32 %h_num_vecs, %hsize, 3;
    mov.u32 %h_vec_stride, %ntid_val;
    mov.u32 %h_vec_idx, %tid_val;

VEC_LOOP:
    setp.ge.u32 %p1, %h_vec_idx, %h_num_vecs;
    @%p1 bra REDUCE;

    cvt.u64.u32 %byte_offset, %h_vec_idx;
    shl.b64 %byte_offset, %byte_offset, 4;

    // Load 8 f16 values from hidden_in ONCE for all 4 rows
    add.u64 %elem_offset, %hidden_ptr, %byte_offset;
    ld.global.v4.u32 {%xv0, %xv1, %xv2, %xv3}, [%elem_offset];

    // Row 0
    shl.b64 %elem_offset, %r0_offset, 1;
    add.u64 %elem_offset, %elem_offset, %embed_ptr;
    add.u64 %elem_offset, %elem_offset, %byte_offset;
    ld.global.v4.u32 {%w0_v0, %w0_v1, %w0_v2, %w0_v3}, [%elem_offset];

    cvt.f32.f16 %f_val, %xv0;
    cvt.f32.f16 %f_w, %w0_v0;
    fma.rn.f32 %sum0, %f_val, %f_w, %sum0;
    shr.u32 %v_idx, %xv0, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w0_v0, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %sum0, %f_val, %f_w, %sum0;

    cvt.f32.f16 %f_val, %xv1;
    cvt.f32.f16 %f_w, %w0_v1;
    fma.rn.f32 %sum0, %f_val, %f_w, %sum0;
    shr.u32 %v_idx, %xv1, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w0_v1, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %sum0, %f_val, %f_w, %sum0;

    cvt.f32.f16 %f_val, %xv2;
    cvt.f32.f16 %f_w, %w0_v2;
    fma.rn.f32 %sum0, %f_val, %f_w, %sum0;
    shr.u32 %v_idx, %xv2, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w0_v2, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %sum0, %f_val, %f_w, %sum0;

    cvt.f32.f16 %f_val, %xv3;
    cvt.f32.f16 %f_w, %w0_v3;
    fma.rn.f32 %sum0, %f_val, %f_w, %sum0;
    shr.u32 %v_idx, %xv3, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w0_v3, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %sum0, %f_val, %f_w, %sum0;

    @!%p_v1 bra NEXT_VEC;

    // Row 1
    shl.b64 %elem_offset, %r1_offset, 1;
    add.u64 %elem_offset, %elem_offset, %embed_ptr;
    add.u64 %elem_offset, %elem_offset, %byte_offset;
    ld.global.v4.u32 {%w1_v0, %w1_v1, %w1_v2, %w1_v3}, [%elem_offset];

    cvt.f32.f16 %f_val, %xv0;
    cvt.f32.f16 %f_w, %w1_v0;
    fma.rn.f32 %sum1, %f_val, %f_w, %sum1;
    shr.u32 %v_idx, %xv0, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w1_v0, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %sum1, %f_val, %f_w, %sum1;

    cvt.f32.f16 %f_val, %xv1;
    cvt.f32.f16 %f_w, %w1_v1;
    fma.rn.f32 %sum1, %f_val, %f_w, %sum1;
    shr.u32 %v_idx, %xv1, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w1_v1, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %sum1, %f_val, %f_w, %sum1;

    cvt.f32.f16 %f_val, %xv2;
    cvt.f32.f16 %f_w, %w1_v2;
    fma.rn.f32 %sum1, %f_val, %f_w, %sum1;
    shr.u32 %v_idx, %xv2, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w1_v2, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %sum1, %f_val, %f_w, %sum1;

    cvt.f32.f16 %f_val, %xv3;
    cvt.f32.f16 %f_w, %w1_v3;
    fma.rn.f32 %sum1, %f_val, %f_w, %sum1;
    shr.u32 %v_idx, %xv3, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w1_v3, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %sum1, %f_val, %f_w, %sum1;

    @!%p_v2 bra NEXT_VEC;

    // Row 2
    shl.b64 %elem_offset, %r2_offset, 1;
    add.u64 %elem_offset, %elem_offset, %embed_ptr;
    add.u64 %elem_offset, %elem_offset, %byte_offset;
    ld.global.v4.u32 {%w2_v0, %w2_v1, %w2_v2, %w2_v3}, [%elem_offset];

    cvt.f32.f16 %f_val, %xv0;
    cvt.f32.f16 %f_w, %w2_v0;
    fma.rn.f32 %sum2, %f_val, %f_w, %sum2;
    shr.u32 %v_idx, %xv0, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w2_v0, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %sum2, %f_val, %f_w, %sum2;

    cvt.f32.f16 %f_val, %xv1;
    cvt.f32.f16 %f_w, %w2_v1;
    fma.rn.f32 %sum2, %f_val, %f_w, %sum2;
    shr.u32 %v_idx, %xv1, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w2_v1, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %sum2, %f_val, %f_w, %sum2;

    cvt.f32.f16 %f_val, %xv2;
    cvt.f32.f16 %f_w, %w2_v2;
    fma.rn.f32 %sum2, %f_val, %f_w, %sum2;
    shr.u32 %v_idx, %xv2, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w2_v2, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %sum2, %f_val, %f_w, %sum2;

    cvt.f32.f16 %f_val, %xv3;
    cvt.f32.f16 %f_w, %w2_v3;
    fma.rn.f32 %sum2, %f_val, %f_w, %sum2;
    shr.u32 %v_idx, %xv3, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w2_v3, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %sum2, %f_val, %f_w, %sum2;

    @!%p_v3 bra NEXT_VEC;

    // Row 3
    shl.b64 %elem_offset, %r3_offset, 1;
    add.u64 %elem_offset, %elem_offset, %embed_ptr;
    add.u64 %elem_offset, %elem_offset, %byte_offset;
    ld.global.v4.u32 {%w3_v0, %w3_v1, %w3_v2, %w3_v3}, [%elem_offset];

    cvt.f32.f16 %f_val, %xv0;
    cvt.f32.f16 %f_w, %w3_v0;
    fma.rn.f32 %sum3, %f_val, %f_w, %sum3;
    shr.u32 %v_idx, %xv0, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w3_v0, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %sum3, %f_val, %f_w, %sum3;

    cvt.f32.f16 %f_val, %xv1;
    cvt.f32.f16 %f_w, %w3_v1;
    fma.rn.f32 %sum3, %f_val, %f_w, %sum3;
    shr.u32 %v_idx, %xv1, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w3_v1, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %sum3, %f_val, %f_w, %sum3;

    cvt.f32.f16 %f_val, %xv2;
    cvt.f32.f16 %f_w, %w3_v2;
    fma.rn.f32 %sum3, %f_val, %f_w, %sum3;
    shr.u32 %v_idx, %xv2, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w3_v2, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %sum3, %f_val, %f_w, %sum3;

    cvt.f32.f16 %f_val, %xv3;
    cvt.f32.f16 %f_w, %w3_v3;
    fma.rn.f32 %sum3, %f_val, %f_w, %sum3;
    shr.u32 %v_idx, %xv3, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w3_v3, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %sum3, %f_val, %f_w, %sum3;

NEXT_VEC:
    add.u32 %h_vec_idx, %h_vec_idx, %h_vec_stride;
    bra VEC_LOOP;

REDUCE:
    // Warp reduction for sum0
    shfl.sync.down.b32 %shfl_val, %sum0, 16, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum0, 8, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum0, 4, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum0, 2, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum0, 1, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;

    // Warp reduction for sum1
    shfl.sync.down.b32 %shfl_val, %sum1, 16, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum1, 8, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum1, 4, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum1, 2, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum1, 1, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;

    // Warp reduction for sum2
    shfl.sync.down.b32 %shfl_val, %sum2, 16, 31, 0xffffffff;
    add.f32 %sum2, %sum2, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum2, 8, 31, 0xffffffff;
    add.f32 %sum2, %sum2, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum2, 4, 31, 0xffffffff;
    add.f32 %sum2, %sum2, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum2, 2, 31, 0xffffffff;
    add.f32 %sum2, %sum2, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum2, 1, 31, 0xffffffff;
    add.f32 %sum2, %sum2, %shfl_val;

    // Warp reduction for sum3
    shfl.sync.down.b32 %shfl_val, %sum3, 16, 31, 0xffffffff;
    add.f32 %sum3, %sum3, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum3, 8, 31, 0xffffffff;
    add.f32 %sum3, %sum3, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum3, 4, 31, 0xffffffff;
    add.f32 %sum3, %sum3, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum3, 2, 31, 0xffffffff;
    add.f32 %sum3, %sum3, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum3, 1, 31, 0xffffffff;
    add.f32 %sum3, %sum3, %shfl_val;

    .shared .f32 s_sum0[32];
    .shared .f32 s_sum1[32];
    .shared .f32 s_sum2[32];
    .shared .f32 s_sum3[32];

    and.b32 %lane_id, %tid_val, 31;
    shr.u32 %warp_id, %tid_val, 5;
    setp.eq.u32 %p0, %lane_id, 0;

    mov.u32 %s_addr, s_sum0;
    shl.b32 %h_vec_stride, %warp_id, 2;
    add.u32 %s_addr, %s_addr, %h_vec_stride;
    @%p0 st.shared.f32 [%s_addr], %sum0;

    mov.u32 %s_addr, s_sum1;
    add.u32 %s_addr, %s_addr, %h_vec_stride;
    @%p0 st.shared.f32 [%s_addr], %sum1;

    mov.u32 %s_addr, s_sum2;
    add.u32 %s_addr, %s_addr, %h_vec_stride;
    @%p0 st.shared.f32 [%s_addr], %sum2;

    mov.u32 %s_addr, s_sum3;
    add.u32 %s_addr, %s_addr, %h_vec_stride;
    @%p0 st.shared.f32 [%s_addr], %sum3;

    bar.sync 0;

    shr.u32 %num_warps, %ntid_val, 5;
    setp.lt.u32 %p0, %tid_val, %num_warps;

    mov.u32 %s_addr, s_sum0;
    shl.b32 %h_vec_stride, %tid_val, 2;
    add.u32 %s_addr, %s_addr, %h_vec_stride;
    @%p0 ld.shared.f32 %sum0, [%s_addr];
    @!%p0 mov.f32 %sum0, 0.0;

    mov.u32 %s_addr, s_sum1;
    add.u32 %s_addr, %s_addr, %h_vec_stride;
    @%p0 ld.shared.f32 %sum1, [%s_addr];
    @!%p0 mov.f32 %sum1, 0.0;

    mov.u32 %s_addr, s_sum2;
    add.u32 %s_addr, %s_addr, %h_vec_stride;
    @%p0 ld.shared.f32 %sum2, [%s_addr];
    @!%p0 mov.f32 %sum2, 0.0;

    mov.u32 %s_addr, s_sum3;
    add.u32 %s_addr, %s_addr, %h_vec_stride;
    @%p0 ld.shared.f32 %sum3, [%s_addr];
    @!%p0 mov.f32 %sum3, 0.0;

    // Block reduction
    shfl.sync.down.b32 %shfl_val, %sum0, 16, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum0, 8, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum0, 4, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum0, 2, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum0, 1, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;

    shfl.sync.down.b32 %shfl_val, %sum1, 16, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum1, 8, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum1, 4, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum1, 2, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum1, 1, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;

    shfl.sync.down.b32 %shfl_val, %sum2, 16, 31, 0xffffffff;
    add.f32 %sum2, %sum2, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum2, 8, 31, 0xffffffff;
    add.f32 %sum2, %sum2, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum2, 4, 31, 0xffffffff;
    add.f32 %sum2, %sum2, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum2, 2, 31, 0xffffffff;
    add.f32 %sum2, %sum2, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum2, 1, 31, 0xffffffff;
    add.f32 %sum2, %sum2, %shfl_val;

    shfl.sync.down.b32 %shfl_val, %sum3, 16, 31, 0xffffffff;
    add.f32 %sum3, %sum3, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum3, 8, 31, 0xffffffff;
    add.f32 %sum3, %sum3, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum3, 4, 31, 0xffffffff;
    add.f32 %sum3, %sum3, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum3, 2, 31, 0xffffffff;
    add.f32 %sum3, %sum3, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum3, 1, 31, 0xffffffff;
    add.f32 %sum3, %sum3, %shfl_val;

    setp.eq.u32 %p0, %tid_val, 0;
    @!%p0 bra EXIT;

    ld.param.u64 %logits_ptr, [logits_out];

    // Store row 0
    cvt.u64.u32 %byte_offset, %v0_idx;
    shl.b64 %byte_offset, %byte_offset, 2;
    add.u64 %elem_offset, %logits_ptr, %byte_offset;
    st.global.f32 [%elem_offset], %sum0;

    // Store row 1
    @!%p_v1 bra EXIT;
    cvt.u64.u32 %byte_offset, %v1_idx;
    shl.b64 %byte_offset, %byte_offset, 2;
    add.u64 %elem_offset, %logits_ptr, %byte_offset;
    st.global.f32 [%elem_offset], %sum1;

    // Store row 2
    @!%p_v2 bra EXIT;
    cvt.u64.u32 %byte_offset, %v2_idx;
    shl.b64 %byte_offset, %byte_offset, 2;
    add.u64 %elem_offset, %logits_ptr, %byte_offset;
    st.global.f32 [%elem_offset], %sum2;

    // Store row 3
    @!%p_v3 bra EXIT;
    cvt.u64.u32 %byte_offset, %v3_idx;
    shl.b64 %byte_offset, %byte_offset, 2;
    add.u64 %elem_offset, %logits_ptr, %byte_offset;
    st.global.f32 [%elem_offset], %sum3;

EXIT:
    ret;
}

.visible .entry lm_head_tied_gemv_fp16_2rows(
    .param .u64 logits_out,
    .param .u64 hidden_in,
    .param .u64 embed_table,
    .param .u32 vocab_size,
    .param .u32 hidden_size
) {
    .reg .pred %p0, %p1, %p_v1;
    .reg .b32 %b_idx, %v0_idx, %v1_idx, %v_idx, %vsize, %hsize, %lane_id, %warp_id, %num_warps, %tid_val, %ntid_val;
    .reg .b32 %h_vec_idx, %h_vec_stride, %h_num_vecs, %s_addr;
    .reg .b64 %logits_ptr, %hidden_ptr, %embed_ptr, %row0_offset, %row1_offset, %elem_offset, %byte_offset;
    .reg .b32 %xv0, %xv1, %xv2, %xv3;
    .reg .b32 %w0_v0, %w0_v1, %w0_v2, %w0_v3;
    .reg .b32 %w1_v0, %w1_v1, %w1_v2, %w1_v3;
    .reg .f32 %f_val, %f_w0, %f_w1, %sum0, %sum1, %shfl_val;

    ld.param.u32 %vsize, [vocab_size];
    ld.param.u32 %hsize, [hidden_size];

    mov.u32 %b_idx, %ctaid.x;
    shl.b32 %v0_idx, %b_idx, 1;
    add.u32 %v1_idx, %v0_idx, 1;

    setp.ge.u32 %p0, %v0_idx, %vsize;
    @%p0 bra EXIT;

    setp.lt.u32 %p_v1, %v1_idx, %vsize;

    mov.u32 %tid_val, %tid.x;
    mov.u32 %ntid_val, %ntid.x;
    ld.param.u64 %hidden_ptr, [hidden_in];
    ld.param.u64 %embed_ptr, [embed_table];

    mov.f32 %sum0, 0.0;
    mov.f32 %sum1, 0.0;

    mul.wide.u32 %row0_offset, %v0_idx, %hsize;
    mul.wide.u32 %row1_offset, %v1_idx, %hsize;

    shr.u32 %h_num_vecs, %hsize, 3;
    mov.u32 %h_vec_stride, %ntid_val;
    mov.u32 %h_vec_idx, %tid_val;

VEC_LOOP:
    setp.ge.u32 %p1, %h_vec_idx, %h_num_vecs;
    @%p1 bra REDUCE;

    cvt.u64.u32 %byte_offset, %h_vec_idx;
    shl.b64 %byte_offset, %byte_offset, 4;

    // Load 8 f16 values from hidden_in ONCE for both rows
    add.u64 %elem_offset, %hidden_ptr, %byte_offset;
    ld.global.v4.u32 {%xv0, %xv1, %xv2, %xv3}, [%elem_offset];

    // Load 8 f16 values from embed_table for row 0
    shl.b64 %elem_offset, %row0_offset, 1;
    add.u64 %elem_offset, %elem_offset, %embed_ptr;
    add.u64 %elem_offset, %elem_offset, %byte_offset;
    ld.global.v4.u32 {%w0_v0, %w0_v1, %w0_v2, %w0_v3}, [%elem_offset];

    // Accumulate row 0
    cvt.f32.f16 %f_val, %xv0;
    cvt.f32.f16 %f_w0, %w0_v0;
    fma.rn.f32 %sum0, %f_val, %f_w0, %sum0;
    shr.u32 %v_idx, %xv0, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w0_v0, 16;
    cvt.f32.f16 %f_w0, %v_idx;
    fma.rn.f32 %sum0, %f_val, %f_w0, %sum0;

    cvt.f32.f16 %f_val, %xv1;
    cvt.f32.f16 %f_w0, %w0_v1;
    fma.rn.f32 %sum0, %f_val, %f_w0, %sum0;
    shr.u32 %v_idx, %xv1, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w0_v1, 16;
    cvt.f32.f16 %f_w0, %v_idx;
    fma.rn.f32 %sum0, %f_val, %f_w0, %sum0;

    cvt.f32.f16 %f_val, %xv2;
    cvt.f32.f16 %f_w0, %w0_v2;
    fma.rn.f32 %sum0, %f_val, %f_w0, %sum0;
    shr.u32 %v_idx, %xv2, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w0_v2, 16;
    cvt.f32.f16 %f_w0, %v_idx;
    fma.rn.f32 %sum0, %f_val, %f_w0, %sum0;

    cvt.f32.f16 %f_val, %xv3;
    cvt.f32.f16 %f_w0, %w0_v3;
    fma.rn.f32 %sum0, %f_val, %f_w0, %sum0;
    shr.u32 %v_idx, %xv3, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w0_v3, 16;
    cvt.f32.f16 %f_w0, %v_idx;
    fma.rn.f32 %sum0, %f_val, %f_w0, %sum0;

    @!%p_v1 bra NEXT_VEC;

    // Load 8 f16 values from embed_table for row 1
    shl.b64 %elem_offset, %row1_offset, 1;
    add.u64 %elem_offset, %elem_offset, %embed_ptr;
    add.u64 %elem_offset, %elem_offset, %byte_offset;
    ld.global.v4.u32 {%w1_v0, %w1_v1, %w1_v2, %w1_v3}, [%elem_offset];

    // Accumulate row 1 using reused %xv registers
    cvt.f32.f16 %f_val, %xv0;
    cvt.f32.f16 %f_w1, %w1_v0;
    fma.rn.f32 %sum1, %f_val, %f_w1, %sum1;
    shr.u32 %v_idx, %xv0, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w1_v0, 16;
    cvt.f32.f16 %f_w1, %v_idx;
    fma.rn.f32 %sum1, %f_val, %f_w1, %sum1;

    cvt.f32.f16 %f_val, %xv1;
    cvt.f32.f16 %f_w1, %w1_v1;
    fma.rn.f32 %sum1, %f_val, %f_w1, %sum1;
    shr.u32 %v_idx, %xv1, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w1_v1, 16;
    cvt.f32.f16 %f_w1, %v_idx;
    fma.rn.f32 %sum1, %f_val, %f_w1, %sum1;

    cvt.f32.f16 %f_val, %xv2;
    cvt.f32.f16 %f_w1, %w1_v2;
    fma.rn.f32 %sum1, %f_val, %f_w1, %sum1;
    shr.u32 %v_idx, %xv2, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w1_v2, 16;
    cvt.f32.f16 %f_w1, %v_idx;
    fma.rn.f32 %sum1, %f_val, %f_w1, %sum1;

    cvt.f32.f16 %f_val, %xv3;
    cvt.f32.f16 %f_w1, %w1_v3;
    fma.rn.f32 %sum1, %f_val, %f_w1, %sum1;
    shr.u32 %v_idx, %xv3, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %w1_v3, 16;
    cvt.f32.f16 %f_w1, %v_idx;
    fma.rn.f32 %sum1, %f_val, %f_w1, %sum1;

NEXT_VEC:
    add.u32 %h_vec_idx, %h_vec_idx, %h_vec_stride;
    bra VEC_LOOP;

REDUCE:
    // Warp reduction for sum0
    shfl.sync.down.b32 %shfl_val, %sum0, 16, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum0, 8, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum0, 4, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum0, 2, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum0, 1, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;

    // Warp reduction for sum1
    shfl.sync.down.b32 %shfl_val, %sum1, 16, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum1, 8, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum1, 4, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum1, 2, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum1, 1, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;

    .shared .f32 s_sum0[32];
    .shared .f32 s_sum1[32];
    and.b32 %lane_id, %tid_val, 31;
    shr.u32 %warp_id, %tid_val, 5;
    setp.eq.u32 %p0, %lane_id, 0;

    mov.u32 %s_addr, s_sum0;
    shl.b32 %h_vec_stride, %warp_id, 2;
    add.u32 %s_addr, %s_addr, %h_vec_stride;
    @%p0 st.shared.f32 [%s_addr], %sum0;

    mov.u32 %s_addr, s_sum1;
    add.u32 %s_addr, %s_addr, %h_vec_stride;
    @%p0 st.shared.f32 [%s_addr], %sum1;

    bar.sync 0;

    shr.u32 %num_warps, %ntid_val, 5;
    setp.lt.u32 %p0, %tid_val, %num_warps;

    mov.u32 %s_addr, s_sum0;
    shl.b32 %h_vec_stride, %tid_val, 2;
    add.u32 %s_addr, %s_addr, %h_vec_stride;
    @%p0 ld.shared.f32 %sum0, [%s_addr];
    @!%p0 mov.f32 %sum0, 0.0;

    mov.u32 %s_addr, s_sum1;
    add.u32 %s_addr, %s_addr, %h_vec_stride;
    @%p0 ld.shared.f32 %sum1, [%s_addr];
    @!%p0 mov.f32 %sum1, 0.0;

    // Block reduction for sum0
    shfl.sync.down.b32 %shfl_val, %sum0, 16, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum0, 8, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum0, 4, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum0, 2, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum0, 1, 31, 0xffffffff;
    add.f32 %sum0, %sum0, %shfl_val;

    // Block reduction for sum1
    shfl.sync.down.b32 %shfl_val, %sum1, 16, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum1, 8, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum1, 4, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum1, 2, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;
    shfl.sync.down.b32 %shfl_val, %sum1, 1, 31, 0xffffffff;
    add.f32 %sum1, %sum1, %shfl_val;

    setp.eq.u32 %p0, %tid_val, 0;
    @!%p0 bra EXIT;

    ld.param.u64 %logits_ptr, [logits_out];

    // Store row 0
    cvt.u64.u32 %byte_offset, %v0_idx;
    shl.b64 %byte_offset, %byte_offset, 2;
    add.u64 %elem_offset, %logits_ptr, %byte_offset;
    st.global.f32 [%elem_offset], %sum0;

    // Store row 1
    @!%p_v1 bra EXIT;
    cvt.u64.u32 %byte_offset, %v1_idx;
    shl.b64 %byte_offset, %byte_offset, 2;
    add.u64 %elem_offset, %logits_ptr, %byte_offset;
    st.global.f32 [%elem_offset], %sum1;

EXIT:
    ret;
}

.visible .entry lm_head_tied_gemv_fp16(
    .param .u64 logits_out,
    .param .u64 hidden_in,
    .param .u64 embed_table,
    .param .u32 vocab_size,
    .param .u32 hidden_size
) {
    .reg .pred %p0, %p1;
    .reg .b32 %vocab_idx, %v_idx, %vsize, %hsize, %lane_id, %warp_id, %num_warps, %tid_val, %ntid_val;
    .reg .b32 %h_vec_idx, %h_vec_stride, %h_num_vecs, %s_addr;
    .reg .b64 %logits_ptr, %hidden_ptr, %embed_ptr, %row_offset, %elem_offset, %byte_offset;
    .reg .b32 %xv0, %xv1, %xv2, %xv3;
    .reg .b32 %wv0, %wv1, %wv2, %wv3;
    .reg .f32 %f_val, %f_w, %f_sum, %shfl_val;

    ld.param.u32 %vsize, [vocab_size];
    ld.param.u32 %hsize, [hidden_size];

    mov.u32 %vocab_idx, %ctaid.x;
    setp.ge.u32 %p0, %vocab_idx, %vsize;
    @%p0 bra EXIT;

    mov.u32 %tid_val, %tid.x;
    mov.u32 %ntid_val, %ntid.x;
    ld.param.u64 %hidden_ptr, [hidden_in];
    ld.param.u64 %embed_ptr, [embed_table];

    mov.f32 %f_sum, 0.0;
    mul.wide.u32 %row_offset, %vocab_idx, %hsize;

    // Vectorized 128-bit loads: 8 FP16s (16 bytes) per vector load
    shr.u32 %h_num_vecs, %hsize, 3;
    mov.u32 %h_vec_stride, %ntid_val;
    mov.u32 %h_vec_idx, %tid_val;

VEC_LOOP:
    setp.ge.u32 %p1, %h_vec_idx, %h_num_vecs;
    @%p1 bra REDUCE;

    // Vector byte offset = h_vec_idx * 16
    cvt.u64.u32 %byte_offset, %h_vec_idx;
    shl.b64 %byte_offset, %byte_offset, 4;

    // Load 8 f16 values from hidden_in
    add.u64 %elem_offset, %hidden_ptr, %byte_offset;
    ld.global.v4.u32 {%xv0, %xv1, %xv2, %xv3}, [%elem_offset];

    // Load 8 f16 values from embed_table[vocab_idx, ...]
    shl.b64 %elem_offset, %row_offset, 1;
    add.u64 %elem_offset, %elem_offset, %embed_ptr;
    add.u64 %elem_offset, %elem_offset, %byte_offset;
    ld.global.v4.u32 {%wv0, %wv1, %wv2, %wv3}, [%elem_offset];

    // Pair 0 & 1 (xv0, wv0)
    cvt.f32.f16 %f_val, %xv0;
    cvt.f32.f16 %f_w, %wv0;
    fma.rn.f32 %f_sum, %f_val, %f_w, %f_sum;
    shr.u32 %v_idx, %xv0, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %wv0, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %f_sum, %f_val, %f_w, %f_sum;

    // Pair 2 & 3 (xv1, wv1)
    cvt.f32.f16 %f_val, %xv1;
    cvt.f32.f16 %f_w, %wv1;
    fma.rn.f32 %f_sum, %f_val, %f_w, %f_sum;
    shr.u32 %v_idx, %xv1, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %wv1, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %f_sum, %f_val, %f_w, %f_sum;

    // Pair 4 & 5 (xv2, wv2)
    cvt.f32.f16 %f_val, %xv2;
    cvt.f32.f16 %f_w, %wv2;
    fma.rn.f32 %f_sum, %f_val, %f_w, %f_sum;
    shr.u32 %v_idx, %xv2, 16;
    cvt.f32.f16 %f_val, %v_idx;
    shr.u32 %v_idx, %wv2, 16;
    cvt.f32.f16 %f_w, %v_idx;
    fma.rn.f32 %f_sum, %f_val, %f_w, %f_sum;

    // Pair 6 & 7 (xv3, wv3)
    cvt.f32.f16 %f_val, %xv3;
    cvt.f32.f16 %f_w, %wv3;
    fma.rn.f32 %f_sum, %f_val, %f_w, %f_sum;
    shr.u32 %v_idx, %xv3, 16;
    cvt.f32.f16 %f_val, %v_idx;
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

    ld.param.u64 %logits_ptr, [logits_out];
    cvt.u64.u32 %row_offset, %vocab_idx;
    shl.b64 %row_offset, %row_offset, 2;
    add.u64 %logits_ptr, %logits_ptr, %row_offset;
    st.global.f32 [%logits_ptr], %f_sum;

EXIT:
    ret;
}

// 2-Stage Argmax Reduction across vocab (vocab_size=128,000)
// Stage 1: Each block reduces 512 elements -> outputs max val and max idx per block
// 250 blocks for 128,000 vocab
.visible .entry argmax_stage1_fp32(
    .param .u64 block_max_vals,
    .param .u64 block_max_indices,
    .param .u64 logits_in,
    .param .u32 vocab_size
) {
    .reg .pred %p0, %p_gt;
    .reg .b32 %t_id, %b_idx, %vsize, %stride, %idx, %warp_id, %lane_id, %num_warps;
    .reg .b32 %cur_idx, %other_idx, %s_addr;
    .reg .b64 %l_ptr, %out_v_ptr, %out_i_ptr, %offset;
    .reg .f32 %cur_val, %other_val, %shfl_val;

    ld.param.u32 %vsize, [vocab_size];
    ld.param.u64 %l_ptr, [logits_in];

    mov.u32 %t_id, %tid.x;
    mov.u32 %b_idx, %ctaid.x;

    // Start with negative infinity
    mov.f32 %cur_val, -1e30;
    mov.u32 %cur_idx, 0;

    // Each block processes elements with block stride
    mul.wide.u32 %offset, %b_idx, 256;
    cvt.u32.u64 %idx, %offset;
    add.u32 %idx, %idx, %t_id;

    setp.ge.u32 %p0, %idx, %vsize;
    @%p0 bra REDUCE_WARP;

    cvt.u64.u32 %offset, %idx;
    shl.b64 %offset, %offset, 2;
    add.u64 %offset, %l_ptr, %offset;
    ld.global.f32 %cur_val, [%offset];
    mov.u32 %cur_idx, %idx;

REDUCE_WARP:
    // Warp shuffle max reduction
    // Delta 16
    shfl.sync.down.b32 %shfl_val, %cur_val, 16, 31, 0xffffffff;
    shfl.sync.down.b32 %other_idx, %cur_idx, 16, 31, 0xffffffff;
    setp.gt.f32 %p_gt, %shfl_val, %cur_val;
    @%p_gt mov.f32 %cur_val, %shfl_val;
    @%p_gt mov.u32 %cur_idx, %other_idx;

    // Delta 8
    shfl.sync.down.b32 %shfl_val, %cur_val, 8, 31, 0xffffffff;
    shfl.sync.down.b32 %other_idx, %cur_idx, 8, 31, 0xffffffff;
    setp.gt.f32 %p_gt, %shfl_val, %cur_val;
    @%p_gt mov.f32 %cur_val, %shfl_val;
    @%p_gt mov.u32 %cur_idx, %other_idx;

    // Delta 4
    shfl.sync.down.b32 %shfl_val, %cur_val, 4, 31, 0xffffffff;
    shfl.sync.down.b32 %other_idx, %cur_idx, 4, 31, 0xffffffff;
    setp.gt.f32 %p_gt, %shfl_val, %cur_val;
    @%p_gt mov.f32 %cur_val, %shfl_val;
    @%p_gt mov.u32 %cur_idx, %other_idx;

    // Delta 2
    shfl.sync.down.b32 %shfl_val, %cur_val, 2, 31, 0xffffffff;
    shfl.sync.down.b32 %other_idx, %cur_idx, 2, 31, 0xffffffff;
    setp.gt.f32 %p_gt, %shfl_val, %cur_val;
    @%p_gt mov.f32 %cur_val, %shfl_val;
    @%p_gt mov.u32 %cur_idx, %other_idx;

    // Delta 1
    shfl.sync.down.b32 %shfl_val, %cur_val, 1, 31, 0xffffffff;
    shfl.sync.down.b32 %other_idx, %cur_idx, 1, 31, 0xffffffff;
    setp.gt.f32 %p_gt, %shfl_val, %cur_val;
    @%p_gt mov.f32 %cur_val, %shfl_val;
    @%p_gt mov.u32 %cur_idx, %other_idx;

    // Shared memory for 8 warps (256 threads / 32 = 8 warps)
    .shared .f32 s_val[8];
    .shared .u32 s_idx[8];
    and.b32 %lane_id, %t_id, 31;
    shr.u32 %warp_id, %t_id, 5;

    setp.eq.u32 %p0, %lane_id, 0;
    mov.u32 %s_addr, s_val;
    shl.b32 %stride, %warp_id, 2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p0 st.shared.f32 [%s_addr], %cur_val;

    mov.u32 %s_addr, s_idx;
    add.u32 %s_addr, %s_addr, %stride;
    @%p0 st.shared.u32 [%s_addr], %cur_idx;
    bar.sync 0;

    // Warp 0 reduces the 8 warps
    setp.lt.u32 %p0, %t_id, 8;
    mov.u32 %s_addr, s_val;
    shl.b32 %stride, %t_id, 2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p0 ld.shared.f32 %cur_val, [%s_addr];
    @!%p0 mov.f32 %cur_val, -1e30;

    mov.u32 %s_addr, s_idx;
    add.u32 %s_addr, %s_addr, %stride;
    @%p0 ld.shared.u32 %cur_idx, [%s_addr];
    @!%p0 mov.u32 %cur_idx, 0;

    // Final reduce across 8 elements in warp 0
    shfl.sync.down.b32 %shfl_val, %cur_val, 4, 31, 0xffffffff;
    shfl.sync.down.b32 %other_idx, %cur_idx, 4, 31, 0xffffffff;
    setp.gt.f32 %p_gt, %shfl_val, %cur_val;
    @%p_gt mov.f32 %cur_val, %shfl_val;
    @%p_gt mov.u32 %cur_idx, %other_idx;

    shfl.sync.down.b32 %shfl_val, %cur_val, 2, 31, 0xffffffff;
    shfl.sync.down.b32 %other_idx, %cur_idx, 2, 31, 0xffffffff;
    setp.gt.f32 %p_gt, %shfl_val, %cur_val;
    @%p_gt mov.f32 %cur_val, %shfl_val;
    @%p_gt mov.u32 %cur_idx, %other_idx;

    shfl.sync.down.b32 %shfl_val, %cur_val, 1, 31, 0xffffffff;
    shfl.sync.down.b32 %other_idx, %cur_idx, 1, 31, 0xffffffff;
    setp.gt.f32 %p_gt, %shfl_val, %cur_val;
    @%p_gt mov.f32 %cur_val, %shfl_val;
    @%p_gt mov.u32 %cur_idx, %other_idx;

    setp.eq.u32 %p0, %t_id, 0;
    @!%p0 bra FINISH;

    ld.param.u64 %out_v_ptr, [block_max_vals];
    ld.param.u64 %out_i_ptr, [block_max_indices];
    cvt.u64.u32 %offset, %b_idx;
    shl.b64 %offset, %offset, 2;
    add.u64 %out_v_ptr, %out_v_ptr, %offset;
    add.u64 %out_i_ptr, %out_i_ptr, %offset;
    st.global.f32 [%out_v_ptr], %cur_val;
    st.global.u32 [%out_i_ptr], %cur_idx;

FINISH:
    ret;
}

// Stage 2: 1 block with 512 threads reduces the 500 block maximums to the single best token
.visible .entry argmax_stage2_fp32(
    .param .u64 best_token_out,
    .param .u64 block_max_vals,
    .param .u64 block_max_indices,
    .param .u32 num_blocks
) {
    .reg .pred %p0, %p_gt;
    .reg .b32 %t_id, %nblocks, %cur_idx, %other_idx, %lane_id, %warp_id, %stride, %s_addr;
    .reg .b64 %out_ptr, %in_v_ptr, %in_i_ptr, %offset;
    .reg .f32 %cur_val, %other_val, %shfl_val;

    ld.param.u32 %nblocks, [num_blocks];
    mov.u32 %t_id, %tid.x;

    mov.f32 %cur_val, -1e30;
    mov.u32 %cur_idx, 0;

    setp.ge.u32 %p0, %t_id, %nblocks;
    @%p0 bra REDUCE_W;

    ld.param.u64 %in_v_ptr, [block_max_vals];
    ld.param.u64 %in_i_ptr, [block_max_indices];
    cvt.u64.u32 %offset, %t_id;
    shl.b64 %offset, %offset, 2;
    add.u64 %in_v_ptr, %in_v_ptr, %offset;
    add.u64 %in_i_ptr, %in_i_ptr, %offset;
    ld.global.f32 %cur_val, [%in_v_ptr];
    ld.global.u32 %cur_idx, [%in_i_ptr];

REDUCE_W:
    shfl.sync.down.b32 %shfl_val, %cur_val, 16, 31, 0xffffffff;
    shfl.sync.down.b32 %other_idx, %cur_idx, 16, 31, 0xffffffff;
    setp.gt.f32 %p_gt, %shfl_val, %cur_val;
    @%p_gt mov.f32 %cur_val, %shfl_val;
    @%p_gt mov.u32 %cur_idx, %other_idx;

    shfl.sync.down.b32 %shfl_val, %cur_val, 8, 31, 0xffffffff;
    shfl.sync.down.b32 %other_idx, %cur_idx, 8, 31, 0xffffffff;
    setp.gt.f32 %p_gt, %shfl_val, %cur_val;
    @%p_gt mov.f32 %cur_val, %shfl_val;
    @%p_gt mov.u32 %cur_idx, %other_idx;

    shfl.sync.down.b32 %shfl_val, %cur_val, 4, 31, 0xffffffff;
    shfl.sync.down.b32 %other_idx, %cur_idx, 4, 31, 0xffffffff;
    setp.gt.f32 %p_gt, %shfl_val, %cur_val;
    @%p_gt mov.f32 %cur_val, %shfl_val;
    @%p_gt mov.u32 %cur_idx, %other_idx;

    shfl.sync.down.b32 %shfl_val, %cur_val, 2, 31, 0xffffffff;
    shfl.sync.down.b32 %other_idx, %cur_idx, 2, 31, 0xffffffff;
    setp.gt.f32 %p_gt, %shfl_val, %cur_val;
    @%p_gt mov.f32 %cur_val, %shfl_val;
    @%p_gt mov.u32 %cur_idx, %other_idx;

    shfl.sync.down.b32 %shfl_val, %cur_val, 1, 31, 0xffffffff;
    shfl.sync.down.b32 %other_idx, %cur_idx, 1, 31, 0xffffffff;
    setp.gt.f32 %p_gt, %shfl_val, %cur_val;
    @%p_gt mov.f32 %cur_val, %shfl_val;
    @%p_gt mov.u32 %cur_idx, %other_idx;

    .shared .f32 s_v2[16];
    .shared .u32 s_i2[16];
    and.b32 %lane_id, %t_id, 31;
    shr.u32 %warp_id, %t_id, 5;

    setp.eq.u32 %p0, %lane_id, 0;
    mov.u32 %s_addr, s_v2;
    shl.b32 %stride, %warp_id, 2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p0 st.shared.f32 [%s_addr], %cur_val;

    mov.u32 %s_addr, s_i2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p0 st.shared.u32 [%s_addr], %cur_idx;
    bar.sync 0;

    // Warp 0 finishes reduction of 16 warps
    setp.lt.u32 %p0, %t_id, 16;
    mov.u32 %s_addr, s_v2;
    shl.b32 %stride, %t_id, 2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p0 ld.shared.f32 %cur_val, [%s_addr];
    @!%p0 mov.f32 %cur_val, -1e30;

    mov.u32 %s_addr, s_i2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p0 ld.shared.u32 %cur_idx, [%s_addr];
    @!%p0 mov.u32 %cur_idx, 0;

    shfl.sync.down.b32 %shfl_val, %cur_val, 8, 31, 0xffffffff;
    shfl.sync.down.b32 %other_idx, %cur_idx, 8, 31, 0xffffffff;
    setp.gt.f32 %p_gt, %shfl_val, %cur_val;
    @%p_gt mov.f32 %cur_val, %shfl_val;
    @%p_gt mov.u32 %cur_idx, %other_idx;

    shfl.sync.down.b32 %shfl_val, %cur_val, 4, 31, 0xffffffff;
    shfl.sync.down.b32 %other_idx, %cur_idx, 4, 31, 0xffffffff;
    setp.gt.f32 %p_gt, %shfl_val, %cur_val;
    @%p_gt mov.f32 %cur_val, %shfl_val;
    @%p_gt mov.u32 %cur_idx, %other_idx;

    shfl.sync.down.b32 %shfl_val, %cur_val, 2, 31, 0xffffffff;
    shfl.sync.down.b32 %other_idx, %cur_idx, 2, 31, 0xffffffff;
    setp.gt.f32 %p_gt, %shfl_val, %cur_val;
    @%p_gt mov.f32 %cur_val, %shfl_val;
    @%p_gt mov.u32 %cur_idx, %other_idx;

    shfl.sync.down.b32 %shfl_val, %cur_val, 1, 31, 0xffffffff;
    shfl.sync.down.b32 %other_idx, %cur_idx, 1, 31, 0xffffffff;
    setp.gt.f32 %p_gt, %shfl_val, %cur_val;
    @%p_gt mov.f32 %cur_val, %shfl_val;
    @%p_gt mov.u32 %cur_idx, %other_idx;

    setp.eq.u32 %p0, %t_id, 0;
    @!%p0 bra DONE;

    ld.param.u64 %out_ptr, [best_token_out];
    st.global.u32 [%out_ptr], %cur_idx;

DONE:
    ret;
}
"#;

pub struct TiedLmHead {
    ptx: PtxModule,
    vocab_size: usize,
    hidden_size: usize,
    stage1_vals: DeviceBuffer<f32>,
    stage1_indices: DeviceBuffer<u32>,
    best_token_buf: DeviceBuffer<u32>,
    pinned_token_buf: HostPinnedBuffer<u32>,
}

impl TiedLmHead {
    pub fn new(gpu: &GpuContext, vocab_size: usize, hidden_size: usize) -> Result<Self> {
        let ptx = PtxModule::load_from_ptx(gpu, LM_HEAD_PTX)?;
        let num_stage1_blocks = (vocab_size + 256 - 1) / 256;
        let stage1_vals = DeviceBuffer::<f32>::zeroed(gpu, num_stage1_blocks)?;
        let stage1_indices = DeviceBuffer::<u32>::zeroed(gpu, num_stage1_blocks)?;
        let best_token_buf = DeviceBuffer::<u32>::zeroed(gpu, 1)?;
        let pinned_token_buf = HostPinnedBuffer::<u32>::zeroed(gpu, 1)?;
        Ok(Self {
            ptx,
            vocab_size,
            hidden_size,
            stage1_vals,
            stage1_indices,
            best_token_buf,
            pinned_token_buf,
        })
    }

    /// Compute real logits [vocab_size] = hidden [hidden_size] @ embed_table [vocab_size, hidden_size]^T
    pub fn compute_logits(
        &self,
        gpu: &GpuContext,
        hidden_state: &DeviceBuffer<half::f16>,
        embed_table: &DeviceBuffer<half::f16>,
        logits_out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let func = self.ptx.get_function("lm_head_tied_gemv_fp16_4rows")?;
        let stream = gpu.compute_stream();

        let mut d_logits = logits_out.inner_mut().cu_deviceptr();
        let mut d_hidden = hidden_state.inner().cu_deviceptr();
        let mut d_embed = embed_table.inner().cu_deviceptr();
        let mut v_size = self.vocab_size as u32;
        let mut h_size = self.hidden_size as u32;

        let mut args: [*mut std::ffi::c_void; 5] = [
            &mut d_logits as *mut _ as *mut std::ffi::c_void,
            &mut d_hidden as *mut _ as *mut std::ffi::c_void,
            &mut d_embed as *mut _ as *mut std::ffi::c_void,
            &mut v_size as *mut _ as *mut std::ffi::c_void,
            &mut h_size as *mut _ as *mut std::ffi::c_void,
        ];

        let block_dim: u32 = 128;
        // Each block processes 4 vocab rows -> 32,000 blocks instead of 128,000!
        let grid_dim: u32 = (self.vocab_size as u32 + 3) / 4;

        unsafe {
            launch_kernel_on_stream(
                &func,
                (grid_dim, 1, 1),
                (block_dim, 1, 1),
                0,
                stream,
                &mut args,
            )?;
        }
        Ok(())
    }

    /// GPU-Side 2-Stage Argmax Reduction: Returns only the single best token ID (u32)
    /// avoiding PCIe transfer of 128,000 logits to host.
    pub fn sample_argmax(
        &mut self,
        gpu: &GpuContext,
        logits: &DeviceBuffer<f32>,
    ) -> Result<u32> {
        let stream = gpu.compute_stream();
        let stage1_func = self.ptx.get_function("argmax_stage1_fp32")?;
        let stage2_func = self.ptx.get_function("argmax_stage2_fp32")?;

        let num_stage1_blocks = ((self.vocab_size + 256 - 1) / 256) as u32;
        let mut d_logits = logits.inner().cu_deviceptr();
        let mut d_s1_vals = self.stage1_vals.inner_mut().cu_deviceptr();
        let mut d_s1_idx = self.stage1_indices.inner_mut().cu_deviceptr();
        let mut v_size = self.vocab_size as u32;

        let mut args1: [*mut std::ffi::c_void; 4] = [
            &mut d_s1_vals as *mut _ as *mut std::ffi::c_void,
            &mut d_s1_idx as *mut _ as *mut std::ffi::c_void,
            &mut d_logits as *mut _ as *mut std::ffi::c_void,
            &mut v_size as *mut _ as *mut std::ffi::c_void,
        ];

        unsafe {
            launch_kernel_on_stream(
                &stage1_func,
                (num_stage1_blocks, 1, 1),
                (256, 1, 1),
                0,
                stream,
                &mut args1,
            )?;
        }

        let mut d_best_token = self.best_token_buf.inner_mut().cu_deviceptr();
        let mut n_blocks = num_stage1_blocks;

        let mut args2: [*mut std::ffi::c_void; 4] = [
            &mut d_best_token as *mut _ as *mut std::ffi::c_void,
            &mut d_s1_vals as *mut _ as *mut std::ffi::c_void,
            &mut d_s1_idx as *mut _ as *mut std::ffi::c_void,
            &mut n_blocks as *mut _ as *mut std::ffi::c_void,
        ];

        unsafe {
            launch_kernel_on_stream(
                &stage2_func,
                (1, 1, 1),
                (512, 1, 1),
                0,
                stream,
                &mut args2,
            )?;
        }

        self.best_token_buf.inner().copy_to_pinned_host(stream, self.pinned_token_buf.inner_mut())?;
        Ok(self.pinned_token_buf.as_slice()[0])
    }
}
