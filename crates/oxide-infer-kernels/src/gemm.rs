//! Fused GEMV for Single-Token Decode ($B=1..4$) and Fused GEMM for Batched Inference.
//! Input multiplication in FP16, accumulation strictly into 32-bit registers (FP32) with 0-cycle latency penalty.

pub const FUSED_GEMV_FP16_FP32ACCUM_PTX: &str = r#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry fused_gemv_fp16_fp32accum(
	.param .u64 fused_gemv_fp16_fp32accum_param_0,
	.param .u64 fused_gemv_fp16_fp32accum_param_1,
	.param .u64 fused_gemv_fp16_fp32accum_param_2,
	.param .u32 fused_gemv_fp16_fp32accum_param_3,
	.param .u32 fused_gemv_fp16_fp32accum_param_4
)
{
	.reg .pred 	%p<12>;
	.reg .b16 	%rs<4>;
	.reg .f32 	%f<22>;
	.reg .b32 	%r<30>;
	.reg .b64 	%rd<16>;
	.shared .align 4 .f32 s_row_sum;

	ld.param.u64 	%rd4, [fused_gemv_fp16_fp32accum_param_0];
	ld.param.u64 	%rd5, [fused_gemv_fp16_fp32accum_param_1];
	ld.param.u64 	%rd6, [fused_gemv_fp16_fp32accum_param_2];
	ld.param.u32 	%r6, [fused_gemv_fp16_fp32accum_param_3];
	ld.param.u32 	%r7, [fused_gemv_fp16_fp32accum_param_4];
	mov.u32 	%r1, %tid.x;
	mov.u32 	%r2, %ctaid.x;
	setp.ge.s32 	%p1, %r2, %r7;
	@%p1 bra 	$L__BB0_10;

	setp.ne.s32 	%p2, %r1, 0;
	@%p2 bra 	$L__BB0_3;

	mov.u32 	%r8, 0;
	st.shared.u32 	[s_row_sum], %r8;

$L__BB0_3:
	bar.sync 	0;
	setp.ge.s32 	%p3, %r1, %r6;
	mov.f32 	%f21, 0f00000000;
	@%p3 bra 	$L__BB0_6;

	mov.u32 	%r3, %ntid.x;
	cvta.to.global.u64 	%rd1, %rd5;
	cvta.to.global.u64 	%rd2, %rd6;
	mul.lo.s32 	%r9, %r2, %r6;
	cvt.s64.s32 	%rd3, %r9;
	mov.f32 	%f21, 0f00000000;
	mov.u32 	%r29, %r1;

$L__BB0_5:
	cvt.s64.s32 	%rd7, %r29;
	mul.wide.s32 	%rd8, %r29, 2;
	add.s64 	%rd9, %rd1, %rd8;
	ld.global.nc.u16 	%rs1, [%rd9];
	cvt.f32.f16 %f7, %rs1;
	add.s64 	%rd10, %rd7, %rd3;
	shl.b64 	%rd11, %rd10, 1;
	add.s64 	%rd12, %rd2, %rd11;
	ld.global.nc.u16 	%rs2, [%rd12];
	cvt.f32.f16 %f8, %rs2;
	fma.rn.f32 	%f21, %f7, %f8, %f21;
	add.s32 	%r29, %r29, %r3;
	setp.lt.s32 	%p4, %r29, %r6;
	@%p4 bra 	$L__BB0_5;

$L__BB0_6:
	mov.b32 	%r10, %f21;
	mov.u32 	%r11, 2;
	mov.u32 	%r12, 31;
	mov.u32 	%r13, 16;
	mov.u32 	%r14, -1;
	shfl.sync.down.b32 	%r15|%p5, %r10, %r13, %r12, %r14;
	mov.b32 	%f9, %r15;
	add.f32 	%f10, %f21, %f9;
	mov.b32 	%r16, %f10;
	mov.u32 	%r17, 8;
	shfl.sync.down.b32 	%r18|%p6, %r16, %r17, %r12, %r14;
	mov.b32 	%f11, %r18;
	add.f32 	%f12, %f10, %f11;
	mov.b32 	%r19, %f12;
	mov.u32 	%r20, 4;
	shfl.sync.down.b32 	%r21|%p7, %r19, %r20, %r12, %r14;
	mov.b32 	%f13, %r21;
	add.f32 	%f14, %f12, %f13;
	mov.b32 	%r22, %f14;
	shfl.sync.down.b32 	%r23|%p8, %r22, %r11, %r12, %r14;
	mov.b32 	%f15, %r23;
	add.f32 	%f16, %f14, %f15;
	mov.b32 	%r24, %f16;
	mov.u32 	%r25, 1;
	shfl.sync.down.b32 	%r26|%p9, %r24, %r25, %r12, %r14;
	mov.b32 	%f17, %r26;
	add.f32 	%f4, %f16, %f17;
	and.b32  	%r27, %r1, 31;
	setp.ne.s32 	%p10, %r27, 0;
	@%p10 bra 	$L__BB0_8;

	mov.u32 	%r28, s_row_sum;
	atom.shared.add.f32 	%f18, [%r28], %f4;

$L__BB0_8:
	bar.sync 	0;
	@%p2 bra 	$L__BB0_10;

	ld.shared.f32 	%f19, [s_row_sum];
	cvt.rn.f16.f32 %rs3, %f19;
	cvta.to.global.u64 	%rd13, %rd4;
	mul.wide.s32 	%rd14, %r2, 2;
	add.s64 	%rd15, %rd13, %rd14;
	st.global.u16 	[%rd15], %rs3;

$L__BB0_10:
	ret;
}

// Fused Vectorized GEMV (8 elements per thread: ld.global.v4.u32 = 128-bit load)
.visible .entry fused_gemv_fp16_vec8(
    .param .u64 y_out,
    .param .u64 x_in,
    .param .u64 w_in,
    .param .u32 k_dim,
    .param .u32 m_dim
) {
    .reg .pred %p0, %p1;
    .reg .b32 %row_idx, %k_idx, %k_size, %m_size, %lane_id, %warp_id, %num_warps, %stride, %s_addr;
    .reg .b64 %y_ptr, %x_ptr, %w_ptr, %row_offset, %k_offset;
    .reg .b32 %x_v0, %x_v1, %x_v2, %x_v3;
    .reg .b32 %w_v0, %w_v1, %w_v2, %w_v3;
    .reg .f32 %fx0, %fx1, %fx2, %fx3, %fx4, %fx5, %fx6, %fx7;
    .reg .f32 %fw0, %fw1, %fw2, %fw3, %fw4, %fw5, %fw6, %fw7;
    .reg .f32 %f_sum, %shfl_val;

    ld.param.u32 %k_size, [k_dim];
    ld.param.u32 %m_size, [m_dim];

    mov.u32 %row_idx, %ctaid.x;
    setp.ge.u32 %p0, %row_idx, %m_size;
    @%p0 bra EXIT;

    ld.param.u64 %x_ptr, [x_in];
    ld.param.u64 %w_ptr, [w_in];

    mul.wide.u32 %row_offset, %row_idx, %k_size;
    shl.b64 %row_offset, %row_offset, 1;
    add.u64 %w_ptr, %w_ptr, %row_offset;

    // tid processes 8 elements at a time
    mov.u32 %k_idx, %tid.x;
    shl.b32 %k_idx, %k_idx, 3;

    mov.f32 %f_sum, 0.0;

VEC_LOOP:
    setp.ge.u32 %p1, %k_idx, %k_size;
    @%p1 bra REDUCE;

    cvt.u64.u32 %k_offset, %k_idx;
    shl.b64 %k_offset, %k_offset, 1;

    // 128-bit vector load from x (8 fp16 values = 4 u32 words)
    add.u64 %row_offset, %x_ptr, %k_offset;
    ld.global.v4.u32 {%x_v0, %x_v1, %x_v2, %x_v3}, [%row_offset];

    // 128-bit vector load from weights
    add.u64 %row_offset, %w_ptr, %k_offset;
    ld.global.v4.u32 {%w_v0, %w_v1, %w_v2, %w_v3}, [%row_offset];

    .reg .b16 %h_a0, %h_a1, %h_b0, %h_b1;

    // Pair 0
    mov.b32 {%h_a0, %h_a1}, %x_v0;
    mov.b32 {%h_b0, %h_b1}, %w_v0;
    cvt.f32.f16 %fx0, %h_a0;
    cvt.f32.f16 %fx1, %h_a1;
    cvt.f32.f16 %fw0, %h_b0;
    cvt.f32.f16 %fw1, %h_b1;
    fma.rn.f32 %f_sum, %fx0, %fw0, %f_sum;
    fma.rn.f32 %f_sum, %fx1, %fw1, %f_sum;

    // Pair 1
    mov.b32 {%h_a0, %h_a1}, %x_v1;
    mov.b32 {%h_b0, %h_b1}, %w_v1;
    cvt.f32.f16 %fx2, %h_a0;
    cvt.f32.f16 %fx3, %h_a1;
    cvt.f32.f16 %fw2, %h_b0;
    cvt.f32.f16 %fw3, %h_b1;
    fma.rn.f32 %f_sum, %fx2, %fw2, %f_sum;
    fma.rn.f32 %f_sum, %fx3, %fw3, %f_sum;

    // Pair 2
    mov.b32 {%h_a0, %h_a1}, %x_v2;
    mov.b32 {%h_b0, %h_b1}, %w_v2;
    cvt.f32.f16 %fx4, %h_a0;
    cvt.f32.f16 %fx5, %h_a1;
    cvt.f32.f16 %fw4, %h_b0;
    cvt.f32.f16 %fw5, %h_b1;
    fma.rn.f32 %f_sum, %fx4, %fw4, %f_sum;
    fma.rn.f32 %f_sum, %fx5, %fw5, %f_sum;

    // Pair 3
    mov.b32 {%h_a0, %h_a1}, %x_v3;
    mov.b32 {%h_b0, %h_b1}, %w_v3;
    cvt.f32.f16 %fx6, %h_a0;
    cvt.f32.f16 %fx7, %h_a1;
    cvt.f32.f16 %fw6, %h_b0;
    cvt.f32.f16 %fw7, %h_b1;
    fma.rn.f32 %f_sum, %fx6, %fw6, %f_sum;
    fma.rn.f32 %f_sum, %fx7, %fw7, %f_sum;

    mov.u32 %stride, %ntid.x;
    shl.b32 %stride, %stride, 3;
    add.u32 %k_idx, %k_idx, %stride;
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

    // Shared memory for block reduction across warps
    .shared .f32 s_gemv_sum[32];
    .reg .b32 %t_val;
    mov.u32 %t_val, %tid.x;
    and.b32 %lane_id, %t_val, 31;
    shr.u32 %warp_id, %t_val, 5;

    setp.eq.u32 %p0, %lane_id, 0;
    mov.u32 %s_addr, s_gemv_sum;
    shl.b32 %stride, %warp_id, 2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p0 st.shared.f32 [%s_addr], %f_sum;
    bar.sync 0;

    mov.u32 %num_warps, %ntid.x;
    shr.u32 %num_warps, %num_warps, 5;
    setp.lt.u32 %p0, %t_val, %num_warps;
    mov.u32 %s_addr, s_gemv_sum;
    shl.b32 %stride, %t_val, 2;
    add.u32 %s_addr, %s_addr, %stride;
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

    setp.eq.u32 %p0, %t_val, 0;
    @!%p0 bra EXIT;

    ld.param.u64 %y_ptr, [y_out];
    cvt.u64.u32 %row_offset, %row_idx;
    shl.b64 %row_offset, %row_offset, 1;
    add.u64 %y_ptr, %y_ptr, %row_offset;

    .reg .b16 %hy;
    cvt.rn.f16.f32 %hy, %f_sum;
    st.global.u16 [%y_ptr], %hy;

EXIT:
    ret;
}

// Fused Dual-GEMV SwiGLU:
// Computes gate = dot(w1[row], x) and up = dot(w3[row], x) simultaneously using 128-bit vector loads,
// then evaluates out[row] = silu(gate) * up = (gate / (1 + exp(-gate))) * up
// Writes directly to activated buffer, completely eliminating intermediate VRAM writes for gate & up!
.visible .entry fused_dual_gemv_swiglu_fp16(
    .param .u64 activated_out,
    .param .u64 x_in,
    .param .u64 w1_gate_in,
    .param .u64 w3_up_in,
    .param .u32 k_dim,
    .param .u32 m_dim
) {
    .reg .pred %p0, %p1;
    .reg .b32 %row_idx, %k_idx, %k_size, %m_size, %lane_id, %warp_id, %num_warps, %stride, %s_addr;
    .reg .b64 %act_ptr, %x_ptr, %w1_ptr, %w3_ptr, %row_offset, %k_offset;
    .reg .b32 %x_v0, %x_v1, %x_v2, %x_v3;
    .reg .b32 %w1_v0, %w1_v1, %w1_v2, %w1_v3;
    .reg .b32 %w3_v0, %w3_v1, %w3_v2, %w3_v3;
    .reg .f32 %fx0, %fx1, %fx2, %fx3, %fx4, %fx5, %fx6, %fx7;
    .reg .f32 %fw1_0, %fw1_1, %fw1_2, %fw1_3, %fw1_4, %fw1_5, %fw1_6, %fw1_7;
    .reg .f32 %fw3_0, %fw3_1, %fw3_2, %fw3_3, %fw3_4, %fw3_5, %fw3_6, %fw3_7;
    .reg .f32 %f_gate_sum, %f_up_sum, %shfl_gate, %shfl_up;
    .reg .f32 %f_neg_gate, %f_exp, %f_sig, %f_act;

    ld.param.u32 %k_size, [k_dim];
    ld.param.u32 %m_size, [m_dim];

    mov.u32 %row_idx, %ctaid.x;
    setp.ge.u32 %p0, %row_idx, %m_size;
    @%p0 bra EXIT;

    ld.param.u64 %x_ptr, [x_in];
    ld.param.u64 %w1_ptr, [w1_gate_in];
    ld.param.u64 %w3_ptr, [w3_up_in];

    mul.wide.u32 %row_offset, %row_idx, %k_size;
    shl.b64 %row_offset, %row_offset, 1;
    add.u64 %w1_ptr, %w1_ptr, %row_offset;
    add.u64 %w3_ptr, %w3_ptr, %row_offset;

    // tid processes 8 elements at a time
    mov.u32 %k_idx, %tid.x;
    shl.b32 %k_idx, %k_idx, 3;

    mov.f32 %f_gate_sum, 0.0;
    mov.f32 %f_up_sum, 0.0;

VEC_LOOP:
    setp.ge.u32 %p1, %k_idx, %k_size;
    @%p1 bra REDUCE;

    cvt.u64.u32 %k_offset, %k_idx;
    shl.b64 %k_offset, %k_offset, 1;

    // 128-bit vector load from x (8 fp16 values = 4 u32 words)
    add.u64 %row_offset, %x_ptr, %k_offset;
    ld.global.v4.u32 {%x_v0, %x_v1, %x_v2, %x_v3}, [%row_offset];

    // 128-bit vector load from w1 (gate)
    add.u64 %row_offset, %w1_ptr, %k_offset;
    ld.global.v4.u32 {%w1_v0, %w1_v1, %w1_v2, %w1_v3}, [%row_offset];

    // 128-bit vector load from w3 (up)
    add.u64 %row_offset, %w3_ptr, %k_offset;
    ld.global.v4.u32 {%w3_v0, %w3_v1, %w3_v2, %w3_v3}, [%row_offset];

    .reg .b16 %h_x0, %h_x1, %h_w1_0, %h_w1_1, %h_w3_0, %h_w3_1;

    // Pair 0
    mov.b32 {%h_x0, %h_x1}, %x_v0;
    mov.b32 {%h_w1_0, %h_w1_1}, %w1_v0;
    mov.b32 {%h_w3_0, %h_w3_1}, %w3_v0;
    cvt.f32.f16 %fx0, %h_x0;
    cvt.f32.f16 %fx1, %h_x1;
    cvt.f32.f16 %fw1_0, %h_w1_0;
    cvt.f32.f16 %fw1_1, %h_w1_1;
    cvt.f32.f16 %fw3_0, %h_w3_0;
    cvt.f32.f16 %fw3_1, %h_w3_1;
    fma.rn.f32 %f_gate_sum, %fx0, %fw1_0, %f_gate_sum;
    fma.rn.f32 %f_gate_sum, %fx1, %fw1_1, %f_gate_sum;
    fma.rn.f32 %f_up_sum, %fx0, %fw3_0, %f_up_sum;
    fma.rn.f32 %f_up_sum, %fx1, %fw3_1, %f_up_sum;

    // Pair 1
    mov.b32 {%h_x0, %h_x1}, %x_v1;
    mov.b32 {%h_w1_0, %h_w1_1}, %w1_v1;
    mov.b32 {%h_w3_0, %h_w3_1}, %w3_v1;
    cvt.f32.f16 %fx2, %h_x0;
    cvt.f32.f16 %fx3, %h_x1;
    cvt.f32.f16 %fw1_2, %h_w1_0;
    cvt.f32.f16 %fw1_3, %h_w1_1;
    cvt.f32.f16 %fw3_2, %h_w3_0;
    cvt.f32.f16 %fw3_3, %h_w3_1;
    fma.rn.f32 %f_gate_sum, %fx2, %fw1_2, %f_gate_sum;
    fma.rn.f32 %f_gate_sum, %fx3, %fw1_3, %f_gate_sum;
    fma.rn.f32 %f_up_sum, %fx2, %fw3_2, %f_up_sum;
    fma.rn.f32 %f_up_sum, %fx3, %fw3_3, %f_up_sum;

    // Pair 2
    mov.b32 {%h_x0, %h_x1}, %x_v2;
    mov.b32 {%h_w1_0, %h_w1_1}, %w1_v2;
    mov.b32 {%h_w3_0, %h_w3_1}, %w3_v2;
    cvt.f32.f16 %fx4, %h_x0;
    cvt.f32.f16 %fx5, %h_x1;
    cvt.f32.f16 %fw1_4, %h_w1_0;
    cvt.f32.f16 %fw1_5, %h_w1_1;
    cvt.f32.f16 %fw3_4, %h_w3_0;
    cvt.f32.f16 %fw3_5, %h_w3_1;
    fma.rn.f32 %f_gate_sum, %fx4, %fw1_4, %f_gate_sum;
    fma.rn.f32 %f_gate_sum, %fx5, %fw1_5, %f_gate_sum;
    fma.rn.f32 %f_up_sum, %fx4, %fw3_4, %f_up_sum;
    fma.rn.f32 %f_up_sum, %fx5, %fw3_5, %f_up_sum;

    // Pair 3
    mov.b32 {%h_x0, %h_x1}, %x_v3;
    mov.b32 {%h_w1_0, %h_w1_1}, %w1_v3;
    mov.b32 {%h_w3_0, %h_w3_1}, %w3_v3;
    cvt.f32.f16 %fx6, %h_x0;
    cvt.f32.f16 %fx7, %h_x1;
    cvt.f32.f16 %fw1_6, %h_w1_0;
    cvt.f32.f16 %fw1_7, %h_w1_1;
    cvt.f32.f16 %fw3_6, %h_w3_0;
    cvt.f32.f16 %fw3_7, %h_w3_1;
    fma.rn.f32 %f_gate_sum, %fx6, %fw1_6, %f_gate_sum;
    fma.rn.f32 %f_gate_sum, %fx7, %fw1_7, %f_gate_sum;
    fma.rn.f32 %f_up_sum, %fx6, %fw3_6, %f_up_sum;
    fma.rn.f32 %f_up_sum, %fx7, %fw3_7, %f_up_sum;

    mov.u32 %stride, %ntid.x;
    shl.b32 %stride, %stride, 3;
    add.u32 %k_idx, %k_idx, %stride;
    bra VEC_LOOP;

REDUCE:
    // Warp shuffle reduce for gate
    shfl.sync.down.b32 %shfl_gate, %f_gate_sum, 16, 31, 0xffffffff;
    add.f32 %f_gate_sum, %f_gate_sum, %shfl_gate;
    shfl.sync.down.b32 %shfl_gate, %f_gate_sum, 8, 31, 0xffffffff;
    add.f32 %f_gate_sum, %f_gate_sum, %shfl_gate;
    shfl.sync.down.b32 %shfl_gate, %f_gate_sum, 4, 31, 0xffffffff;
    add.f32 %f_gate_sum, %f_gate_sum, %shfl_gate;
    shfl.sync.down.b32 %shfl_gate, %f_gate_sum, 2, 31, 0xffffffff;
    add.f32 %f_gate_sum, %f_gate_sum, %shfl_gate;
    shfl.sync.down.b32 %shfl_gate, %f_gate_sum, 1, 31, 0xffffffff;
    add.f32 %f_gate_sum, %f_gate_sum, %shfl_gate;

    // Warp shuffle reduce for up
    shfl.sync.down.b32 %shfl_up, %f_up_sum, 16, 31, 0xffffffff;
    add.f32 %f_up_sum, %f_up_sum, %shfl_up;
    shfl.sync.down.b32 %shfl_up, %f_up_sum, 8, 31, 0xffffffff;
    add.f32 %f_up_sum, %f_up_sum, %shfl_up;
    shfl.sync.down.b32 %shfl_up, %f_up_sum, 4, 31, 0xffffffff;
    add.f32 %f_up_sum, %f_up_sum, %shfl_up;
    shfl.sync.down.b32 %shfl_up, %f_up_sum, 2, 31, 0xffffffff;
    add.f32 %f_up_sum, %f_up_sum, %shfl_up;
    shfl.sync.down.b32 %shfl_up, %f_up_sum, 1, 31, 0xffffffff;
    add.f32 %f_up_sum, %f_up_sum, %shfl_up;

    // Shared memory for block reduction (gate & up)
    .shared .f32 s_gate[16];
    .shared .f32 s_up[16];
    .reg .b32 %t_val;
    mov.u32 %t_val, %tid.x;
    and.b32 %lane_id, %t_val, 31;
    shr.u32 %warp_id, %t_val, 5;

    setp.eq.u32 %p0, %lane_id, 0;
    mov.u32 %s_addr, s_gate;
    shl.b32 %stride, %warp_id, 2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p0 st.shared.f32 [%s_addr], %f_gate_sum;

    mov.u32 %s_addr, s_up;
    add.u32 %s_addr, %s_addr, %stride;
    @%p0 st.shared.f32 [%s_addr], %f_up_sum;
    bar.sync 0;

    mov.u32 %num_warps, %ntid.x;
    shr.u32 %num_warps, %num_warps, 5;
    setp.lt.u32 %p0, %t_val, %num_warps;

    mov.u32 %s_addr, s_gate;
    shl.b32 %stride, %t_val, 2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p0 ld.shared.f32 %f_gate_sum, [%s_addr];
    @!%p0 mov.f32 %f_gate_sum, 0.0;

    mov.u32 %s_addr, s_up;
    add.u32 %s_addr, %s_addr, %stride;
    @%p0 ld.shared.f32 %f_up_sum, [%s_addr];
    @!%p0 mov.f32 %f_up_sum, 0.0;

    // Final reduce across warps
    shfl.sync.down.b32 %shfl_gate, %f_gate_sum, 16, 31, 0xffffffff;
    add.f32 %f_gate_sum, %f_gate_sum, %shfl_gate;
    shfl.sync.down.b32 %shfl_gate, %f_gate_sum, 8, 31, 0xffffffff;
    add.f32 %f_gate_sum, %f_gate_sum, %shfl_gate;
    shfl.sync.down.b32 %shfl_gate, %f_gate_sum, 4, 31, 0xffffffff;
    add.f32 %f_gate_sum, %f_gate_sum, %shfl_gate;
    shfl.sync.down.b32 %shfl_gate, %f_gate_sum, 2, 31, 0xffffffff;
    add.f32 %f_gate_sum, %f_gate_sum, %shfl_gate;
    shfl.sync.down.b32 %shfl_gate, %f_gate_sum, 1, 31, 0xffffffff;
    add.f32 %f_gate_sum, %f_gate_sum, %shfl_gate;

    shfl.sync.down.b32 %shfl_up, %f_up_sum, 16, 31, 0xffffffff;
    add.f32 %f_up_sum, %f_up_sum, %shfl_up;
    shfl.sync.down.b32 %shfl_up, %f_up_sum, 8, 31, 0xffffffff;
    add.f32 %f_up_sum, %f_up_sum, %shfl_up;
    shfl.sync.down.b32 %shfl_up, %f_up_sum, 4, 31, 0xffffffff;
    add.f32 %f_up_sum, %f_up_sum, %shfl_up;
    shfl.sync.down.b32 %shfl_up, %f_up_sum, 2, 31, 0xffffffff;
    add.f32 %f_up_sum, %f_up_sum, %shfl_up;
    shfl.sync.down.b32 %shfl_up, %f_up_sum, 1, 31, 0xffffffff;
    add.f32 %f_up_sum, %f_up_sum, %shfl_up;

    setp.eq.u32 %p0, %t_val, 0;
    @!%p0 bra EXIT;

    // Fused SiLU * up:
    // sig = 1.0 / (1.0 + exp(-gate))
    // act = (gate * sig) * up
    neg.f32 %f_neg_gate, %f_gate_sum;
    mul.f32 %f_neg_gate, %f_neg_gate, 1.44269504; // log2(e)
    ex2.approx.ftz.f32 %f_exp, %f_neg_gate;
    add.f32 %f_exp, %f_exp, 1.0;
    rcp.approx.ftz.f32 %f_sig, %f_exp;

    mul.f32 %f_act, %f_gate_sum, %f_sig;
    mul.f32 %f_act, %f_act, %f_up_sum;

    ld.param.u64 %act_ptr, [activated_out];
    cvt.u64.u32 %row_offset, %row_idx;
    shl.b64 %row_offset, %row_offset, 1;
    add.u64 %act_ptr, %act_ptr, %row_offset;

    .reg .b16 %h_act;
    cvt.rn.f16.f32 %h_act, %f_act;
    st.global.u16 [%act_ptr], %h_act;

EXIT:
    ret;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use oxide_infer_core::{GpuContext, DeviceBuffer};
    use crate::ptx::PtxModule;
    use cuda_core::launch_kernel_on_stream;

    #[test]
    fn test_fused_gemv_fp32accum_execution() {
        let gpu = GpuContext::init(0).expect("Init GPU");
        let module = PtxModule::load_from_ptx(&gpu, FUSED_GEMV_FP16_FP32ACCUM_PTX).expect("Load PTX");
        let func = module.get_function("fused_gemv_fp16_fp32accum").expect("Get func");

        let k_dim = 64usize;
        let n_dim = 16usize;

        // Vector of 1.0s, matrix of 0.5s -> each dot product should equal 64 * 0.5 = 32.0
        let in_vec_host: Vec<half::f16> = (0..k_dim).map(|_| half::f16::from_f32(1.0)).collect();
        let weight_host: Vec<half::f16> = (0..n_dim * k_dim).map(|_| half::f16::from_f32(0.5)).collect();

        let d_in = DeviceBuffer::from_slice(&gpu, &in_vec_host).unwrap();
        let d_weight = DeviceBuffer::from_slice(&gpu, &weight_host).unwrap();
        let mut d_out = DeviceBuffer::<half::f16>::zeroed(&gpu, n_dim).unwrap();

        let stream = gpu.compute_stream();
        
        let mut d_out_ptr = d_out.inner_mut().cu_deviceptr();
        let mut d_in_ptr = d_in.inner().cu_deviceptr();
        let mut d_weight_ptr = d_weight.inner().cu_deviceptr();
        let mut k_dim_u32 = k_dim as u32;
        let mut n_dim_u32 = n_dim as u32;

        let mut args: [*mut std::ffi::c_void; 5] = [
            &mut d_out_ptr as *mut _ as *mut std::ffi::c_void,
            &mut d_in_ptr as *mut _ as *mut std::ffi::c_void,
            &mut d_weight_ptr as *mut _ as *mut std::ffi::c_void,
            &mut k_dim_u32 as *mut _ as *mut std::ffi::c_void,
            &mut n_dim_u32 as *mut _ as *mut std::ffi::c_void,
        ];

        unsafe {
            launch_kernel_on_stream(
                &func,
                (n_dim as u32, 1, 1),
                (64, 1, 1),
                4, // shared memory for s_row_sum
                stream,
                &mut args,
            ).expect("Launch GEMV failed");
        }

        stream.synchronize().expect("Sync stream");

        let result = d_out.inner().to_host_vec(stream).expect("Copy back");
        assert_eq!(result.len(), n_dim);
        
        for val in result {
            let f = val.to_f32();
            assert!((f - 32.0).abs() < 1e-1, "Expected 32.0, got {}", f);
        }
    }
}

// Batched Vectorized GEMV / Small-M GEMM for concurrent decode across B batches
// y_out: [B, m_dim], x_in: [B, k_dim], w_in: [m_dim, k_dim] (weights shared!)
pub const BATCHED_GEMV_FP16_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry batched_gemv_fp16_vec8(
    .param .u64 y_out,
    .param .u64 x_in,
    .param .u64 w_in,
    .param .u32 k_dim,
    .param .u32 m_dim,
    .param .u32 batch_size
) {
    .reg .pred %p0, %p1, %p_batch;
    .reg .b32 %row_idx, %batch_idx, %k_idx, %k_size, %m_size, %b_size, %lane_id, %stride;
    .reg .b64 %y_ptr, %x_ptr, %w_ptr, %w_row_offset, %x_batch_offset, %y_batch_offset, %k_offset;
    .reg .b32 %x_v0, %x_v1, %x_v2, %x_v3;
    .reg .b32 %w_v0, %w_v1, %w_v2, %w_v3;
    .reg .f32 %fx0, %fx1, %fx2, %fx3, %fx4, %fx5, %fx6, %fx7;
    .reg .f32 %fw0, %fw1, %fw2, %fw3, %fw4, %fw5, %fw6, %fw7;
    .reg .f32 %f_sum, %shfl_val;

    ld.param.u32 %k_size, [k_dim];
    ld.param.u32 %m_size, [m_dim];
    ld.param.u32 %b_size, [batch_size];

    mov.u32 %row_idx, %ctaid.x;
    mov.u32 %batch_idx, %ctaid.y;

    setp.ge.u32 %p0, %row_idx, %m_size;
    setp.ge.u32 %p_batch, %batch_idx, %b_size;
    or.pred %p0, %p0, %p_batch;
    @%p0 bra EXIT;

    ld.param.u64 %x_ptr, [x_in];
    ld.param.u64 %w_ptr, [w_in];
    ld.param.u64 %y_ptr, [y_out];

    // Compute pointer to weight row: w_ptr + row_idx * k_size * 2
    mul.wide.u32 %w_row_offset, %row_idx, %k_size;
    shl.b64 %w_row_offset, %w_row_offset, 1;
    add.u64 %w_ptr, %w_ptr, %w_row_offset;

    // Compute pointer to input batch row: x_ptr + batch_idx * k_size * 2
    mul.wide.u32 %x_batch_offset, %batch_idx, %k_size;
    shl.b64 %x_batch_offset, %x_batch_offset, 1;
    add.u64 %x_ptr, %x_ptr, %x_batch_offset;

    // tid processes 8 elements at a time
    mov.u32 %k_idx, %tid.x;
    shl.b32 %k_idx, %k_idx, 3;

    mov.f32 %f_sum, 0.0;

VEC_LOOP:
    setp.ge.u32 %p1, %k_idx, %k_size;
    @%p1 bra REDUCE;

    cvt.u64.u32 %k_offset, %k_idx;
    shl.b64 %k_offset, %k_offset, 1;

    // 128-bit vector load for x (16 bytes)
    .reg .b64 %cur_x_ptr;
    add.u64 %cur_x_ptr, %x_ptr, %k_offset;
    ld.global.v4.u32 {%x_v0, %x_v1, %x_v2, %x_v3}, [%cur_x_ptr];

    // 128-bit vector load for weight (16 bytes)
    .reg .b64 %cur_w_ptr;
    add.u64 %cur_w_ptr, %w_ptr, %k_offset;
    ld.global.v4.u32 {%w_v0, %w_v1, %w_v2, %w_v3}, [%cur_w_ptr];

    // Unpack FP16 pairs
    .reg .b16 %xi0, %xi1, %xi2, %xi3, %xi4, %xi5, %xi6, %xi7;
    .reg .b16 %wi0, %wi1, %wi2, %wi3, %wi4, %wi5, %wi6, %wi7;
    mov.b32 {%xi0, %xi1}, %x_v0;
    mov.b32 {%xi2, %xi3}, %x_v1;
    mov.b32 {%xi4, %xi5}, %x_v2;
    mov.b32 {%xi6, %xi7}, %x_v3;

    mov.b32 {%wi0, %wi1}, %w_v0;
    mov.b32 {%wi2, %wi3}, %w_v1;
    mov.b32 {%wi4, %wi5}, %w_v2;
    mov.b32 {%wi6, %wi7}, %w_v3;

    cvt.f32.f16 %fx0, %xi0;
    cvt.f32.f16 %fx1, %xi1;
    cvt.f32.f16 %fx2, %xi2;
    cvt.f32.f16 %fx3, %xi3;
    cvt.f32.f16 %fx4, %xi4;
    cvt.f32.f16 %fx5, %xi5;
    cvt.f32.f16 %fx6, %xi6;
    cvt.f32.f16 %fx7, %xi7;

    cvt.f32.f16 %fw0, %wi0;
    cvt.f32.f16 %fw1, %wi1;
    cvt.f32.f16 %fw2, %wi2;
    cvt.f32.f16 %fw3, %wi3;
    cvt.f32.f16 %fw4, %wi4;
    cvt.f32.f16 %fw5, %wi5;
    cvt.f32.f16 %fw6, %wi6;
    cvt.f32.f16 %fw7, %wi7;

    fma.rn.f32 %f_sum, %fx0, %fw0, %f_sum;
    fma.rn.f32 %f_sum, %fx1, %fw1, %f_sum;
    fma.rn.f32 %f_sum, %fx2, %fw2, %f_sum;
    fma.rn.f32 %f_sum, %fx3, %fw3, %f_sum;
    fma.rn.f32 %f_sum, %fx4, %fw4, %f_sum;
    fma.rn.f32 %f_sum, %fx5, %fw5, %f_sum;
    fma.rn.f32 %f_sum, %fx6, %fw6, %f_sum;
    fma.rn.f32 %f_sum, %fx7, %fw7, %f_sum;

    mov.u32 %stride, %ntid.x;
    shl.b32 %stride, %stride, 3;
    add.u32 %k_idx, %k_idx, %stride;
    bra VEC_LOOP;

REDUCE:
    // In-warp butterfly reduction
    shfl.sync.bfly.b32 %shfl_val, %f_sum, 16, 31, 0xffffffff;
    add.f32 %f_sum, %f_sum, %shfl_val;
    shfl.sync.bfly.b32 %shfl_val, %f_sum, 8, 31, 0xffffffff;
    add.f32 %f_sum, %f_sum, %shfl_val;
    shfl.sync.bfly.b32 %shfl_val, %f_sum, 4, 31, 0xffffffff;
    add.f32 %f_sum, %f_sum, %shfl_val;
    shfl.sync.bfly.b32 %shfl_val, %f_sum, 2, 31, 0xffffffff;
    add.f32 %f_sum, %f_sum, %shfl_val;
    shfl.sync.bfly.b32 %shfl_val, %f_sum, 1, 31, 0xffffffff;
    add.f32 %f_sum, %f_sum, %shfl_val;

    mov.u32 %lane_id, %laneid;
    setp.ne.u32 %p0, %lane_id, 0;
    @%p0 bra EXIT;

    // Thread 0 writes output to y_out[batch_idx * m_dim + row_idx]
    mul.wide.u32 %y_batch_offset, %batch_idx, %m_size;
    cvt.u64.u32 %w_row_offset, %row_idx;
    add.u64 %y_batch_offset, %y_batch_offset, %w_row_offset;
    shl.b64 %y_batch_offset, %y_batch_offset, 1;
    add.u64 %y_ptr, %y_ptr, %y_batch_offset;

    .reg .b16 %h_out;
    cvt.rn.f16.f32 %h_out, %f_sum;
    st.global.u16 [%y_ptr], %h_out;

EXIT:
    ret;
}
"#;
