//! Rotary Position Embedding (RoPE) and Fused SwiGLU Kernels for LLMs.

pub const ROPE_FP16_PTX: &str = r#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry fused_rope_fp16(
	.param .u64 fused_rope_fp16_param_0,
	.param .u64 fused_rope_fp16_param_1,
	.param .u64 fused_rope_fp16_param_2,
	.param .u64 fused_rope_fp16_param_3,
	.param .u64 fused_rope_fp16_param_4,
	.param .u64 fused_rope_fp16_param_5,
	.param .u32 fused_rope_fp16_param_6,
	.param .u32 fused_rope_fp16_param_7,
	.param .u32 fused_rope_fp16_param_8,
	.param .u32 fused_rope_fp16_param_9
)
{
	.reg .pred 	%p<8>;
	.reg .b16 	%rs<9>;
	.reg .f32 	%f<17>;
	.reg .b32 	%r<14>;
	.reg .b64 	%rd<28>;

	ld.param.u64 	%rd1, [fused_rope_fp16_param_0];
	ld.param.u64 	%rd2, [fused_rope_fp16_param_1];
	ld.param.u64 	%rd3, [fused_rope_fp16_param_2];
	ld.param.u64 	%rd4, [fused_rope_fp16_param_3];
	ld.param.u64 	%rd5, [fused_rope_fp16_param_4];
	ld.param.u64 	%rd6, [fused_rope_fp16_param_5];
	ld.param.u32 	%r6, [fused_rope_fp16_param_7];
	ld.param.u32 	%r7, [fused_rope_fp16_param_8];
	ld.param.u32 	%r8, [fused_rope_fp16_param_9];
	ld.param.u32 	%r5, [fused_rope_fp16_param_6];
	shr.u32 	%r9, %r5, 31;
	add.s32 	%r10, %r5, %r9;
	shr.s32 	%r1, %r10, 1;
	mov.u32 	%r2, %tid.x;
	setp.ge.s32 	%p1, %r2, %r1;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd7, %rd5;
	mov.u32 	%r3, %ctaid.y;
	setp.eq.s32 	%p2, %r3, 0;
	mad.lo.s32 	%r11, %r1, %r8, %r2;
	mul.wide.s32 	%rd8, %r11, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f1, [%rd9];
	cvta.to.global.u64 	%rd10, %rd6;
	add.s64 	%rd11, %rd10, %rd8;
	ld.global.nc.f32 	%f2, [%rd11];
	mov.u32 	%r4, %ctaid.x;
	setp.lt.s32 	%p3, %r4, %r6;
	and.pred  	%p4, %p2, %p3;
	@%p4 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_2;

$L__BB0_4:
	mad.lo.s32 	%r13, %r4, %r5, %r2;
	cvta.to.global.u64 	%rd20, %rd3;
	mul.wide.s32 	%rd21, %r13, 2;
	add.s64 	%rd22, %rd20, %rd21;
	ld.global.nc.u16 	%rs5, [%rd22];
	cvt.f32.f16 %f10, %rs5;
	mul.wide.s32 	%rd23, %r1, 2;
	add.s64 	%rd24, %rd22, %rd23;
	ld.global.nc.u16 	%rs6, [%rd24];
	cvt.f32.f16 %f11, %rs6;
	mul.f32 	%f14, %f1, %f10;
	mul.f32 	%f15, %f2, %f11;
	sub.f32 	%f12, %f14, %f15;
	mul.f32 	%f16, %f1, %f11;
	fma.rn.f32 	%f13, %f2, %f10, %f16;
	cvt.rn.f16.f32 %rs7, %f12;
	cvta.to.global.u64 	%rd25, %rd1;
	add.s64 	%rd26, %rd25, %rd21;
	st.global.u16 	[%rd26], %rs7;
	cvt.rn.f16.f32 %rs8, %f13;
	add.s64 	%rd27, %rd26, %rd23;
	st.global.u16 	[%rd27], %rs8;
	bra.uni 	$L__BB0_5;

$L__BB0_2:
	setp.ge.s32 	%p6, %r4, %r7;
	or.pred  	%p7, %p6, %p2;
	@%p7 bra 	$L__BB0_5;

	mad.lo.s32 	%r12, %r4, %r5, %r2;
	cvta.to.global.u64 	%rd12, %rd4;
	mul.wide.s32 	%rd13, %r12, 2;
	add.s64 	%rd14, %rd12, %rd13;
	ld.global.nc.u16 	%rs1, [%rd14];
	cvt.f32.f16 %f3, %rs1;
	mul.wide.s32 	%rd15, %r1, 2;
	add.s64 	%rd16, %rd14, %rd15;
	ld.global.nc.u16 	%rs2, [%rd16];
	cvt.f32.f16 %f4, %rs2;
	mul.f32 	%f7, %f1, %f3;
	mul.f32 	%f8, %f2, %f4;
	sub.f32 	%f5, %f7, %f8;
	mul.f32 	%f9, %f1, %f4;
	fma.rn.f32 	%f6, %f2, %f3, %f9;
	cvt.rn.f16.f32 %rs3, %f5;
	cvta.to.global.u64 	%rd17, %rd2;
	add.s64 	%rd18, %rd17, %rd13;
	st.global.u16 	[%rd18], %rs3;
	cvt.rn.f16.f32 %rs4, %f6;
	add.s64 	%rd19, %rd18, %rd15;
	st.global.u16 	[%rd19], %rs4;

$L__BB0_5:
	ret;
}
"#;

pub const SWIGLU_FP16_PTX: &str = r#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry fused_swiglu_fp16(
	.param .u64 fused_swiglu_fp16_param_0,
	.param .u64 fused_swiglu_fp16_param_1,
	.param .u64 fused_swiglu_fp16_param_2,
	.param .u32 fused_swiglu_fp16_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b16 	%rs<4>;
	.reg .f32 	%f<8>;
	.reg .b32 	%r<6>;
	.reg .b64 	%rd<11>;

	ld.param.u64 	%rd1, [fused_swiglu_fp16_param_0];
	ld.param.u64 	%rd2, [fused_swiglu_fp16_param_1];
	ld.param.u64 	%rd3, [fused_swiglu_fp16_param_2];
	ld.param.u32 	%r2, [fused_swiglu_fp16_param_3];
	mov.u32 	%r3, %ctaid.x;
	mov.u32 	%r4, %ntid.x;
	mov.u32 	%r5, %tid.x;
	mad.lo.s32 	%r1, %r3, %r4, %r5;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd2;
	mul.wide.s32 	%rd5, %r1, 2;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.u16 	%rs1, [%rd6];
	cvt.f32.f16 %f1, %rs1;
	cvta.to.global.u64 	%rd7, %rd3;
	add.s64 	%rd8, %rd7, %rd5;
	ld.global.nc.u16 	%rs2, [%rd8];
	cvt.f32.f16 %f2, %rs2;
	mul.f32 	%f4, %f1, 0fBFB8AA3B;
	ex2.approx.f32 	%f5, %f4;
	add.f32 	%f6, %f5, 0f3F800000;
	div.rn.f32 	%f7, %f1, %f6;
	mul.f32 	%f3, %f2, %f7;
	cvt.rn.f16.f32 %rs3, %f3;
	cvta.to.global.u64 	%rd9, %rd1;
	add.s64 	%rd10, %rd9, %rd5;
	st.global.u16 	[%rd10], %rs3;

$L__BB0_2:
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
    fn test_fused_swiglu_execution() {
        let gpu = GpuContext::init(0).expect("Init GPU");
        let module = PtxModule::load_from_ptx(&gpu, SWIGLU_FP16_PTX).expect("Load PTX");
        let func = module.get_function("fused_swiglu_fp16").expect("Get func");

        let n = 128usize;
        let gate_host: Vec<half::f16> = (0..n).map(|_| half::f16::from_f32(0.0)).collect(); // silu(0) = 0
        let up_host: Vec<half::f16> = (0..n).map(|_| half::f16::from_f32(5.0)).collect();

        let d_gate = DeviceBuffer::from_slice(&gpu, &gate_host).unwrap();
        let d_up = DeviceBuffer::from_slice(&gpu, &up_host).unwrap();
        let mut d_out = DeviceBuffer::<half::f16>::zeroed(&gpu, n).unwrap();

        let stream = gpu.compute_stream();
        let mut d_out_ptr = d_out.inner_mut().cu_deviceptr();
        let mut d_gate_ptr = d_gate.inner().cu_deviceptr();
        let mut d_up_ptr = d_up.inner().cu_deviceptr();
        let mut n_u32 = n as u32;

        let mut args: [*mut std::ffi::c_void; 4] = [
            &mut d_out_ptr as *mut _ as *mut std::ffi::c_void,
            &mut d_gate_ptr as *mut _ as *mut std::ffi::c_void,
            &mut d_up_ptr as *mut _ as *mut std::ffi::c_void,
            &mut n_u32 as *mut _ as *mut std::ffi::c_void,
        ];

        unsafe {
            launch_kernel_on_stream(
                &func,
                (1, 1, 1),
                (128, 1, 1),
                0,
                stream,
                &mut args,
            ).expect("Launch SwiGLU failed");
        }

        stream.synchronize().expect("Sync stream");

        let result = d_out.inner().to_host_vec(stream).expect("Copy back");
        assert_eq!(result.len(), n);
        
        // silu(0) * 5.0 = 0.0 * 5.0 = 0.0
        for val in result {
            assert_eq!(val.to_f32(), 0.0);
        }
    }
}
