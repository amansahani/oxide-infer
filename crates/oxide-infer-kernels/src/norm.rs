//! High-Performance Fused Root Mean Square Normalization (RMSNorm)
//! In-warp shuffle reduction with FP32 arithmetic accumulation and zeroed shared memory.

pub const RMS_NORM_FP16_PTX: &str = r#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry rms_norm_fp16(
	.param .u64 rms_norm_fp16_param_0,
	.param .u64 rms_norm_fp16_param_1,
	.param .u64 rms_norm_fp16_param_2,
	.param .f32 rms_norm_fp16_param_3,
	.param .u32 rms_norm_fp16_param_4
)
{
	.reg .pred 	%p<12>;
	.reg .b16 	%rs<5>;
	.reg .f32 	%f<30>;
	.reg .b32 	%r<33>;
	.reg .b64 	%rd<19>;
	.shared .align 4 .f32 s_mean;

	ld.param.u64 	%rd5, [rms_norm_fp16_param_0];
	ld.param.u64 	%rd7, [rms_norm_fp16_param_1];
	ld.param.u64 	%rd6, [rms_norm_fp16_param_2];
	ld.param.f32 	%f6, [rms_norm_fp16_param_3];
	ld.param.u32 	%r8, [rms_norm_fp16_param_4];
	cvta.to.global.u64 	%rd1, %rd7;
	mov.u32 	%r32, %tid.x;
	setp.ne.s32 	%p1, %r32, 0;
	@%p1 bra 	$L__BB0_2;

	mov.u32 	%r9, 0;
	st.shared.u32 	[s_mean], %r9;

$L__BB0_2:
	bar.sync 	0;
	mov.u32 	%r10, %ctaid.x;
	mul.lo.s32 	%r11, %r10, %r8;
	cvt.s64.s32 	%rd2, %r11;
	setp.ge.s32 	%p2, %r32, %r8;
	mov.f32 	%f29, 0f00000000;
	@%p2 bra 	$L__BB0_5;

	mov.f32 	%f29, 0f00000000;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r31, %r32;

$L__BB0_4:
	cvt.s64.s32 	%rd8, %r31;
	add.s64 	%rd9, %rd8, %rd2;
	shl.b64 	%rd10, %rd9, 1;
	add.s64 	%rd11, %rd1, %rd10;
	ld.global.nc.u16 	%rs1, [%rd11];
	cvt.f32.f16 %f9, %rs1;
	fma.rn.f32 	%f29, %f9, %f9, %f29;
	add.s32 	%r31, %r31, %r2;
	setp.lt.s32 	%p3, %r31, %r8;
	@%p3 bra 	$L__BB0_4;

$L__BB0_5:
	mov.b32 	%r12, %f29;
	mov.u32 	%r13, 2;
	mov.u32 	%r14, 31;
	mov.u32 	%r15, 16;
	mov.u32 	%r16, -1;
	shfl.sync.down.b32 	%r17|%p4, %r12, %r15, %r14, %r16;
	mov.b32 	%f10, %r17;
	add.f32 	%f11, %f29, %f10;
	mov.b32 	%r18, %f11;
	mov.u32 	%r19, 8;
	shfl.sync.down.b32 	%r20|%p5, %r18, %r19, %r14, %r16;
	mov.b32 	%f12, %r20;
	add.f32 	%f13, %f11, %f12;
	mov.b32 	%r21, %f13;
	mov.u32 	%r22, 4;
	shfl.sync.down.b32 	%r23|%p6, %r21, %r22, %r14, %r16;
	mov.b32 	%f14, %r23;
	add.f32 	%f15, %f13, %f14;
	mov.b32 	%r24, %f15;
	shfl.sync.down.b32 	%r25|%p7, %r24, %r13, %r14, %r16;
	mov.b32 	%f16, %r25;
	add.f32 	%f17, %f15, %f16;
	mov.b32 	%r26, %f17;
	mov.u32 	%r27, 1;
	shfl.sync.down.b32 	%r28|%p8, %r26, %r27, %r14, %r16;
	mov.b32 	%f18, %r28;
	add.f32 	%f4, %f17, %f18;
	and.b32  	%r29, %r32, 31;
	setp.ne.s32 	%p9, %r29, 0;
	@%p9 bra 	$L__BB0_7;

	mov.u32 	%r30, s_mean;
	atom.shared.add.f32 	%f19, [%r30], %f4;

$L__BB0_7:
	bar.sync 	0;
	@%p2 bra 	$L__BB0_10;

	ld.shared.f32 	%f20, [s_mean];
	mov.u32 	%r5, %ntid.x;
	cvta.to.global.u64 	%rd3, %rd5;
	cvta.to.global.u64 	%rd4, %rd6;
	cvt.rn.f32.s32 	%f21, %r8;
	div.rn.f32 	%f22, %f20, %f21;
	add.f32 	%f23, %f22, %f6;
	rsqrt.approx.f32 	%f5, %f23;

$L__BB0_9:
	cvt.s64.s32 	%rd12, %r32;
	add.s64 	%rd13, %rd12, %rd2;
	shl.b64 	%rd14, %rd13, 1;
	add.s64 	%rd15, %rd1, %rd14;
	ld.global.nc.u16 	%rs2, [%rd15];
	cvt.f32.f16 %f24, %rs2;
	mul.wide.s32 	%rd16, %r32, 2;
	add.s64 	%rd17, %rd4, %rd16;
	ld.global.nc.u16 	%rs3, [%rd17];
	cvt.f32.f16 %f25, %rs3;
	mul.f32 	%f27, %f5, %f24;
	mul.f32 	%f26, %f27, %f25;
	cvt.rn.f16.f32 %rs4, %f26;
	add.s64 	%rd18, %rd3, %rd14;
	st.global.u16 	[%rd18], %rs4;
	add.s32 	%r32, %r32, %r5;
	setp.lt.s32 	%p11, %r32, %r8;
	@%p11 bra 	$L__BB0_9;

$L__BB0_10:
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
    fn test_rms_norm_execution() {
        let gpu = GpuContext::init(0).expect("Init GPU");
        let module = PtxModule::load_from_ptx(&gpu, RMS_NORM_FP16_PTX).expect("Load PTX");
        let func = module.get_function("rms_norm_fp16").expect("Get func");

        let hidden_dim = 128usize;
        let mut epsilon = 1e-5f32;
        let mut hidden_dim_u32 = hidden_dim as u32;

        let input_host: Vec<half::f16> = (0..hidden_dim).map(|_| half::f16::from_f32(1.0)).collect();
        let weight_host: Vec<half::f16> = (0..hidden_dim).map(|_| half::f16::from_f32(1.0)).collect();

        let d_in = DeviceBuffer::from_slice(&gpu, &input_host).unwrap();
        let d_weight = DeviceBuffer::from_slice(&gpu, &weight_host).unwrap();
        let mut d_out = DeviceBuffer::<half::f16>::zeroed(&gpu, hidden_dim).unwrap();

        let stream = gpu.compute_stream();
        
        let mut d_out_ptr = d_out.inner_mut().cu_deviceptr();
        let mut d_in_ptr = d_in.inner().cu_deviceptr();
        let mut d_weight_ptr = d_weight.inner().cu_deviceptr();

        let mut args: [*mut std::ffi::c_void; 5] = [
            &mut d_out_ptr as *mut _ as *mut std::ffi::c_void,
            &mut d_in_ptr as *mut _ as *mut std::ffi::c_void,
            &mut d_weight_ptr as *mut _ as *mut std::ffi::c_void,
            &mut epsilon as *mut _ as *mut std::ffi::c_void,
            &mut hidden_dim_u32 as *mut _ as *mut std::ffi::c_void,
        ];

        unsafe {
            launch_kernel_on_stream(
                &func,
                (1, 1, 1),
                (128, 1, 1),
                4, // shared memory for s_mean
                stream,
                &mut args,
            ).expect("Kernel launch failed");
        }

        stream.synchronize().expect("Sync stream");

        let result = d_out.inner().to_host_vec(stream).expect("Copy back");
        assert_eq!(result.len(), hidden_dim);
        
        // For input of all 1.0s and weight of 1.0s, RMS is 1.0, so output should be 1.0 * (1 / 1) * 1.0 = ~1.0
        let val = result[0].to_f32();
        assert!((val - 1.0).abs() < 1e-2, "Expected ~1.0, got {}", val);
    }
}
