//! 1D Causal Convolution Kernel for Linear-Recurrent and Hybrid Layers.

pub const CAUSAL_CONV1D_FP16_PTX: &str = r#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry causal_conv1d_fp16(
	.param .u64 causal_conv1d_fp16_param_0,
	.param .u64 causal_conv1d_fp16_param_1,
	.param .u64 causal_conv1d_fp16_param_2,
	.param .u64 causal_conv1d_fp16_param_3,
	.param .u32 causal_conv1d_fp16_param_4,
	.param .u32 causal_conv1d_fp16_param_5
)
{
	.reg .pred 	%p<12>;
	.reg .b16 	%rs<20>;
	.reg .f32 	%f<34>;
	.reg .b32 	%r<44>;
	.reg .b64 	%rd<69>;

	ld.param.u64 	%rd27, [causal_conv1d_fp16_param_0];
	ld.param.u64 	%rd29, [causal_conv1d_fp16_param_1];
	ld.param.u64 	%rd28, [causal_conv1d_fp16_param_2];
	ld.param.u64 	%rd30, [causal_conv1d_fp16_param_3];
	ld.param.u32 	%r23, [causal_conv1d_fp16_param_4];
	ld.param.u32 	%r22, [causal_conv1d_fp16_param_5];
	cvta.to.global.u64 	%rd1, %rd29;
	cvta.to.global.u64 	%rd2, %rd30;
	mov.u32 	%r24, %ntid.x;
	mov.u32 	%r25, %ctaid.x;
	mov.u32 	%r26, %tid.x;
	mad.lo.s32 	%r1, %r25, %r24, %r26;
	setp.ge.s32 	%p1, %r1, %r23;
	@%p1 bra 	$L__BB0_17;

	add.s32 	%r2, %r22, -1;
	mul.lo.s32 	%r27, %r2, %r1;
	cvt.s64.s32 	%rd3, %r27;
	mul.lo.s32 	%r28, %r1, %r22;
	cvt.s64.s32 	%rd4, %r28;
	setp.gt.s32 	%p2, %r22, 1;
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	add.s32 	%r30, %r22, -2;
	and.b32  	%r39, %r2, 3;
	setp.lt.u32 	%p3, %r30, 3;
	mov.f32 	%f33, 0f00000000;
	mov.u32 	%r38, 0;
	@%p3 bra 	$L__BB0_6;

	sub.s32 	%r36, %r22, %r39;
	shl.b64 	%rd31, %rd3, 1;
	add.s64 	%rd32, %rd1, %rd31;
	add.s64 	%rd63, %rd32, 4;
	shl.b64 	%rd33, %rd4, 1;
	add.s64 	%rd34, %rd2, %rd33;
	add.s64 	%rd62, %rd34, 4;
	mov.f32 	%f33, 0f00000000;
	mov.u32 	%r38, 0;

$L__BB0_5:
	ld.global.u16 	%rs2, [%rd63+-4];
	cvt.f32.f16 %f13, %rs2;
	ld.global.nc.u16 	%rs3, [%rd62+-4];
	cvt.f32.f16 %f14, %rs3;
	fma.rn.f32 	%f21, %f13, %f14, %f33;
	ld.global.u16 	%rs4, [%rd63+-2];
	cvt.f32.f16 %f15, %rs4;
	ld.global.nc.u16 	%rs5, [%rd62+-2];
	cvt.f32.f16 %f16, %rs5;
	fma.rn.f32 	%f22, %f15, %f16, %f21;
	ld.global.u16 	%rs6, [%rd63];
	cvt.f32.f16 %f17, %rs6;
	ld.global.nc.u16 	%rs7, [%rd62];
	cvt.f32.f16 %f18, %rs7;
	fma.rn.f32 	%f23, %f17, %f18, %f22;
	ld.global.u16 	%rs8, [%rd63+2];
	cvt.f32.f16 %f19, %rs8;
	ld.global.nc.u16 	%rs9, [%rd62+2];
	cvt.f32.f16 %f20, %rs9;
	fma.rn.f32 	%f33, %f19, %f20, %f23;
	add.s32 	%r38, %r38, 4;
	add.s64 	%rd63, %rd63, 8;
	add.s64 	%rd62, %rd62, 8;
	add.s32 	%r36, %r36, -4;
	setp.ne.s32 	%p4, %r36, 1;
	@%p4 bra 	$L__BB0_5;

$L__BB0_6:
	setp.eq.s32 	%p5, %r39, 0;
	@%p5 bra 	$L__BB0_9;

	cvt.s64.s32 	%rd35, %r38;
	add.s64 	%rd36, %rd35, %rd4;
	shl.b64 	%rd37, %rd36, 1;
	add.s64 	%rd65, %rd2, %rd37;
	add.s64 	%rd38, %rd35, %rd3;
	shl.b64 	%rd39, %rd38, 1;
	add.s64 	%rd64, %rd1, %rd39;

$L__BB0_8:
	.pragma "nounroll";
	ld.global.u16 	%rs10, [%rd64];
	cvt.f32.f16 %f24, %rs10;
	ld.global.nc.u16 	%rs11, [%rd65];
	cvt.f32.f16 %f25, %rs11;
	fma.rn.f32 	%f33, %f24, %f25, %f33;
	add.s64 	%rd65, %rd65, 2;
	add.s64 	%rd64, %rd64, 2;
	add.s32 	%r39, %r39, -1;
	setp.ne.s32 	%p6, %r39, 0;
	@%p6 bra 	$L__BB0_8;
	bra.uni 	$L__BB0_9;

$L__BB0_2:
	mov.f32 	%f33, 0f00000000;

$L__BB0_9:
	add.s32 	%r12, %r22, -2;
	cvt.s64.s32 	%rd17, %r1;
	cvta.to.global.u64 	%rd40, %rd28;
	mul.wide.s32 	%rd41, %r1, 2;
	add.s64 	%rd42, %rd40, %rd41;
	ld.global.nc.u16 	%rs1, [%rd42];
	cvt.f32.f16 %f26, %rs1;
	cvt.s64.s32 	%rd43, %r2;
	add.s64 	%rd44, %rd4, %rd43;
	shl.b64 	%rd45, %rd44, 1;
	add.s64 	%rd46, %rd2, %rd45;
	ld.global.nc.u16 	%rs13, [%rd46];
	cvt.f32.f16 %f27, %rs13;
	fma.rn.f32 	%f8, %f26, %f27, %f33;
	setp.lt.s32 	%p7, %r22, 3;
	@%p7 bra 	$L__BB0_16;

	add.s32 	%r33, %r22, -3;
	and.b32  	%r43, %r12, 3;
	setp.lt.u32 	%p8, %r33, 3;
	mov.u32 	%r42, 0;
	@%p8 bra 	$L__BB0_13;

	sub.s32 	%r40, %r22, %r43;
	shl.b64 	%rd47, %rd3, 1;
	add.s64 	%rd48, %rd1, %rd47;
	add.s64 	%rd66, %rd48, 4;
	mov.u32 	%r42, 0;

$L__BB0_12:
	ld.global.u16 	%rs14, [%rd66+-2];
	st.global.u16 	[%rd66+-4], %rs14;
	ld.global.u16 	%rs15, [%rd66];
	st.global.u16 	[%rd66+-2], %rs15;
	ld.global.u16 	%rs16, [%rd66+2];
	st.global.u16 	[%rd66], %rs16;
	add.s32 	%r42, %r42, 4;
	ld.global.u16 	%rs17, [%rd66+4];
	st.global.u16 	[%rd66+2], %rs17;
	add.s64 	%rd66, %rd66, 8;
	add.s32 	%r40, %r40, -4;
	setp.ne.s32 	%p9, %r40, 2;
	@%p9 bra 	$L__BB0_12;

$L__BB0_13:
	setp.eq.s32 	%p10, %r43, 0;
	@%p10 bra 	$L__BB0_16;

	add.s32 	%r35, %r42, 1;
	cvt.s64.s32 	%rd49, %r35;
	add.s64 	%rd50, %rd3, %rd49;
	shl.b64 	%rd51, %rd50, 1;
	add.s64 	%rd68, %rd1, %rd51;
	cvt.u64.u32 	%rd52, %r42;
	add.s64 	%rd53, %rd3, %rd52;
	shl.b64 	%rd54, %rd53, 1;
	add.s64 	%rd67, %rd1, %rd54;

$L__BB0_15:
	.pragma "nounroll";
	ld.global.u16 	%rs18, [%rd68];
	st.global.u16 	[%rd67], %rs18;
	add.s64 	%rd68, %rd68, 2;
	add.s64 	%rd67, %rd67, 2;
	add.s32 	%r43, %r43, -1;
	setp.ne.s32 	%p11, %r43, 0;
	@%p11 bra 	$L__BB0_15;

$L__BB0_16:
	cvt.s64.s32 	%rd55, %r12;
	add.s64 	%rd56, %rd3, %rd55;
	shl.b64 	%rd57, %rd56, 1;
	add.s64 	%rd58, %rd1, %rd57;
	st.global.u16 	[%rd58], %rs1;
	cvt.rn.f16.f32 %rs19, %f8;
	cvta.to.global.u64 	%rd59, %rd27;
	shl.b64 	%rd60, %rd17, 1;
	add.s64 	%rd61, %rd59, %rd60;
	st.global.u16 	[%rd61], %rs19;

$L__BB0_17:
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
    fn test_causal_conv1d_execution() {
        let gpu = GpuContext::init(0).expect("Init GPU");
        let module = PtxModule::load_from_ptx(&gpu, CAUSAL_CONV1D_FP16_PTX).expect("Load PTX");
        let func = module.get_function("causal_conv1d_fp16").expect("Get func");

        let dim = 64usize;
        let k_len = 7usize;

        // Weights all 1.0; state cache all 0.0; in_val all 2.0
        // Expected output = 0.0 * 6 + 2.0 * 1.0 = 2.0
        let weight_host: Vec<half::f16> = (0..dim * k_len).map(|_| half::f16::from_f32(1.0)).collect();
        let in_host: Vec<half::f16> = (0..dim).map(|_| half::f16::from_f32(2.0)).collect();

        let d_weight = DeviceBuffer::from_slice(&gpu, &weight_host).unwrap();
        let d_in = DeviceBuffer::from_slice(&gpu, &in_host).unwrap();
        let mut d_cache = DeviceBuffer::<half::f16>::zeroed(&gpu, dim * (k_len - 1)).unwrap();
        let mut d_out = DeviceBuffer::<half::f16>::zeroed(&gpu, dim).unwrap();

        let stream = gpu.compute_stream();
        let mut d_out_ptr = d_out.inner_mut().cu_deviceptr();
        let mut d_cache_ptr = d_cache.inner_mut().cu_deviceptr();
        let mut d_in_ptr = d_in.inner().cu_deviceptr();
        let mut d_weight_ptr = d_weight.inner().cu_deviceptr();
        let mut dim_u32 = dim as u32;
        let mut k_len_u32 = k_len as u32;

        let mut args: [*mut std::ffi::c_void; 6] = [
            &mut d_out_ptr as *mut _ as *mut std::ffi::c_void,
            &mut d_cache_ptr as *mut _ as *mut std::ffi::c_void,
            &mut d_in_ptr as *mut _ as *mut std::ffi::c_void,
            &mut d_weight_ptr as *mut _ as *mut std::ffi::c_void,
            &mut dim_u32 as *mut _ as *mut std::ffi::c_void,
            &mut k_len_u32 as *mut _ as *mut std::ffi::c_void,
        ];

        unsafe {
            launch_kernel_on_stream(
                &func,
                (1, 1, 1),
                (64, 1, 1),
                0,
                stream,
                &mut args,
            ).expect("Launch Causal Conv1D failed");
        }

        stream.synchronize().expect("Sync stream");

        let result = d_out.inner().to_host_vec(stream).expect("Copy back");
        assert_eq!(result.len(), dim);
        for val in result {
            assert_eq!(val.to_f32(), 2.0);
        }
    }
}
