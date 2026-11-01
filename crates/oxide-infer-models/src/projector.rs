use anyhow::Result;
use oxide_infer_core::{DeviceBuffer, GpuContext, SafeTensorsLoader};
use oxide_infer_kernels::PtxModule;
use oxide_infer_kernels::gemm::FUSED_GEMV_FP16_FP32ACCUM_PTX;
use cuda_core::launch_kernel_on_stream;

const GELU_BIAS_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry fused_bias_gelu_fp16(
    .param .u64 x,
    .param .u64 bias,
    .param .u32 num_elements
) {
    .reg .pred %p0;
    .reg .b32 %idx, %n, %b_idx, %dim_stride;
    .reg .b64 %x_ptr, %bias_ptr, %offset;
    .reg .b16 %h_val, %h_bias;
    .reg .f32 %f_val, %f_bias, %f_x, %f_cube, %f_inner, %f_tanh, %f_out;

    ld.param.u32 %n, [num_elements];
    mov.u32 %idx, %tid.x;
    mov.u32 %b_idx, %ctaid.x;
    mov.u32 %dim_stride, %ntid.x;
    mad.lo.u32 %idx, %b_idx, %dim_stride, %idx;

    setp.ge.u32 %p0, %idx, %n;
    @%p0 bra EXIT;

    ld.param.u64 %x_ptr, [x];
    ld.param.u64 %bias_ptr, [bias];

    cvt.u64.u32 %offset, %idx;
    shl.b64 %offset, %offset, 1;

    add.u64 %bias_ptr, %bias_ptr, %offset;
    ld.global.u16 %h_bias, [%bias_ptr];

    add.u64 %x_ptr, %x_ptr, %offset;
    ld.global.u16 %h_val, [%x_ptr];

    cvt.f32.f16 %f_val, %h_val;
    cvt.f32.f16 %f_bias, %h_bias;
    add.f32 %f_x, %f_val, %f_bias;

    // Fast GELU approximation: 0.5 * x * (1.0 + tanh(0.79788456 * (x + 0.044715 * x^3)))
    mul.f32 %f_cube, %f_x, %f_x;
    mul.f32 %f_cube, %f_cube, %f_x;
    fma.rn.f32 %f_inner, 0.044715, %f_cube, %f_x;
    mul.f32 %f_inner, %f_inner, 0.79788456;

    mul.f32 %f_inner, %f_inner, 2.88539008; // 2.0 / ln(2)
    ex2.approx.ftz.f32 %f_tanh, %f_inner;
    sub.f32 %f_out, %f_tanh, 1.0;
    add.f32 %f_tanh, %f_tanh, 1.0;
    rcp.approx.ftz.f32 %f_tanh, %f_tanh;
    mul.f32 %f_tanh, %f_out, %f_tanh;

    add.f32 %f_tanh, %f_tanh, 1.0;
    mul.f32 %f_tanh, %f_tanh, 0.5;
    mul.f32 %f_out, %f_x, %f_tanh;

    cvt.rn.f16.f32 %h_val, %f_out;
    st.global.u16 [%x_ptr], %h_val;

EXIT:
    ret;
}

.visible .entry add_bias_fp16(
    .param .u64 x,
    .param .u64 bias,
    .param .u32 num_elements
) {
    .reg .pred %p0;
    .reg .b32 %idx, %n, %b_idx, %dim_stride;
    .reg .b64 %x_ptr, %bias_ptr, %offset;
    .reg .b16 %h_val, %h_bias;
    .reg .f32 %f_val, %f_bias, %f_res;

    ld.param.u32 %n, [num_elements];
    mov.u32 %idx, %tid.x;
    mov.u32 %b_idx, %ctaid.x;
    mov.u32 %dim_stride, %ntid.x;
    mad.lo.u32 %idx, %b_idx, %dim_stride, %idx;

    setp.ge.u32 %p0, %idx, %n;
    @%p0 bra EXIT;

    ld.param.u64 %x_ptr, [x];
    ld.param.u64 %bias_ptr, [bias];

    cvt.u64.u32 %offset, %idx;
    shl.b64 %offset, %offset, 1;

    add.u64 %bias_ptr, %bias_ptr, %offset;
    ld.global.u16 %h_bias, [%bias_ptr];

    add.u64 %x_ptr, %x_ptr, %offset;
    ld.global.u16 %h_val, [%x_ptr];

    cvt.f32.f16 %f_val, %h_val;
    cvt.f32.f16 %f_bias, %h_bias;
    add.f32 %f_res, %f_val, %f_bias;

    cvt.rn.f16.f32 %h_val, %f_res;
    st.global.u16 [%x_ptr], %h_val;

EXIT:
    ret;
}
"#;

pub struct MultiModalProjector {
    pub linear_1_weight: DeviceBuffer<half::f16>,
    pub linear_1_bias: DeviceBuffer<half::f16>,
    pub linear_2_weight: DeviceBuffer<half::f16>,
    pub linear_2_bias: DeviceBuffer<half::f16>,
    pub intermediate_buf: DeviceBuffer<half::f16>,
    gemv_ptx: PtxModule,
    gelu_ptx: PtxModule,
}

impl MultiModalProjector {
    pub fn load_from_safetensors(gpu: &GpuContext, loader: &SafeTensorsLoader) -> Result<Self> {
        let stream = gpu.compute_stream();
        let convert_bf16_to_f16 = |buf: &DeviceBuffer<half::bf16>| -> Result<DeviceBuffer<half::f16>> {
            let host_bf16 = buf.inner().to_host_vec(stream)?;
            let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
            DeviceBuffer::from_slice(gpu, &host_f16)
        };

        let l1_w = loader.load_tensor::<half::bf16>(gpu, "model.multi_modal_projector.linear_1.weight")?;
        let l1_b = loader.load_tensor::<half::bf16>(gpu, "model.multi_modal_projector.linear_1.bias")?;
        let l2_w = loader.load_tensor::<half::bf16>(gpu, "model.multi_modal_projector.linear_2.weight")?;
        let l2_b = loader.load_tensor::<half::bf16>(gpu, "model.multi_modal_projector.linear_2.bias")?;

        let linear_1_weight = convert_bf16_to_f16(&l1_w)?;
        let linear_1_bias = convert_bf16_to_f16(&l1_b)?;
        let linear_2_weight = convert_bf16_to_f16(&l2_w)?;
        let linear_2_bias = convert_bf16_to_f16(&l2_b)?;

        let intermediate_buf = DeviceBuffer::zeroed(gpu, 4096)?;

        let gemv_ptx = PtxModule::load_from_ptx(gpu, FUSED_GEMV_FP16_FP32ACCUM_PTX)?;
        let gelu_ptx = PtxModule::load_from_ptx(gpu, GELU_BIAS_PTX)?;

        Ok(Self {
            linear_1_weight,
            linear_1_bias,
            linear_2_weight,
            linear_2_bias,
            intermediate_buf,
            gemv_ptx,
            gelu_ptx,
        })
    }

    /// Project downsampled 4096-dim image feature into 2048-dim language embedding space.
    pub fn project(
        &mut self,
        gpu: &GpuContext,
        input_merged_features: &DeviceBuffer<half::f16>,
        out_token_embedding: &mut DeviceBuffer<half::f16>,
    ) -> Result<()> {
        let stream = gpu.compute_stream();

        // 1. linear_1 GEMV: [4096, 4096] x [4096] -> [4096]
        let mut m1: u32 = 4096;
        let mut k1: u32 = 4096;
        let gemv_func = self.gemv_ptx.get_function("fused_gemv_fp16_fp32accum")?;

        let mut d_l1_out = self.intermediate_buf.inner_mut().cu_deviceptr();
        let mut d_input = input_merged_features.inner().cu_deviceptr();
        let mut d_w1 = self.linear_1_weight.inner().cu_deviceptr();

        let mut args_w1: [*mut std::ffi::c_void; 5] = [
            &mut d_l1_out as *mut _ as *mut std::ffi::c_void,
            &mut d_input as *mut _ as *mut std::ffi::c_void,
            &mut d_w1 as *mut _ as *mut std::ffi::c_void,
            &mut k1 as *mut _ as *mut std::ffi::c_void,
            &mut m1 as *mut _ as *mut std::ffi::c_void,
        ];

        unsafe {
            launch_kernel_on_stream(
                &gemv_func,
                (m1, 1, 1),
                (64, 1, 1),
                4,
                stream,
                &mut args_w1,
            )?;
        }

        // 2. fused bias add + GELU on intermediate_buf
        let gelu_func = self.gelu_ptx.get_function("fused_bias_gelu_fp16")?;
        let mut d_b1 = self.linear_1_bias.inner().cu_deviceptr();
        let mut args_gelu: [*mut std::ffi::c_void; 3] = [
            &mut d_l1_out as *mut _ as *mut std::ffi::c_void,
            &mut d_b1 as *mut _ as *mut std::ffi::c_void,
            &mut m1 as *mut _ as *mut std::ffi::c_void,
        ];

        let block_dim: u32 = 256;
        let grid_dim: u32 = (m1 + block_dim - 1) / block_dim;
        unsafe {
            launch_kernel_on_stream(
                &gelu_func,
                (grid_dim, 1, 1),
                (block_dim, 1, 1),
                0,
                stream,
                &mut args_gelu,
            )?;
        }

        // 3. linear_2 GEMV: [2048, 4096] x [4096] -> [2048]
        let mut m2: u32 = 2048;
        let mut k2: u32 = 4096;
        let mut d_out = out_token_embedding.inner_mut().cu_deviceptr();
        let mut d_w2 = self.linear_2_weight.inner().cu_deviceptr();

        let mut args_w2: [*mut std::ffi::c_void; 5] = [
            &mut d_out as *mut _ as *mut std::ffi::c_void,
            &mut d_l1_out as *mut _ as *mut std::ffi::c_void,
            &mut d_w2 as *mut _ as *mut std::ffi::c_void,
            &mut k2 as *mut _ as *mut std::ffi::c_void,
            &mut m2 as *mut _ as *mut std::ffi::c_void,
        ];

        unsafe {
            launch_kernel_on_stream(
                &gemv_func,
                (m2, 1, 1),
                (64, 1, 1),
                4,
                stream,
                &mut args_w2,
            )?;
        }

        // 4. bias add on out_token_embedding
        let bias_func = self.gelu_ptx.get_function("add_bias_fp16")?;
        let mut d_b2 = self.linear_2_bias.inner().cu_deviceptr();
        let mut args_bias: [*mut std::ffi::c_void; 3] = [
            &mut d_out as *mut _ as *mut std::ffi::c_void,
            &mut d_b2 as *mut _ as *mut std::ffi::c_void,
            &mut m2 as *mut _ as *mut std::ffi::c_void,
        ];
        let grid_dim2: u32 = (m2 + block_dim - 1) / block_dim;

        unsafe {
            launch_kernel_on_stream(
                &bias_func,
                (grid_dim2, 1, 1),
                (block_dim, 1, 1),
                0,
                stream,
                &mut args_bias,
            )?;
        }

        Ok(())
    }
}
