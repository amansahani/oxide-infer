use anyhow::Result;
use oxide_infer_core::{DeviceBuffer, GpuContext, SafeTensorsLoader};
use oxide_infer_kernels::PtxModule;
use oxide_infer_kernels::norm::RMS_NORM_FP16_PTX;
use oxide_infer_kernels::gemm::FUSED_GEMV_FP16_FP32ACCUM_PTX;
use oxide_infer_kernels::activations::SWIGLU_FP16_PTX;
use cuda_core::launch_kernel_on_stream;

/// SwiGLU Gated Feed-Forward Network Layer.
/// Computes: `out = (w1(x) * silu) * w3(x) -> w2(...)`
pub struct FfnLayer {
    pub w1_gate: DeviceBuffer<half::f16>,
    pub w2_down: DeviceBuffer<half::f16>,
    pub w3_up: DeviceBuffer<half::f16>,
    pub ffn_norm: DeviceBuffer<half::f16>,
    pub activated_buf: DeviceBuffer<half::f16>,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    rms_module: PtxModule,
    pub gemv_module: PtxModule,
    swiglu_module: PtxModule,
}

impl FfnLayer {
    pub fn load_from_safetensors(
        gpu: &GpuContext,
        loader: &SafeTensorsLoader,
        layer_idx: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{}.feed_forward", layer_idx);
        let norm_prefix = format!("model.language_model.layers.{}.ffn_norm.weight", layer_idx);

        let w1_bf16 = loader.load_tensor::<half::bf16>(gpu, &format!("{}.w1.weight", prefix))?;
        let w2_bf16 = loader.load_tensor::<half::bf16>(gpu, &format!("{}.w2.weight", prefix))?;
        let w3_bf16 = loader.load_tensor::<half::bf16>(gpu, &format!("{}.w3.weight", prefix))?;
        let norm_bf16 = loader.load_tensor::<half::bf16>(gpu, &norm_prefix)?;

        let stream = gpu.compute_stream();
        let convert_bf16_to_f16 = |buf: &DeviceBuffer<half::bf16>| -> Result<DeviceBuffer<half::f16>> {
            let host_bf16 = buf.inner().to_host_vec(stream)?;
            let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
            DeviceBuffer::from_slice(gpu, &host_f16)
        };

        let w1_gate = convert_bf16_to_f16(&w1_bf16)?;
        let w2_down = convert_bf16_to_f16(&w2_bf16)?;
        let w3_up = convert_bf16_to_f16(&w3_bf16)?;
        let ffn_norm = convert_bf16_to_f16(&norm_bf16)?;
        let activated_buf = DeviceBuffer::<half::f16>::zeroed(gpu, intermediate_size)?;

        let rms_module = PtxModule::load_from_ptx(gpu, RMS_NORM_FP16_PTX)?;
        let gemv_module = PtxModule::load_from_ptx(gpu, FUSED_GEMV_FP16_FP32ACCUM_PTX)?;
        let swiglu_module = PtxModule::load_from_ptx(gpu, SWIGLU_FP16_PTX)?;

        Ok(Self {
            w1_gate,
            w2_down,
            w3_up,
            ffn_norm,
            activated_buf,
            intermediate_size,
            hidden_size,
            rms_module,
            gemv_module,
            swiglu_module,
        })
    }

    /// Forward pass through the FFN block without internal norm (expects already normed input).
    pub fn forward(
        &mut self,
        gpu: &GpuContext,
        normed_x: &DeviceBuffer<half::f16>,
        out: &mut DeviceBuffer<half::f16>,
    ) -> Result<()> {
        let stream = gpu.compute_stream();
        let mut d_normed_ptr = normed_x.inner().cu_deviceptr();

        // 1. Single-Pass Fused Dual-GEMV SwiGLU:
        // Computes gate = w1(x) and up = w3(x) simultaneously with 128-bit vector loads,
        // evaluates activated = silu(gate) * up in registers, and writes directly to pre-allocated activated_buf!
        let fused_swiglu_func = self.gemv_module.get_function("fused_dual_gemv_swiglu_fp16")?;

        let mut k_dim_u32 = self.hidden_size as u32;
        let mut n_dim_u32 = self.intermediate_size as u32;

        let mut d_act_ptr = self.activated_buf.inner_mut().cu_deviceptr();
        let mut d_w1_ptr = self.w1_gate.inner().cu_deviceptr();
        let mut d_w3_ptr = self.w3_up.inner().cu_deviceptr();

        let mut args_fused: [*mut std::ffi::c_void; 6] = [
            &mut d_act_ptr as *mut _ as *mut std::ffi::c_void,
            &mut d_normed_ptr as *mut _ as *mut std::ffi::c_void,
            &mut d_w1_ptr as *mut _ as *mut std::ffi::c_void,
            &mut d_w3_ptr as *mut _ as *mut std::ffi::c_void,
            &mut k_dim_u32 as *mut _ as *mut std::ffi::c_void,
            &mut n_dim_u32 as *mut _ as *mut std::ffi::c_void,
        ];

        unsafe {
            launch_kernel_on_stream(&fused_swiglu_func, (self.intermediate_size as u32, 1, 1), (64, 1, 1), 0, stream, &mut args_fused)?;
        }

        // 2. Down projection: out = w2(activated) via 128-bit Vectorized GEMV
        let gemv_func = self.gemv_module.get_function("fused_gemv_fp16_vec8")?;
        let mut d_out_ptr = out.inner_mut().cu_deviceptr();
        let mut d_w2_ptr = self.w2_down.inner().cu_deviceptr();
        let mut k_down_u32 = self.intermediate_size as u32;
        let mut n_down_u32 = self.hidden_size as u32;

        let mut args_w2: [*mut std::ffi::c_void; 5] = [
            &mut d_out_ptr as *mut _ as *mut std::ffi::c_void,
            &mut d_act_ptr as *mut _ as *mut std::ffi::c_void,
            &mut d_w2_ptr as *mut _ as *mut std::ffi::c_void,
            &mut k_down_u32 as *mut _ as *mut std::ffi::c_void,
            &mut n_down_u32 as *mut _ as *mut std::ffi::c_void,
        ];

        unsafe {
            launch_kernel_on_stream(&gemv_func, (self.hidden_size as u32, 1, 1), (64, 1, 1), 0, stream, &mut args_w2)?;
        }

        Ok(())
    }
}
