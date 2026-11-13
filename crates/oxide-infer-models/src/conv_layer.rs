use anyhow::Result;
use oxide_infer_kernels::quant_gemv::{QuantGemv, supported as quant_gemv_supported};
use oxide_infer_core::{DeviceBuffer, GpuContext, SafeTensorsLoader};
use oxide_infer_kernels::PtxModule;
use oxide_infer_kernels::conv::CAUSAL_CONV1D_FP16_PTX;
use oxide_infer_kernels::gemm::FUSED_GEMV_FP16_FP32ACCUM_PTX;
use cuda_core::launch_kernel_on_stream;

// Fused Causal Conv1D + B*x gating + C*conv_out gating
// Computes B[i]*x[i], evaluates 1D convolution with state_cache, shifts state, and multiplies by C[i]
// completely in registers without intermediate DRAM buffers (bx_buffer / conv_out_buffer).
pub const FUSED_CONV_GATING_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry fused_causal_conv1d_gating_fp16(
    .param .u64 y_out,
    .param .u64 state_cache,
    .param .u64 bcx_in,
    .param .u64 conv_weight,
    .param .u32 hidden_size,
    .param .u32 k_len
) {
    .reg .pred %p0, %p_loop;
    .reg .b32 %idx, %hsize, %klen, %b_idx, %stride, %c_offset, %x_offset;
    .reg .b32 %k, %k_minus_1, %row_offset_32;
    .reg .b64 %y_ptr, %cache_ptr, %bcx_ptr, %w_ptr;
    .reg .b64 %c_ptr, %x_ptr, %b_ptr, %offset, %c_state_ptr, %c_w_ptr;
    .reg .b16 %hb, %hx, %hc, %hin, %hw, %hs;
    .reg .f32 %fb, %fx, %fin, %fc, %fw, %fs, %fsum;

    ld.param.u32 %hsize, [hidden_size];
    ld.param.u32 %klen, [k_len];
    mov.u32 %idx, %tid.x;
    mov.u32 %b_idx, %ctaid.x;
    mov.u32 %stride, %ntid.x;
    mad.lo.u32 %idx, %b_idx, %stride, %idx;

    setp.ge.u32 %p0, %idx, %hsize;
    @%p0 bra EXIT;

    ld.param.u64 %bcx_ptr, [bcx_in];
    ld.param.u64 %cache_ptr, [state_cache];
    ld.param.u64 %w_ptr, [conv_weight];
    ld.param.u64 %y_ptr, [y_out];

    // 1. Read B[idx] (at offset idx)
    cvt.u64.u32 %offset, %idx;
    shl.b64 %offset, %offset, 1;
    add.u64 %b_ptr, %bcx_ptr, %offset;
    ld.global.u16 %hb, [%b_ptr];

    // Read x[idx] (at offset 2 * hidden_size + idx)
    shl.b32 %x_offset, %hsize, 1;
    add.u32 %x_offset, %x_offset, %idx;
    cvt.u64.u32 %offset, %x_offset;
    shl.b64 %offset, %offset, 1;
    add.u64 %x_ptr, %bcx_ptr, %offset;
    ld.global.u16 %hx, [%x_ptr];

    // Compute in_val = B * x
    cvt.f32.f16 %fb, %hb;
    cvt.f32.f16 %fx, %hx;
    mul.f32 %fin, %fb, %fx;
    cvt.rn.f16.f32 %hin, %fin;

    // 2. Convolution over state_cache and weights
    // Cache layout: [hidden_size, k_len - 1]
    // Weight layout: [hidden_size, k_len]
    sub.u32 %k_minus_1, %klen, 1;
    
    // Base pointers for this row
    // cache_row = cache_ptr + idx * (k_len - 1) * 2
    mul.lo.u32 %row_offset_32, %idx, %k_minus_1;
    cvt.u64.u32 %offset, %row_offset_32;
    shl.b64 %offset, %offset, 1;
    add.u64 %c_state_ptr, %cache_ptr, %offset;

    // weight_row = w_ptr + idx * k_len * 2
    mul.lo.u32 %row_offset_32, %idx, %klen;
    cvt.u64.u32 %offset, %row_offset_32;
    shl.b64 %offset, %offset, 1;
    add.u64 %c_w_ptr, %w_ptr, %offset;

    mov.f32 %fsum, 0.0;
    mov.u32 %k, 0;

LOOP_K:
    setp.ge.u32 %p_loop, %k, %k_minus_1;
    @%p_loop bra AFTER_LOOP;

    // Read state_cache[idx, k]
    cvt.u64.u32 %offset, %k;
    shl.b64 %offset, %offset, 1;
    add.u64 %b_ptr, %c_state_ptr, %offset;
    ld.global.u16 %hs, [%b_ptr];
    cvt.f32.f16 %fs, %hs;

    // Read weight[idx, k]
    add.u64 %x_ptr, %c_w_ptr, %offset;
    ld.global.u16 %hw, [%x_ptr];
    cvt.f32.f16 %fw, %hw;

    fma.rn.f32 %fsum, %fs, %fw, %fsum;

    add.u32 %k, %k, 1;
    bra LOOP_K;

AFTER_LOOP:
    // Add current input term: in_val * weight[idx, k_len - 1]
    cvt.u64.u32 %offset, %k_minus_1;
    shl.b64 %offset, %offset, 1;
    add.u64 %x_ptr, %c_w_ptr, %offset;
    ld.global.u16 %hw, [%x_ptr];
    cvt.f32.f16 %fw, %hw;
    fma.rn.f32 %fsum, %fin, %fw, %fsum;

    // Shift state_cache: state[k] = state[k+1] for k in 0..k_len-3; state[k_len-2] = in_val
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

    // 4. Multiply by C[idx] (at offset hidden_size + idx)
    add.u32 %c_offset, %hsize, %idx;
    cvt.u64.u32 %offset, %c_offset;
    shl.b64 %offset, %offset, 1;
    add.u64 %c_ptr, %bcx_ptr, %offset;
    ld.global.u16 %hc, [%c_ptr];
    cvt.f32.f16 %fc, %hc;

    mul.f32 %fsum, %fsum, %fc;
    cvt.rn.f16.f32 %hc, %fsum;

    // Store directly to y_out[idx]
    cvt.u64.u32 %offset, %idx;
    shl.b64 %offset, %offset, 1;
    add.u64 %c_ptr, %y_ptr, %offset;
    st.global.u16 [%c_ptr], %hc;

EXIT:
    ret;
}
"#;

pub struct CausalConvLayer {
    pub in_proj: DeviceBuffer<half::f16>,
    pub conv_weight: DeviceBuffer<half::f16>,
    pub out_proj: DeviceBuffer<half::f16>,
    /// Packed ggml bytes (additive)
    pub in_packed: Option<DeviceBuffer<u8>>,
    pub out_packed: Option<DeviceBuffer<u8>>,
    pub quant_type: Option<oxide_infer_core::quant::QuantType>,
    quant_gemv: Option<QuantGemv>,
    pub state_cache: DeviceBuffer<half::f16>,
    pub bcx_buffer: DeviceBuffer<half::f16>,
    pub bx_buffer: DeviceBuffer<half::f16>,
    pub conv_out_buffer: DeviceBuffer<half::f16>,
    pub y_buffer: DeviceBuffer<half::f16>,
    pub hidden_size: usize,
    pub k_len: usize,
    gemv_module: PtxModule,
    fused_conv_module: PtxModule,
}

impl CausalConvLayer {
    pub fn load_from_safetensors(
        gpu: &GpuContext,
        loader: &SafeTensorsLoader,
        layer_idx: usize,
        hidden_size: usize,
        k_len: usize,
    ) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{}.conv", layer_idx);
        let stream = gpu.compute_stream();

        let convert_bf16_to_f16 = |buf: &DeviceBuffer<half::bf16>| -> Result<DeviceBuffer<half::f16>> {
            let host_bf16 = buf.inner().to_host_vec(stream)?;
            let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
            DeviceBuffer::from_slice(gpu, &host_f16)
        };

        let in_proj_bf16 = loader.load_tensor::<half::bf16>(gpu, &format!("{}.in_proj.weight", prefix))?;
        let conv_bf16 = loader.load_tensor::<half::bf16>(gpu, &format!("{}.conv.weight", prefix))?;
        let out_proj_bf16 = loader.load_tensor::<half::bf16>(gpu, &format!("{}.out_proj.weight", prefix))?;

        let in_proj = convert_bf16_to_f16(&in_proj_bf16)?;
        let conv_weight = convert_bf16_to_f16(&conv_bf16)?;
        let out_proj = convert_bf16_to_f16(&out_proj_bf16)?;

        let quant_type = loader.quant_type();
        let (in_packed, out_packed, quant_gemv) = match quant_type {
            Some(qt) if quant_gemv_supported(qt) => (
                Some(loader.load_tensor_packed_gpu(gpu, &format!("{}.in_proj.weight", prefix))?),
                Some(loader.load_tensor_packed_gpu(gpu, &format!("{}.out_proj.weight", prefix))?),
                Some(QuantGemv::new(gpu)?),
            ),
            _ => (None, None, None),
        };

        let state_cache = DeviceBuffer::zeroed(gpu, hidden_size * k_len)?;
        let bcx_buffer = DeviceBuffer::zeroed(gpu, 3 * hidden_size)?;
        let bx_buffer = DeviceBuffer::zeroed(gpu, hidden_size)?;
        let conv_out_buffer = DeviceBuffer::zeroed(gpu, hidden_size)?;
        let y_buffer = DeviceBuffer::zeroed(gpu, hidden_size)?;

        let gemv_module = PtxModule::load_from_ptx(gpu, FUSED_GEMV_FP16_FP32ACCUM_PTX)?;
        let fused_conv_module = PtxModule::load_from_ptx(gpu, FUSED_CONV_GATING_PTX)?;

        Ok(Self {
            in_proj,
            conv_weight,
            out_proj,
            in_packed,
            out_packed,
            quant_type,
            quant_gemv,
            state_cache,
            bcx_buffer,
            bx_buffer,
            conv_out_buffer,
            y_buffer,
            hidden_size,
            k_len,
            gemv_module,
            fused_conv_module,
        })
    }

    /// Forward pass through the exact LFM3 ShortConv block:
    /// 1. BCx = in_proj(x) -> [B, C, x]
    /// 2. Fused Causal Conv1D + B*x gating + C*conv_out gating -> y
    /// 3. out = out_proj(y)
    pub fn forward(
        &mut self,
        gpu: &GpuContext,
        x: &DeviceBuffer<half::f16>,
        out: &mut DeviceBuffer<half::f16>,
    ) -> Result<()> {
        let stream = gpu.compute_stream();
        let gemv_func = self.gemv_module.get_function("fused_gemv_fp16_vec8")?;
        let fused_conv_func = self.fused_conv_module.get_function("fused_causal_conv1d_gating_fp16")?;

        let mut h_dim = self.hidden_size as u32;
        let mut in_proj_rows = (3 * self.hidden_size) as u32;

        let mut d_bcx = self.bcx_buffer.inner_mut().cu_deviceptr();
        let mut d_x = x.inner().cu_deviceptr();
        let mut d_in_w = self.in_proj.inner().cu_deviceptr();

        let mut args_in: [*mut std::ffi::c_void; 5] = [
            &mut d_bcx as *mut _ as *mut std::ffi::c_void,
            &mut d_x as *mut _ as *mut std::ffi::c_void,
            &mut d_in_w as *mut _ as *mut std::ffi::c_void,
            &mut h_dim as *mut _ as *mut std::ffi::c_void,
            &mut in_proj_rows as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&gemv_func, (in_proj_rows, 1, 1), (64, 1, 1), 0, stream, &mut args_in)?;
        }

        let block_dim: u32 = 128;
        let grid_dim: u32 = (h_dim + block_dim - 1) / block_dim;
        let mut d_y = self.y_buffer.inner_mut().cu_deviceptr();
        let mut d_cache = self.state_cache.inner_mut().cu_deviceptr();
        let mut d_cw = self.conv_weight.inner().cu_deviceptr();
        let mut k_len_u32 = self.k_len as u32;

        let mut args_fused: [*mut std::ffi::c_void; 6] = [
            &mut d_y as *mut _ as *mut std::ffi::c_void,
            &mut d_cache as *mut _ as *mut std::ffi::c_void,
            &mut d_bcx as *mut _ as *mut std::ffi::c_void,
            &mut d_cw as *mut _ as *mut std::ffi::c_void,
            &mut h_dim as *mut _ as *mut std::ffi::c_void,
            &mut k_len_u32 as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&fused_conv_func, (grid_dim, 1, 1), (block_dim, 1, 1), 0, stream, &mut args_fused)?;
        }

        let mut d_out = out.inner_mut().cu_deviceptr();
        let mut d_ow = self.out_proj.inner().cu_deviceptr();

        let mut args_out_proj: [*mut std::ffi::c_void; 5] = [
            &mut d_out as *mut _ as *mut std::ffi::c_void,
            &mut d_y as *mut _ as *mut std::ffi::c_void,
            &mut d_ow as *mut _ as *mut std::ffi::c_void,
            &mut h_dim as *mut _ as *mut std::ffi::c_void,
            &mut h_dim as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&gemv_func, (h_dim, 1, 1), (64, 1, 1), 0, stream, &mut args_out_proj)?;
        }

        Ok(())
    }

    pub fn forward_with_cache(
        &mut self,
        gpu: &GpuContext,
        x: &DeviceBuffer<half::f16>,
        out: &mut DeviceBuffer<half::f16>,
        state_cache: &mut DeviceBuffer<half::f16>,
    ) -> Result<()> {
        let stream = gpu.compute_stream();
        let gemv_func = self.gemv_module.get_function("fused_gemv_fp16_vec8")?;
        let fused_conv_func = self.fused_conv_module.get_function("fused_causal_conv1d_gating_fp16")?;

        let mut h_dim = self.hidden_size as u32;
        let mut in_proj_rows = (3 * self.hidden_size) as u32;

        // 1. in_proj GEMV: [6144, 2048] x [2048] -> [6144] (BCx) with 128-bit vectorization
        let mut d_bcx = self.bcx_buffer.inner_mut().cu_deviceptr();
        let mut d_x = x.inner().cu_deviceptr();
        let mut d_in_w = self.in_proj.inner().cu_deviceptr();

        let mut args_in: [*mut std::ffi::c_void; 5] = [
            &mut d_bcx as *mut _ as *mut std::ffi::c_void,
            &mut d_x as *mut _ as *mut std::ffi::c_void,
            &mut d_in_w as *mut _ as *mut std::ffi::c_void,
            &mut h_dim as *mut _ as *mut std::ffi::c_void,
            &mut in_proj_rows as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&gemv_func, (in_proj_rows, 1, 1), (64, 1, 1), 0, stream, &mut args_in)?;
        }

        // 2. Fused Causal Conv1D + in/out gating directly into y_buffer (eliminating bx & conv_out DRAM writes)
        let block_dim: u32 = 128;
        let grid_dim: u32 = (h_dim + block_dim - 1) / block_dim;
        let mut d_y = self.y_buffer.inner_mut().cu_deviceptr();
        let mut d_cache = state_cache.inner_mut().cu_deviceptr();
        let mut d_cw = self.conv_weight.inner().cu_deviceptr();
        let mut k_len_u32 = self.k_len as u32;

        let mut args_fused: [*mut std::ffi::c_void; 6] = [
            &mut d_y as *mut _ as *mut std::ffi::c_void,
            &mut d_cache as *mut _ as *mut std::ffi::c_void,
            &mut d_bcx as *mut _ as *mut std::ffi::c_void,
            &mut d_cw as *mut _ as *mut std::ffi::c_void,
            &mut h_dim as *mut _ as *mut std::ffi::c_void,
            &mut k_len_u32 as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&fused_conv_func, (grid_dim, 1, 1), (block_dim, 1, 1), 0, stream, &mut args_fused)?;
        }

        // 3. out_proj GEMV: [2048, 2048] x [2048] -> [2048] with 128-bit vectorization
        let mut d_out = out.inner_mut().cu_deviceptr();
        let mut d_ow = self.out_proj.inner().cu_deviceptr();

        let mut args_out_proj: [*mut std::ffi::c_void; 5] = [
            &mut d_out as *mut _ as *mut std::ffi::c_void,
            &mut d_y as *mut _ as *mut std::ffi::c_void,
            &mut d_ow as *mut _ as *mut std::ffi::c_void,
            &mut h_dim as *mut _ as *mut std::ffi::c_void,
            &mut h_dim as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&gemv_func, (h_dim, 1, 1), (64, 1, 1), 0, stream, &mut args_out_proj)?;
        }

        Ok(())
    }
}
