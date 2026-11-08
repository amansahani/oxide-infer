use anyhow::Result;
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext, SafeTensorsLoader};
use oxide_infer_kernels::vision::{VISION_BIAS_ADD_PTX, VISION_GELU_PTX, VISION_LAYER_NORM_PTX};
use oxide_infer_kernels::PtxModule;
use cuda_core::launch_kernel_on_stream;

fn main() -> Result<()> {
    println!("Testing Rust Vision MultiModalProjector on GPU vs PyTorch reference...");
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();

    // Load weights
    let loader = SafeTensorsLoader::open("/mnt/hdd/models/LFM3-VL-2B/model.safetensors")?;

    let convert_bf16_to_f16 = |name: &str| -> Result<DeviceBuffer<f16>> {
        let buf = loader.load_tensor::<half::bf16>(&gpu, name)?;
        let host_bf16 = buf.inner().to_host_vec(stream)?;
        let host_f16: Vec<f16> = host_bf16.into_iter().map(|b| f16::from_f32(b.to_f32())).collect();
        DeviceBuffer::from_slice(&gpu, &host_f16)
    };

    let ln_w = convert_bf16_to_f16("model.multi_modal_projector.layer_norm.weight")?;
    let ln_b = convert_bf16_to_f16("model.multi_modal_projector.layer_norm.bias")?;
    let l1_w = convert_bf16_to_f16("model.multi_modal_projector.linear_1.weight")?;
    let l1_b = convert_bf16_to_f16("model.multi_modal_projector.linear_1.bias")?;
    let l2_w = convert_bf16_to_f16("model.multi_modal_projector.linear_2.weight")?;
    let l2_b = convert_bf16_to_f16("model.multi_modal_projector.linear_2.bias")?;

    // Load PyTorch reference last_hidden_state (shape: 256, 1024)
    let lhs_bytes = std::fs::read("/tmp/vis_ref_last_hidden_state.npy")?;
    let header_len = 128usize;
    let lhs_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(
            lhs_bytes[header_len..].as_ptr() as *const f32,
            (lhs_bytes.len() - header_len) / 4,
        )
    };
    let lhs_f16: Vec<f16> = lhs_slice.iter().map(|&x| f16::from_f32(x)).collect();
    let d_lhs = DeviceBuffer::from_slice(&gpu, &lhs_f16)?;

    // Step 1: LayerNorm(1024) over 256 tokens
    let mut d_normed = DeviceBuffer::<f16>::zeroed(&gpu, 256 * 1024)?;
    let ln_mod = PtxModule::load_from_ptx(&gpu, VISION_LAYER_NORM_PTX)?;
    let ln_func = ln_mod.get_function("layer_norm_fp16")?;

    let mut d_normed_ptr = d_normed.inner_mut().cu_deviceptr();
    let mut d_lhs_ptr = d_lhs.inner().cu_deviceptr();
    let mut d_ln_w_ptr = ln_w.inner().cu_deviceptr();
    let mut d_ln_b_ptr = ln_b.inner().cu_deviceptr();
    let mut rows: i32 = 256;
    let mut cols: i32 = 1024;
    let mut eps: f32 = 1e-5;

    let mut args_ln: [*mut std::ffi::c_void; 7] = [
        &mut d_normed_ptr as *mut _ as *mut std::ffi::c_void,
        &mut d_lhs_ptr as *mut _ as *mut std::ffi::c_void,
        &mut d_ln_w_ptr as *mut _ as *mut std::ffi::c_void,
        &mut d_ln_b_ptr as *mut _ as *mut std::ffi::c_void,
        &mut rows as *mut _ as *mut std::ffi::c_void,
        &mut cols as *mut _ as *mut std::ffi::c_void,
        &mut eps as *mut _ as *mut std::ffi::c_void,
    ];

    unsafe {
        launch_kernel_on_stream(
            &ln_func,
            (rows as u32, 1, 1),
            (256, 1, 1),
            0,
            stream,
            &mut args_ln,
        )?;
    }

    // Step 2: Linear 1 GEMM [64, 4096] x [4096, 4096] -> [64, 4096]
    // We can evaluate row-by-row with fused_gemv or full GEMM. Let's do row-by-row GEMV first:
    let gemv_mod = PtxModule::load_from_ptx(&gpu, oxide_infer_kernels::gemm::FUSED_GEMV_FP16_FP32ACCUM_PTX)?;
    let gemv_func = gemv_mod.get_function("fused_gemv_fp16_fp32accum")?;

    let mut d_l1_out = DeviceBuffer::<f16>::zeroed(&gpu, 64 * 4096)?;
    let num_tokens: usize = 64;
    let m1: u32 = 4096;
    let k1: u32 = 4096;

    for tok in 0..num_tokens {
        let in_offset = tok * 4096;
        let out_offset = tok * 4096;
        let mut d_out_tok = d_l1_out.inner().cu_deviceptr() + (out_offset * 2) as u64;
        let mut d_in_tok = d_normed.inner().cu_deviceptr() + (in_offset * 2) as u64;
        let mut d_w1 = l1_w.inner().cu_deviceptr();
        let mut k1_arg = k1;
        let mut m1_arg = m1;

        let mut args_w1: [*mut std::ffi::c_void; 5] = [
            &mut d_out_tok as *mut _ as *mut std::ffi::c_void,
            &mut d_in_tok as *mut _ as *mut std::ffi::c_void,
            &mut d_w1 as *mut _ as *mut std::ffi::c_void,
            &mut k1_arg as *mut _ as *mut std::ffi::c_void,
            &mut m1_arg as *mut _ as *mut std::ffi::c_void,
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
    }

    // Step 3: Bias add + GELU on [64, 4096]
    let bias_mod = PtxModule::load_from_ptx(&gpu, VISION_BIAS_ADD_PTX)?;
    let bias_func = bias_mod.get_function("bias_add_fp16")?;

    let mut d_l1_ptr = d_l1_out.inner_mut().cu_deviceptr();
    let mut d_b1_ptr = l1_b.inner().cu_deviceptr();
    let mut num_rows_i32: i32 = 64;
    let mut cols_i32: i32 = 4096;

    let mut args_b1: [*mut std::ffi::c_void; 4] = [
        &mut d_l1_ptr as *mut _ as *mut std::ffi::c_void,
        &mut d_b1_ptr as *mut _ as *mut std::ffi::c_void,
        &mut num_rows_i32 as *mut _ as *mut std::ffi::c_void,
        &mut cols_i32 as *mut _ as *mut std::ffi::c_void,
    ];

    let block_dim: u32 = 256;
    let total_elems1 = (num_rows_i32 * cols_i32) as u32;
    let grid_dim1 = (total_elems1 + block_dim - 1) / block_dim;
    unsafe {
        launch_kernel_on_stream(
            &bias_func,
            (grid_dim1, 1, 1),
            (block_dim, 1, 1),
            0,
            stream,
            &mut args_b1,
        )?;
    }

    let gelu_mod = PtxModule::load_from_ptx(&gpu, VISION_GELU_PTX)?;
    let gelu_func = gelu_mod.get_function("gelu_fp16")?;

    let mut d_gelu_out = DeviceBuffer::<f16>::zeroed(&gpu, 64 * 4096)?;
    let mut d_gelu_out_ptr = d_gelu_out.inner_mut().cu_deviceptr();
    let mut total_elems_u32 = total_elems1;

    let mut args_gelu: [*mut std::ffi::c_void; 3] = [
        &mut d_gelu_out_ptr as *mut _ as *mut std::ffi::c_void,
        &mut d_l1_ptr as *mut _ as *mut std::ffi::c_void,
        &mut total_elems_u32 as *mut _ as *mut std::ffi::c_void,
    ];

    unsafe {
        launch_kernel_on_stream(
            &gelu_func,
            (grid_dim1, 1, 1),
            (block_dim, 1, 1),
            0,
            stream,
            &mut args_gelu,
        )?;
    }

    // Step 4: Linear 2 GEMM [64, 4096] x [4096, 2048] -> [64, 2048]
    let mut d_l2_out = DeviceBuffer::<f16>::zeroed(&gpu, 64 * 2048)?;
    let m2: u32 = 2048;
    let k2: u32 = 4096;

    for tok in 0..num_tokens {
        let in_offset = tok * 4096;
        let out_offset = tok * 2048;
        let mut d_out_tok = d_l2_out.inner().cu_deviceptr() + (out_offset * 2) as u64;
        let mut d_in_tok = d_gelu_out.inner().cu_deviceptr() + (in_offset * 2) as u64;
        let mut d_w2 = l2_w.inner().cu_deviceptr();
        let mut k2_arg = k2;
        let mut m2_arg = m2;

        let mut args_w2: [*mut std::ffi::c_void; 5] = [
            &mut d_out_tok as *mut _ as *mut std::ffi::c_void,
            &mut d_in_tok as *mut _ as *mut std::ffi::c_void,
            &mut d_w2 as *mut _ as *mut std::ffi::c_void,
            &mut k2_arg as *mut _ as *mut std::ffi::c_void,
            &mut m2_arg as *mut _ as *mut std::ffi::c_void,
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
    }

    // Step 5: Bias add on [64, 2048]
    let mut d_l2_ptr = d_l2_out.inner_mut().cu_deviceptr();
    let mut d_b2_ptr = l2_b.inner().cu_deviceptr();
    let mut cols_i32_2: i32 = 2048;

    let mut args_b2: [*mut std::ffi::c_void; 4] = [
        &mut d_l2_ptr as *mut _ as *mut std::ffi::c_void,
        &mut d_b2_ptr as *mut _ as *mut std::ffi::c_void,
        &mut num_rows_i32 as *mut _ as *mut std::ffi::c_void,
        &mut cols_i32_2 as *mut _ as *mut std::ffi::c_void,
    ];

    let total_elems2 = (num_rows_i32 * cols_i32_2) as u32;
    let grid_dim2 = (total_elems2 + block_dim - 1) / block_dim;
    unsafe {
        launch_kernel_on_stream(
            &bias_func,
            (grid_dim2, 1, 1),
            (block_dim, 1, 1),
            0,
            stream,
            &mut args_b2,
        )?;
    }

    stream.synchronize()?;

    // Read back and compare against PyTorch reference /tmp/vis_ref_projected.npy
    let out_host = d_l2_out.inner().to_host_vec(stream)?;
    let ref_proj_bytes = std::fs::read("/tmp/vis_ref_projected.npy")?;
    let ref_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(
            ref_proj_bytes[header_len..].as_ptr() as *const f32,
            (ref_proj_bytes.len() - header_len) / 4,
        )
    };

    let mut max_diff = 0.0f32;
    for (i, &val) in out_host.iter().enumerate() {
        let diff = (val.to_f32() - ref_slice[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("Projector output count: {} floats", out_host.len());
    println!("Max difference between Rust GPU projector and PyTorch reference: {:.6}", max_diff);
    assert!(max_diff < 0.005, "Projector diff exceeds tolerance: {}", max_diff);
    println!("✅ MultiModalProjector parity confirmed on GPU!");

    Ok(())
}
