use std::sync::Arc;
use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_kernels::PtxModule;
use oxide_infer_kernels::gemm::BATCHED_GEMV_FP16_PTX;
use cuda_core::launch_kernel_on_stream;

fn main() -> anyhow::Result<()> {
    let gpu = Arc::new(GpuContext::init(0)?);
    let stream = gpu.compute_stream();
    let module = PtxModule::load_from_ptx(&gpu, BATCHED_GEMV_FP16_PTX)?;
    let batched_func = module.get_function("batched_gemv_fp16_vec8")?;

    let loader = SafeTensorsLoader::open("/mnt/hdd/models/LFM3-VL-2B/model.safetensors")?;
    let convert_bf16_to_f16 = |buf: &DeviceBuffer<half::bf16>| -> anyhow::Result<DeviceBuffer<half::f16>> {
        let host_bf16 = buf.inner().to_host_vec(stream)?;
        let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
        DeviceBuffer::from_slice(&gpu, &host_f16)
    };

    println!("Loading layer 0 w2 down projection weight...");
    let w2_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.layers.0.feed_forward.w2.weight")?;
    let w2_down = convert_bf16_to_f16(&w2_bf16)?;

    let b: usize = 4;
    let k: usize = 7680;
    let m: usize = 2048;

    let x_host: Vec<half::f16> = (0..b*k).map(|i| half::f16::from_f32((i % 7) as f32 * 0.05)).collect();
    let d_x = DeviceBuffer::from_slice(&gpu, &x_host)?;
    let mut d_y = DeviceBuffer::<half::f16>::zeroed(&gpu, b * m)?;

    let mut d_y_ptr = d_y.inner_mut().cu_deviceptr();
    let mut d_x_ptr = d_x.inner().cu_deviceptr();
    let mut d_w_ptr = w2_down.inner().cu_deviceptr();
    let mut k_dim_u32 = k as u32;
    let mut m_dim_u32 = m as u32;
    let mut b_dim_u32 = b as u32;

    let mut args: [*mut std::ffi::c_void; 6] = [
        &mut d_y_ptr as *mut _ as *mut std::ffi::c_void,
        &mut d_x_ptr as *mut _ as *mut std::ffi::c_void,
        &mut d_w_ptr as *mut _ as *mut std::ffi::c_void,
        &mut k_dim_u32 as *mut _ as *mut std::ffi::c_void,
        &mut m_dim_u32 as *mut _ as *mut std::ffi::c_void,
        &mut b_dim_u32 as *mut _ as *mut std::ffi::c_void,
    ];

    println!("Launching batched_gemv_fp16_vec8 with B={}, K={}, M={}...", b, k, m);
    unsafe {
        launch_kernel_on_stream(
            &batched_func,
            (m as u32, b as u32, 1),
            (32, 1, 1),
            0,
            stream,
            &mut args,
        )?;
    }
    stream.synchronize()?;

    let y_res = d_y.inner().to_host_vec(stream)?;
    println!("Batched GEMV executed on real model weights! Output length: {} (4 x 2048)", y_res.len());
    for s in 0..b {
        let slice = &y_res[s*m..s*m+5];
        println!("  Batch {} output slice: {:?}", s, slice.iter().map(|v| v.to_f32()).collect::<Vec<_>>());
    }
    Ok(())
}
