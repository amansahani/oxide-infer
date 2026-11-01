use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_models::{EmbeddingTable, CausalConvLayer};
use oxide_infer_kernels::PtxModule;
use oxide_infer_kernels::norm::RMS_NORM_FP16_PTX;
use cuda_core::launch_kernel_on_stream;

fn main() -> anyhow::Result<()> {
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();
    let loader = SafeTensorsLoader::open("/mnt/hdd/models/LFM3-VL-2B/model.safetensors")?;

    let embed_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embed_tokens.weight")?;
    let host_bf16 = embed_bf16.inner().to_host_vec(stream)?;
    let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let embed_table = EmbeddingTable::new(&gpu, &host_f16, 128000, 2048)?;

    let mut d_hidden = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let d_tok0 = DeviceBuffer::from_slice(&gpu, &[124894u32])?;
    embed_table.lookup(&gpu, &d_tok0, 1, &mut d_hidden)?;
    let h_embed = d_hidden.inner().to_host_vec(stream)?;
    println!("1. Rust embeds first 5: {:?}", &h_embed[..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());

    let op_norm_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.layers.0.operator_norm.weight")?;
    let host_op: Vec<half::f16> = op_norm_bf16.inner().to_host_vec(stream)?.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let op_norm_w = DeviceBuffer::from_slice(&gpu, &host_op)?;

    let norm_ptx = PtxModule::load_from_ptx(&gpu, RMS_NORM_FP16_PTX)?;
    let rms_func = norm_ptx.get_function("rms_norm_fp16")?;

    let mut d_normed = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut d_n = d_normed.inner_mut().cu_deviceptr();
    let mut d_h = d_hidden.inner().cu_deviceptr();
    let mut d_w = op_norm_w.inner().cu_deviceptr();
    let mut eps: f32 = 1e-5;
    let mut hsize: u32 = 2048;

    let mut args: [*mut std::ffi::c_void; 5] = [
        &mut d_n as *mut _ as *mut std::ffi::c_void,
        &mut d_h as *mut _ as *mut std::ffi::c_void,
        &mut d_w as *mut _ as *mut std::ffi::c_void,
        &mut eps as *mut _ as *mut std::ffi::c_void,
        &mut hsize as *mut _ as *mut std::ffi::c_void,
    ];
    unsafe {
        launch_kernel_on_stream(&rms_func, (1, 1, 1), (128, 1, 1), 4, stream, &mut args)?;
    }
    stream.synchronize()?;
    let h_normed = d_normed.inner().to_host_vec(stream)?;
    println!("2. Rust op_norm first 5: {:?}", &h_normed[..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());

    let mut conv_layer = CausalConvLayer::load_from_safetensors(&gpu, &loader, 0, 2048, 7)?;
    let mut conv_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    conv_layer.forward(&gpu, &d_normed, &mut conv_out)?;
    stream.synchronize()?;
    let h_conv_out = conv_out.inner().to_host_vec(stream)?;
    println!("6. Rust out_proj first 5: {:?}", &h_conv_out[..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());

    Ok(())
}
