use anyhow::Result;
use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_models::CausalConvLayer;

#[test]
fn test_lfm3_conv_forward_pass_on_real_weights() -> Result<()> {
    let gpu = GpuContext::init(0).expect("Init GPU");
    let model_path = "/mnt/hdd/models/LFM3-VL-2B/model.safetensors";

    println!("Loading layer 0 conv weights from {}", model_path);
    let loader = SafeTensorsLoader::open(model_path)?;

    // Hidden size = 2048, K_len = 7
    let mut conv = CausalConvLayer::load_from_safetensors(&gpu, &loader, 0, 2048, 7)?;
    println!("Causal Conv layer 0 weights successfully loaded into GPU!");

    // Input token representation x
    let x_host: Vec<half::f16> = (0..2048).map(|_| half::f16::from_f32(0.5)).collect();
    let d_x = DeviceBuffer::from_slice(&gpu, &x_host)?;
    let mut d_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;

    println!("Running GPU Causal Conv1D forward pass...");
    conv.forward(&gpu, &d_x, &mut d_out)?;

    let out_host = d_out.inner().to_host_vec(gpu.compute_stream())?;
    assert_eq!(out_host.len(), 2048);

    println!("Conv1D Forward pass completed! First 5 output values: {:?}", 
        &out_host[..5].iter().map(|v| v.to_f32()).collect::<Vec<_>>());

    Ok(())
}
