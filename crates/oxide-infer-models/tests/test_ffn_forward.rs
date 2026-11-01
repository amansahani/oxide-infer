use anyhow::Result;
use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_models::FfnLayer;

#[test]
fn test_lfm3_ffn_forward_pass_on_real_weights() -> Result<()> {
    let gpu = GpuContext::init(0).expect("Init GPU");
    let model_path = "/mnt/hdd/models/LFM3-VL-2B/model.safetensors";

    println!("Loading layer 0 real weights from {}", model_path);
    let loader = SafeTensorsLoader::open(model_path)?;

    // Hidden size = 2048, Intermediate size = 7680
    let mut ffn = FfnLayer::load_from_safetensors(&gpu, &loader, 0, 2048, 7680)?;
    println!("FFN layer 0 weights successfully mapped and converted to GPU!");

    // Create synthetic input hidden vector x = 1.0
    let x_host: Vec<half::f16> = (0..2048).map(|_| half::f16::from_f32(0.1)).collect();
    let d_x = DeviceBuffer::from_slice(&gpu, &x_host)?;
    let mut d_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;

    println!("Running GPU FFN forward pass (RMSNorm -> Gate/Up GEMV -> SwiGLU -> Down GEMV)...");
    ffn.forward(&gpu, &d_x, &mut d_out)?;

    let out_host = d_out.inner().to_host_vec(gpu.compute_stream())?;
    assert_eq!(out_host.len(), 2048);

    println!("FFN Forward pass completed successfully! First 5 output values: {:?}", 
        &out_host[..5].iter().map(|v| v.to_f32()).collect::<Vec<_>>());

    Ok(())
}
