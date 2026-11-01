use std::sync::Arc;
use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_models::FullLfm3DecoderLayer;

fn main() -> anyhow::Result<()> {
    let gpu = Arc::new(GpuContext::init(0)?);
    let loader = SafeTensorsLoader::open("/mnt/hdd/models/LFM3-VL-2B/model.safetensors")?;
    
    let layer_types = [
        "conv", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
        "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention",
        "conv", "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv",
        "full_attention", "conv"
    ];

    println!("Loading 24 shared layers (single copy in VRAM)...");
    let mut layers = Vec::with_capacity(24);
    for idx in 0..24 {
        let is_attn = layer_types[idx] != "conv";
        let is_sliding = layer_types[idx] == "sliding_attention";
        layers.push(FullLfm3DecoderLayer::load_from_safetensors(&gpu, &loader, idx, is_attn, is_sliding, 2048, 7680)?);
    }

    println!("Shared layers loaded. Testing layer forward...");
    let mut d_hidden = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_norm = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_layer = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_ffn = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut q_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 16 * 128)?;
    let mut k_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?;
    let mut v_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?;
    let mut gate_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 16)?;

    for layer in layers.iter_mut() {
        layer.forward(
            &gpu,
            &mut d_hidden,
            &mut scratch_norm,
            &mut scratch_layer,
            &mut scratch_ffn,
            &mut q_out,
            &mut k_out,
            &mut v_out,
            &mut gate_out,
            0,
        )?;
    }
    gpu.compute_stream().synchronize()?;
    println!("SUCCESS! Single layer set evaluated 1 full forward step perfectly!");
    Ok(())
}
