use anyhow::Result;
use oxide_infer_core::{GpuContext, SafeTensorsLoader};

#[test]
fn test_load_real_lfm3_weights() -> Result<()> {
    let gpu = GpuContext::init(0).expect("Failed to initialize GPU 0");
    let model_path = "/mnt/hdd/models/LFM3-VL-2B/model.safetensors";

    println!("Opening model weights at {}", model_path);
    let loader = SafeTensorsLoader::open(model_path)?;

    // 1. Verify embedding norm weights [2048]
    println!("Loading embedding_norm.weight...");
    let d_norm = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embedding_norm.weight")?;
    assert_eq!(d_norm.len(), 2048);
    println!("Successfully loaded embedding_norm.weight into GPU VRAM! ({} elements)", d_norm.len());

    // 2. Verify layer 0 FFN gate weight w1 [7680, 2048]
    println!("Loading layer 0 FFN w1.weight (gate)...");
    let d_w1 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.layers.0.feed_forward.w1.weight")?;
    assert_eq!(d_w1.len(), 7680 * 2048);
    println!("Successfully loaded layer 0 FFN w1 into GPU VRAM! ({} elements = {:.2} MB)", 
             d_w1.len(), (d_w1.len() * 2) as f64 / 1_048_576.0);

    // 3. Verify layer 0 Causal Conv weights [2048, 1, 7]
    println!("Loading layer 0 conv.conv.weight...");
    let d_conv = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.layers.0.conv.conv.weight")?;
    assert_eq!(d_conv.len(), 2048 * 7);
    println!("Successfully loaded layer 0 conv filter into GPU VRAM! ({} elements)", d_conv.len());

    Ok(())
}
