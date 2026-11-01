use oxide_infer_core::GpuContext;
use oxide_infer_kernels::PtxModule;
use oxide_infer_models::{RotaryEmbeddingTable, GqaAttentionEngine};

fn main() -> anyhow::Result<()> {
    let gpu = GpuContext::init(0)?;
    println!("Testing RotaryEmbeddingTable PTX load...");
    let _ = RotaryEmbeddingTable::new(&gpu, 1024, 128, 1000000.0, 0.5)?;
    println!("RotaryEmbeddingTable loaded successfully!");

    println!("Testing GqaAttentionEngine PTX load...");
    let _ = GqaAttentionEngine::new(&gpu)?;
    println!("GqaAttentionEngine loaded successfully!");

    Ok(())
}
