use oxide_infer_core::GpuContext;
use oxide_infer_models::RotaryEmbeddingTable;

fn main() -> anyhow::Result<()> {
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();
    let rope = RotaryEmbeddingTable::new(&gpu, 1024, 128, 1000000.0, 0.5)?;

    let cos_host = rope.cos_table.inner().to_host_vec(stream)?;
    let sin_host = rope.sin_table.inner().to_host_vec(stream)?;

    // Pos 1:
    let p1_cos: Vec<f32> = cos_host[128..256].iter().map(|x| x.to_f32()).collect();
    let p1_sin: Vec<f32> = sin_host[128..256].iter().map(|x| x.to_f32()).collect();

    println!("Rust pos 1 cos first 10: {:?}", &p1_cos[..10]);
    println!("Rust pos 1 cos 60..70: {:?}", &p1_cos[60..70]);
    println!("Rust pos 1 sin first 10: {:?}", &p1_sin[..10]);
    println!("Rust pos 1 sin 60..70: {:?}", &p1_sin[60..70]);

    Ok(())
}
