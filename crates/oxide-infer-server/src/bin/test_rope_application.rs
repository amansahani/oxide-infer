use oxide_infer_core::{GpuContext, DeviceBuffer};
use oxide_infer_models::RotaryEmbeddingTable;

fn main() -> anyhow::Result<()> {
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();
    let rope = RotaryEmbeddingTable::new(&gpu, 1024, 128, 1000000.0, 0.5, 0)?;

    let qn_slice: Vec<half::f16> = vec![
        half::f16::from_f32(-0.34814453125),
        half::f16::from_f32(-0.5205078125),
        half::f16::from_f32(0.2227783203125),
        half::f16::from_f32(-0.1943359375),
        half::f16::from_f32(0.79443359375),
    ];
    let mut q_full = vec![half::f16::from_f32(0.0); 128];
    for i in 0..5 { q_full[i] = qn_slice[i]; }

    let mut d_q = DeviceBuffer::from_slice(&gpu, &q_full)?;
    rope.apply(&gpu, &mut d_q, 1, 1)?;
    stream.synchronize()?;

    let out = d_q.inner().to_host_vec(stream)?;
    println!("Rust applied RoPE at pos 1: first 5: {:?}", &out[..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());
    Ok(())
}
