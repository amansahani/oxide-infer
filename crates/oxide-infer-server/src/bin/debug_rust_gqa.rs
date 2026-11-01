use oxide_infer_core::GpuContext;
use oxide_infer_core::DeviceBuffer;
use oxide_infer_models::GqaAttentionEngine;

fn main() -> anyhow::Result<()> {
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();
    let gqa = GqaAttentionEngine::new(&gpu)?;

    let host_q: Vec<half::f16> = vec![half::f16::from_f32(1.0); 16 * 128];
    let host_k: Vec<half::f16> = vec![half::f16::from_f32(1.0); 2 * 4 * 128];
    let mut host_v: Vec<half::f16> = vec![half::f16::from_f32(1.0); 2 * 4 * 128];
    for i in 512..1024 {
        host_v[i] = half::f16::from_f32(2.0);
    }

    let q = DeviceBuffer::from_slice(&gpu, &host_q)?;
    let k_cache = DeviceBuffer::from_slice(&gpu, &host_k)?;
    let v_cache = DeviceBuffer::from_slice(&gpu, &host_v)?;
    let mut ctx = DeviceBuffer::<half::f16>::zeroed(&gpu, 16 * 128)?;

    gqa.compute_attention(&gpu, &mut ctx, &q, &k_cache, &v_cache, 2, 16, 4, 128)?;
    stream.synchronize()?;

    let h_ctx = ctx.inner().to_host_vec(stream)?;
    println!("GQA test output head 0 first 5: {:?}", &h_ctx[..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());
    Ok(())
}
