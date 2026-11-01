use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_models::Lfm3AttentionLayer;

fn main() -> anyhow::Result<()> {
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();
    let loader = SafeTensorsLoader::open("/mnt/hdd/models/LFM3-VL-2B/model.safetensors")?;

    let attn = Lfm3AttentionLayer::load_from_safetensors(&gpu, &loader, 2, 2048, 16, 4, 128)?;

    // Let's test with a constant normed input: all 0.1
    let host_in: Vec<half::f16> = vec![half::f16::from_f32(0.1); 2048];
    let d_in = DeviceBuffer::from_slice(&gpu, &host_in)?;

    let mut q_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 16 * 128)?;
    let mut k_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?;
    let mut v_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?;
    let mut gate_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 16)?;

    attn.forward_projections(&gpu, &d_in, &mut q_out, &mut k_out, &mut v_out, &mut gate_out)?;
    stream.synchronize()?;

    println!("q_out first 5 (after q_norm): {:?}", &q_out.inner().to_host_vec(stream)?[..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());
    println!("k_out first 5 (after k_norm): {:?}", &k_out.inner().to_host_vec(stream)?[..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());
    println!("v_out first 5: {:?}", &v_out.inner().to_host_vec(stream)?[..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());
    println!("gate_out first 5: {:?}", &gate_out.inner().to_host_vec(stream)?[..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());

    Ok(())
}
