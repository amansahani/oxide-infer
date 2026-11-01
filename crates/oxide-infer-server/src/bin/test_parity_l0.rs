use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_models::{EmbeddingTable, FullLfm3DecoderLayer};

fn main() -> anyhow::Result<()> {
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();
    let loader = SafeTensorsLoader::open("/mnt/hdd/models/LFM3-VL-2B/model.safetensors")?;

    let embed_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embed_tokens.weight")?;
    let host_bf16 = embed_bf16.inner().to_host_vec(stream)?;
    let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let embed_table = EmbeddingTable::new(&gpu, &host_f16, 128000, 2048)?;

    let mut layer0 = FullLfm3DecoderLayer::load_from_safetensors(&gpu, &loader, 0, false, false, 2048, 7680)?;

    let mut d_hidden = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_norm = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_layer = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_ffn = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;

    let mut q_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 16 * 128)?;
    let mut k_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?;
    let mut v_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?;
    let mut gate_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 16)?;

    // Step 0: token 124894
    let d_tok0 = DeviceBuffer::from_slice(&gpu, &[124894u32])?;
    embed_table.lookup(&gpu, &d_tok0, 1, &mut d_hidden)?;
    layer0.forward(
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
    stream.synchronize()?;
    let out0 = d_hidden.inner().to_host_vec(stream)?;
    let f32_out0: Vec<f32> = out0.iter().map(|x| x.to_f32()).collect();
    let mean0: f32 = f32_out0.iter().sum::<f32>() / f32_out0.len() as f32;
    println!("Rust Layer 0 Step 0: mean={:+.6}, first 5={:?}", mean0, &f32_out0[..5]);

    // Step 1: token 124899
    let d_tok1 = DeviceBuffer::from_slice(&gpu, &[124899u32])?;
    embed_table.lookup(&gpu, &d_tok1, 1, &mut d_hidden)?;
    layer0.forward(
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
    stream.synchronize()?;
    let out1 = d_hidden.inner().to_host_vec(stream)?;
    let f32_out1: Vec<f32> = out1.iter().map(|x| x.to_f32()).collect();
    let mean1: f32 = f32_out1.iter().sum::<f32>() / f32_out1.len() as f32;
    println!("Rust Layer 0 Step 1: mean={:+.6}, first 5={:?}", mean1, &f32_out1[..5]);

    Ok(())
}
