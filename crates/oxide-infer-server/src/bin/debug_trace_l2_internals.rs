use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_models::FullLfm3DecoderLayer;

fn main() -> anyhow::Result<()> {
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();
    let loader = SafeTensorsLoader::open("/mnt/hdd/models/LFM3-VL-2B/model.safetensors")?;

    let mut layer0 = FullLfm3DecoderLayer::load_from_safetensors(&gpu, &loader, 0, false, false, 2048, 7680)?;
    let mut layer1 = FullLfm3DecoderLayer::load_from_safetensors(&gpu, &loader, 1, false, false, 2048, 7680)?;
    let mut layer2 = FullLfm3DecoderLayer::load_from_safetensors(&gpu, &loader, 2, true, false, 2048, 7680)?;

    let embed_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embed_tokens.weight")?;
    let host_bf16 = embed_bf16.inner().to_host_vec(stream)?;
    let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let embed_table = oxide_infer_models::EmbeddingTable::new(&gpu, &host_f16, 128000, 2048)?;

    let mut d_hidden = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_norm = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_layer = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_ffn = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;

    let mut q_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 16 * 128)?;
    let mut k_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?;
    let mut v_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?;
    let mut gate_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 16)?;

    for (pos_id, &tok) in [124894u32, 124899u32].iter().enumerate() {
        let d_tok = DeviceBuffer::from_slice(&gpu, &[tok])?;
        embed_table.lookup(&gpu, &d_tok, 1, &mut d_hidden)?;

        layer0.forward(&gpu, &mut d_hidden, &mut scratch_norm, &mut scratch_layer, &mut scratch_ffn,
            &mut q_out, &mut k_out, &mut v_out, &mut gate_out, pos_id)?;
        layer1.forward(&gpu, &mut d_hidden, &mut scratch_norm, &mut scratch_layer, &mut scratch_ffn,
            &mut q_out, &mut k_out, &mut v_out, &mut gate_out, pos_id)?;

        layer2.forward(&gpu, &mut d_hidden, &mut scratch_norm, &mut scratch_layer, &mut scratch_ffn,
            &mut q_out, &mut k_out, &mut v_out, &mut gate_out, pos_id)?;

        if pos_id == 1 {
            println!("Step 1 q_out head 0 first 5: {:?}", &q_out.inner().to_host_vec(stream)?[..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());
            println!("Step 1 k_out head 0 first 5: {:?}", &k_out.inner().to_host_vec(stream)?[..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());
            println!("Step 1 v_out head 0 first 5: {:?}", &v_out.inner().to_host_vec(stream)?[..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());
            println!("Step 1 gate_out first 5: {:?}", &gate_out.inner().to_host_vec(stream)?[..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());
        }
    }

    Ok(())
}
