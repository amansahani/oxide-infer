use anyhow::Result;
use oxide_infer_core::{DeviceBuffer, GpuContext, SafeTensorsLoader};
use oxide_infer_models::{EmbeddingTable, FullLfm3DecoderLayer};

#[test]
fn test_verify_layer0_and_layer1_parity() -> Result<()> {
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();
    let loader = SafeTensorsLoader::open("/mnt/hdd/models/LFM3-VL-2B/model.safetensors")?;

    let token_ids = vec![124894u32, 124899, 5922, 207, 2992, 355, 47300, 296, 734, 12683, 39, 124900, 207, 124899, 63514, 207, 124901, 207];

    let embed_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embed_tokens.weight")?;
    let host_bf16 = embed_bf16.inner().to_host_vec(stream)?;
    let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let embed_table = EmbeddingTable::new(&gpu, &host_f16, 128000, 2048)?;

    let mut layer0 = FullLfm3DecoderLayer::load_from_safetensors(&gpu, &loader, 0, false, false, 2048, 7680)?;
    let mut layer1 = FullLfm3DecoderLayer::load_from_safetensors(&gpu, &loader, 1, false, false, 2048, 7680)?;

    let mut d_hidden = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_norm = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_layer = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_ffn = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut q_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 16 * 128)?;
    let mut k_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?;
    let mut v_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?;
    let mut gate_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 16)?;

    for (pos_id, &tok) in token_ids.iter().enumerate() {
        let d_tok = DeviceBuffer::from_slice(&gpu, &[tok])?;
        embed_table.lookup(&gpu, &d_tok, 1, &mut d_hidden)?;

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
            pos_id,
        )?;

        layer1.forward(
            &gpu,
            &mut d_hidden,
            &mut scratch_norm,
            &mut scratch_layer,
            &mut scratch_ffn,
            &mut q_out,
            &mut k_out,
            &mut v_out,
            &mut gate_out,
            pos_id,
        )?;
    }
    stream.synchronize()?;

    let final_host = d_hidden.inner().to_host_vec(stream)?;
    let sum: f32 = final_host.iter().map(|x| x.to_f32()).sum();
    let mean = sum / final_host.len() as f32;
    let var: f32 = final_host.iter().map(|x| (x.to_f32() - mean).powi(2)).sum::<f32>() / final_host.len() as f32;
    let std = var.sqrt();

    println!("Oxide-Infer Layer 1 output: mean = {}, std = {}", mean, std);
    // Ground truth PyTorch: mean = -0.000509, std = 0.009177
    assert!((mean - (-0.000509)).abs() < 0.001);
    assert!((std - 0.009177).abs() < 0.002);
    Ok(())
}
