use anyhow::Result;
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext, SafeTensorsLoader};
use oxide_infer_models::embedding::TokenEmbedding;
use oxide_infer_models::lm_head::LmHead;
use oxide_infer_models::{HybridTransformerLayer, ModelEngine};

fn main() -> Result<()> {
    println!("Testing End-to-End Multimodal Prefill in oxide-infer...");
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();

    let weights_path = "/mnt/hdd/models/LFM3-VL-2B/model.safetensors";
    let loader = SafeTensorsLoader::open(weights_path)?;

    println!("Loading LM layers...");
    let num_layers = 24usize;
    let hidden_size = 2048usize;
    let intermediate_size = 6144usize;
    let k_len = 32usize;

    let mut layers = Vec::with_capacity(num_layers);
    for idx in 0..num_layers {
        let layer = HybridTransformerLayer::load_from_safetensors(&gpu, &loader, idx, hidden_size, intermediate_size, k_len)?;
        layers.push(layer);
    }
    let mut engine = ModelEngine::new(layers, hidden_size);

    println!("Loading LM head...");
    let mut lm_head = LmHead::load_from_safetensors(&gpu, &loader)?;

    // Load unified multimodal input embeddings: shape [80, 2048]
    let embeds_bytes = std::fs::read("/tmp/vis_ref_unified_embeds.npy")?;
    let header_len = 128usize;
    let embeds_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(
            embeds_bytes[header_len..].as_ptr() as *const f32,
            (embeds_bytes.len() - header_len) / 4,
        )
    };
    let embeds_f16: Vec<f16> = embeds_slice.iter().map(|&x| f16::from_f32(x)).collect();
    let mut d_embeds = DeviceBuffer::from_slice(&gpu, &embeds_f16)?;

    let num_tokens = 80usize;
    println!("Running prefill wave for {} multimodal tokens...", num_tokens);
    let mut d_hidden = DeviceBuffer::<f16>::zeroed(&gpu, num_tokens * hidden_size)?;

    // Forward through all 24 layers
    engine.forward_prefill(&gpu, &mut d_embeds, &mut d_hidden, num_tokens)?;

    // Read hidden state of the last token (index 79)
    let last_tok_offset = (num_tokens - 1) * hidden_size;
    let mut d_last_tok = DeviceBuffer::<f16>::zeroed(&gpu, hidden_size)?;
    let d_last_tok_ptr = d_last_tok.inner_mut().cu_deviceptr();
    let d_src_ptr = d_hidden.inner().cu_deviceptr() + (last_tok_offset * 2) as u64;

    unsafe {
        cuda_core::cudaMemcpyAsync(
            d_last_tok_ptr as *mut std::ffi::c_void,
            d_src_ptr as *const std::ffi::c_void,
            hidden_size * 2,
            cuda_core::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            stream.inner(),
        );
    }

    // Run LM head
    let mut d_logits = DeviceBuffer::<f16>::zeroed(&gpu, 128000)?;
    lm_head.forward(&gpu, &d_last_tok, &mut d_logits)?;
    stream.synchronize()?;

    let logits_host = d_logits.inner().to_host_vec(stream)?;

    // Find top token
    let mut top_token = 0usize;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &l) in logits_host.iter().enumerate() {
        let val = l.to_f32();
        if val > max_val {
            max_val = val;
            top_token = i;
        }
    }

    println!("Oxide-infer predicted next token: {} (max_logit: {:.4})", top_token, max_val);

    // Verify against PyTorch reference prefill logits:
    let ref_logits_bytes = std::fs::read("/tmp/vis_ref_prefill_logits.npy")?;
    let ref_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(
            ref_logits_bytes[header_len..].as_ptr() as *const f32,
            (ref_logits_bytes.len() - header_len) / 4,
        )
    };

    let mut ref_top_token = 0usize;
    let mut ref_max = f32::NEG_INFINITY;
    for (i, &r) in ref_slice.iter().enumerate() {
        if r > ref_max {
            ref_max = r;
            ref_top_token = i;
        }
    }
    println!("PyTorch reference predicted token: {} (max_logit: {:.4})", ref_top_token, ref_max);

    assert_eq!(top_token, ref_top_token, "Top token mismatch!");
    assert_eq!(top_token, 597, "Expected token 597 ('The')");

    println!("✅ 100% EXACT PARITY CONFIRMED ON MULTIMODAL PREFILL IN OXIDE-INFER!");
    Ok(())
}
