use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_models::{EmbeddingTable, FullLfm3DecoderLayer, TiedLmHead};
use oxide_infer_kernels::PtxModule;
use oxide_infer_kernels::norm::RMS_NORM_FP16_PTX;
use oxide_infer_server::tokenizer::OxideTokenizer;
use cuda_core::launch_kernel_on_stream;

fn main() -> anyhow::Result<()> {
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();
    let loader = SafeTensorsLoader::open("/mnt/hdd/models/LFM3-VL-2B/model.safetensors")?;

    let tokenizer = OxideTokenizer::from_file("/mnt/hdd/models/LFM3-VL-2B/tokenizer.json")?;
    let test_prompt = "<|startoftext|><|im_start|>user\nWhat is Rust in one sentence?<|im_end|>\n<|im_start|>assistant\n<think>\n";
    let token_ids = tokenizer.encode(test_prompt)?;
    println!("Prompt tokens ({} tokens): {:?}", token_ids.len(), token_ids);

    let embed_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embed_tokens.weight")?;
    let host_bf16 = embed_bf16.inner().to_host_vec(stream)?;
    let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let embed_table = EmbeddingTable::new(&gpu, &host_f16, 128000, 2048)?;

    let layer_types = [
        "conv", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
        "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention",
        "conv", "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv",
        "full_attention", "conv"
    ];

    let mut layers = Vec::with_capacity(24);
    for idx in 0..24 {
        let is_attn = layer_types[idx] != "conv";
        let is_sliding = layer_types[idx] == "sliding_attention";
        let layer = FullLfm3DecoderLayer::load_from_safetensors(&gpu, &loader, idx, is_attn, is_sliding, 2048, 7680)?;
        layers.push(layer);
    }

    let mut d_hidden = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_norm = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_layer = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_ffn = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;

    let mut q_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 16 * 128)?;
    let mut k_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?;
    let mut v_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?;
    let mut gate_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 16)?;

    // Step through each prompt token sequentially
    for (pos_id, &tok) in token_ids.iter().enumerate() {
        let d_tok = DeviceBuffer::from_slice(&gpu, &[tok])?;
        embed_table.lookup(&gpu, &d_tok, 1, &mut d_hidden)?;

        for layer in layers.iter_mut() {
            layer.forward(
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
    }
    stream.synchronize()?;

    // Final norm & LM head
    let norm_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embedding_norm.weight")?;
    let norm_f16: Vec<half::f16> = norm_bf16.inner().to_host_vec(stream)?.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let embedding_norm = DeviceBuffer::from_slice(&gpu, &norm_f16)?;
    let norm_ptx = PtxModule::load_from_ptx(&gpu, RMS_NORM_FP16_PTX)?;
    let rms_func = norm_ptx.get_function("rms_norm_fp16")?;

    let mut d_normed_final = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut d_out_final = d_normed_final.inner_mut().cu_deviceptr();
    let mut d_hid = d_hidden.inner().cu_deviceptr();
    let mut d_emb_w = embedding_norm.inner().cu_deviceptr();
    let mut epsilon: f32 = 1e-5;
    let mut h_size: u32 = 2048;

    let mut args_final: [*mut std::ffi::c_void; 5] = [
        &mut d_out_final as *mut _ as *mut std::ffi::c_void,
        &mut d_hid as *mut _ as *mut std::ffi::c_void,
        &mut d_emb_w as *mut _ as *mut std::ffi::c_void,
        &mut epsilon as *mut _ as *mut std::ffi::c_void,
        &mut h_size as *mut _ as *mut std::ffi::c_void,
    ];
    unsafe { launch_kernel_on_stream(&rms_func, (1, 1, 1), (128, 1, 1), 4, stream, &mut args_final)?; }

    let lm_head = TiedLmHead::new(&gpu, 128000, 2048)?;
    let mut d_logits = DeviceBuffer::<f32>::zeroed(&gpu, 128000)?;
    lm_head.compute_logits(&gpu, &d_normed_final, &embed_table.table, &mut d_logits)?;
    stream.synchronize()?;

    let logits_vec = d_logits.inner().to_host_vec(stream)?;
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nPrompt completion top 5 token IDs: {:?}", indexed[..5].iter().map(|x| x.0).collect::<Vec<_>>());
    println!("Prompt completion top 5 values: {:?}", indexed[..5].iter().map(|x| x.1).collect::<Vec<_>>());
    for item in &indexed[..5] {
        let dec = tokenizer.decode(&[item.0 as u32]).unwrap_or_default();
        println!("{} (id {}): {:?}", item.1, item.0, dec);
    }

    Ok(())
}
