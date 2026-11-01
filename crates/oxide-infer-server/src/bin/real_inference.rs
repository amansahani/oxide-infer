use std::sync::Arc;
use std::time::Instant;
use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_server::tokenizer::OxideTokenizer;
use oxide_infer_models::{EmbeddingTable, FullLfm3DecoderLayer, TiedLmHead};
use oxide_infer_kernels::norm::RMS_NORM_FP16_PTX;
use oxide_infer_kernels::PtxModule;
use oxide_infer_scheduler::sample_greedy;
use cuda_core::launch_kernel_on_stream;

fn main() -> anyhow::Result<()> {
    println!("================================================================================");
    println!("       OXIDE-INFER: COMPLETE 24-LAYER HYBRID MODEL INFERENCE ON HARDWARE         ");
    println!("       Model: /mnt/hdd/models/LFM3-VL-2B (RTX 3060 12GB - cuda-oxide)           ");
    println!("================================================================================\n");

    let gpu = Arc::new(GpuContext::init(0)?);
    let stream = gpu.compute_stream();

    // 1. Initialize HuggingFace Tokenizer with Chat Template Prompt
    println!("[1/5] Initializing Tokenizer...");
    let tokenizer = OxideTokenizer::from_file("/mnt/hdd/models/LFM3-VL-2B/tokenizer.json")?;
    let test_prompt = "<|startoftext|><|im_start|>user\nWhat is Rust in one sentence?<|im_end|>\n<|im_start|>assistant\n<think>\n";
    let token_ids = tokenizer.encode(test_prompt)?;
    println!("      Prompt Tokens ({} tokens): {:?}\n", token_ids.len(), token_ids);

    // 2. Load Safetensors
    println!("[2/5] Loading full Safetensors weights into VRAM...");
    let t_load = Instant::now();
    let loader = SafeTensorsLoader::open("/mnt/hdd/models/LFM3-VL-2B/model.safetensors")?;

    let embed_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embed_tokens.weight")?;
    let host_bf16 = embed_bf16.inner().to_host_vec(stream)?;
    let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let embed_table = EmbeddingTable::new(&gpu, &host_f16, 128000, 2048)?;
    let mut lm_head = TiedLmHead::new(&gpu, 128000, 2048)?;

    // Embedding Norm (RMSNorm before LM head)
    let norm_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embedding_norm.weight")?;
    let norm_f16: Vec<half::f16> = norm_bf16.inner().to_host_vec(stream)?
        .into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let embedding_norm = DeviceBuffer::from_slice(&gpu, &norm_f16)?;
    let norm_ptx = PtxModule::load_from_ptx(&gpu, RMS_NORM_FP16_PTX)?;

    let layer_types = [
        "conv", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
        "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention",
        "conv", "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv",
        "full_attention", "conv"
    ];

    println!("      Loading all 24 hybrid layers (16 Conv1D + 8 GQA Attention)...");
    let mut layers = Vec::with_capacity(24);
    for idx in 0..24 {
        let is_attn = layer_types[idx] != "conv";
        let is_sliding = layer_types[idx] == "sliding_attention";
        let layer = FullLfm3DecoderLayer::load_from_safetensors(&gpu, &loader, idx, is_attn, is_sliding, 2048, 7680)?;
        layers.push(layer);
    }
    println!("      All 24 layers loaded in {:.2?} s\n", t_load.elapsed().as_secs_f64());

    // 3. Scratch buffers
    let mut d_hidden = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_norm = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_layer = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_ffn = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;

    let mut q_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 16 * 128)?;
    let mut k_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?;
    let mut v_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?;
    let mut gate_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 16)?;

    let mut d_normed_final = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut d_logits = DeviceBuffer::<f32>::zeroed(&gpu, 128000)?;

    // 4. Prefill prompt
    println!("[3/5] Running Prefill across {} prompt tokens through all 24 layers...", token_ids.len());
    let t_prefill = Instant::now();

    for (pos_id, &tok) in token_ids.iter().enumerate() {
        let d_tok = DeviceBuffer::from_slice(&gpu, &[tok])?;
        embed_table.lookup(&gpu, &d_tok, 1, &mut d_hidden)?;

        for layer in &mut layers {
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
    println!("      Prefill complete in {:.2?} ms\n", t_prefill.elapsed().as_secs_f64() * 1000.0);

    // 5. Autoregressive Generation
    let max_new_tokens = 30;
    println!("[4/5] Generating {} tokens with full 24-layer forward steps...", max_new_tokens);
    let mut current_token = *token_ids.last().unwrap();
    let mut generated_tokens = Vec::new();

    let rms_func = norm_ptx.get_function("rms_norm_fp16")?;
    let mut epsilon: f32 = 1e-5;
    let mut h_size: u32 = 2048;

    // Pre-allocate single token input device buffer to completely eliminate CUDA driver allocations in the generation loop
    let mut d_tok = DeviceBuffer::<u32>::zeroed(&gpu, 1)?;

    let t_gen = Instant::now();
    for step_idx in 0..max_new_tokens {
        let pos_id = token_ids.len() + step_idx;
        // Embed current token: reuse existing pre-allocated device buffer asynchronously
        d_tok.copy_from_slice(&gpu, &[current_token])?;
        embed_table.lookup(&gpu, &d_tok, 1, &mut d_hidden)?;

        // Forward through all 24 layers
        for layer in &mut layers {
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

        // Apply final embedding_norm
        let mut d_out_final = d_normed_final.inner_mut().cu_deviceptr();
        let mut d_hid = d_hidden.inner().cu_deviceptr();
        let mut d_emb_w = embedding_norm.inner().cu_deviceptr();

        let mut args_final: [*mut std::ffi::c_void; 5] = [
            &mut d_out_final as *mut _ as *mut std::ffi::c_void,
            &mut d_hid as *mut _ as *mut std::ffi::c_void,
            &mut d_emb_w as *mut _ as *mut std::ffi::c_void,
            &mut epsilon as *mut _ as *mut std::ffi::c_void,
            &mut h_size as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&rms_func, (1, 1, 1), (128, 1, 1), 4, stream, &mut args_final)?;
        }

        // LM Head Logits calculation (128,000 vocab) + GPU-side Argmax Reduction
        lm_head.compute_logits(&gpu, &d_normed_final, &embed_table.table, &mut d_logits)?;
        let best_token = lm_head.sample_argmax(&gpu, &d_logits)?;

        generated_tokens.push(best_token);
        current_token = best_token;

        let decoded = tokenizer.decode(&[best_token]).unwrap_or_else(|_| " ".to_string());
        print!("{}", decoded);
        use std::io::Write;
        std::io::stdout().flush().unwrap();

        // Stop on EOS or IM_END
        if best_token == 124900 || best_token == 124896 {
            break;
        }
    }
    println!("\n");
    let gen_elapsed = t_gen.elapsed();
    let tps = (generated_tokens.len() as f64) / gen_elapsed.as_secs_f64();
    println!("      Generated {} tokens in {:.2?} ({:.1} tok/s)\n",
        generated_tokens.len(), gen_elapsed, tps);

    // 6. Display Full Decoded Text
    println!("[5/5] Final Real Inference Output:");
    let full_text = tokenizer.decode(&generated_tokens)?;
    println!("--------------------------------------------------------------------------------");
    println!("GENERATED TEXT:\n{}", full_text);
    println!("--------------------------------------------------------------------------------");

    Ok(())
}
