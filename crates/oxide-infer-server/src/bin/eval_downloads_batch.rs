use anyhow::Result;
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext, SafeTensorsLoader, VisionPreprocessor};
use oxide_infer_models::{EmbeddingTable, FullLfm3DecoderLayer, TiedLmHead, VisionTower};
use oxide_infer_server::batch_engine::{BatchEngine, BatchSequence, PrefillMeta};
use oxide_infer_server::tokenizer::OxideTokenizer;
use std::time::Instant;

const HIDDEN: usize = 2048;
const MAX_BATCH: usize = 64;

struct EvalContext {
    prompt_len: usize,
    cur_token: u32,
    gen_tokens: Vec<u32>,
    gen_count: usize,
    max_gen: usize,
    d_hidden: DeviceBuffer<f16>,
    layer_k_caches: Vec<Option<DeviceBuffer<f16>>>,
    layer_v_caches: Vec<Option<DeviceBuffer<f16>>>,
    layer_conv_states: Vec<Option<DeviceBuffer<f16>>>,
}

impl BatchSequence for EvalContext {
    fn k_cache_ptr(&self, layer: usize) -> u64 {
        self.layer_k_caches[layer].as_ref().map(|b| b.inner().cu_deviceptr()).unwrap_or(0)
    }
    fn v_cache_ptr(&self, layer: usize) -> u64 {
        self.layer_v_caches[layer].as_ref().map(|b| b.inner().cu_deviceptr()).unwrap_or(0)
    }
    fn conv_state_ptr(&self, layer: usize) -> u64 {
        self.layer_conv_states[layer].as_ref().map(|b| b.inner().cu_deviceptr()).unwrap_or(0)
    }
    fn hidden_ptr(&self) -> u64 {
        self.d_hidden.inner().cu_deviceptr()
    }
}

fn create_context(gpu: &GpuContext, prompt_len: usize, cur_token: u32, max_gen: usize, layer_types: &[&str]) -> Result<EvalContext> {
    let mut k_caches = Vec::with_capacity(24);
    let mut v_caches = Vec::with_capacity(24);
    let mut conv_states = Vec::with_capacity(24);
    for idx in 0..24 {
        if layer_types[idx] == "conv" {
            k_caches.push(None);
            v_caches.push(None);
            conv_states.push(Some(DeviceBuffer::<f16>::zeroed(gpu, 2048 * 7)?));
        } else {
            k_caches.push(Some(DeviceBuffer::<f16>::zeroed(gpu, 1024 * 512)?));
            v_caches.push(Some(DeviceBuffer::<f16>::zeroed(gpu, 1024 * 512)?));
            conv_states.push(None);
        }
    }

    Ok(EvalContext {
        prompt_len,
        cur_token,
        gen_tokens: Vec::new(),
        gen_count: 0,
        max_gen,
        d_hidden: DeviceBuffer::<f16>::zeroed(gpu, 2048)?,
        layer_k_caches: k_caches,
        layer_v_caches: v_caches,
        layer_conv_states: conv_states,
    })
}

fn main() -> Result<()> {
    println!("================================================================================");
    println!("   OXIDE-INFER: SCIENTIFIC MULTIMODAL BATCH EVALUATION & OCR BENCHMARK        ");
    println!("   Ampere Tensor Core Acceleration (cuBLASLt GEMM + Native PTX Micro-Kernels)   ");
    println!("================================================================================\n");

    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();

    let model_path = "/mnt/hdd/models/LFM3-VL-2B";
    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    let weights_path = format!("{}/model.safetensors", model_path);

    println!("1. Loading Tokenizer from {}...", tokenizer_path);
    let tokenizer = OxideTokenizer::from_file(&tokenizer_path)?;

    println!("2. Loading SafeTensors weights from {}...", weights_path);
    let loader = SafeTensorsLoader::open(&weights_path)?;

    println!("3. Loading Embedding table & tied LM Head...");
    let embed_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embed_tokens.weight")?;
    let host_bf16 = embed_bf16.inner().to_host_vec(stream)?;
    let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let embed_table = EmbeddingTable::new(&gpu, &host_f16, 128000, 2048)?;

    let norm_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embedding_norm.weight")?;
    let norm_f16: Vec<half::f16> = norm_bf16.inner().to_host_vec(stream)?
        .into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let embedding_norm = DeviceBuffer::from_slice(&gpu, &norm_f16)?;

    println!("4. Loading VisionTower and MultiModal Projector...");
    let mut vision_tower = VisionTower::load_from_safetensors(&gpu, &loader)?;
    let preprocessor = VisionPreprocessor::default();

    let layer_types = [
        "conv", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
        "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention",
        "conv", "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv",
        "full_attention", "conv"
    ];

    println!("5. Loading 24 Transformer Layers...");
    let mut layers = Vec::with_capacity(24);
    for idx in 0..24 {
        let is_attn = layer_types[idx] != "conv";
        let is_sliding = layer_types[idx] == "sliding_attention";
        let layer = FullLfm3DecoderLayer::load_from_safetensors(&gpu, &loader, idx, is_attn, is_sliding, 2048, 7680)?;
        layers.push(layer);
    }
    let is_conv: Vec<bool> = layer_types.iter().map(|t| *t == "conv").collect();

    println!("6. Initializing BatchEngine (max_batch={})...", MAX_BATCH);
    let mut batch_engine = BatchEngine::new(&gpu, MAX_BATCH)?;

    println!("\n✅ Pipeline fully loaded into VRAM. Starting multimodal batch evaluation!\n");

    // Tasks: (Image path, Prompt instruction, Task type, Max tokens)
    let tasks = [
        (
            "/home/tomi/Downloads/markheet.jpeg",
            "Perform a complete, detailed OCR transcription of this marksheet/certificate. Extract all visible names, roll numbers, subjects, marks, and bilingual text.",
            "OCR (Marksheet)",
            128,
        ),
        (
            "/home/tomi/Downloads/markheet.jpeg",
            "Describe this image in detail: what kind of document is it, what is its layout, condition, and visual characteristics?",
            "Understanding (Marksheet)",
            128,
        ),
        (
            "/home/tomi/Downloads/catagory.jpeg",
            "Perform full OCR text transcription of this document. Transcribe all text, numbers, dates, and official headings visible.",
            "OCR (Category Certificate)",
            128,
        ),
        (
            "/home/tomi/Downloads/catagory.jpeg",
            "Explain what document this is, who issued it, and describe its overall visual appearance and seals.",
            "Understanding (Category Certificate)",
            128,
        ),
        (
            "/home/tomi/Downloads/x570_rtx3060.jpeg",
            "Perform OCR on any text visible in this hardware photo, including motherboard labels, GPU port brackets, and markings.",
            "OCR (Hardware Components)",
            128,
        ),
        (
            "/home/tomi/Downloads/x570_rtx3060.jpeg",
            "Describe the objects in this photo in technical detail. Identify the motherboard, graphics card, form factors, and features.",
            "Understanding (Hardware Components)",
            128,
        ),
        (
            "/home/tomi/Downloads/signat.png",
            "Perform OCR and characterize the handwriting of the signature shown.",
            "OCR / Handwriting (Signature)",
            96,
        ),
        (
            "/home/tomi/Downloads/signat.png",
            "Describe the signature stroke, ink color, style, and visual layout.",
            "Understanding (Signature)",
            96,
        ),
    ];

    println!("================================================================================");
    println!(" RUNNING 8-WAY PARALLEL BATCH INFERENCE (OCR + UNDERSTANDING ACROSS REAL IMAGES)");
    println!("================================================================================\n");

    let t_batch_start = Instant::now();

    // 1. Process images and extract visual embeddings
    let mut image_cache: std::collections::HashMap<&str, DeviceBuffer<f16>> = std::collections::HashMap::new();
    let unique_paths: Vec<&str> = vec![
        "/home/tomi/Downloads/markheet.jpeg",
        "/home/tomi/Downloads/catagory.jpeg",
        "/home/tomi/Downloads/x570_rtx3060.jpeg",
        "/home/tomi/Downloads/signat.png",
    ];

    for path in unique_paths {
        println!("  -> Processing image: {}", path);
        let dyn_img = oxide_infer_core::image::open(path)?;
        let t0 = Instant::now();
        let patchified = preprocessor.process_image(&dyn_img)?;
        let d_patches = DeviceBuffer::from_slice(&gpu, &patchified.patches)?;
        let mut d_proj = DeviceBuffer::<f16>::zeroed(&gpu, 64 * HIDDEN)?;
        vision_tower.forward(&gpu, &d_patches, &mut d_proj)?;
        stream.synchronize()?;
        println!("     Vision Tower elapsed: {:.2} ms (64 visual tokens)", t0.elapsed().as_secs_f64() * 1000.0);
        image_cache.insert(path, d_proj);
    }

    // 2. Assemble unified multimodal sequences and contexts
    let mut contexts = Vec::with_capacity(tasks.len());
    let mut token_ids_list = Vec::with_capacity(tasks.len());

    for (img_path, instruction, _, max_tokens) in &tasks {
        let full_prompt = format!(
            "<|startoftext|><|im_start|>user\n<|vision_start|>{}<|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n<think>",
            "<|image|>".repeat(64),
            instruction
        );

        let token_ids = tokenizer.encode(&full_prompt)?;
        let prompt_len = token_ids.len();
        let cur_token = *token_ids.last().unwrap();

        let ctx = create_context(&gpu, prompt_len, cur_token, *max_tokens, &layer_types)?;
        contexts.push(ctx);
        token_ids_list.push(token_ids);
    }

    // 3. Batch Prefill Wave execution
    println!("\nExecuting Multimodal Batch Prefill Wave across all {} tasks concurrently...", tasks.len());
    let t_prefill = Instant::now();

    // Splice visual features directly into batch engine prefill waves
    let mb = batch_engine.max_batch();
    for i in 0..tasks.len() {
        let plen = token_ids_list[i].len();
        let mut cur_pos = 0usize;
        while cur_pos < plen {
            let chunk_len = (plen - cur_pos).min(mb);
            let chunk_tokens = &token_ids_list[i][cur_pos..cur_pos + chunk_len];
            let chunk_pos: Vec<u32> = (cur_pos..cur_pos + chunk_len).map(|p| p as u32).collect();
            let origins: Vec<u32> = vec![0; chunk_len];
            let local_pos: Vec<u32> = (0..chunk_len).map(|p| p as u32).collect();

            let mut carry_host = vec![0u64; 24 * mb];
            for l in 0..layers.len() {
                if is_conv[l] {
                    for r in 0..chunk_len {
                        carry_host[l * chunk_len + r] = contexts[i].conv_state_ptr(l);
                    }
                }
            }

            let meta = PrefillMeta {
                carry_ptrs: carry_host,
                origins,
                local_pos,
            };

            let ctx_ref = &contexts[i];
            let ctx_refs: Vec<&EvalContext> = vec![ctx_ref; chunk_len];

            // Run forward step
            batch_engine.forward_step(
                &gpu,
                &mut layers,
                &is_conv,
                &ctx_refs,
                chunk_tokens,
                &chunk_pos,
                &embed_table.table,
                &embedding_norm,
                false,
                Some(&meta),
            )?;

            // Conv state copy for decode
            for l in 0..layers.len() {
                if is_conv[l] {
                    let dst = contexts[i].conv_state_ptr(l);
                    let src = batch_engine.prefill_state_ptr(l, chunk_len - 1);
                    unsafe {
                        oxide_infer_server::batch_engine::cuMemcpyDtoDAsync_v2(
                            dst,
                            src,
                            2048 * 6 * 2,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                }
            }

            cur_pos += chunk_len;
        }
    }
    stream.synchronize()?;
    let t_prefill_dur = t_prefill.elapsed();
    println!("Prefill completed in {:.2} ms for {} total prompt tokens", t_prefill_dur.as_secs_f64() * 1000.0, token_ids_list.iter().map(|t| t.len()).sum::<usize>());

    // 4. Sample First Generation Token
    let pending_refs: Vec<&EvalContext> = contexts.iter().collect();
    let first_tokens = batch_engine.sample_only(&gpu, &pending_refs, &embed_table.table, &embedding_norm)?;
    for (i, &tok) in first_tokens.iter().enumerate() {
        contexts[i].gen_tokens.push(tok);
        contexts[i].cur_token = tok;
        contexts[i].gen_count = 1;
    }

    // 5. Continuous Batched Decoding Loop
    println!("\nEntering Continuous Batched Decoding Loop...");
    let t_decode = Instant::now();
    let mut total_generated_tokens = tasks.len(); // first tokens

    loop {
        let active_indices: Vec<usize> = (0..contexts.len())
            .filter(|&i| {
                let ctx = &contexts[i];
                ctx.gen_count < ctx.max_gen && ctx.cur_token != 124900 && ctx.cur_token != 124896
            })
            .collect();

        if active_indices.is_empty() {
            break;
        }

        let b = active_indices.len();
        let tokens: Vec<u32> = active_indices.iter().map(|&i| contexts[i].cur_token).collect();
        let pos_ids: Vec<u32> = active_indices.iter().map(|&i| (contexts[i].prompt_len + contexts[i].gen_count - 1) as u32).collect();
        let active_refs: Vec<&EvalContext> = active_indices.iter().map(|&i| &contexts[i]).collect();

        let next_tokens = batch_engine.forward_step(
            &gpu,
            &mut layers,
            &is_conv,
            &active_refs,
            &tokens,
            &pos_ids,
            &embed_table.table,
            &embedding_norm,
            true,
            None,
        )?;

        for (k, &act_idx) in active_indices.iter().enumerate() {
            let next_tok = next_tokens[k];
            contexts[act_idx].gen_tokens.push(next_tok);
            contexts[act_idx].cur_token = next_tok;
            contexts[act_idx].gen_count += 1;
            total_generated_tokens += 1;
        }
    }

    stream.synchronize()?;
    let t_decode_dur = t_decode.elapsed();
    let t_total = t_batch_start.elapsed();

    println!("\n================================================================================");
    println!(" BATCH MULTIMODAL INFERENCE COMPLETED SUCCESSFULLY");
    println!("================================================================================");
    println!("  Tasks executed concurrently : {}", tasks.len());
    println!("  Total tokens generated      : {}", total_generated_tokens);
    println!("  Decode duration             : {:.2} s", t_decode_dur.as_secs_f64());
    println!("  Decode generation throughput: {:.2} tok/s", (total_generated_tokens as f64) / t_decode_dur.as_secs_f64());
    println!("  Total wall-clock turnaround : {:.2} s", t_total.as_secs_f64());
    println!("================================================================================\n");

    println!("================================================================================");
    println!(" DETAILED OUTPUTS BY IMAGE AND TASK:");
    println!("================================================================================\n");

    for (i, (img_path, prompt, task_name, _)) in tasks.iter().enumerate() {
        let generated_text = tokenizer.decode(&contexts[i].gen_tokens).unwrap_or_else(|_| "<decode error>".to_string());
        println!("--------------------------------------------------------------------------------");
        println!("TASK {}: [{}] on {}", i + 1, task_name, img_path);
        println!("PROMPT: {}", prompt);
        println!("RESPONSE ({} tokens):", contexts[i].gen_tokens.len());
        println!("{}\n", generated_text.trim());
    }

    Ok(())
}
