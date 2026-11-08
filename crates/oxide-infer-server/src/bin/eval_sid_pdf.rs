use anyhow::Result;
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext, SafeTensorsLoader, VisionPreprocessor};
use oxide_infer_models::{EmbeddingTable, FullLfm3DecoderLayer, TiedLmHead, VisionTower};
use oxide_infer_server::batch_engine::{BatchEngine, BatchSequence, PrefillMeta};
use oxide_infer_server::tokenizer::OxideTokenizer;
use std::time::Instant;

const HIDDEN: usize = 2048;

struct PdfPageContext {
    page_num: usize,
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

impl BatchSequence for PdfPageContext {
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

fn create_context(gpu: &GpuContext, page_num: usize, prompt_len: usize, cur_token: u32, max_gen: usize, layer_types: &[&str]) -> Result<PdfPageContext> {
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

    Ok(PdfPageContext {
        page_num,
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
    println!(" OXIDE-INFER: NATIVE MULTIMODAL BENCHMARK ON SEBI SCHEME INFORMATION DOCUMENT  ");
    println!(" Target Document: SID_quant Business Cycle Fund_March 2025.pdf (Pages 1-3)      ");
    println!(" Pure Rust + Ampere PTX Kernels + cuBLASLt GEMM (Zero Python Serving Path)     ");
    println!("================================================================================\n");

    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();

    let model_path = "/mnt/hdd/models/LFM3-VL-2B";
    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    let weights_path = format!("{}/model.safetensors", model_path);

    println!("1. Loading Tokenizer & Weights...");
    let tokenizer = OxideTokenizer::from_file(&tokenizer_path)?;
    let loader = SafeTensorsLoader::open(&weights_path)?;

    println!("2. Loading Embedding Table & Tied LM Head...");
    let embed_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embed_tokens.weight")?;
    let host_bf16 = embed_bf16.inner().to_host_vec(stream)?;
    let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let embed_table = EmbeddingTable::new(&gpu, &host_f16, 128000, 2048)?;

    let norm_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embedding_norm.weight")?;
    let norm_f16: Vec<half::f16> = norm_bf16.inner().to_host_vec(stream)?
        .into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let embedding_norm = DeviceBuffer::from_slice(&gpu, &norm_f16)?;

    println!("3. Loading VisionTower...");
    let mut vision_tower = VisionTower::load_from_safetensors(&gpu, &loader)?;
    let preprocessor = VisionPreprocessor::default();

    let layer_types = [
        "conv", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
        "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention",
        "conv", "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv",
        "full_attention", "conv"
    ];

    println!("4. Loading 24 Transformer Layers...");
    let mut layers = Vec::with_capacity(24);
    for idx in 0..24 {
        let is_attn = layer_types[idx] != "conv";
        let is_sliding = layer_types[idx] == "sliding_attention";
        let layer = FullLfm3DecoderLayer::load_from_safetensors(&gpu, &loader, idx, is_attn, is_sliding, 2048, 7680)?;
        layers.push(layer);
    }
    let is_conv: Vec<bool> = layer_types.iter().map(|t| *t == "conv").collect();

    println!("5. Initializing BatchEngine (max_batch=128)...");
    let mut batch_engine = BatchEngine::new(&gpu, 128)?;

    let page_files = [
        (1, "/tmp/sid_page-01.png", "Transcribe the prominent title and header text visible on this page."),
        (2, "/tmp/sid_page-02.png", "Transcribe the table of contents and section headings on this page."),
        (3, "/tmp/sid_page-03.png", "Extract the highlights, summary, and fund details from this page."),
    ];

    let t_total_bench = Instant::now();

    // 1. Process Vision Tower for each PDF page
    println!("\n--------------------------------------------------------------------------------");
    println!(" PHASE 1: NATIVE PURE-RUST VISION TOWER EXECUTION");
    println!("--------------------------------------------------------------------------------");
    let mut page_vis_embeds = Vec::with_capacity(page_files.len());
    let mut total_vision_ms = 0.0;

    for (pnum, pfile, _) in &page_files {
        let dyn_img = oxide_infer_core::image::open(pfile)?;
        let t0 = Instant::now();
        let patchified = preprocessor.process_image(&dyn_img)?;
        let d_patches = DeviceBuffer::from_slice(&gpu, &patchified.patches)?;
        let mut d_proj = DeviceBuffer::<f16>::zeroed(&gpu, 64 * HIDDEN)?;
        vision_tower.forward(&gpu, &d_patches, &mut d_proj)?;
        stream.synchronize()?;
        let dur_ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_vision_ms += dur_ms;
        println!("  [Page {}] Vision preprocessor + ViT + Spatial Merge: {:.2} ms (64 visual tokens)", pnum, dur_ms);
        page_vis_embeds.push(d_proj);
    }
    println!(" Total Vision Tower execution time for {} pages: {:.2} ms (avg {:.2} ms/page)",
        page_files.len(), total_vision_ms, total_vision_ms / page_files.len() as f64);

    // 2. Build unified multimodal sequences
    println!("\n--------------------------------------------------------------------------------");
    println!(" PHASE 2: UNIFIED MULTIMODAL SEQUENCE ASSEMBLY & WAVE PREFILL");
    println!("--------------------------------------------------------------------------------");

    let embed_table_host = embed_table.table.inner().to_host_vec(stream)?;
    let mut contexts = Vec::with_capacity(page_files.len());
    let mut token_ids_list = Vec::with_capacity(page_files.len());
    let mut unified_embeds_list = Vec::with_capacity(page_files.len());

    let max_gen_tokens = 64usize;

    for (i, (pnum, _, prompt_instr)) in page_files.iter().enumerate() {
        let prompt_text = format!(
            "<|im_start|>user\n<|image|>{}<|im_end|>\n<|im_start|>assistant\n<think>",
            prompt_instr
        );
        let raw_tokens = tokenizer.encode(&prompt_text)?;
        // Replace single <|image|> token (125030) with 64 <|image|> tokens
        let mut token_ids: Vec<u32> = Vec::new();
        for &tok in &raw_tokens {
            if tok == 125030 {
                token_ids.extend(vec![125030; 64]);
            } else {
                token_ids.push(tok);
            }
        }

        let plen = token_ids.len();
        let cur_tok = *token_ids.last().unwrap();
        let ctx = create_context(&gpu, *pnum, plen, cur_tok, max_gen_tokens, &layer_types)?;

        // Splice visual features into unified embeddings
        let vis_proj_host = page_vis_embeds[i].inner().to_host_vec(stream)?;
        let mut unified_embeds: Vec<f16> = Vec::with_capacity(plen * HIDDEN);
        let mut vis_tok_idx = 0usize;

        for &tok in &token_ids {
            if tok == 125030 && vis_tok_idx < 64 {
                let v_slice = &vis_proj_host[vis_tok_idx * HIDDEN..(vis_tok_idx + 1) * HIDDEN];
                unified_embeds.extend_from_slice(v_slice);
                vis_tok_idx += 1;
            } else {
                let t_slice = &embed_table_host[(tok as usize) * HIDDEN..((tok as usize) + 1) * HIDDEN];
                unified_embeds.extend_from_slice(t_slice);
            }
        }
        let d_unified = DeviceBuffer::from_slice(&gpu, &unified_embeds)?;

        contexts.push(ctx);
        token_ids_list.push(token_ids);
        unified_embeds_list.push(d_unified);
    }

    // Prefill each sequence via forward_step_with_embeds
    let t_prefill = Instant::now();
    for i in 0..page_files.len() {
        let plen = token_ids_list[i].len();
        let pos_ids: Vec<u32> = (0..plen as u32).collect();
        let origins = vec![0u32; plen];
        let local_pos: Vec<u32> = (0..plen as u32).collect();

        let mut carry_host = vec![0u64; 24 * 128];
        for l in 0..layers.len() {
            if is_conv[l] {
                for r in 0..plen {
                    carry_host[l * plen + r] = contexts[i].conv_state_ptr(l);
                }
            }
        }
        let meta = PrefillMeta {
            carry_ptrs: carry_host,
            origins,
            local_pos,
        };

        let ctx_ref = &contexts[i];
        let ctx_refs: Vec<&PdfPageContext> = vec![ctx_ref; plen];

        batch_engine.forward_step_with_embeds(
            &gpu,
            &mut layers,
            &is_conv,
            &ctx_refs,
            &token_ids_list[i],
            &pos_ids,
            &embed_table.table,
            &embedding_norm,
            false,
            Some(&meta),
            Some(&unified_embeds_list[i]),
        )?;

        // Carry forward conv state to sequence context for decode
        for l in 0..layers.len() {
            if is_conv[l] {
                let dst = contexts[i].conv_state_ptr(l);
                let src = batch_engine.prefill_state_ptr(l, plen - 1);
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
    }
    stream.synchronize()?;
    let dur_prefill = t_prefill.elapsed();
    let total_prompt_tokens: usize = token_ids_list.iter().map(|t| t.len()).sum();
    println!("  Prefill completed in {:.2} ms for {} total multimodal tokens ({:.1} tok/s)",
        dur_prefill.as_secs_f64() * 1000.0, total_prompt_tokens,
        (total_prompt_tokens as f64) / dur_prefill.as_secs_f64());

    // Sample First Generation Tokens
    let pending_refs: Vec<&PdfPageContext> = contexts.iter().collect();
    let first_tokens = batch_engine.sample_only(&gpu, &pending_refs, &embed_table.table, &embedding_norm)?;
    for (i, &tok) in first_tokens.iter().enumerate() {
        contexts[i].gen_tokens.push(tok);
        contexts[i].cur_token = tok;
        contexts[i].gen_count = 1;
    }

    // 3. Continuous Batched Decoding Loop
    println!("\n--------------------------------------------------------------------------------");
    println!(" PHASE 3: CONTIGUOUS BATCHED DECODING LOOP");
    println!("--------------------------------------------------------------------------------");

    let t_decode = Instant::now();
    let mut total_generated_tokens = page_files.len();

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
        let active_refs: Vec<&PdfPageContext> = active_indices.iter().map(|&i| &contexts[i]).collect();

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
    let dur_decode = t_decode.elapsed();
    let dur_total = t_total_bench.elapsed();

    println!("\n================================================================================");
    println!(" BENCHMARK PERFORMANCE SUMMARY ON SEBI SCHEME INFORMATION DOCUMENT");
    println!("================================================================================");
    println!("  Pages Processed Concurrently : {}", page_files.len());
    println!("  Total Multimodal Tokens Prefilled : {}", total_prompt_tokens);
    println!("  Prefill Throughput           : {:.2} tok/s ({:.2} ms total)",
        (total_prompt_tokens as f64) / dur_prefill.as_secs_f64(), dur_prefill.as_secs_f64() * 1000.0);
    println!("  Total Generation Tokens      : {}", total_generated_tokens);
    println!("  Batched Decode Throughput    : {:.2} tok/s ({:.2} s total)",
        (total_generated_tokens as f64) / dur_decode.as_secs_f64(), dur_decode.as_secs_f64());
    println!("  Total Turnaround Time        : {:.2} s", dur_total.as_secs_f64());
    println!("================================================================================\n");

    println!("================================================================================");
    println!(" TRANSCRIBED & UNDERSTOOD OUTPUTS (OXIDE-INFER PURE RUST ENGINE):");
    println!("================================================================================\n");

    for (i, (pnum, pfile, prompt)) in page_files.iter().enumerate() {
        let text = tokenizer.decode(&contexts[i].gen_tokens).unwrap_or_else(|_| "<decode error>".to_string());
        println!(">>> [PAGE {}] ({})", pnum, pfile);
        println!("PROMPT: {}", prompt);
        println!("OXIDE-INFER OUTPUT ({} tokens):", contexts[i].gen_tokens.len());
        println!("{}\n", text.trim());
    }

    Ok(())
}
