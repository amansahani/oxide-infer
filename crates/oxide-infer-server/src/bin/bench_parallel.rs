use std::sync::Arc;
use std::time::Instant;
use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_server::tokenizer::OxideTokenizer;
use oxide_infer_models::{EmbeddingTable, FullLfm3DecoderLayer, TiedLmHead};
use oxide_infer_kernels::norm::RMS_NORM_FP16_PTX;
use oxide_infer_kernels::PtxModule;
use cuda_core::launch_kernel_on_stream;

struct ClientSession {
    token_ids: Vec<u32>,
    layers: Vec<FullLfm3DecoderLayer>,
    d_hidden: DeviceBuffer<half::f16>,
    scratch_norm: DeviceBuffer<half::f16>,
    scratch_layer: DeviceBuffer<half::f16>,
    scratch_ffn: DeviceBuffer<half::f16>,
    q_out: DeviceBuffer<half::f16>,
    k_out: DeviceBuffer<half::f16>,
    v_out: DeviceBuffer<half::f16>,
    gate_out: DeviceBuffer<half::f16>,
    d_normed_final: DeviceBuffer<half::f16>,
    d_tok: DeviceBuffer<u32>,
    current_token: u32,
    generated_tokens: Vec<u32>,
}

fn main() -> anyhow::Result<()> {
    println!("================================================================================");
    println!("        OXIDE-INFER: PARALLEL CONCURRENT SERVING BENCHMARK                      ");
    println!("        Model: /mnt/hdd/models/LFM3-VL-2B (RTX 3060 12GB - cuda-oxide)          ");
    println!("================================================================================\n");

    let gpu = Arc::new(GpuContext::init(0)?);
    let stream = gpu.compute_stream();

    let tokenizer = OxideTokenizer::from_file("/mnt/hdd/models/LFM3-VL-2B/tokenizer.json")?;
    let test_prompts = [
        "<|startoftext|><|im_start|>user\nWhat is Rust in one sentence?<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "<|startoftext|><|im_start|>user\nExplain GPU tensor cores briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "<|startoftext|><|im_start|>user\nWhat is the difference between TCP and UDP?<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "<|startoftext|><|im_start|>user\nDescribe asynchronous programming in Rust.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    ];

    println!("[1/4] Loading model weights into shared VRAM...");
    let loader = SafeTensorsLoader::open("/mnt/hdd/models/LFM3-VL-2B/model.safetensors")?;
    let embed_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embed_tokens.weight")?;
    let host_bf16 = embed_bf16.inner().to_host_vec(stream)?;
    let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let embed_table = EmbeddingTable::new(&gpu, &host_f16, 128000, 2048)?;
    let mut lm_head = TiedLmHead::new(&gpu, 128000, 2048)?;

    let norm_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embedding_norm.weight")?;
    let norm_f16: Vec<half::f16> = norm_bf16.inner().to_host_vec(stream)?
        .into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let embedding_norm = DeviceBuffer::from_slice(&gpu, &norm_f16)?;
    let norm_ptx = PtxModule::load_from_ptx(&gpu, RMS_NORM_FP16_PTX)?;
    let rms_func = norm_ptx.get_function("rms_norm_fp16")?;
    let mut d_logits = DeviceBuffer::<f32>::zeroed(&gpu, 128000)?;

    let layer_types = [
        "conv", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
        "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention",
        "conv", "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv",
        "full_attention", "conv"
    ];

    println!("      Weights ready.");

    for &concurrency in &[1, 2, 4] {
        println!("\n--------------------------------------------------------------------------------");
        println!(">>> BENCHMARKING CONCURRENCY = {} PARALLEL SESSIONS", concurrency);
        println!("--------------------------------------------------------------------------------");

        let mut clients: Vec<ClientSession> = Vec::with_capacity(concurrency);
        for c_idx in 0..concurrency {
            let prompt = test_prompts[c_idx % test_prompts.len()];
            let token_ids = tokenizer.encode(prompt)?;

            let mut layers = Vec::with_capacity(24);
            for idx in 0..24 {
                let is_attn = layer_types[idx] != "conv";
                let is_sliding = layer_types[idx] == "sliding_attention";
                let layer = FullLfm3DecoderLayer::load_from_safetensors(&gpu, &loader, idx, is_attn, is_sliding, 2048, 7680)?;
                layers.push(layer);
            }

            clients.push(ClientSession {
                current_token: *token_ids.last().unwrap(),
                token_ids,
                layers,
                d_hidden: DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?,
                scratch_norm: DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?,
                scratch_layer: DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?,
                scratch_ffn: DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?,
                q_out: DeviceBuffer::<half::f16>::zeroed(&gpu, 16 * 128)?,
                k_out: DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?,
                v_out: DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?,
                gate_out: DeviceBuffer::<half::f16>::zeroed(&gpu, 16)?,
                d_normed_final: DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?,
                d_tok: DeviceBuffer::<u32>::zeroed(&gpu, 1)?,
                generated_tokens: Vec::new(),
            });
        }

        // Prefill all clients
        let t_prefill_start = Instant::now();
        let mut total_prefill_tokens = 0;
        for client in &mut clients {
            total_prefill_tokens += client.token_ids.len();
            for (pos_id, &tok) in client.token_ids.iter().enumerate() {
                client.d_tok.copy_from_slice(&gpu, &[tok])?;
                embed_table.lookup(&gpu, &client.d_tok, 1, &mut client.d_hidden)?;
                for layer in &mut client.layers {
                    layer.forward(
                        &gpu,
                        &mut client.d_hidden,
                        &mut client.scratch_norm,
                        &mut client.scratch_layer,
                        &mut client.scratch_ffn,
                        &mut client.q_out,
                        &mut client.k_out,
                        &mut client.v_out,
                        &mut client.gate_out,
                        pos_id,
                    )?;
                }
            }
        }
        stream.synchronize()?;
        let prefill_dur = t_prefill_start.elapsed();
        let prefill_tps = (total_prefill_tokens as f64) / prefill_dur.as_secs_f64();
        println!("      [Prefill] {} tokens processed across {} streams in {:.2?} ({:.1} tok/s)",
            total_prefill_tokens, concurrency, prefill_dur, prefill_tps);

        // Batched / Concurrent Decode Loop
        let n_gen_steps = 32;
        let mut epsilon: f32 = 1e-5;
        let mut h_size: u32 = 2048;

        let t_gen_start = Instant::now();
        for step in 0..n_gen_steps {
            for client in &mut clients {
                let pos_id = client.token_ids.len() + step;
                client.d_tok.copy_from_slice(&gpu, &[client.current_token])?;
                embed_table.lookup(&gpu, &client.d_tok, 1, &mut client.d_hidden)?;

                for layer in &mut client.layers {
                    layer.forward(
                        &gpu,
                        &mut client.d_hidden,
                        &mut client.scratch_norm,
                        &mut client.scratch_layer,
                        &mut client.scratch_ffn,
                        &mut client.q_out,
                        &mut client.k_out,
                        &mut client.v_out,
                        &mut client.gate_out,
                        pos_id,
                    )?;
                }

                let mut d_out_final = client.d_normed_final.inner_mut().cu_deviceptr();
                let mut d_hid = client.d_hidden.inner().cu_deviceptr();
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

                lm_head.compute_logits(&gpu, &client.d_normed_final, &embed_table.table, &mut d_logits)?;
                let next_tok = lm_head.sample_argmax(&gpu, &d_logits)?;
                client.generated_tokens.push(next_tok);
                client.current_token = next_tok;
            }
        }
        stream.synchronize()?;
        let gen_dur = t_gen_start.elapsed();
        let total_gen_tokens = concurrency * n_gen_steps;
        let gen_tps = (total_gen_tokens as f64) / gen_dur.as_secs_f64();
        let avg_latency_per_step = gen_dur.as_secs_f64() / (n_gen_steps as f64);
        let per_stream_tps = (n_gen_steps as f64) / gen_dur.as_secs_f64();

        println!("      [Decode]  {} tokens generated across {} streams in {:.2?} ({:.1} total tok/s)",
            total_gen_tokens, concurrency, gen_dur, gen_tps);
        println!("                Per-stream rate: {:.1} tok/s | Inter-token step latency: {:.2} ms",
            per_stream_tps, avg_latency_per_step * 1000.0);
    }

    Ok(())
}
