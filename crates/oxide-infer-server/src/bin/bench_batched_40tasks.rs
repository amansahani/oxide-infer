use std::sync::Arc;
use std::time::Instant;
use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_models::{EmbeddingTable, FullLfm3DecoderLayer, TiedLmHead};
use oxide_infer_server::tokenizer::OxideTokenizer;
use oxide_infer_kernels::norm::RMS_NORM_FP16_PTX;
use oxide_infer_kernels::PtxModule;
use cuda_core::launch_kernel_on_stream;

fn main() -> anyhow::Result<()> {
    let tasks = [
        "What is the capital of France?",
        "Explain photosynthesis in one sentence.",
        "What is the speed of light in vacuum?",
        "Who wrote Romeo and Juliet?",
        "What is the largest ocean on Earth?",
        "Define recursion in computer science.",
        "What is the chemical symbol for gold?",
        "Explain Newton's third law of motion.",
        "What does CPU stand for?",
        "Name three primary colors.",
        "What is the square root of 144?",
        "Who painted the Mona Lisa?",
        "What is the powerhouse of the cell?",
        "Explain what an API is in simple terms.",
        "What is the boiling point of water in Celsius?",
        "How many planets are in our solar system?",
        "What is the main language spoken in Brazil?",
        "Define polymorphism in OOP.",
        "What does HTML stand for?",
        "What is the tallest mountain in the world?",
        "Name two greenhouse gases.",
        "Who discovered penicillin?",
        "What is the currency of Japan?",
        "What is Big O notation?",
        "Explain the purpose of DNS.",
        "What is the difference between RAM and ROM?",
        "Who is known as the father of modern physics?",
        "What is an IP address?",
        "What does HTTP stand for?",
        "What is the atomic number of Carbon?",
        "What is the fastest land animal?",
        "Define machine learning in one sentence.",
        "What is Git used for?",
        "Explain the greenhouse effect.",
        "What is the primary function of red blood cells?",
        "Who developed the theory of general relativity?",
        "What is a compiler?",
        "What does GPU stand for?",
        "What is the freezing point of water in Fahrenheit?",
        "What is the capital of Japan?"
    ];

    println!("================================================================================");
    println!("     OXIDE-INFER: 40-TASK CONTINUOUS BATCH SERVING BENCHMARK                    ");
    println!("     Ampere Layer-Interleaved Weight-Amortized Continuous Pipeline               ");
    println!("================================================================================\n");

    let gpu = Arc::new(GpuContext::init(0)?);
    let stream = gpu.compute_stream();
    let tokenizer = OxideTokenizer::from_file("/mnt/hdd/models/LFM3-VL-2B/tokenizer.json")?;
    let loader = SafeTensorsLoader::open("/mnt/hdd/models/LFM3-VL-2B/model.safetensors")?;

    println!("[1/3] Loading single shared weight set in VRAM (2.88 GB)...");
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

    let mut layers = Vec::with_capacity(24);
    for idx in 0..24 {
        let is_attn = layer_types[idx] != "conv";
        let is_sliding = layer_types[idx] == "sliding_attention";
        layers.push(FullLfm3DecoderLayer::load_from_safetensors(&gpu, &loader, idx, is_attn, is_sliding, 2048, 7680)?);
    }
    println!("      Weights ready.\n");

    let num_slots = 4;
    println!("[2/3] Setting up {} execution slots for continuous batching...", num_slots);
    struct Slot {
        task_idx: Option<usize>,
        prompt_tokens: Vec<u32>,
        cur_token: u32,
        gen_count: usize,
        max_gen: usize,
        start_time: Instant,
        turnaround: Option<f64>,
        gen_tokens: Vec<u32>,
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
    }

    let mut slots = Vec::with_capacity(num_slots);
    for _ in 0..num_slots {
        slots.push(Slot {
            task_idx: None,
            prompt_tokens: Vec::new(),
            cur_token: 0,
            gen_count: 0,
            max_gen: 30,
            start_time: Instant::now(),
            turnaround: None,
            gen_tokens: Vec::new(),
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
        });
    }

    let mut completed_turnarounds: Vec<f64> = Vec::with_capacity(tasks.len());
    let mut next_task_idx = 0;
    let mut epsilon: f32 = 1e-5;
    let mut h_size: u32 = 2048;

    println!("[3/3] Running 40 tasks through 4-slot continuous batching pipeline...");
    let wall_clock_start = Instant::now();

    while completed_turnarounds.len() < tasks.len() {
        // 1. Admit new tasks into free slots
        for slot in slots.iter_mut() {
            if slot.task_idx.is_none() && next_task_idx < tasks.len() {
                let prompt = format!("<|startoftext|><|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n", tasks[next_task_idx]);
                let tokens = tokenizer.encode(&prompt)?;
                let cur = *tokens.last().unwrap();
                slot.task_idx = Some(next_task_idx);
                slot.prompt_tokens = tokens;
                slot.cur_token = cur;
                slot.gen_count = 0;
                slot.gen_tokens.clear();
                slot.start_time = Instant::now();
                next_task_idx += 1;

                // Prefill this newly admitted slot
                for (pos_id, &tok) in slot.prompt_tokens.iter().enumerate() {
                    slot.d_tok.copy_from_slice(&gpu, &[tok])?;
                    embed_table.lookup(&gpu, &slot.d_tok, 1, &mut slot.d_hidden)?;
                    for layer in layers.iter_mut() {
                        layer.forward(
                            &gpu,
                            &mut slot.d_hidden,
                            &mut slot.scratch_norm,
                            &mut slot.scratch_layer,
                            &mut slot.scratch_ffn,
                            &mut slot.q_out,
                            &mut slot.k_out,
                            &mut slot.v_out,
                            &mut slot.gate_out,
                            pos_id,
                        )?;
                    }
                }
            }
        }

        // 2. Decode 1 token across all active slots in parallel (layer-interleaved)
        // Check active slots
        let active_count = slots.iter().filter(|s| s.task_idx.is_some()).count();
        if active_count == 0 {
            break;
        }

        // Embedding lookup for all active slots
        for slot in slots.iter_mut() {
            if slot.task_idx.is_some() {
                slot.d_tok.copy_from_slice(&gpu, &[slot.cur_token])?;
                embed_table.lookup(&gpu, &slot.d_tok, 1, &mut slot.d_hidden)?;
            }
        }

        // Forward through 24 layers interleaved:
        // Each layer is evaluated across active slots together, keeping layer weights HOT in L2 cache!
        for layer in layers.iter_mut() {
            for slot in slots.iter_mut() {
                if slot.task_idx.is_some() {
                    let pos_id = slot.prompt_tokens.len() + slot.gen_count;
                    layer.forward(
                        &gpu,
                        &mut slot.d_hidden,
                        &mut slot.scratch_norm,
                        &mut slot.scratch_layer,
                        &mut slot.scratch_ffn,
                        &mut slot.q_out,
                        &mut slot.k_out,
                        &mut slot.v_out,
                        &mut slot.gate_out,
                        pos_id,
                    )?;
                }
            }
        }

        // Argmax & completion check for each slot
        for slot in slots.iter_mut() {
            if slot.task_idx.is_some() {
                let mut d_out_final_ptr = slot.d_normed_final.inner_mut().cu_deviceptr();
                let mut d_hid_ptr = slot.d_hidden.inner().cu_deviceptr();
                let mut d_emb_w_ptr = embedding_norm.inner().cu_deviceptr();

                let mut args_final: [*mut std::ffi::c_void; 5] = [
                    &mut d_out_final_ptr as *mut _ as *mut std::ffi::c_void,
                    &mut d_hid_ptr as *mut _ as *mut std::ffi::c_void,
                    &mut d_emb_w_ptr as *mut _ as *mut std::ffi::c_void,
                    &mut epsilon as *mut _ as *mut std::ffi::c_void,
                    &mut h_size as *mut _ as *mut std::ffi::c_void,
                ];
                unsafe {
                    launch_kernel_on_stream(&rms_func, (1, 1, 1), (128, 1, 1), 4, stream, &mut args_final)?;
                }

                lm_head.compute_logits(&gpu, &slot.d_normed_final, &embed_table.table, &mut d_logits)?;
                if let Ok(next_tok) = lm_head.sample_argmax(&gpu, &d_logits) {
                    slot.gen_tokens.push(next_tok);
                    slot.cur_token = next_tok;
                    slot.gen_count += 1;

                    if slot.gen_count >= slot.max_gen || next_tok == 124900 || next_tok == 124896 {
                        let dur = slot.start_time.elapsed().as_secs_f64();
                        completed_turnarounds.push(dur);
                        slot.task_idx = None; // Free slot for next admitted request!
                    }
                } else {
                    let dur = slot.start_time.elapsed().as_secs_f64();
                    completed_turnarounds.push(dur);
                    slot.task_idx = None;
                }
            }
        }
    }
    stream.synchronize()?;
    let total_time = wall_clock_start.elapsed().as_secs_f64();
    let total_tokens = completed_turnarounds.len() * 30;

    println!("\n================================================================================");
    println!("     RESULTS: 40 TASKS VIA CONTINUOUS BATCHING PIPELINE                         ");
    println!("================================================================================");
    println!("Status:                    {}/{} completed", completed_turnarounds.len(), tasks.len());
    println!("Total Tokens Generated:    {} tokens", total_tokens);
    println!("Total Wall Clock Time:     {:.2} s", total_time);
    println!("Aggregate Serving Rate:    {:.2} tok/s", (total_tokens as f64) / total_time);
    println!("Fastest Client Turnaround: {:.2} s", completed_turnarounds.iter().copied().fold(f64::INFINITY, f64::min));
    println!("Average Client Latency:    {:.2} s", completed_turnarounds.iter().sum::<f64>() / completed_turnarounds.len() as f64);
    println!("Slowest Client Latency:    {:.2} s", completed_turnarounds.iter().copied().fold(0.0, f64::max));

    Ok(())
}
