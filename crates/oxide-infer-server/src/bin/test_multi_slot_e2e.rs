use std::sync::Arc;
use std::time::Instant;
use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_models::{EmbeddingTable, FullLfm3DecoderLayer, TiedLmHead};
use oxide_infer_server::tokenizer::OxideTokenizer;
use oxide_infer_kernels::norm::RMS_NORM_FP16_PTX;
use oxide_infer_kernels::PtxModule;
use cuda_core::launch_kernel_on_stream;

fn main() -> anyhow::Result<()> {
    let gpu = Arc::new(GpuContext::init(0)?);
    let stream = gpu.compute_stream();
    let tokenizer = OxideTokenizer::from_file("/mnt/hdd/models/LFM3-VL-2B/tokenizer.json")?;
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

    println!("Loading 1 set of 24 layers...");
    let mut layers = Vec::with_capacity(24);
    for idx in 0..24 {
        let is_attn = layer_types[idx] != "conv";
        let is_sliding = layer_types[idx] == "sliding_attention";
        layers.push(FullLfm3DecoderLayer::load_from_safetensors(&gpu, &loader, idx, is_attn, is_sliding, 2048, 7680)?);
    }

    let prompts = [
        "<|startoftext|><|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "<|startoftext|><|im_start|>user\nExplain photosynthesis in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "<|startoftext|><|im_start|>user\nWhat is the speed of light in vacuum?<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "<|startoftext|><|im_start|>user\nWho wrote Romeo and Juliet?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    ];

    println!("Setting up 4 active client streams...");
    struct StreamSlot {
        prompt_tokens: Vec<u32>,
        cur_token: u32,
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

    let mut slots = Vec::new();
    for p in prompts.iter() {
        let tokens = tokenizer.encode(p)?;
        let cur = *tokens.last().unwrap();
        slots.push(StreamSlot {
            cur_token: cur,
            prompt_tokens: tokens,
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

    println!("Prefilling 4 streams...");
    for slot in slots.iter_mut() {
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
    stream.synchronize()?;
    println!("Prefill done. Decoding 30 tokens across 4 streams interleaved per layer...");

    let mut epsilon: f32 = 1e-5;
    let mut h_size: u32 = 2048;

    let t_start = Instant::now();
    for step in 0..30 {
        // Interleaved layer execution: for each layer, evaluate all 4 active streams!
        // This keeps layer weights hot in GPU L2 cache across all 4 streams!
        for slot in slots.iter_mut() {
            let pos_id = slot.prompt_tokens.len() + step;
            slot.d_tok.copy_from_slice(&gpu, &[slot.cur_token])?;
            embed_table.lookup(&gpu, &slot.d_tok, 1, &mut slot.d_hidden)?;
        }

        for layer in layers.iter_mut() {
            for slot in slots.iter_mut() {
                let pos_id = slot.prompt_tokens.len() + step;
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

        for slot in slots.iter_mut() {
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
            let next_tok = lm_head.sample_argmax(&gpu, &d_logits)?;
            slot.gen_tokens.push(next_tok);
            slot.cur_token = next_tok;
        }
    }
    stream.synchronize()?;
    let elapsed = t_start.elapsed();
    let total_tokens = 4 * 30;
    println!("SUCCESS: Generated {} tokens across 4 concurrent streams in {:.2?} ({:.2} tok/s)!",
        total_tokens, elapsed, (total_tokens as f64) / elapsed.as_secs_f64());
    
    for (i, slot) in slots.iter().enumerate() {
        let text = tokenizer.decode(&slot.gen_tokens)?;
        println!("Stream {}: {:?}", i, text);
    }
    Ok(())
}
