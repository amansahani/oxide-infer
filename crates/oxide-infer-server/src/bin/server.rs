use oxide_infer_kernels::gemm::BATCHED_GEMV_FP16_PTX;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};
use axum::{
    extract::State,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_server::tokenizer::OxideTokenizer;
use oxide_infer_models::{EmbeddingTable, FullLfm3DecoderLayer, TiedLmHead};
use oxide_infer_kernels::norm::RMS_NORM_FP16_PTX;
use oxide_infer_kernels::PtxModule;
use cuda_core::launch_kernel_on_stream;

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Option<Vec<ChatMessage>>,
    pub prompt: Option<String>,
    pub max_tokens: Option<usize>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: ChatUsage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

struct ServingRequest {
    prompt: String,
    max_tokens: usize,
    response_tx: oneshot::Sender<(String, usize, usize)>,
}

#[derive(Clone)]
struct ServerState {
    tx: mpsc::Sender<ServingRequest>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("================================================================================");
    println!("   OXIDE-INFER: CONTINUOUS BATCHING CONCURRENT SERVING ENGINE                   ");
    println!("   Ampere Native Micro-Kernels with Layer-Interleaved Weight Amortization      ");
    println!("================================================================================\n");

    let host = std::env::var("OXIDE_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("OXIDE_PORT").unwrap_or_else(|_| "8090".to_string());
    let addr = format!("{}:{}", host, port);

    let model_path = std::env::var("OXIDE_MODEL").unwrap_or_else(|_| "/mnt/hdd/models/LFM3-VL-2B".to_string());
    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    let weights_path = format!("{}/model.safetensors", model_path);

    println!("[1/3] Initializing Tokenizer from {}...", tokenizer_path);
    let tokenizer = Arc::new(OxideTokenizer::from_file(&tokenizer_path)?);

    println!("[2/3] Initializing GPU Context & Single Shared Model Weights (2.88 GB)...");
    let gpu = Arc::new(GpuContext::init(0)?);
    let stream = gpu.compute_stream();

    let loader = SafeTensorsLoader::open(&weights_path)?;
    let embed_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embed_tokens.weight")?;
    let host_bf16 = embed_bf16.inner().to_host_vec(stream)?;
    let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let embed_table = Arc::new(EmbeddingTable::new(&gpu, &host_f16, 128000, 2048)?);
    let mut lm_head = TiedLmHead::new(&gpu, 128000, 2048)?;

    let norm_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embedding_norm.weight")?;
    let norm_f16: Vec<half::f16> = norm_bf16.inner().to_host_vec(stream)?
        .into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let embedding_norm = Arc::new(DeviceBuffer::from_slice(&gpu, &norm_f16)?);
    let norm_ptx = Arc::new(PtxModule::load_from_ptx(&gpu, RMS_NORM_FP16_PTX)?);
    let rms_func = norm_ptx.get_function("rms_norm_fp16")?;
    let mut d_logits = DeviceBuffer::<f32>::zeroed(&gpu, 128000)?;

    let layer_types = [
        "conv", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
        "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention",
        "conv", "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv",
        "full_attention", "conv"
    ];

    println!("      Loading 24 hybrid layers (Conv1D + GQA Attention)...");
    let mut layers = Vec::with_capacity(24);
    for idx in 0..24 {
        let is_attn = layer_types[idx] != "conv";
        let is_sliding = layer_types[idx] == "sliding_attention";
        let layer = FullLfm3DecoderLayer::load_from_safetensors(&gpu, &loader, idx, is_attn, is_sliding, 2048, 7680)?;
        layers.push(layer);
    }
    println!("      All weights loaded into GPU VRAM.\n");

    let num_slots = 4;
    println!("[3/3] Setting up {} continuous execution slots with unified weights...", num_slots);

    struct ActiveSlot {
        req: Option<ServingRequest>,
        prompt_len: usize,
        cur_token: u32,
        gen_tokens: Vec<u32>,
        gen_count: usize,
        max_gen: usize,
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
        slots.push(ActiveSlot {
            req: None,
            prompt_len: 0,
            cur_token: 0,
            gen_tokens: Vec::new(),
            gen_count: 0,
            max_gen: 0,
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

    let (req_tx, mut req_rx) = mpsc::channel::<ServingRequest>(1024);

    let tokenizer_clone = tokenizer.clone();
    let gpu_clone = gpu.clone();
    let embed_clone = embed_table.clone();
    let emb_norm_clone = embedding_norm.clone();

    // Dedicated asynchronous GPU continuous batching worker
    tokio::task::spawn_blocking(move || {
        let stream = gpu_clone.compute_stream();
        let batched_ptx = PtxModule::load_from_ptx(&gpu_clone, BATCHED_GEMV_FP16_PTX).expect("Failed to load BATCHED_GEMV_FP16_PTX");
        let batched_gemv_func = batched_ptx.get_function("batched_gemv_fp16_vec8").expect("batched_gemv_fp16_vec8 not found");
        let mut d_batch_act = DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 16 * 7680).expect("Failed to alloc d_batch_act");
        let mut d_batch_ffn_out = DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 16 * 2048).expect("Failed to alloc d_batch_ffn_out");

        let mut epsilon: f32 = 1e-5;
        let mut h_size: u32 = 2048;

        loop {
            // 1. Drain pending requests into free slots
            if !slots.iter().any(|s| s.req.is_some()) {
                // If completely idle, wait for the first incoming request
                match req_rx.blocking_recv() {
                    Some(req) => {
                        let slot = &mut slots[0];
                        let token_ids = tokenizer_clone.encode(&req.prompt).unwrap_or_else(|_| vec![124894, 2048]);
                        slot.prompt_len = token_ids.len();
                        slot.cur_token = *token_ids.last().unwrap();
                        slot.gen_tokens.clear();
                        slot.gen_count = 0;
                        slot.max_gen = req.max_tokens;

                        for (pos_id, &tok) in token_ids.iter().enumerate() {
                            let _ = slot.d_tok.copy_from_slice(&gpu_clone, &[tok]);
                            let _ = embed_clone.lookup(&gpu_clone, &slot.d_tok, 1, &mut slot.d_hidden);
                            for layer in layers.iter_mut() {
                                let _ = layer.forward(
                                    &gpu_clone,
                                    &mut slot.d_hidden,
                                    &mut slot.scratch_norm,
                                    &mut slot.scratch_layer,
                                    &mut slot.scratch_ffn,
                                    &mut slot.q_out,
                                    &mut slot.k_out,
                                    &mut slot.v_out,
                                    &mut slot.gate_out,
                                    pos_id,
                                );
                            }
                        }
                        slot.req = Some(req);
                    }
                    None => break, // Channel closed, shutdown worker
                }
            }

            // Drain any additional pending requests into remaining free slots
            for idx in 0..slots.len() {
                if slots[idx].req.is_none() {
                    if let Ok(req) = req_rx.try_recv() {
                        let token_ids = tokenizer_clone.encode(&req.prompt).unwrap_or_else(|_| vec![124894, 2048]);
                        let slot = &mut slots[idx];
                        slot.prompt_len = token_ids.len();
                        slot.cur_token = *token_ids.last().unwrap();
                        slot.gen_tokens.clear();
                        slot.gen_count = 0;
                        slot.max_gen = req.max_tokens;

                        for (pos_id, &tok) in token_ids.iter().enumerate() {
                            let _ = slot.d_tok.copy_from_slice(&gpu_clone, &[tok]);
                            let _ = embed_clone.lookup(&gpu_clone, &slot.d_tok, 1, &mut slot.d_hidden);
                            for layer in layers.iter_mut() {
                                let _ = layer.forward(
                                    &gpu_clone,
                                    &mut slot.d_hidden,
                                    &mut slot.scratch_norm,
                                    &mut slot.scratch_layer,
                                    &mut slot.scratch_ffn,
                                    &mut slot.q_out,
                                    &mut slot.k_out,
                                    &mut slot.v_out,
                                    &mut slot.gate_out,
                                    pos_id,
                                );
                            }
                        }
                        slot.req = Some(req);
                    }
                }
            }

            // Check if any slots are active
            let active_count = slots.iter().filter(|s| s.req.is_some()).count();
            if active_count == 0 {
                continue;
            }

            // 2. Decode 1 token across all active slots with shared layer weight amortization
            for slot in slots.iter_mut() {
                if slot.req.is_some() {
                    let _ = slot.d_tok.copy_from_slice(&gpu_clone, &[slot.cur_token]);
                    let _ = embed_clone.lookup(&gpu_clone, &slot.d_tok, 1, &mut slot.d_hidden);
                }
            }

            // Layer-interleaved execution across all active slots:
            // For each layer, evaluate all active slots back-to-back to keep weights in L2 cache
            for layer in layers.iter_mut() {
                for slot in slots.iter_mut() {
                    if slot.req.is_some() {
                        let pos_id = slot.prompt_len + slot.gen_count;
                        let _ = layer.forward(
                            &gpu_clone,
                            &mut slot.d_hidden,
                            &mut slot.scratch_norm,
                            &mut slot.scratch_layer,
                            &mut slot.scratch_ffn,
                            &mut slot.q_out,
                            &mut slot.k_out,
                            &mut slot.v_out,
                            &mut slot.gate_out,
                            pos_id,
                        );
                    }
                }
            }

            // Final norm, LM Head & sampling for each active slot
            for slot in slots.iter_mut() {
                if let Some(ref req) = slot.req {
                    let mut d_out_final_ptr = slot.d_normed_final.inner_mut().cu_deviceptr();
                    let mut d_hid_ptr = slot.d_hidden.inner().cu_deviceptr();
                    let mut d_emb_w_ptr = emb_norm_clone.inner().cu_deviceptr();

                    let mut args_final: [*mut std::ffi::c_void; 5] = [
                        &mut d_out_final_ptr as *mut _ as *mut std::ffi::c_void,
                        &mut d_hid_ptr as *mut _ as *mut std::ffi::c_void,
                        &mut d_emb_w_ptr as *mut _ as *mut std::ffi::c_void,
                        &mut epsilon as *mut _ as *mut std::ffi::c_void,
                        &mut h_size as *mut _ as *mut std::ffi::c_void,
                    ];
                    unsafe {
                        let _ = launch_kernel_on_stream(&rms_func, (1, 1, 1), (128, 1, 1), 4, stream, &mut args_final);
                    }

                    let _ = lm_head.compute_logits(&gpu_clone, &slot.d_normed_final, &embed_clone.table, &mut d_logits);
                    if let Ok(next_tok) = lm_head.sample_argmax(&gpu_clone, &d_logits) {
                        slot.gen_tokens.push(next_tok);
                        slot.cur_token = next_tok;
                        slot.gen_count += 1;

                        if slot.gen_count >= slot.max_gen || next_tok == 124900 || next_tok == 124896 {
                            let _ = stream.synchronize();
                            let decoded = tokenizer_clone.decode(&slot.gen_tokens).unwrap_or_default();
                            let finished_req = slot.req.take().unwrap();
                            let _ = finished_req.response_tx.send((decoded, slot.prompt_len, slot.gen_tokens.len()));
                        }
                    } else {
                        let _ = stream.synchronize();
                        let decoded = tokenizer_clone.decode(&slot.gen_tokens).unwrap_or_default();
                        let finished_req = slot.req.take().unwrap();
                        let _ = finished_req.response_tx.send((decoded, slot.prompt_len, slot.gen_tokens.len()));
                    }
                }
            }
        }
    });

    let state = ServerState { tx: req_tx };

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let listener = TcpListener::bind(&addr).await?;
    println!("\n🚀 OxideInfer Continuous Batching Server listening on http://{}", addr);
    println!("   API Endpoints (OpenAI / vLLM compatible):");
    println!("     - GET  http://{}/health", addr);
    println!("     - POST http://{}/v1/chat/completions\n", addr);

    axum::serve(listener, app).await?;
    Ok(())
}

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "engine": "oxide-infer",
        "mode": "continuous_batching_4slots",
        "model": "LFM3-VL-2B"
    }))
}

async fn chat_completions(
    State(state): State<ServerState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    let prompt = if let Some(messages) = payload.messages {
        let mut full_prompt = String::from("<|startoftext|>");
        for msg in messages {
            full_prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
        }
        full_prompt.push_str("<|im_start|>assistant\n<think>\n");
        full_prompt
    } else if let Some(p) = payload.prompt {
        p
    } else {
        return Json(serde_json::json!({"error": "No prompt or messages provided"})).into_response();
    };

    let max_tokens = payload.max_tokens.unwrap_or(32);
    let (resp_tx, resp_rx) = oneshot::channel();

    let req = ServingRequest {
        prompt,
        max_tokens,
        response_tx: resp_tx,
    };

    if state.tx.send(req).await.is_err() {
        return Json(serde_json::json!({"error": "Engine worker channel closed"})).into_response();
    }

    match resp_rx.await {
        Ok((text, prompt_tokens, completion_tokens)) => {
            let response = ChatCompletionResponse {
                id: format!("chatcmpl-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos()),
                object: "chat.completion".to_string(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                model: "LFM3-VL-2B".to_string(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatResponseMessage {
                        role: "assistant".to_string(),
                        content: text,
                    },
                    finish_reason: "stop".to_string(),
                }],
                usage: ChatUsage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
            };
            Json(response).into_response()
        }
        Err(_) => Json(serde_json::json!({"error": "Inference worker dropped response"})).into_response(),
    }
}
