use std::sync::Arc;
use std::ffi::c_void;
use std::time::Instant;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};
use axum::{
    extract::State,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer, VisionPreprocessor};
use oxide_infer_server::tokenizer::OxideTokenizer;
use oxide_infer_models::{EmbeddingTable, FullLfm3DecoderLayer, TiedLmHead, VisionTower, LayerType, RESIDUAL_ADD_PTX};
use oxide_infer_server::batch_engine::{BatchEngine, BatchSequence, PrefillMeta};
use oxide_infer_kernels::norm::RMS_NORM_FP16_PTX;
use oxide_infer_kernels::PtxModule;
use cuda_core::launch_kernel_on_stream;

pub type CublasHandle = *mut c_void;
pub type CublasStatus = i32;

#[link(name = "cublas")]
#[link(name = "cuda")]
extern "C" {
    pub fn cublasCreate_v2(handle: *mut CublasHandle) -> CublasStatus;
    pub fn cublasDestroy_v2(handle: CublasHandle) -> CublasStatus;
    pub fn cublasSetStream_v2(handle: CublasHandle, stream: *mut c_void) -> CublasStatus;
    pub fn cublasHgemm(
        handle: CublasHandle,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const half::f16,
        A: *const c_void,
        lda: i32,
        B: *const c_void,
        ldb: i32,
        beta: *const half::f16,
        C: *mut c_void,
        ldc: i32,
    ) -> CublasStatus;
    pub fn cuMemcpyDtoDAsync_v2(dst: u64, src: u64, byte_count: usize, stream: *mut c_void) -> i32;
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub part_type: String,
    pub text: Option<String>,
    pub image_url: Option<ImageUrl>,
}

#[derive(Debug, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
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
    image_bytes: Option<Vec<u8>>,
    max_tokens: usize,
    response_tx: oneshot::Sender<(String, usize, usize)>,
}

#[derive(Clone)]
struct ServerState {
    tx: mpsc::Sender<ServingRequest>,
}

const MAX_BATCH_SIZE: usize = 128;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("================================================================================");
    println!("   OXIDE-INFER: HIGH-PERFORMANCE CONTINUOUS BATCHING SERVING ENGINE            ");
    println!("   Ampere Tensor Core Acceleration (cuBLAS HGEMM + Native Micro-Kernels)       ");
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

    println!("      Loading VisionTower & MultiModal Projector...");
    let vision_tower = VisionTower::load_from_safetensors(&gpu, &loader)?;
    let preprocessor = VisionPreprocessor::default();

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

    println!("[3/3] Setting up dynamic continuous batch pool (Max capacity: {} concurrent sequences)...", MAX_BATCH_SIZE);

    struct ActiveContext {
        req: ServingRequest,
        prompt_len: usize,
        cur_token: u32,
        gen_tokens: Vec<u32>,
        gen_count: usize,
        max_gen: usize,
        d_hidden: DeviceBuffer<half::f16>,
        scratch_norm: DeviceBuffer<half::f16>,
        scratch_layer: DeviceBuffer<half::f16>,
        q_out: DeviceBuffer<half::f16>,
        k_out: DeviceBuffer<half::f16>,
        v_out: DeviceBuffer<half::f16>,
        gate_out: DeviceBuffer<half::f16>,
        d_normed_final: DeviceBuffer<half::f16>,
        d_tok: DeviceBuffer<u32>,
        // Per-request isolated state buffers: 24 layers of KV or Conv state
        layer_k_caches: Vec<Option<DeviceBuffer<half::f16>>>,
        layer_v_caches: Vec<Option<DeviceBuffer<half::f16>>>,
        layer_conv_states: Vec<Option<DeviceBuffer<half::f16>>>,
    }

    impl BatchSequence for ActiveContext {
        fn k_cache_ptr(&self, layer: usize) -> u64 {
            self.layer_k_caches[layer]
                .as_ref()
                .expect("missing k cache")
                .inner()
                .cu_deviceptr()
        }
        fn v_cache_ptr(&self, layer: usize) -> u64 {
            self.layer_v_caches[layer]
                .as_ref()
                .expect("missing v cache")
                .inner()
                .cu_deviceptr()
        }
        fn conv_state_ptr(&self, layer: usize) -> u64 {
            self.layer_conv_states[layer]
                .as_ref()
                .expect("missing conv state")
                .inner()
                .cu_deviceptr()
        }
        fn hidden_ptr(&self) -> u64 {
            self.d_hidden.inner().cu_deviceptr()
        }
    }

    let (req_tx, mut req_rx) = mpsc::channel::<ServingRequest>(4096);

    let tokenizer_clone = tokenizer.clone();
    let gpu_clone = gpu.clone();
    let embed_clone = embed_table.clone();
    let emb_norm_clone = embedding_norm.clone();
    let mut vision_tower = vision_tower;
    let preprocessor = preprocessor;

    // Dedicated asynchronous GPU continuous batching worker
    tokio::task::spawn_blocking(move || {
        let stream = gpu_clone.compute_stream();
        let mut cublas_handle: CublasHandle = std::ptr::null_mut();
        unsafe {
            assert_eq!(cublasCreate_v2(&mut cublas_handle), 0);
            assert_eq!(cublasSetStream_v2(cublas_handle, stream.cu_stream() as *mut c_void), 0);
        }
        let batched_ptx = PtxModule::load_from_ptx(&gpu_clone, RESIDUAL_ADD_PTX).expect("Failed to load RESIDUAL_ADD_PTX");
        let swiglu_func = batched_ptx.get_function("batched_swiglu_separate_fp16").expect("batched_swiglu_separate_fp16 not found");

        
        const BATCHED_FUSED_CONV_GATING_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry batched_fused_causal_conv1d_gating_fp16(
    .param .u64 y_out,
    .param .u64 state_cache,
    .param .u64 bcx_in,
    .param .u64 conv_weight,
    .param .u32 hidden_size,
    .param .u32 k_len,
    .param .u32 batch_size
) {
    .reg .pred %p0, %p1, %p_loop;
    .reg .b32 %idx, %hsize, %klen, %b_idx, %stride, %c_offset, %x_offset, %batch;
    .reg .b32 %k, %k_minus_1, %row_offset_32;
    .reg .b64 %y_ptr, %cache_ptr, %bcx_ptr, %w_ptr;
    .reg .b64 %c_ptr, %x_ptr, %b_ptr, %offset, %c_state_ptr, %c_w_ptr, %batch_off_bcx, %batch_off_cache, %batch_off_y;
    .reg .b16 %hb, %hx, %hc, %hin, %hw, %hs;
    .reg .f32 %fb, %fx, %fin, %fc, %fw, %fs, %fsum;

    ld.param.u32 %hsize, [hidden_size];
    ld.param.u32 %klen, [k_len];
    ld.param.u32 %batch, [batch_size];

    mov.u32 %b_idx, %ctaid.y;
    setp.ge.u32 %p0, %b_idx, %batch;
    @%p0 bra EXIT;

    mov.u32 %idx, %tid.x;
    .reg .b32 %bx;
    mov.u32 %bx, %ctaid.x;
    mov.u32 %stride, %ntid.x;
    mad.lo.u32 %idx, %bx, %stride, %idx;

    setp.ge.u32 %p0, %idx, %hsize;
    @%p0 bra EXIT;

    ld.param.u64 %bcx_ptr, [bcx_in];
    ld.param.u64 %cache_ptr, [state_cache];
    ld.param.u64 %w_ptr, [conv_weight];
    ld.param.u64 %y_ptr, [y_out];

    mul.wide.u32 %batch_off_bcx, %b_idx, %hsize;
    mul.lo.u64 %batch_off_bcx, %batch_off_bcx, 3;
    shl.b64 %batch_off_bcx, %batch_off_bcx, 1;
    add.u64 %bcx_ptr, %bcx_ptr, %batch_off_bcx;

    sub.u32 %k_minus_1, %klen, 1;
    mul.lo.u32 %row_offset_32, %hsize, %k_minus_1;
    mul.wide.u32 %batch_off_cache, %b_idx, %row_offset_32;
    shl.b64 %batch_off_cache, %batch_off_cache, 1;
    add.u64 %cache_ptr, %cache_ptr, %batch_off_cache;

    mul.wide.u32 %batch_off_y, %b_idx, %hsize;
    shl.b64 %batch_off_y, %batch_off_y, 1;
    add.u64 %y_ptr, %y_ptr, %batch_off_y;

    cvt.u64.u32 %offset, %idx;
    shl.b64 %offset, %offset, 1;
    add.u64 %b_ptr, %bcx_ptr, %offset;
    ld.global.u16 %hb, [%b_ptr];

    shl.b32 %x_offset, %hsize, 1;
    add.u32 %x_offset, %x_offset, %idx;
    cvt.u64.u32 %offset, %x_offset;
    shl.b64 %offset, %offset, 1;
    add.u64 %x_ptr, %bcx_ptr, %offset;
    ld.global.u16 %hx, [%x_ptr];

    cvt.f32.f16 %fb, %hb;
    cvt.f32.f16 %fx, %hx;
    mul.f32 %fin, %fb, %fx;
    cvt.rn.f16.f32 %hin, %fin;

    mul.lo.u32 %row_offset_32, %idx, %k_minus_1;
    cvt.u64.u32 %offset, %row_offset_32;
    shl.b64 %offset, %offset, 1;
    add.u64 %c_state_ptr, %cache_ptr, %offset;

    mul.lo.u32 %row_offset_32, %idx, %klen;
    cvt.u64.u32 %offset, %row_offset_32;
    shl.b64 %offset, %offset, 1;
    add.u64 %c_w_ptr, %w_ptr, %offset;

    mov.f32 %fsum, 0.0;
    mov.u32 %k, 0;

LOOP_K:
    setp.ge.u32 %p_loop, %k, %k_minus_1;
    @%p_loop bra AFTER_LOOP;

    cvt.u64.u32 %offset, %k;
    shl.b64 %offset, %offset, 1;
    add.u64 %b_ptr, %c_state_ptr, %offset;
    ld.global.u16 %hs, [%b_ptr];
    cvt.f32.f16 %fs, %hs;

    add.u64 %x_ptr, %c_w_ptr, %offset;
    ld.global.u16 %hw, [%x_ptr];
    cvt.f32.f16 %fw, %hw;

    fma.rn.f32 %fsum, %fs, %fw, %fsum;

    add.u32 %k, %k, 1;
    bra LOOP_K;

AFTER_LOOP:
    cvt.u64.u32 %offset, %k_minus_1;
    shl.b64 %offset, %offset, 1;
    add.u64 %x_ptr, %c_w_ptr, %offset;
    ld.global.u16 %hw, [%x_ptr];
    cvt.f32.f16 %fw, %hw;
    fma.rn.f32 %fsum, %fin, %fw, %fsum;

    mov.u32 %k, 0;
SHIFT_LOOP:
    add.u32 %stride, %k, 1;
    setp.ge.u32 %p_loop, %stride, %k_minus_1;
    @%p_loop bra SHIFT_LAST;

    cvt.u64.u32 %offset, %stride;
    shl.b64 %offset, %offset, 1;
    add.u64 %x_ptr, %c_state_ptr, %offset;
    ld.global.u16 %hs, [%x_ptr];

    cvt.u64.u32 %offset, %k;
    shl.b64 %offset, %offset, 1;
    add.u64 %b_ptr, %c_state_ptr, %offset;
    st.global.u16 [%b_ptr], %hs;

    add.u32 %k, %k, 1;
    bra SHIFT_LOOP;

SHIFT_LAST:
    sub.u32 %stride, %k_minus_1, 1;
    cvt.u64.u32 %offset, %stride;
    shl.b64 %offset, %offset, 1;
    add.u64 %b_ptr, %c_state_ptr, %offset;
    st.global.u16 [%b_ptr], %hin;

    add.u32 %c_offset, %hsize, %idx;
    cvt.u64.u32 %offset, %c_offset;
    shl.b64 %offset, %offset, 1;
    add.u64 %c_ptr, %bcx_ptr, %offset;
    ld.global.u16 %hc, [%c_ptr];
    cvt.f32.f16 %fc, %hc;

    mul.f32 %fsum, %fsum, %fc;
    cvt.rn.f16.f32 %hin, %fsum;

    cvt.u64.u32 %offset, %idx;
    shl.b64 %offset, %offset, 1;
    add.u64 %b_ptr, %y_ptr, %offset;
    st.global.u16 [%b_ptr], %hin;

EXIT:
    ret;
}
"#;

        const BATCHED_ARGMAX_DIRECT_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry batched_argmax_direct_f16(
    .param .u64 tokens_out,
    .param .u64 logits_in,
    .param .u32 vocab_size
) {
    .reg .pred %p_loop, %p_swap, %p_lane0;
    .reg .b32 %batch_idx, %t_id, %nt_id, %vsize, %idx;
    .reg .b64 %out_ptr, %in_ptr, %offset, %byte_off;
    .reg .b16 %h_val;
    .reg .f32 %f_val, %max_val, %shfl_val;
    .reg .b32 %best_idx, %shfl_idx, %lane_id, %warp_id, %s_addr, %stride;
    .shared .f32 s_warp_max[8];
    .shared .b32 s_warp_idx[8];

    ld.param.u32 %vsize, [vocab_size];
    mov.u32 %batch_idx, %ctaid.x;
    mov.u32 %t_id, %tid.x;
    mov.u32 %nt_id, %ntid.x;

    mov.f32 %max_val, -1.0e30;
    mov.u32 %best_idx, 0;

    ld.param.u64 %in_ptr, [logits_in];
    mul.wide.u32 %offset, %batch_idx, %vsize;
    shl.b64 %offset, %offset, 1;
    add.u64 %in_ptr, %in_ptr, %offset;

    mov.u32 %idx, %t_id;

LOOP:
    setp.ge.u32 %p_loop, %idx, %vsize;
    @%p_loop bra REDUCE;

    cvt.u64.u32 %byte_off, %idx;
    shl.b64 %byte_off, %byte_off, 1;
    add.u64 %offset, %in_ptr, %byte_off;
    ld.global.u16 %h_val, [%offset];
    cvt.f32.f16 %f_val, %h_val;

    setp.gt.f32 %p_swap, %f_val, %max_val;
    @%p_swap mov.f32 %max_val, %f_val;
    @%p_swap mov.u32 %best_idx, %idx;

    add.u32 %idx, %idx, %nt_id;
    bra LOOP;

REDUCE:
    shfl.sync.down.b32 %shfl_val, %max_val, 16, 31, 0xffffffff;
    shfl.sync.down.b32 %shfl_idx, %best_idx, 16, 31, 0xffffffff;
    setp.gt.f32 %p_swap, %shfl_val, %max_val;
    @%p_swap mov.f32 %max_val, %shfl_val;
    @%p_swap mov.u32 %best_idx, %shfl_idx;

    shfl.sync.down.b32 %shfl_val, %max_val, 8, 31, 0xffffffff;
    shfl.sync.down.b32 %shfl_idx, %best_idx, 8, 31, 0xffffffff;
    setp.gt.f32 %p_swap, %shfl_val, %max_val;
    @%p_swap mov.f32 %max_val, %shfl_val;
    @%p_swap mov.u32 %best_idx, %shfl_idx;

    shfl.sync.down.b32 %shfl_val, %max_val, 4, 31, 0xffffffff;
    shfl.sync.down.b32 %shfl_idx, %best_idx, 4, 31, 0xffffffff;
    setp.gt.f32 %p_swap, %shfl_val, %max_val;
    @%p_swap mov.f32 %max_val, %shfl_val;
    @%p_swap mov.u32 %best_idx, %shfl_idx;

    shfl.sync.down.b32 %shfl_val, %max_val, 2, 31, 0xffffffff;
    shfl.sync.down.b32 %shfl_idx, %best_idx, 2, 31, 0xffffffff;
    setp.gt.f32 %p_swap, %shfl_val, %max_val;
    @%p_swap mov.f32 %max_val, %shfl_val;
    @%p_swap mov.u32 %best_idx, %shfl_idx;

    shfl.sync.down.b32 %shfl_val, %max_val, 1, 31, 0xffffffff;
    shfl.sync.down.b32 %shfl_idx, %best_idx, 1, 31, 0xffffffff;
    setp.gt.f32 %p_swap, %shfl_val, %max_val;
    @%p_swap mov.f32 %max_val, %shfl_val;
    @%p_swap mov.u32 %best_idx, %shfl_idx;

    and.b32 %lane_id, %t_id, 31;
    shr.u32 %warp_id, %t_id, 5;
    setp.eq.u32 %p_lane0, %lane_id, 0;

    mov.u32 %s_addr, s_warp_max;
    shl.b32 %stride, %warp_id, 2;
    add.u32 %s_addr, %s_addr, %stride;
    @%p_lane0 st.shared.f32 [%s_addr], %max_val;

    mov.u32 %s_addr, s_warp_idx;
    add.u32 %s_addr, %s_addr, %stride;
    @%p_lane0 st.shared.b32 [%s_addr], %best_idx;

    bar.sync 0;

    setp.lt.u32 %p_lane0, %t_id, 8;
    @!%p_lane0 bra FINISH;

    mov.u32 %s_addr, s_warp_max;
    shl.b32 %stride, %t_id, 2;
    add.u32 %s_addr, %s_addr, %stride;
    ld.shared.f32 %max_val, [%s_addr];

    mov.u32 %s_addr, s_warp_idx;
    add.u32 %s_addr, %s_addr, %stride;
    ld.shared.b32 %best_idx, [%s_addr];

    shfl.sync.down.b32 %shfl_val, %max_val, 4, 31, 0xffffffff;
    shfl.sync.down.b32 %shfl_idx, %best_idx, 4, 31, 0xffffffff;
    setp.gt.f32 %p_swap, %shfl_val, %max_val;
    @%p_swap mov.f32 %max_val, %shfl_val;
    @%p_swap mov.u32 %best_idx, %shfl_idx;

    shfl.sync.down.b32 %shfl_val, %max_val, 2, 31, 0xffffffff;
    shfl.sync.down.b32 %shfl_idx, %best_idx, 2, 31, 0xffffffff;
    setp.gt.f32 %p_swap, %shfl_val, %max_val;
    @%p_swap mov.f32 %max_val, %shfl_val;
    @%p_swap mov.u32 %best_idx, %shfl_idx;

    shfl.sync.down.b32 %shfl_val, %max_val, 1, 31, 0xffffffff;
    shfl.sync.down.b32 %shfl_idx, %best_idx, 1, 31, 0xffffffff;
    setp.gt.f32 %p_swap, %shfl_val, %max_val;
    @%p_swap mov.f32 %max_val, %shfl_val;
    @%p_swap mov.u32 %best_idx, %shfl_idx;

    setp.eq.u32 %p_lane0, %t_id, 0;
    @!%p_lane0 bra FINISH;

    ld.param.u64 %out_ptr, [tokens_out];
    cvt.u64.u32 %offset, %batch_idx;
    shl.b64 %offset, %offset, 2;
    add.u64 %out_ptr, %out_ptr, %offset;
    st.global.u32 [%out_ptr], %best_idx;

FINISH:
    ret;
}
"#;
        let argmax_ptx = PtxModule::load_from_ptx(&gpu_clone, BATCHED_ARGMAX_DIRECT_PTX).expect("Failed to load BATCHED_ARGMAX_DIRECT_PTX");
        let batched_argmax_func = argmax_ptx.get_function("batched_argmax_direct_f16").expect("batched_argmax_direct_f16 not found");
        let conv_ptx = PtxModule::load_from_ptx(&gpu_clone, BATCHED_FUSED_CONV_GATING_PTX).expect("Failed to load BATCHED_FUSED_CONV_GATING_PTX");
        let batched_conv_func = conv_ptx.get_function("batched_fused_causal_conv1d_gating_fp16").expect("batched_fused_causal_conv1d_gating_fp16 not found");


        let mut d_batch_x = DeviceBuffer::<half::f16>::zeroed(&gpu_clone, MAX_BATCH_SIZE * 2048).expect("d_batch_x");
        let mut d_batch_gate = DeviceBuffer::<half::f16>::zeroed(&gpu_clone, MAX_BATCH_SIZE * 7680).expect("d_batch_gate");
        let mut d_batch_up = DeviceBuffer::<half::f16>::zeroed(&gpu_clone, MAX_BATCH_SIZE * 7680).expect("d_batch_up");
        let mut d_batch_act = DeviceBuffer::<half::f16>::zeroed(&gpu_clone, MAX_BATCH_SIZE * 7680).expect("d_batch_act");
        let mut d_batch_ffn_out = DeviceBuffer::<half::f16>::zeroed(&gpu_clone, MAX_BATCH_SIZE * 2048).expect("d_batch_ffn_out");
        let mut d_batch_logits_f16 = DeviceBuffer::<half::f16>::zeroed(&gpu_clone, MAX_BATCH_SIZE * 128000).expect("d_batch_logits_f16");
        let mut d_batch_tokens = DeviceBuffer::<u32>::zeroed(&gpu_clone, MAX_BATCH_SIZE).expect("d_batch_tokens");
        let mut d_batch_bcx = DeviceBuffer::<half::f16>::zeroed(&gpu_clone, MAX_BATCH_SIZE * 6144).expect("d_batch_bcx");
        let mut d_batch_conv_y = DeviceBuffer::<half::f16>::zeroed(&gpu_clone, MAX_BATCH_SIZE * 2048).expect("d_batch_conv_y");
        // Conv state layout is [batch, hidden, k_len - 1] = [batch, 2048, 6].
        // The per-request buffers are allocated with 7 slots, but only 6 are live.
        let mut d_batch_state_pool = DeviceBuffer::<half::f16>::zeroed(&gpu_clone, MAX_BATCH_SIZE * 2048 * 6).expect("d_batch_state_pool");


        // Temporary fidelity-isolation switches. Remove once the batched paths are verified.
        let scalar_batch = std::env::var("OXIDE_SCALAR_BATCH").is_ok();
        let scalar_conv = scalar_batch || std::env::var("OXIDE_SCALAR_CONV").is_ok();
        let scalar_ffn = scalar_batch || std::env::var("OXIDE_SCALAR_FFN").is_ok();
        let scalar_lm = scalar_batch || std::env::var("OXIDE_SCALAR_LM").is_ok();
        let scalar_prefill = scalar_batch || std::env::var("OXIDE_SCALAR_PREFILL").is_ok();
        let profile = std::env::var("OXIDE_PROFILE").is_ok();
        let debug_top = std::env::var("OXIDE_DEBUG_TOP").is_ok();
        let dump_hidden = std::env::var("OXIDE_DUMP_HIDDEN").ok();
        let (mut prof_embed, mut prof_conv, mut prof_attn, mut prof_ffn, mut prof_lm, mut prof_steps) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0u32);

        // Fully batched engine (default). `OXIDE_SCALAR_BATCH=1` keeps the old
        // per-sequence reference path for fidelity checks.
        let mut batch_engine = if scalar_batch {
            None
        } else {
            Some(BatchEngine::new(&gpu_clone, MAX_BATCH_SIZE).expect("batched engine init"))
        };
        let is_conv: Vec<bool> = layer_types.iter().map(|t| *t == "conv").collect();

        let mut active_contexts: Vec<ActiveContext> = Vec::with_capacity(MAX_BATCH_SIZE);
        let mut epsilon: f32 = 1e-5;
        let mut h_size: u32 = 2048;
        let alpha = half::f16::from_f32(1.0);
        let beta = half::f16::from_f32(0.0);
        let op_t = 1; // CUBLAS_OP_T
        let op_n = 0; // CUBLAS_OP_N

        loop {
            // 1. Drain incoming requests into newly admitted contexts
            let mut new_contexts: Vec<(ActiveContext, Vec<u32>, Option<Vec<half::f16>>)> = Vec::new();
            if active_contexts.is_empty() {
                match req_rx.blocking_recv() {
                    Some(req) => {
                        let mut token_ids = tokenizer_clone.encode(&req.prompt).unwrap_or_else(|_| vec![124894, 2048]);
                        
                        // If request has image bytes, process through VisionPreprocessor + VisionTower
                        let custom_embeds = if let Some(ref img_bytes) = req.image_bytes {
                            match oxide_infer_core::image::load_from_memory(img_bytes) {
                                Ok(dyn_img) => {
                                    match preprocessor.process_image(&dyn_img) {
                                        Ok(patchified) => {
                                            match DeviceBuffer::from_slice(&gpu_clone, &patchified.patches) {
                                                Ok(d_patches) => {
                                                    let num_tokens = (patchified.grid_h * patchified.grid_w) / 4;
                                                    let mut d_proj = DeviceBuffer::<half::f16>::zeroed(&gpu_clone, num_tokens * 2048)
                                                        .expect("alloc vision proj");
                                                    if let Err(e) = vision_tower.forward(&gpu_clone, &d_patches, &mut d_proj) {
                                                        eprintln!("[vision] VisionTower error: {}", e);
                                                        None
                                                    } else {
                                                        let _ = stream.synchronize();
                                                        let vis_host = d_proj.inner().to_host_vec(stream).unwrap_or_default();
                                                        let embed_table_host = embed_clone.table.inner().to_host_vec(stream).unwrap_or_default();
                                                        
                                                        // Expand <|image|> token placeholders if needed or replace canonical count
                                                        let mut expanded_tokens = Vec::with_capacity(token_ids.len() + num_tokens);
                                                        let mut seen_first_img = false;
                                                        for &tok in &token_ids {
                                                            if tok == 125030 {
                                                                if !seen_first_img {
                                                                    expanded_tokens.extend(vec![125030; num_tokens]);
                                                                    seen_first_img = true;
                                                                }
                                                            } else {
                                                                expanded_tokens.push(tok);
                                                            }
                                                        }
                                                        if seen_first_img {
                                                            token_ids = expanded_tokens;
                                                        }

                                                        let mut seq_embeds = Vec::with_capacity(token_ids.len() * 2048);
                                                        let mut v_idx = 0usize;
                                                        for &tok in &token_ids {
                                                            if tok == 125030 && v_idx < num_tokens {
                                                                let slice = &vis_host[v_idx * 2048..(v_idx + 1) * 2048];
                                                                seq_embeds.extend_from_slice(slice);
                                                                v_idx += 1;
                                                            } else {
                                                                let tidx = tok as usize;
                                                                if (tidx + 1) * 2048 <= embed_table_host.len() {
                                                                    let slice = &embed_table_host[tidx * 2048..(tidx + 1) * 2048];
                                                                    seq_embeds.extend_from_slice(slice);
                                                                } else {
                                                                    seq_embeds.extend(vec![half::f16::ZERO; 2048]);
                                                                }
                                                            }
                                                        }
                                                        Some(seq_embeds)
                                                    }
                                                }
                                                Err(e) => {
                                                    eprintln!("[vision] patch alloc error: {}", e);
                                                    None
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!("[vision] preprocessor error: {}", e);
                                            None
                                        }
                                    }
                                }
                                Err(e) => {
                                    eprintln!("[vision] image decode error: {}", e);
                                    None
                                }
                            }
                        } else {
                            None
                        };

                        let prompt_len = token_ids.len();
                        let cur_token = *token_ids.last().unwrap();
                        let max_gen = req.max_tokens;
                        if debug_top {
                            let ids: Vec<u32> = token_ids.iter().cloned().collect();
                            eprintln!("[tok] n={} ids={:?}", prompt_len, ids);
                        }

                        let mut k_caches = Vec::with_capacity(24);
                        let mut v_caches = Vec::with_capacity(24);
                        let mut conv_states = Vec::with_capacity(24);
                        for idx in 0..24 {
                            if layer_types[idx] == "conv" {
                                k_caches.push(None);
                                v_caches.push(None);
                                conv_states.push(Some(DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 2048 * 7).expect("alloc")));
                            } else {
                                k_caches.push(Some(DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 1024 * 512).expect("alloc")));
                                v_caches.push(Some(DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 1024 * 512).expect("alloc")));
                                conv_states.push(None);
                            }
                        }

                        let ctx = ActiveContext {
                            req,
                            prompt_len,
                            cur_token,
                            gen_tokens: Vec::new(),
                            gen_count: 0,
                            max_gen,
                            d_hidden: DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 2048).expect("alloc"),
                            scratch_norm: DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 2048).expect("alloc"),
                            scratch_layer: DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 2048).expect("alloc"),
                            q_out: DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 16 * 128).expect("alloc"),
                            k_out: DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 4 * 128).expect("alloc"),
                            v_out: DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 4 * 128).expect("alloc"),
                            gate_out: DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 16).expect("alloc"),
                            d_normed_final: DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 2048).expect("alloc"),
                            d_tok: DeviceBuffer::<u32>::zeroed(&gpu_clone, 1).expect("alloc"),
                            layer_k_caches: k_caches,
                            layer_v_caches: v_caches,
                            layer_conv_states: conv_states,
                        };
                        new_contexts.push((ctx, token_ids, custom_embeds));
                    }
                    None => break,
                }
            }

            while active_contexts.len() + new_contexts.len() < MAX_BATCH_SIZE {
                match req_rx.try_recv() {
                    Ok(req) => {
                        let mut token_ids = tokenizer_clone.encode(&req.prompt).unwrap_or_else(|_| vec![124894, 2048]);
                        
                        let custom_embeds = if let Some(ref img_bytes) = req.image_bytes {
                            match oxide_infer_core::image::load_from_memory(img_bytes) {
                                Ok(dyn_img) => {
                                    match preprocessor.process_image(&dyn_img) {
                                        Ok(patchified) => {
                                            match DeviceBuffer::from_slice(&gpu_clone, &patchified.patches) {
                                                Ok(d_patches) => {
                                                    let num_tokens = (patchified.grid_h * patchified.grid_w) / 4;
                                                    let mut d_proj = DeviceBuffer::<half::f16>::zeroed(&gpu_clone, num_tokens * 2048)
                                                        .expect("alloc vision proj");
                                                    if let Err(e) = vision_tower.forward(&gpu_clone, &d_patches, &mut d_proj) {
                                                        eprintln!("[vision] VisionTower error: {}", e);
                                                        None
                                                    } else {
                                                        let _ = stream.synchronize();
                                                        let vis_host = d_proj.inner().to_host_vec(stream).unwrap_or_default();
                                                        let embed_table_host = embed_clone.table.inner().to_host_vec(stream).unwrap_or_default();
                                                        
                                                        let mut expanded_tokens = Vec::with_capacity(token_ids.len() + num_tokens);
                                                        let mut seen_first_img = false;
                                                        for &tok in &token_ids {
                                                            if tok == 125030 {
                                                                if !seen_first_img {
                                                                    expanded_tokens.extend(vec![125030; num_tokens]);
                                                                    seen_first_img = true;
                                                                }
                                                            } else {
                                                                expanded_tokens.push(tok);
                                                            }
                                                        }
                                                        if seen_first_img {
                                                            token_ids = expanded_tokens;
                                                        }

                                                        let mut seq_embeds = Vec::with_capacity(token_ids.len() * 2048);
                                                        let mut v_idx = 0usize;
                                                        for &tok in &token_ids {
                                                            if tok == 125030 && v_idx < num_tokens {
                                                                let slice = &vis_host[v_idx * 2048..(v_idx + 1) * 2048];
                                                                seq_embeds.extend_from_slice(slice);
                                                                v_idx += 1;
                                                            } else {
                                                                let tidx = tok as usize;
                                                                if (tidx + 1) * 2048 <= embed_table_host.len() {
                                                                    let slice = &embed_table_host[tidx * 2048..(tidx + 1) * 2048];
                                                                    seq_embeds.extend_from_slice(slice);
                                                                } else {
                                                                    seq_embeds.extend(vec![half::f16::ZERO; 2048]);
                                                                }
                                                            }
                                                        }
                                                        Some(seq_embeds)
                                                    }
                                                }
                                                Err(e) => {
                                                    eprintln!("[vision] patch alloc error: {}", e);
                                                    None
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!("[vision] preprocessor error: {}", e);
                                            None
                                        }
                                    }
                                }
                                Err(e) => {
                                    eprintln!("[vision] image decode error: {}", e);
                                    None
                                }
                            }
                        } else {
                            None
                        };

                        let prompt_len = token_ids.len();
                        let cur_token = *token_ids.last().unwrap();
                        let max_gen = req.max_tokens;
                        if debug_top {
                            let ids: Vec<u32> = token_ids.iter().cloned().collect();
                            eprintln!("[tok] n={} ids={:?}", prompt_len, ids);
                        }

                        let mut k_caches = Vec::with_capacity(24);
                        let mut v_caches = Vec::with_capacity(24);
                        let mut conv_states = Vec::with_capacity(24);
                        for idx in 0..24 {
                            if layer_types[idx] == "conv" {
                                k_caches.push(None);
                                v_caches.push(None);
                                conv_states.push(Some(DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 2048 * 7).expect("alloc")));
                            } else {
                                k_caches.push(Some(DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 1024 * 512).expect("alloc")));
                                v_caches.push(Some(DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 1024 * 512).expect("alloc")));
                                conv_states.push(None);
                            }
                        }

                        let ctx = ActiveContext {
                            req,
                            prompt_len,
                            cur_token,
                            gen_tokens: Vec::new(),
                            gen_count: 0,
                            max_gen,
                            d_hidden: DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 2048).expect("alloc"),
                            scratch_norm: DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 2048).expect("alloc"),
                            scratch_layer: DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 2048).expect("alloc"),
                            q_out: DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 16 * 128).expect("alloc"),
                            k_out: DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 4 * 128).expect("alloc"),
                            v_out: DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 4 * 128).expect("alloc"),
                            gate_out: DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 16).expect("alloc"),
                            d_normed_final: DeviceBuffer::<half::f16>::zeroed(&gpu_clone, 2048).expect("alloc"),
                            d_tok: DeviceBuffer::<u32>::zeroed(&gpu_clone, 1).expect("alloc"),
                            layer_k_caches: k_caches,
                            layer_v_caches: v_caches,
                            layer_conv_states: conv_states,
                        };
                        new_contexts.push((ctx, token_ids, custom_embeds));
                    }
                    Err(_) => break,
                }
            }

            // ---- Fully batched engine path ----
            if let Some(engine) = batch_engine.as_mut() {
                if !new_contexts.is_empty() {
                    let prefill_t0 = if profile {
                        let _ = stream.synchronize();
                        Some(Instant::now())
                    } else {
                        None
                    };
                    // Token-batched prefill: each wave processes up to
                    // `max_batch` prompt tokens as one row-batch, so every
                    // layer's weights are streamed once per wave instead of
                    // once per token. Prompts longer than a wave are split
                    // into contiguous chunks; the request's persistent conv
                    // state carries the 6-token window across chunk bounds.
                    new_contexts.sort_by_key(|(_, t, _)| t.len());
                    let mb = engine.max_batch();
                    // (request index, chunk start) in execution order.
                    let mut chunks: Vec<(usize, usize)> = Vec::new();
                    for (ri, (_, t, _)) in new_contexts.iter().enumerate() {
                        let mut start = 0usize;
                        while start < t.len() {
                            chunks.push((ri, start));
                            start += mb;
                        }
                    }
                    let mut ci = 0usize;
                    while ci < chunks.len() {
                        // Pack whole chunks into this wave, preserving
                        // contiguity of each request's chunk rows.
                        let mut wave: Vec<(usize, usize, usize)> = Vec::new(); // (ri, start, len)
                        let mut used = 0usize;
                        while ci + wave.len() < chunks.len() {
                            let (ri, start) = chunks[ci + wave.len()];
                            let len = (new_contexts[ri].1.len() - start).min(mb);
                            if used + len > mb {
                                break;
                            }
                            wave.push((ri, start, len));
                            used += len;
                        }
                        if wave.is_empty() {
                            continue;
                        }
                        let mut tokens: Vec<u32> = Vec::with_capacity(used);
                        let mut pos_ids: Vec<u32> = Vec::with_capacity(used);
                        let mut origins: Vec<u32> = Vec::with_capacity(used);
                        let mut local_pos: Vec<u32> = Vec::with_capacity(used);
                        let mut row_ctx: Vec<usize> = Vec::with_capacity(used);
                        for (ri, start, len) in wave.iter() {
                            let origin = tokens.len() as u32;
                            for j in 0..*len {
                                tokens.push(new_contexts[*ri].1[start + j]);
                                pos_ids.push((start + j) as u32);
                                origins.push(origin);
                                local_pos.push(j as u32);
                                row_ctx.push(*ri);
                            }
                        }
                        // Per-layer carry pointers: each row uses its
                        // request's persistent conv state for that layer.
                        let mut carry_host = vec![0u64; 24 * mb];
                        for l in 0..layers.len() {
                            if is_conv[l] {
                                for (r, ri) in row_ctx.iter().enumerate() {
                                    carry_host[l * used + r] = new_contexts[*ri]
                                        .0
                                        .conv_state_ptr(l);
                                }
                            }
                        }
                        let meta = PrefillMeta {
                            carry_ptrs: carry_host,
                            origins,
                            local_pos,
                        };
                        let ctx_refs: Vec<&ActiveContext> =
                            row_ctx.iter().map(|ri| &new_contexts[*ri].0).collect();
                        let mut wave_has_custom_embeds = false;
                        for (ri, _, _) in wave.iter() {
                            if new_contexts[*ri].2.is_some() {
                                wave_has_custom_embeds = true;
                                break;
                            }
                        }

                        let d_wave_embeds = if wave_has_custom_embeds {
                            let embed_table_host = embed_clone.table.inner().to_host_vec(stream).unwrap_or_default();
                            let mut wave_embed_vec = Vec::with_capacity(used * 2048);
                            for (ri, start, len) in wave.iter() {
                                if let Some(ref custom_vec) = new_contexts[*ri].2 {
                                    let slice = &custom_vec[start * 2048..(start + len) * 2048];
                                    wave_embed_vec.extend_from_slice(slice);
                                } else {
                                    for j in 0..*len {
                                        let tok = new_contexts[*ri].1[start + j] as usize;
                                        if (tok + 1) * 2048 <= embed_table_host.len() {
                                            wave_embed_vec.extend_from_slice(&embed_table_host[tok * 2048..(tok + 1) * 2048]);
                                        } else {
                                            wave_embed_vec.extend(vec![half::f16::ZERO; 2048]);
                                        }
                                    }
                                }
                            }
                            Some(DeviceBuffer::from_slice(&gpu_clone, &wave_embed_vec).expect("alloc wave embeds"))
                        } else {
                            None
                        };

                        engine
                            .forward_step_with_embeds(
                                &gpu_clone,
                                &mut layers,
                                &is_conv,
                                &ctx_refs,
                                &tokens,
                                &pos_ids,
                                &embed_clone.table,
                                &emb_norm_clone,
                                false,
                                Some(&meta),
                                d_wave_embeds.as_ref(),
                            )
                            .expect("batched prefill");
                        // Copy each chunk's last-row conv state back into the
                        // request's persistent buffers (decode-ready state).
                        for (ri, start, len) in wave.iter() {
                            // wave row index of this chunk's last token
                            let last_row = {
                                let mut acc = 0usize;
                                let mut found = 0usize;
                                for (ri2, start2, len2) in wave.iter() {
                                    if ri2 == ri && start2 == start {
                                        found = acc + len2 - 1;
                                        break;
                                    }
                                    acc += len2;
                                }
                                found
                            };
                            for l in 0..layers.len() {
                                if is_conv[l] {
                                    let dst = new_contexts[*ri]
                                        .0
                                        .conv_state_ptr(l);
                                    let src = engine.prefill_state_ptr(l, last_row);
                                    unsafe {
                                        let _ = cuMemcpyDtoDAsync_v2(
                                            dst,
                                            src,
                                            2048 * 6 * 2,
                                            stream.cu_stream() as *mut c_void,
                                        );
                                    }
                                }
                            }
                        }
                        ci += wave.len();
                    }
                    if let Some(t0) = prefill_t0 {
                        let _ = stream.synchronize();
                        eprintln!(
                            "[profile] engine prefill: {:.1} ms",
                            t0.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                    for (ctx, _, _) in new_contexts.drain(..) {
                        active_contexts.push(ctx);
                    }

                    // Sample the first token of freshly prefilled sequences
                    // directly from their persisted final-prompt hidden
                    // (ctx.d_hidden, written by the last prefill forward_step).
                    // Without this, a "decode" step would re-embed the
                    // prompt's last token at position prompt_len
                    // (double-processing) and shift every later position.
                    let mut pending_idx: Vec<usize> = active_contexts
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| c.gen_count == 0)
                        .map(|(i, _)| i)
                        .collect();
                    if !pending_idx.is_empty() {
                        let pending_refs: Vec<&ActiveContext> =
                            pending_idx.iter().map(|&i| &active_contexts[i]).collect();
                        let first_tokens = engine
                            .sample_only(&gpu_clone, &pending_refs, &embed_clone.table, &emb_norm_clone)
                            .expect("post-prefill sample");
                        for (j, &idx) in pending_idx.iter().enumerate() {
                            let tok = first_tokens[j];
                            let ctx = &mut active_contexts[idx];
                            ctx.gen_tokens.push(tok);
                            ctx.cur_token = tok;
                            ctx.gen_count = 1;
                        }
                        // Short-circuit completions (EOS or max_tokens == 1).
                        let mut early_done: Vec<usize> = Vec::new();
                        for &idx in pending_idx.iter() {
                            let ctx = &mut active_contexts[idx];
                            let tok = *ctx.gen_tokens.last().unwrap();
                            if ctx.gen_count >= ctx.max_gen || tok == 124900 || tok == 124896 {
                                early_done.push(idx);
                            }
                        }
                        let _ = stream.synchronize();
                        for &idx in early_done.iter().rev() {
                            let ctx = active_contexts.remove(idx);
                            let decoded =
                                tokenizer_clone.decode(&ctx.gen_tokens).unwrap_or_default();
                            let _ = ctx
                                .req
                                .response_tx
                                .send((decoded, ctx.prompt_len, ctx.gen_tokens.len()));
                        }
                    }
                }

                let b = active_contexts.len();
                if b == 0 {
                    continue;
                }

                let step_t0 = if profile {
                    let _ = stream.synchronize();
                    Some(Instant::now())
                } else {
                    None
                };
                let tokens: Vec<u32> = active_contexts.iter().map(|c| c.cur_token).collect();
                // Every active sequence has gen_count >= 1 here; a decode
                // step processes the previously generated token at its own
                // position: prompt_len + gen_count - 1.
                let pos_ids: Vec<u32> = active_contexts
                    .iter()
                    .map(|c| (c.prompt_len + c.gen_count - 1) as u32)
                    .collect();
                let next_tokens = {
                    let ctx_refs: Vec<&ActiveContext> = active_contexts.iter().collect();
                    engine
                        .forward_step(
                            &gpu_clone,
                            &mut layers,
                            &is_conv,
                            &ctx_refs,
                            &tokens,
                            &pos_ids,
                            &embed_clone.table,
                            &emb_norm_clone,
                            true,
                            None,
                        )
                        .expect("batched decode")
                };
                if let Some(t0) = step_t0 {
                    let _ = stream.synchronize();
                    eprintln!(
                        "[profile] engine step B={}: {:.2} ms",
                        b,
                        t0.elapsed().as_secs_f64() * 1000.0
                    );
                }

                let mut completed_indices = Vec::new();
                for (idx, ctx) in active_contexts.iter_mut().enumerate() {
                    let next_tok = next_tokens[idx];
                    ctx.gen_tokens.push(next_tok);
                    ctx.cur_token = next_tok;
                    ctx.gen_count += 1;
                    if ctx.gen_count >= ctx.max_gen || next_tok == 124900 || next_tok == 124896 {
                        completed_indices.push(idx);
                    }
                }

                if !completed_indices.is_empty() {
                    let _ = stream.synchronize();
                    for &idx in completed_indices.iter().rev() {
                        let ctx = active_contexts.remove(idx);
                        let decoded = tokenizer_clone.decode(&ctx.gen_tokens).unwrap_or_default();
                        let _ = ctx
                            .req
                            .response_tx
                            .send((decoded, ctx.prompt_len, ctx.gen_tokens.len()));
                    }
                }
                continue;
            }

            // 2. High-Fidelity Batched Prefill for all newly admitted requests
            let prefill_t0 = if profile && !new_contexts.is_empty() {
                let _ = stream.synchronize();
                Some(Instant::now())
            } else {
                None
            };
            if !new_contexts.is_empty() {
                let nb = new_contexts.len();
                if nb == 1 || scalar_prefill {
                    // Scalar reference path: process each admitted request independently.
                    for (ctx, token_ids, _) in new_contexts.iter_mut() {
                        for (pos_id, &tok) in token_ids.iter().enumerate() {
                            let _ = ctx.d_tok.copy_from_slice(&gpu_clone, &[tok]);
                            let _ = embed_clone.lookup(&gpu_clone, &ctx.d_tok, 1, &mut ctx.d_hidden);
                            let dump_this_pos =
                                dump_hidden.is_some() && pos_id + 1 == token_ids.len();
                            if let Some(dir) = &dump_hidden {
                                if dump_this_pos {
                                    if let Ok(host) = ctx.d_hidden.inner().to_host_vec(stream) {
                                        let bytes = unsafe {
                                            std::slice::from_raw_parts(
                                                host.as_ptr() as *const u8,
                                                host.len() * 2,
                                            )
                                        };
                                        let _ = std::fs::write(
                                            format!("{}/pos{}_emb.bin", dir, pos_id),
                                            bytes,
                                        );
                                    }
                                }
                            }
                            for (l_idx, layer) in layers.iter_mut().enumerate() {
                                let _ = layer.forward_pre_ffn_with_cache(
                                    &gpu_clone,
                                    &mut ctx.d_hidden,
                                    &mut ctx.scratch_norm,
                                    &mut ctx.scratch_layer,
                                    &mut ctx.q_out,
                                    &mut ctx.k_out,
                                    &mut ctx.v_out,
                                    &mut ctx.gate_out,
                                    pos_id,
                                    ctx.layer_k_caches[l_idx].as_mut(),
                                    ctx.layer_v_caches[l_idx].as_mut(),
                                    ctx.layer_conv_states[l_idx].as_mut(),
                                );
                                let _ = layer.ffn.forward(&gpu_clone, &ctx.scratch_norm, &mut ctx.scratch_layer);
                                let _ = layer.forward_post_ffn(&gpu_clone, &mut ctx.d_hidden, &mut ctx.scratch_layer);
                                if dump_this_pos {
                                    if let Some(dir) = &dump_hidden {
                                        let _ = stream.synchronize();
                                        if let Ok(host) = ctx.d_hidden.inner().to_host_vec(stream) {
                                            let bytes = unsafe {
                                                std::slice::from_raw_parts(
                                                    host.as_ptr() as *const u8,
                                                    host.len() * 2,
                                                )
                                            };
                                            let _ = std::fs::write(
                                                format!("{}/pos{}_layer{}.bin", dir, pos_id, l_idx),
                                                bytes,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // Batched Prefill with Tensor Core HGEMM across all newly admitted requests
                    let max_plen = new_contexts.iter().map(|(_, t, _)| t.len()).max().unwrap_or(0);
                    for pos_id in 0..max_plen {
                        // (a) Embedding lookup for active positions
                        for (ctx, token_ids, _) in new_contexts.iter_mut() {
                            if pos_id < token_ids.len() {
                                let tok = token_ids[pos_id];
                                let _ = ctx.d_tok.copy_from_slice(&gpu_clone, &[tok]);
                                let _ = embed_clone.lookup(&gpu_clone, &ctx.d_tok, 1, &mut ctx.d_hidden);
                            }
                        }

                        // (b) Forward through 24 layers with batched FFN
                        for (l_idx, layer) in layers.iter_mut().enumerate() {
                            let mut active_prefill_indices: Vec<usize> = Vec::with_capacity(nb);
                            for (i, (ctx, token_ids, _)) in new_contexts.iter_mut().enumerate() {
                                if pos_id < token_ids.len() {
                                    active_prefill_indices.push(i);
                                    let _ = layer.forward_pre_ffn_with_cache(
                                        &gpu_clone,
                                        &mut ctx.d_hidden,
                                        &mut ctx.scratch_norm,
                                        &mut ctx.scratch_layer,
                                        &mut ctx.q_out,
                                        &mut ctx.k_out,
                                        &mut ctx.v_out,
                                        &mut ctx.gate_out,
                                        pos_id,
                                        ctx.layer_k_caches[l_idx].as_mut(),
                                        ctx.layer_v_caches[l_idx].as_mut(),
                                        ctx.layer_conv_states[l_idx].as_mut(),
                                    );
                                }
                            }

                            let cur_b = active_prefill_indices.len();
                            if cur_b == 1 {
                                let i = active_prefill_indices[0];
                                let ctx = &mut new_contexts[i].0;
                                let _ = layer.ffn.forward(&gpu_clone, &ctx.scratch_norm, &mut ctx.scratch_layer);
                                let _ = layer.forward_post_ffn(&gpu_clone, &mut ctx.d_hidden, &mut ctx.scratch_layer);
                            } else if cur_b > 1 {
                                let d_batch_x_ptr = d_batch_x.inner_mut().cu_deviceptr();
                                for (slot_idx, &ctx_idx) in active_prefill_indices.iter().enumerate() {
                                    let dst = d_batch_x_ptr + (slot_idx * 2048 * 2) as u64;
                                    let src = new_contexts[ctx_idx].0.scratch_norm.inner().cu_deviceptr();
                                    unsafe {
                                        let _ = cuMemcpyDtoDAsync_v2(dst, src, 2048 * 2, stream.cu_stream() as *mut c_void);
                                    }
                                }

                                unsafe {
                                    cublasHgemm(
                                        cublas_handle,
                                        op_t,
                                        op_n,
                                        7680,
                                        cur_b as i32,
                                        2048,
                                        &alpha,
                                        layer.ffn.w1_gate.inner().cu_deviceptr() as *const c_void,
                                        2048,
                                        d_batch_x.inner().cu_deviceptr() as *const c_void,
                                        2048,
                                        &beta,
                                        d_batch_gate.inner_mut().cu_deviceptr() as *mut c_void,
                                        7680,
                                    );

                                    cublasHgemm(
                                        cublas_handle,
                                        op_t,
                                        op_n,
                                        7680,
                                        cur_b as i32,
                                        2048,
                                        &alpha,
                                        layer.ffn.w3_up.inner().cu_deviceptr() as *const c_void,
                                        2048,
                                        d_batch_x.inner().cu_deviceptr() as *const c_void,
                                        2048,
                                        &beta,
                                        d_batch_up.inner_mut().cu_deviceptr() as *mut c_void,
                                        7680,
                                    );
                                }

                                let mut d_act_ptr = d_batch_act.inner_mut().cu_deviceptr();
                                let mut d_gate_ptr = d_batch_gate.inner().cu_deviceptr();
                                let mut d_up_ptr = d_batch_up.inner().cu_deviceptr();
                                let mut inter_u32 = 7680u32;
                                let mut b_u32 = cur_b as u32;

                                let mut swiglu_args: [*mut c_void; 5] = [
                                    &mut d_act_ptr as *mut _ as *mut c_void,
                                    &mut d_gate_ptr as *mut _ as *mut c_void,
                                    &mut d_up_ptr as *mut _ as *mut c_void,
                                    &mut inter_u32 as *mut _ as *mut c_void,
                                    &mut b_u32 as *mut _ as *mut c_void,
                                ];
                                let block_dim = 256;
                                let grid_x = (inter_u32 + block_dim - 1) / block_dim;
                                let grid_y = b_u32;
                                unsafe {
                                    let _ = launch_kernel_on_stream(&swiglu_func, (grid_x, grid_y, 1), (block_dim, 1, 1), 0, stream, &mut swiglu_args);
                                }

                                unsafe {
                                    cublasHgemm(
                                        cublas_handle,
                                        op_t,
                                        op_n,
                                        2048,
                                        cur_b as i32,
                                        7680,
                                        &alpha,
                                        layer.ffn.w2_down.inner().cu_deviceptr() as *const c_void,
                                        7680,
                                        d_batch_act.inner().cu_deviceptr() as *const c_void,
                                        7680,
                                        &beta,
                                        d_batch_ffn_out.inner_mut().cu_deviceptr() as *mut c_void,
                                        2048,
                                    );
                                }

                                let d_batch_ffn_ptr = d_batch_ffn_out.inner().cu_deviceptr();
                                for (slot_idx, &ctx_idx) in active_prefill_indices.iter().enumerate() {
                                    let src = d_batch_ffn_ptr + (slot_idx * 2048 * 2) as u64;
                                    let dst = new_contexts[ctx_idx].0.scratch_layer.inner_mut().cu_deviceptr();
                                    unsafe {
                                        let _ = cuMemcpyDtoDAsync_v2(dst, src, 2048 * 2, stream.cu_stream() as *mut c_void);
                                    }
                                    let ctx = &mut new_contexts[ctx_idx].0;
                                    let _ = layer.forward_post_ffn(&gpu_clone, &mut ctx.d_hidden, &mut ctx.scratch_layer);
                                }
                            }
                        }
                    }
                }

                for (ctx, _, _) in new_contexts {
                    active_contexts.push(ctx);
                }

                // Sample the first token of freshly prefilled sequences
                // directly from their prefill-final hidden (ctx.d_hidden).
                // Without this, the decode step below would re-embed the
                // prompt's last token at position prompt_len
                // (double-processing) and shift every later position.
                let mut legacy_pending: Vec<usize> = active_contexts
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.gen_count == 0)
                    .map(|(i, _)| i)
                    .collect();
                for &idx in legacy_pending.iter() {
                    let ctx = &mut active_contexts[idx];
                    let mut d_out_final_ptr = ctx.d_normed_final.inner_mut().cu_deviceptr();
                    let mut d_hid_ptr = ctx.d_hidden.inner().cu_deviceptr();
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
                    let _ = lm_head.compute_logits(&gpu_clone, &ctx.d_normed_final, &embed_clone.table, &mut d_logits);
                    if debug_top {
                        let _ = stream.synchronize();
                        if let Ok(host_logits) = d_logits.inner().to_host_vec(stream) {
                            let mut order: Vec<usize> = (0..host_logits.len()).collect();
                            order.sort_unstable_by(|&a, &b| {
                                host_logits[b].partial_cmp(&host_logits[a]).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let top: Vec<(usize, f32)> = order
                                .iter()
                                .take(10)
                                .map(|&i| (i, host_logits[i]))
                                .collect();
                            eprintln!("[top] post-prefill pos={} {:?}", ctx.prompt_len - 1, top);
                        }
                    }
                    match lm_head.sample_argmax(&gpu_clone, &d_logits) {
                        Ok(next_tok) => {
                            ctx.gen_tokens.push(next_tok);
                            ctx.cur_token = next_tok;
                            ctx.gen_count = 1;
                        }
                        Err(_) => {}
                    }
                }
                // Short-circuit completions from the post-prefill sample.
                let mut legacy_early_done: Vec<usize> = Vec::new();
                for &idx in legacy_pending.iter() {
                    let ctx = &mut active_contexts[idx];
                    let tok = *ctx.gen_tokens.last().unwrap();
                    if ctx.gen_count >= ctx.max_gen || tok == 124900 || tok == 124896 {
                        legacy_early_done.push(idx);
                    }
                }
                if !legacy_early_done.is_empty() {
                    let _ = stream.synchronize();
                    for &idx in legacy_early_done.iter().rev() {
                        let ctx = active_contexts.remove(idx);
                        let decoded = tokenizer_clone.decode(&ctx.gen_tokens).unwrap_or_default();
                        let _ = ctx
                            .req
                            .response_tx
                            .send((decoded, ctx.prompt_len, ctx.gen_tokens.len()));
                    }
                }
            }
            if let Some(t0) = prefill_t0 {
                let _ = stream.synchronize();
                eprintln!("[profile] batched prefill: {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0);
            }

            let b = active_contexts.len();
            if b == 0 {
                continue;
            }

            // 3. Batched Embedding lookup
            let mut prof_mark = Instant::now();
            if profile {
                let _ = stream.synchronize();
                prof_mark = Instant::now();
            }
            for ctx in active_contexts.iter_mut() {
                let _ = ctx.d_tok.copy_from_slice(&gpu_clone, &[ctx.cur_token]);
                let _ = embed_clone.lookup(&gpu_clone, &ctx.d_tok, 1, &mut ctx.d_hidden);
            }
            if profile {
                let _ = stream.synchronize();
                prof_embed += prof_mark.elapsed().as_secs_f64();
                prof_mark = Instant::now();
            }

            // 4. Decode 1 token step through all 24 layers
            // FFN blocks are computed in full batched Tensor Core HGEMMs
            for (l_idx, layer) in layers.iter_mut().enumerate() {
                // (a) Pre-FFN: Operator Norm, Conv/Attn Projections & GQA, Post-Op Norm, Residual 1 + FFN Norm
                if b > 1 && !scalar_conv && layer_types[l_idx] == "conv" {
                    // Optimized Batched Conv Acceleration (cuBLAS HGEMM in_proj + 2D vectorized conv + cuBLAS HGEMM out_proj)
                    let rms_func_layer = layer.add_ptx.get_function("rms_norm_vec8_fp16").expect("rms");
                    let fused_res_norm_func = layer.add_ptx.get_function("fused_residual_rms_norm_vec8_fp16").expect("res_norm");

                    // 1. operator_norm on each sequence
                    for ctx in active_contexts.iter_mut() {
                        let mut d_normed = ctx.scratch_norm.inner_mut().cu_deviceptr();
                        let mut d_hidden = ctx.d_hidden.inner().cu_deviceptr();
                        let mut d_op_w = layer.operator_norm.inner().cu_deviceptr();
                        let mut args_norm: [*mut std::ffi::c_void; 5] = [
                            &mut d_normed as *mut _ as *mut std::ffi::c_void,
                            &mut d_hidden as *mut _ as *mut std::ffi::c_void,
                            &mut d_op_w as *mut _ as *mut std::ffi::c_void,
                            &mut epsilon as *mut _ as *mut std::ffi::c_void,
                            &mut h_size as *mut _ as *mut std::ffi::c_void,
                        ];
                        unsafe {
                            let _ = launch_kernel_on_stream(&rms_func_layer, (1, 1, 1), (256, 1, 1), 0, stream, &mut args_norm);
                        }
                    }

                    // Pack scratch_norm into d_batch_x
                    let d_batch_x_ptr = d_batch_x.inner_mut().cu_deviceptr();
                    for (i, ctx) in active_contexts.iter().enumerate() {
                        let dst = d_batch_x_ptr + (i * 2048 * 2) as u64;
                        let src = ctx.scratch_norm.inner().cu_deviceptr();
                        unsafe {
                            let _ = cuMemcpyDtoDAsync_v2(dst, src, 2048 * 2, stream.cu_stream() as *mut c_void);
                        }
                    }

                    // 2. Batched in_proj HGEMM
                    let (in_w, out_w, cw) = match &layer.inner_layer {
                        LayerType::Conv(c) => (c.in_proj.inner().cu_deviceptr(), c.out_proj.inner().cu_deviceptr(), c.conv_weight.inner().cu_deviceptr()),
                        _ => panic!(),
                    };

                    unsafe {
                        cublasHgemm(
                            cublas_handle, op_t, op_n, 6144, b as i32, 2048,
                            &alpha, in_w as *const c_void, 2048,
                            d_batch_x.inner().cu_deviceptr() as *const c_void, 2048,
                            &beta, d_batch_bcx.inner_mut().cu_deviceptr() as *mut c_void, 6144,
                        );
                    }

                    // Pack conv states into d_batch_state_pool
                    let d_state_pool_ptr = d_batch_state_pool.inner_mut().cu_deviceptr();
                    for (i, ctx) in active_contexts.iter().enumerate() {
                        let dst = d_state_pool_ptr + (i * 2048 * 6 * 2) as u64;
                        let src = ctx.layer_conv_states[l_idx].as_ref().unwrap().inner().cu_deviceptr();
                        unsafe {
                            let _ = cuMemcpyDtoDAsync_v2(dst, src, 2048 * 6 * 2, stream.cu_stream() as *mut c_void);
                        }
                    }

                    // 3. Batched Fused Conv Kernel
                    let mut d_y_ptr = d_batch_conv_y.inner_mut().cu_deviceptr();
                    let mut d_cache_ptr = d_batch_state_pool.inner_mut().cu_deviceptr();
                    let mut d_bcx_ptr = d_batch_bcx.inner().cu_deviceptr();
                    let mut d_cw_ptr = cw;
                    let mut k_len_u32 = 7u32;
                    let mut b_u32 = b as u32;

                    let mut args_conv: [*mut std::ffi::c_void; 7] = [
                        &mut d_y_ptr as *mut _ as *mut std::ffi::c_void,
                        &mut d_cache_ptr as *mut _ as *mut std::ffi::c_void,
                        &mut d_bcx_ptr as *mut _ as *mut std::ffi::c_void,
                        &mut d_cw_ptr as *mut _ as *mut std::ffi::c_void,
                        &mut h_size as *mut _ as *mut std::ffi::c_void,
                        &mut k_len_u32 as *mut _ as *mut std::ffi::c_void,
                        &mut b_u32 as *mut _ as *mut std::ffi::c_void,
                    ];
                    let block_dim = 128u32;
                    let grid_x = (h_size + block_dim - 1) / block_dim;
                    let grid_y = b_u32;
                    unsafe {
                        let _ = launch_kernel_on_stream(&batched_conv_func, (grid_x, grid_y, 1), (block_dim, 1, 1), 0, stream, &mut args_conv);
                    }

                    // Copy updated conv state back to contexts
                    for (i, ctx) in active_contexts.iter_mut().enumerate() {
                        let src = d_state_pool_ptr + (i * 2048 * 6 * 2) as u64;
                        let dst = ctx.layer_conv_states[l_idx].as_mut().unwrap().inner_mut().cu_deviceptr();
                        unsafe {
                            let _ = cuMemcpyDtoDAsync_v2(dst, src, 2048 * 6 * 2, stream.cu_stream() as *mut c_void);
                        }
                    }

                    // 4. Batched out_proj HGEMM directly into d_batch_x
                    unsafe {
                        cublasHgemm(
                            cublas_handle, op_t, op_n, 2048, b as i32, 2048,
                            &alpha, out_w as *const c_void, 2048,
                            d_batch_conv_y.inner().cu_deviceptr() as *const c_void, 2048,
                            &beta, d_batch_x.inner_mut().cu_deviceptr() as *mut c_void, 2048,
                        );
                    }

                    // 5. Unpack out_proj, post_operator_norm, and fused residual 1 + ffn_norm
                    for (i, ctx) in active_contexts.iter_mut().enumerate() {
                        let src = d_batch_x_ptr + (i * 2048 * 2) as u64;
                        let dst = ctx.scratch_layer.inner_mut().cu_deviceptr();
                        unsafe {
                            let _ = cuMemcpyDtoDAsync_v2(dst, src, 2048 * 2, stream.cu_stream() as *mut c_void);
                        }

                        // post_operator_norm
                        let mut d_layer_out = ctx.scratch_layer.inner_mut().cu_deviceptr();
                        let mut d_post_op = layer.post_operator_norm.inner().cu_deviceptr();
                        let mut args_post_op: [*mut std::ffi::c_void; 5] = [
                            &mut d_layer_out as *mut _ as *mut std::ffi::c_void,
                            &mut d_layer_out as *mut _ as *mut std::ffi::c_void,
                            &mut d_post_op as *mut _ as *mut std::ffi::c_void,
                            &mut epsilon as *mut _ as *mut std::ffi::c_void,
                            &mut h_size as *mut _ as *mut std::ffi::c_void,
                        ];
                        unsafe {
                            let _ = launch_kernel_on_stream(&rms_func_layer, (1, 1, 1), (256, 1, 1), 0, stream, &mut args_post_op);
                        }

                        // fused residual 1 + ffn_norm
                        let mut d_hidden_mut = ctx.d_hidden.inner_mut().cu_deviceptr();
                        let mut d_ffn_w = layer.ffn_norm.inner().cu_deviceptr();
                        let mut d_normed = ctx.scratch_norm.inner_mut().cu_deviceptr();
                        let mut args_fused_res1: [*mut std::ffi::c_void; 6] = [
                            &mut d_hidden_mut as *mut _ as *mut std::ffi::c_void,
                            &mut d_layer_out as *mut _ as *mut std::ffi::c_void,
                            &mut d_normed as *mut _ as *mut std::ffi::c_void,
                            &mut d_ffn_w as *mut _ as *mut std::ffi::c_void,
                            &mut epsilon as *mut _ as *mut std::ffi::c_void,
                            &mut h_size as *mut _ as *mut std::ffi::c_void,
                        ];
                        unsafe {
                            let _ = launch_kernel_on_stream(&fused_res_norm_func, (1, 1, 1), (256, 1, 1), 0, stream, &mut args_fused_res1);
                        }
                    }
                } else {
                    for ctx in active_contexts.iter_mut() {
                        // gen_count >= 1 (first sample happened right after
                        // prefill); process the previously generated token at
                        // its own position.
                        let pos_id = ctx.prompt_len + ctx.gen_count - 1;
                        let _ = layer.forward_pre_ffn_with_cache(
                            &gpu_clone,
                            &mut ctx.d_hidden,
                            &mut ctx.scratch_norm,
                            &mut ctx.scratch_layer,
                            &mut ctx.q_out,
                            &mut ctx.k_out,
                            &mut ctx.v_out,
                            &mut ctx.gate_out,
                            pos_id,
                            ctx.layer_k_caches[l_idx].as_mut(),
                            ctx.layer_v_caches[l_idx].as_mut(),
                            ctx.layer_conv_states[l_idx].as_mut(),
                        );
                    }
                }
                if profile {
                    let _ = stream.synchronize();
                    let dt = prof_mark.elapsed().as_secs_f64();
                    if layer_types[l_idx] == "conv" {
                        prof_conv += dt;
                    } else {
                        prof_attn += dt;
                    }
                    prof_mark = Instant::now();
                }

                if b == 1 || scalar_ffn {
                    // Scalar reference path.
                    for ctx in active_contexts.iter_mut() {
                        let _ = layer.ffn.forward(&gpu_clone, &ctx.scratch_norm, &mut ctx.scratch_layer);
                        let _ = layer.forward_post_ffn(&gpu_clone, &mut ctx.d_hidden, &mut ctx.scratch_layer);
                    }
                } else {
                    // Bulk path for multiple concurrent requests (vLLM parity with Tensor Core HGEMM)
                    // Pack B scratch_norm into d_batch_x
                    let d_batch_x_ptr = d_batch_x.inner_mut().cu_deviceptr();
                    for (i, ctx) in active_contexts.iter().enumerate() {
                        let dst = d_batch_x_ptr + (i * 2048 * 2) as u64;
                        let src = ctx.scratch_norm.inner().cu_deviceptr();
                        unsafe {
                            let _ = cuMemcpyDtoDAsync_v2(dst, src, 2048 * 2, stream.cu_stream() as *mut c_void);
                        }
                    }

                    // W1 (Gate) & W3 (Up) GEMMs
                    unsafe {
                        cublasHgemm(
                            cublas_handle,
                            op_t,
                            op_n,
                            7680,
                            b as i32,
                            2048,
                            &alpha,
                            layer.ffn.w1_gate.inner().cu_deviceptr() as *const c_void,
                            2048,
                            d_batch_x.inner().cu_deviceptr() as *const c_void,
                            2048,
                            &beta,
                            d_batch_gate.inner_mut().cu_deviceptr() as *mut c_void,
                            7680,
                        );

                        cublasHgemm(
                            cublas_handle,
                            op_t,
                            op_n,
                            7680,
                            b as i32,
                            2048,
                            &alpha,
                            layer.ffn.w3_up.inner().cu_deviceptr() as *const c_void,
                            2048,
                            d_batch_x.inner().cu_deviceptr() as *const c_void,
                            2048,
                            &beta,
                            d_batch_up.inner_mut().cu_deviceptr() as *mut c_void,
                            7680,
                        );
                    }

                    // Separate SwiGLU activation: silu(gate) * up
                    let mut d_act_ptr = d_batch_act.inner_mut().cu_deviceptr();
                    let mut d_gate_ptr = d_batch_gate.inner().cu_deviceptr();
                    let mut d_up_ptr = d_batch_up.inner().cu_deviceptr();
                    let mut inter_u32 = 7680u32;
                    let mut b_u32 = b as u32;

                    let mut swiglu_args: [*mut c_void; 5] = [
                        &mut d_act_ptr as *mut _ as *mut c_void,
                        &mut d_gate_ptr as *mut _ as *mut c_void,
                        &mut d_up_ptr as *mut _ as *mut c_void,
                        &mut inter_u32 as *mut _ as *mut c_void,
                        &mut b_u32 as *mut _ as *mut c_void,
                    ];
                    let block_dim = 256;
                    let grid_x = (inter_u32 + block_dim - 1) / block_dim;
                    let grid_y = b_u32;
                    unsafe {
                        let _ = launch_kernel_on_stream(&swiglu_func, (grid_x, grid_y, 1), (block_dim, 1, 1), 0, stream, &mut swiglu_args);
                    }

                    // W2 (Down) GEMM
                    unsafe {
                        cublasHgemm(
                            cublas_handle,
                            op_t,
                            op_n,
                            2048,
                            b as i32,
                            7680,
                            &alpha,
                            layer.ffn.w2_down.inner().cu_deviceptr() as *const c_void,
                            7680,
                            d_batch_act.inner().cu_deviceptr() as *const c_void,
                            7680,
                            &beta,
                            d_batch_ffn_out.inner_mut().cu_deviceptr() as *mut c_void,
                            2048,
                        );
                    }

                    // Unpack d_batch_ffn_out into each context's scratch_layer and run post-FFN norm + residual 2
                    let d_batch_ffn_ptr = d_batch_ffn_out.inner().cu_deviceptr();
                    for (i, ctx) in active_contexts.iter_mut().enumerate() {
                        let src = d_batch_ffn_ptr + (i * 2048 * 2) as u64;
                        let dst = ctx.scratch_layer.inner_mut().cu_deviceptr();
                        unsafe {
                            let _ = cuMemcpyDtoDAsync_v2(dst, src, 2048 * 2, stream.cu_stream() as *mut c_void);
                        }
                        let _ = layer.forward_post_ffn(&gpu_clone, &mut ctx.d_hidden, &mut ctx.scratch_layer);
                    }
                }
                if profile {
                    let _ = stream.synchronize();
                    prof_ffn += prof_mark.elapsed().as_secs_f64();
                    prof_mark = Instant::now();
                }
            }

            // 5. Final Norm, LM Head & Sampling for active contexts
            let mut completed_indices = Vec::new();
            if b == 1 || scalar_lm {
                // Scalar reference path.
                for (idx, ctx) in active_contexts.iter_mut().enumerate() {
                    let mut d_out_final_ptr = ctx.d_normed_final.inner_mut().cu_deviceptr();
                    let mut d_hid_ptr = ctx.d_hidden.inner().cu_deviceptr();
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

                    let _ = lm_head.compute_logits(&gpu_clone, &ctx.d_normed_final, &embed_clone.table, &mut d_logits);
                    if debug_top {
                        let _ = stream.synchronize();
                        if let Ok(host_logits) = d_logits.inner().to_host_vec(stream) {
                            let mut order: Vec<usize> = (0..host_logits.len()).collect();
                            order.sort_unstable_by(|&a, &b| {
                                host_logits[b].partial_cmp(&host_logits[a]).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let top: Vec<(usize, f32)> = order
                                .iter()
                                .take(10)
                                .map(|&i| (i, host_logits[i]))
                                .collect();
                            eprintln!(
                                "[top] step={} pos={} {:?}",
                                ctx.gen_count,
                                ctx.prompt_len + ctx.gen_count,
                                top
                            );
                        }
                    }
                    match lm_head.sample_argmax(&gpu_clone, &d_logits) {
                        Ok(next_tok) => {
                            ctx.gen_tokens.push(next_tok);
                            ctx.cur_token = next_tok;
                            ctx.gen_count += 1;

                            if ctx.gen_count >= ctx.max_gen || next_tok == 124900 || next_tok == 124896 {
                                completed_indices.push(idx);
                            }
                        }
                        Err(_) => completed_indices.push(idx),
                    }
                }
            } else {
                // Bulk path for multiple concurrent requests (cuBLAS HGEMM LM Head + Batched Argmax)
                // (a) Run final RMS norm for each context
                for ctx in active_contexts.iter_mut() {
                    let mut d_out_final_ptr = ctx.d_normed_final.inner_mut().cu_deviceptr();
                    let mut d_hid_ptr = ctx.d_hidden.inner().cu_deviceptr();
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
                }

                // (b) Pack normalized hidden states into d_batch_x
                let d_batch_x_ptr = d_batch_x.inner_mut().cu_deviceptr();
                for (i, ctx) in active_contexts.iter().enumerate() {
                    let dst = d_batch_x_ptr + (i * 2048 * 2) as u64;
                    let src = ctx.d_normed_final.inner().cu_deviceptr();
                    unsafe {
                        let _ = cuMemcpyDtoDAsync_v2(dst, src, 2048 * 2, stream.cu_stream() as *mut c_void);
                    }
                }

                // (c) cuBLAS HGEMM over entire vocab (128,000) for all B requests in 1 single dispatch
                unsafe {
                    cublasHgemm(
                        cublas_handle,
                        op_t,
                        op_n,
                        128000,
                        b as i32,
                        2048,
                        &alpha,
                        embed_clone.table.inner().cu_deviceptr() as *const c_void,
                        2048,
                        d_batch_x.inner().cu_deviceptr() as *const c_void,
                        2048,
                        &beta,
                        d_batch_logits_f16.inner_mut().cu_deviceptr() as *mut c_void,
                        128000,
                    );
                }

                // (d) Direct GPU Batched Argmax Kernel
                let mut d_tok_ptr = d_batch_tokens.inner_mut().cu_deviceptr();
                let mut d_log_ptr = d_batch_logits_f16.inner().cu_deviceptr();
                let mut v_u32 = 128000u32;
                let mut argmax_args: [*mut c_void; 3] = [
                    &mut d_tok_ptr as *mut _ as *mut c_void,
                    &mut d_log_ptr as *mut _ as *mut c_void,
                    &mut v_u32 as *mut _ as *mut c_void,
                ];

                unsafe {
                    let _ = launch_kernel_on_stream(
                        &batched_argmax_func,
                        (b as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        stream,
                        &mut argmax_args,
                    );
                }

                // (e) Copy generated tokens back to host
                let tokens_vec = d_batch_tokens.inner().to_host_vec(stream).unwrap_or_else(|_| vec![0; b]);
                for (idx, ctx) in active_contexts.iter_mut().enumerate() {
                    let next_tok = tokens_vec[idx];
                    ctx.gen_tokens.push(next_tok);
                    ctx.cur_token = next_tok;
                    ctx.gen_count += 1;

                    if ctx.gen_count >= ctx.max_gen || next_tok == 124900 || next_tok == 124896 {
                        completed_indices.push(idx);
                    }
                }
            }

            if profile {
                let _ = stream.synchronize();
                prof_lm += prof_mark.elapsed().as_secs_f64();
                prof_steps += 1;
                if prof_steps % 20 == 0 {
                    let tot = prof_embed + prof_conv + prof_attn + prof_ffn + prof_lm;
                    eprintln!(
                        "[profile] steps={} B={} | embed {:.1} | conv {:.1} | attn {:.1} | ffn {:.1} | lm {:.1} ms (total {:.1})",
                        prof_steps,
                        b,
                        prof_embed * 1000.0,
                        prof_conv * 1000.0,
                        prof_attn * 1000.0,
                        prof_ffn * 1000.0,
                        prof_lm * 1000.0,
                        tot * 1000.0
                    );
                }
            }

            // 6. Retire finished requests in reverse order to preserve indices
            if !completed_indices.is_empty() {
                let _ = stream.synchronize();
                for &idx in completed_indices.iter().rev() {
                    let ctx = active_contexts.remove(idx);
                    let decoded = tokenizer_clone.decode(&ctx.gen_tokens).unwrap_or_default();
                    let _ = ctx.req.response_tx.send((decoded, ctx.prompt_len, ctx.gen_tokens.len()));
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
    println!("\n🚀 OxideInfer Dynamic Continuous Batching Server listening on http://{}", addr);
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
        "mode": "dynamic_continuous_batching_cublas_hgemm",
        "max_batch_size": MAX_BATCH_SIZE,
        "model": "LFM3-VL-2B"
    }))
}

async fn chat_completions(
    State(state): State<ServerState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    let mut extracted_image_bytes: Option<Vec<u8>> = None;

    let prompt = if let Some(messages) = payload.messages {
        let mut full_prompt = String::from("<|startoftext|>");
        for msg in messages {
            match msg.content {
                MessageContent::Text(text) => {
                    full_prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, text));
                }
                MessageContent::Parts(parts) => {
                    let mut role_content = String::new();
                    for part in parts {
                        if part.part_type == "text" {
                            if let Some(t) = part.text {
                                role_content.push_str(&t);
                            }
                        } else if part.part_type == "image_url" || part.part_type == "image" {
                            if let Some(ref img_obj) = part.image_url {
                                let url = &img_obj.url;
                                if let Some(comma_pos) = url.find(',') {
                                    if url.starts_with("data:") {
                                        let b64_str = &url[comma_pos + 1..];
                                        use base64::Engine;
                                        if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64_str) {
                                            extracted_image_bytes = Some(bytes);
                                        }
                                    }
                                } else if std::path::Path::new(url).exists() {
                                    if let Ok(bytes) = std::fs::read(url) {
                                        extracted_image_bytes = Some(bytes);
                                    }
                                }
                            }
                            role_content.push_str("<|image|>");
                        }
                    }
                    full_prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, role_content));
                }
            }
        }
        full_prompt.push_str("<|im_start|>assistant\n<think>");
        full_prompt
    } else if let Some(p) = payload.prompt {
        p
    } else {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            "Either messages or prompt must be provided",
        )
            .into_response();
    };

    let max_tokens = payload.max_tokens.unwrap_or(64);
    let (tx, rx) = oneshot::channel();

    let req = ServingRequest {
        prompt,
        image_bytes: extracted_image_bytes,
        max_tokens,
        response_tx: tx,
    };

    if let Err(e) = state.tx.send(req).await {
        return (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Worker channel unavailable: {}", e),
        )
            .into_response();
    }

    match rx.await {
        Ok((text, prompt_tokens, comp_tokens)) => {
            let resp = ChatCompletionResponse {
                id: format!("chatcmpl-{}", uuid_simple()),
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
                    completion_tokens: comp_tokens,
                    total_tokens: prompt_tokens + comp_tokens,
                },
            };
            Json(resp).into_response()
        }
        Err(_) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "Worker dropped request",
        )
            .into_response(),
    }
}

fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:x}", nanos)
}
