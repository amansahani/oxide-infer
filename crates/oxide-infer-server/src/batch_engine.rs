//! Fully batched decode/prefill engine.
//!
//! Every per-sequence operation in the transformer layer is either a batched
//! tensor-core GEMM or a batched PTX kernel. Per-sequence KV caches and Conv1D
//! states are addressed through device-side pointer tables, so no per-request
//! packing or unpacking happens in the hot path.

use anyhow::{anyhow, Result};
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext};
use oxide_infer_kernels::batched_attn::{
    BATCHED_ARGMAX_DIRECT_PTX, BATCHED_ATTN_GATE_SCATTER_PTX, BATCHED_ATTN_PACK_Q_PTX,
    BATCHED_ATTN_SOFTMAX_MASK_PTX, BATCHED_CONV_PTR_PTX, BATCHED_EMBED_LOOKUP_PTX,
    BATCHED_GQA_ATTENTION_ONLINE_PTX, BATCHED_GQA_ATTENTION_SPLIT4_PTX,
    BATCHED_GQA_ATTENTION_SPLIT_PTX, BATCHED_KV_COPY_PTX,
    BATCHED_QK_RMSNORM_PTX, BATCHED_ROPE_PTX, BATCHED_SIGMOID_GATE_PTX, CONV_STATE_PREFILL_PTX,
};
use oxide_infer_kernels::PtxModule;
use oxide_infer_kernels::quant_gemv::QuantGemv;
use oxide_infer_models::{FullLfm3DecoderLayer, LayerType};
use cuda_core::{launch_kernel_on_stream, CudaEvent, Function, Stream as CuStream};
use std::ffi::c_void;

use crate::lt_gemm::LtGemm;

/// Remaining `OXIDE_ATTN_DEBUG` GEMM-vs-online diffs (one per attention
/// layer of the first prefill step).
static ATTN_DEBUG_LEFT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(40);

extern "C" {
    /// CUDA driver D2D copy, declared in server.rs too; duplicated here so
    /// the engine can persist/restore per-context hidden states.
    pub fn cuMemcpyDtoDAsync_v2(dst: u64, src: u64, byte_count: usize, stream: *mut c_void) -> i32;
    /// Enqueue a host-to-device copy without synchronizing. Used only for
    /// decode metadata, where the sampling tail synchronizes the stream
    /// before this frame (and the host staging vectors) go out of scope.
    fn cuMemcpyHtoDAsync_v2(dst: u64, src: u64, byte_count: usize, stream: *mut c_void) -> i32;
}

pub const HIDDEN: usize = 2048;
pub const INTER: usize = 7680;
pub const KV_DIM: usize = 512;
pub const NUM_Q_HEADS: usize = 16;
pub const NUM_KV_HEADS: usize = 4;
pub const HEAD_DIM: usize = 128;
/// Keys per block in split-K decode attention. Fixed so chunk boundaries (and
/// therefore the summation order) never depend on the batch composition: a
/// row sees identical arithmetic whether it runs alone or next to 127 others.
const ATTN_CHUNK_KEYS: usize = 128;
/// Largest key count a GEMM attention chunk can score. Must cover the
/// server's sequence limit (`OXIDE_MAX_SEQ_TOKENS`, default 8192) because a
/// chunk never needs more keys than its end position, and the score scratch
/// is sized for it. Configurable because the scratch is 4 * prefill_rows *
/// keys f32 + the same in f16, which grows quickly.
fn max_attn_keys() -> usize {
    std::env::var("OXIDE_MAX_ATTN_KEYS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8192)
}
/// Chunks below this row count stay on the online kernel: with fewer rows the
/// attention GEMMs are launch-bound and the SIMT kernel is cheaper.
const GEMM_ATTN_MIN_ROWS: usize = 64;
/// Hard cap on attention chunks; with 128-key chunks this covers 8192 tokens.
/// The decode split-K path clamps `nchunks` to this, so it must be at least
/// ceil(max_seq_tokens / ATTN_CHUNK_KEYS) or decode would silently score only
/// the first few thousand keys of a long context. Configurable for the same
/// reason as `max_attn_keys`: the partials are max_batch * chunks * heads.
fn max_attn_chunks() -> usize {
    std::env::var("OXIDE_MAX_ATTN_CHUNKS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64)
}
pub const VOCAB: usize = 128_000;

/// Per-sequence GPU state needed by the batched engine.
pub trait BatchSequence {
    fn k_cache_ptr(&self, layer: usize) -> u64;
    fn v_cache_ptr(&self, layer: usize) -> u64;
    fn conv_state_ptr(&self, layer: usize) -> u64;
    /// Pointer to the sequence's persistent 2048-element f16 hidden buffer.
    /// Prefill steps persist their per-position hidden into it, so the
    /// post-prefill first token can be sampled without running any layer
    /// twice (the engine re-embeds tokens at every decode step).
    fn hidden_ptr(&self) -> u64;
}

impl<T: BatchSequence + ?Sized> BatchSequence for &T {
    fn k_cache_ptr(&self, layer: usize) -> u64 {
        (**self).k_cache_ptr(layer)
    }
    fn v_cache_ptr(&self, layer: usize) -> u64 {
        (**self).v_cache_ptr(layer)
    }
    fn conv_state_ptr(&self, layer: usize) -> u64 {
        (**self).conv_state_ptr(layer)
    }
    fn hidden_ptr(&self) -> u64 {
        (**self).hidden_ptr()
    }
}

struct BatchScratch {
    hidden: DeviceBuffer<f16>,
    norm: DeviceBuffer<f16>,
    layer: DeviceBuffer<f16>,
    q: DeviceBuffer<f16>,
    k: DeviceBuffer<f16>,
    v: DeviceBuffer<f16>,
    gate: DeviceBuffer<f16>,
    ctx: DeviceBuffer<f16>,
    ffn_gate: DeviceBuffer<f16>,
    ffn_up: DeviceBuffer<f16>,
    ffn_act: DeviceBuffer<f16>,
    ffn_out: DeviceBuffer<f16>,
    /// Scratch for the int8 MMVQ path: q8_1-quantized activations.
    /// `q8_1_hidden` covers HIDDEN-dim k, `q8_1_inter` covers INTER-dim k.
    /// Each block is 36 bytes (f16 d, f16 s, 32 x i8).
    q8_1_hidden: DeviceBuffer<u8>,
    q8_1_inter: DeviceBuffer<u8>,
    bcx: DeviceBuffer<f16>,
    conv_y: DeviceBuffer<f16>,
    logits: DeviceBuffer<f16>,
    tokens_in: DeviceBuffer<u32>,
    tokens_out: DeviceBuffer<u32>,
    pos_ids: DeviceBuffer<u32>,
    seq_lens: DeviceBuffer<u32>,
    /// Split-K attention partials: `[row][chunk][q_head][head_dim]` (f32) and
    /// `[row][chunk][q_head][max, normaliser]` (f32).
    attn_part_acc: DeviceBuffer<f32>,
    attn_part_stats: DeviceBuffer<f32>,
    /// GEMM prefill attention: Q packed to `[head][row][head_dim]`, the
    /// unnormalised `P * V` in the same order, one KV head's f32 score matrix
    /// `[4 * rows][keys]`, its f16 weights, and the per-column softmax sum
    /// `[head][row]`.
    attn_qpack: DeviceBuffer<f16>,
    attn_tmp: DeviceBuffer<f16>,
    attn_scores: DeviceBuffer<f32>,
    attn_p: DeviceBuffer<f16>,
    attn_sums: DeviceBuffer<f32>,
    k_ptrs: DeviceBuffer<u64>,
    v_ptrs: DeviceBuffer<u64>,
    conv_ptrs: DeviceBuffer<u64>,
    /// Per-layer slices of prefill conv-state rows. One slice per model layer
    /// (only the conv layers are used), each with `rows` rows of
    /// `HIDDEN * 6` f16. Reused by every prefill wave.
    conv_pool: DeviceBuffer<f16>,
    pf_carry: DeviceBuffer<u64>,
    pf_origin: DeviceBuffer<u32>,
    pf_pos: DeviceBuffer<u32>,
}

/// Token-batched prefill metadata for one wave.
///
/// `carry_ptrs` uses the same `[layer * rows + row]` layout as the
/// per-layer pointer tables and must hold each request's persistent conv
/// state pointer for that layer. `origins[row]` is the wave row where the
/// request's chunk begins (used to locate window tokens inside `bcx`) and
/// `local_pos[row]` is the row's position within its chunk.
pub struct PrefillMeta {
    pub carry_ptrs: Vec<u64>,
    pub origins: Vec<u32>,
    pub local_pos: Vec<u32>,
}

/// Batched transformer engine. All scratch buffers are allocated once.
///
/// `max_batch` bounds the decode/sampling batch (one row per sequence).
/// `rows` bounds prefill waves, which pack prompt *tokens* rather than
/// sequences; a wave of `rows` tokens re-streams each weight once while
/// keeping the GEMMs compute-bound, so it is usually much larger than
/// `max_batch`. Logits stay at decode size because prefill never runs the LM
/// head.
pub struct BatchEngine {
    scratch: BatchScratch,
    max_batch: usize,
    rows: usize,
    /// Sequence-limit ceilings, read once at construction so the scratch
    /// buffers and the runtime checks agree.
    max_attn_keys: usize,
    max_attn_chunks: usize,
    lt: LtGemm,
    rms_norm: Function,
    residual_add: Function,
    swiglu: Function,
    conv: Function,
    qk_norm: Function,
    rope: Function,
    kv_copy: Function,
    gqa: Function,
    attn_pack_q: Function,
    attn_softmax: Function,
    attn_gate_scatter: Function,
    gqa_split: Function,
    /// Four-key-per-iteration variant; bit-identical output, more ILP.
    gqa_split4: Function,
    gqa_combine: Function,
    sigmoid_gate: Function,
    embed: Function,
    argmax: Function,
    conv_state_prefill: Function,
    quant_gemv: QuantGemv,
}

/// Enqueue an exact-length host-to-device copy on the compute stream without
/// waiting. The staging slice must outlive the next stream synchronize.
fn upload_async<T>(buf: &DeviceBuffer<T>, src: &[T], stream: &CuStream)
where
    T: cuda_core::DeviceCopy,
{
    debug_assert_eq!(buf.len(), src.len());
    let rc = unsafe {
        cuMemcpyHtoDAsync_v2(
            buf.inner().cu_deviceptr(),
            src.as_ptr() as u64,
            std::mem::size_of_val(src),
            stream.cu_stream() as *mut c_void,
        )
    };
    debug_assert_eq!(rc, 0, "cuMemcpyHtoDAsync failed with {rc}");
}

/// Print the phase totals accumulated by `OXIDE_PROFILE_PHASES`.
///
/// `ev[0]` is recorded before the embedding lookup and `ev[1 + 7i + k]` are
/// the seven per-layer boundaries, so the segments are: operator norm,
/// qkv/in-proj gemm, attention setup (or conv state prep), attention (or conv)
/// kernel, out-proj gemm, ffn gemms + swiglu, post-ffn norm + residual.
fn report_phases(ev: &[CudaEvent], n_layers: usize, rows: usize, is_conv: &[bool]) {
    // `elapsed_time` rejects an unfinished end event, and the prefill path
    // returns without a synchronize, so wait once here (profiling only).
    if let Some(last) = ev.last() {
        let _ = last.synchronize();
    }
    let mut cats = [0f64; 7];
    for i in 0..n_layers {
        let base = 1 + i * 7;
        let mut prev = if i == 0 { 0 } else { base - 1 };
        for (k, c) in cats.iter_mut().enumerate() {
            let dt = match ev[prev].elapsed_time(&ev[base + k]) {
                Ok(ms) => ms as f64,
                Err(_) => 0.0,
            };
            *c += dt;
            prev = base + k;
        }
    }
    let total = ev[0].elapsed_time(&ev[ev.len() - 1]).unwrap_or(0.0) as f64;
    let sum: f64 = cats.iter().sum();
    if std::env::var("OXIDE_PROFILE_LAYERS").is_ok() && rows == 1 {
        for i in 0..n_layers {
            let base = 1 + i * 7;
            let mut prev = if i == 0 { 0 } else { base - 1 };
            let mut seg = [0f64; 7];
            for (k, c) in seg.iter_mut().enumerate() {
                *c = ev[prev].elapsed_time(&ev[base + k]).unwrap_or(0.0) as f64;
                prev = base + k;
            }
            eprintln!(
                "[layer] {:3} {:4} norm={:5.2} qkv={:6.2} setup={:5.2} attn={:6.2} oproj={:5.2} ffn={:6.2} post={:5.2}",
                i,
                if is_conv[i] { "conv" } else { "attn" },
                seg[0], seg[1], seg[2], seg[3], seg[4], seg[5], seg[6]
            );
        }
    }
    eprintln!(
        "[phases] rows={rows:4} op_norm={:6.1} qkv={:6.1} setup={:6.1} attn={:6.1} oproj={:6.1} ffn={:6.1} post={:5.1} other={:5.1} total={:7.1} ms",
        cats[0], cats[1], cats[2], cats[3], cats[4], cats[5], cats[6], total - sum, total
    );
}

fn fptr(buf: &DeviceBuffer<f16>) -> u64 {
    buf.inner().cu_deviceptr()
}

fn fptr_mut(buf: &mut DeviceBuffer<f16>) -> u64 {
    buf.inner_mut().cu_deviceptr()
}

impl BatchEngine {
    /// Packed projection GEMV dispatch. At batch = 1 with the int8 MMVQ path
    /// enabled, activations are quantized to q8_1 (36B per 32 elements) and the
    /// weights are consumed by a dp4a matvec kernel - the same kernel
    /// llama.cpp launches for batch = 1 on sm_86. Otherwise the f32 packed
    /// kernels are used. The q8_1 scratch is reused across calls in a wave
    /// (stream-ordered).
    #[allow(clippy::too_many_arguments)]
    fn gemv_dispatch(
        &self,
        stream: &CuStream,
        qt: oxide_infer_core::quant::QuantType,
        w_ptr_and_len: (u64, u64),
        x_ptr_and_len: (u64, u64),
        y_ptr_and_len: (u64, u64),
        hidden: u32,
        n_elems: u32,
        b: u32,
    ) -> anyhow::Result<()> {
        use oxide_infer_kernels::quant_gemv::{int8_mmvq_enabled, int8_mmvq_supported};
        let proj_int8 = std::env::var("OXIDE_MMVQ_INT8_PROJ").map(|v| v != "0").unwrap_or(true);
        if int8_mmvq_enabled() && proj_int8 && int8_mmvq_supported(qt) && b == 1 {
            let (x_ptr, x_len) = x_ptr_and_len;
            let q8_ptr = self.scratch.q8_1_hidden.inner().cu_deviceptr();
            let q8_len = (hidden as u64 / 32) * 36;
            self.quant_gemv.quantize_q8_1(stream, (x_ptr, x_len), (q8_ptr, q8_len), hidden, b)?;
            return self
                .quant_gemv
                .mmvq_gemv_q4_0(stream, w_ptr_and_len, (q8_ptr, q8_len), y_ptr_and_len, hidden, n_elems, b);
        }
        self.quant_gemv.gemv(stream, qt, w_ptr_and_len, x_ptr_and_len, y_ptr_and_len, hidden, n_elems, b)
    }

    pub fn new(gpu: &GpuContext, max_batch: usize, prefill_rows: usize) -> Result<Self> {
        let rows = prefill_rows.max(max_batch);
        let max_attn_keys = max_attn_keys();
        let max_attn_chunks = max_attn_chunks();
        // Row-sized scratch. `logits`/`tokens_out` are only touched by the
        // sampling tail (decode batch or `sample_only`), so they stay at
        // `max_batch` and keep the 128k-vocab buffer from exploding.
        let b = rows;
        let alloc_f16 = |n: usize| -> Result<DeviceBuffer<f16>> {
            DeviceBuffer::<f16>::zeroed(gpu, n).map_err(|e| anyhow!("alloc: {e}"))
        };
        let alloc_u8 = |n: usize| -> Result<DeviceBuffer<u8>> {
            DeviceBuffer::<u8>::zeroed(gpu, n).map_err(|e| anyhow!("alloc: {e}"))
        };

        let norm_ptx = PtxModule::load_from_ptx(gpu, oxide_infer_models::RESIDUAL_ADD_PTX)
            .map_err(|e| anyhow!("norm PTX: {e}"))?;
        let attn_ptx = PtxModule::load_from_ptx(gpu, BATCHED_QK_RMSNORM_PTX)
            .map_err(|e| anyhow!("qk_norm PTX: {e}"))?;
        let rope_ptx = PtxModule::load_from_ptx(gpu, BATCHED_ROPE_PTX)
            .map_err(|e| anyhow!("rope PTX: {e}"))?;
        let kv_ptx = PtxModule::load_from_ptx(gpu, BATCHED_KV_COPY_PTX)
            .map_err(|e| anyhow!("kv_copy PTX: {e}"))?;
        let gqa_ptx = PtxModule::load_from_ptx(gpu, BATCHED_GQA_ATTENTION_ONLINE_PTX)
            .map_err(|e| anyhow!("gqa PTX: {e}"))?;
        let split_ptx = PtxModule::load_from_ptx(gpu, BATCHED_GQA_ATTENTION_SPLIT_PTX)
            .map_err(|e| anyhow!("gqa split PTX: {e}"))?;
        let split4_ptx = PtxModule::load_from_ptx(gpu, BATCHED_GQA_ATTENTION_SPLIT4_PTX)
            .map_err(|e| anyhow!("gqa split4 PTX: {e}"))?;
        let gate_ptx = PtxModule::load_from_ptx(gpu, BATCHED_SIGMOID_GATE_PTX)
            .map_err(|e| anyhow!("sigmoid gate PTX: {e}"))?;
        let gemm_attn_ptx = PtxModule::load_from_ptx(gpu, BATCHED_ATTN_PACK_Q_PTX)
            .map_err(|e| anyhow!("gemm attn pack PTX: {e}"))?;
        let softmax_ptx = PtxModule::load_from_ptx(gpu, BATCHED_ATTN_SOFTMAX_MASK_PTX)
            .map_err(|e| anyhow!("gemm attn softmax PTX: {e}"))?;
        let scatter_ptx = PtxModule::load_from_ptx(gpu, BATCHED_ATTN_GATE_SCATTER_PTX)
            .map_err(|e| anyhow!("gemm attn scatter PTX: {e}"))?;
        let embed_ptx = PtxModule::load_from_ptx(gpu, BATCHED_EMBED_LOOKUP_PTX)
            .map_err(|e| anyhow!("embed PTX: {e}"))?;
        let conv_ptx = PtxModule::load_from_ptx(gpu, BATCHED_CONV_PTR_PTX)
            .map_err(|e| anyhow!("conv PTX: {e}"))?;
        let argmax_ptx = PtxModule::load_from_ptx(gpu, BATCHED_ARGMAX_DIRECT_PTX)
            .map_err(|e| anyhow!("argmax PTX: {e}"))?;
        let pf_ptx = PtxModule::load_from_ptx(gpu, CONV_STATE_PREFILL_PTX)
            .map_err(|e| anyhow!("conv prefill PTX: {e}"))?;

        let scratch = BatchScratch {
            hidden: alloc_f16(b * HIDDEN)?,
            norm: alloc_f16(b * HIDDEN)?,
            layer: alloc_f16(b * HIDDEN)?,
            q: alloc_f16(b * HIDDEN)?,
            k: alloc_f16(b * KV_DIM)?,
            v: alloc_f16(b * KV_DIM)?,
            gate: alloc_f16(b * NUM_Q_HEADS)?,
            ctx: alloc_f16(b * HIDDEN)?,
            ffn_gate: alloc_f16(b * INTER)?,
            ffn_up: alloc_f16(b * INTER)?,
            ffn_act: alloc_f16(b * INTER)?,
            ffn_out: alloc_f16(b * HIDDEN)?,
            q8_1_hidden: alloc_u8(b * HIDDEN / 32 * 36)?,
            q8_1_inter: alloc_u8(b * INTER / 32 * 36)?,
            bcx: alloc_f16(b * 3 * HIDDEN)?,
            conv_y: alloc_f16(b * HIDDEN)?,
            logits: alloc_f16(max_batch * VOCAB)?,
            tokens_in: DeviceBuffer::<u32>::zeroed(gpu, b).map_err(|e| anyhow!("alloc: {e}"))?,
            tokens_out: DeviceBuffer::<u32>::zeroed(gpu, max_batch).map_err(|e| anyhow!("alloc: {e}"))?,
            pos_ids: DeviceBuffer::<u32>::zeroed(gpu, b).map_err(|e| anyhow!("alloc: {e}"))?,
            seq_lens: DeviceBuffer::<u32>::zeroed(gpu, b).map_err(|e| anyhow!("alloc: {e}"))?,
            attn_part_acc: DeviceBuffer::<f32>::zeroed(
                gpu,
                max_batch * max_attn_chunks * NUM_Q_HEADS * HEAD_DIM,
            )
            .map_err(|e| anyhow!("alloc: {e}"))?,
            attn_part_stats: DeviceBuffer::<f32>::zeroed(
                gpu,
                max_batch * max_attn_chunks * NUM_Q_HEADS * 2,
            )
            .map_err(|e| anyhow!("alloc: {e}"))?,
            attn_qpack: alloc_f16(b * HIDDEN)?,
            attn_tmp: alloc_f16(b * HIDDEN)?,
            attn_scores: DeviceBuffer::<f32>::zeroed(gpu, 4 * b * max_attn_keys)
                .map_err(|e| anyhow!("alloc: {e}"))?,
            attn_p: alloc_f16(4 * b * max_attn_keys)?,
            attn_sums: DeviceBuffer::<f32>::zeroed(gpu, b * NUM_Q_HEADS)
                .map_err(|e| anyhow!("alloc: {e}"))?,
            k_ptrs: DeviceBuffer::<u64>::zeroed(gpu, 24 * b).map_err(|e| anyhow!("alloc: {e}"))?,
            v_ptrs: DeviceBuffer::<u64>::zeroed(gpu, 24 * b).map_err(|e| anyhow!("alloc: {e}"))?,
            conv_ptrs: DeviceBuffer::<u64>::zeroed(gpu, 24 * b).map_err(|e| anyhow!("alloc: {e}"))?,
            conv_pool: alloc_f16(24 * b * HIDDEN * 6)?,
            pf_carry: DeviceBuffer::<u64>::zeroed(gpu, 24 * b).map_err(|e| anyhow!("alloc: {e}"))?,
            pf_origin: DeviceBuffer::<u32>::zeroed(gpu, b).map_err(|e| anyhow!("alloc: {e}"))?,
            pf_pos: DeviceBuffer::<u32>::zeroed(gpu, b).map_err(|e| anyhow!("alloc: {e}"))?,
        };

        let mut lt = LtGemm::new(gpu, max_batch)?;
        // Pay the cuBLASLt algorithm search and kernel-module load for every
        // weight shape here instead of inside the first prefill wave.
        lt.warm(
            gpu,
            &[
                (3 * HIDDEN, HIDDEN),
                (HIDDEN, HIDDEN),
                (KV_DIM, HIDDEN),
                (NUM_Q_HEADS, HIDDEN),
                (INTER, HIDDEN),
                (HIDDEN, INTER),
            ],
        )?;
        Ok(Self {
            scratch,
            max_batch,
            rows,
            max_attn_keys,
            max_attn_chunks,
            lt,
            rms_norm: norm_ptx.get_function("batched_rms_norm_vec8_fp16")?,
            residual_add: norm_ptx.get_function("batched_residual_add_fp16")?,
            swiglu: norm_ptx.get_function("batched_swiglu_separate_fp16")?,
            conv: conv_ptx.get_function("batched_fused_causal_conv1d_gating_ptr_fp16")?,
            qk_norm: attn_ptx.get_function("batched_qk_rmsnorm_fp16")?,
            rope: rope_ptx.get_function("batched_rope_fp16")?,
            kv_copy: kv_ptx.get_function("batched_kv_copy_fp16")?,
            gqa: gqa_ptx.get_function("batched_gqa_attention_online_fp16")?,
            attn_pack_q: gemm_attn_ptx.get_function("batched_attn_pack_q_fp16")?,
            attn_softmax: softmax_ptx.get_function("batched_attn_softmax_mask_fp16")?,
            attn_gate_scatter: scatter_ptx.get_function("batched_attn_gate_scatter_fp16")?,
            gqa_split: split_ptx.get_function("batched_gqa_attention_split_fp16")?,
            gqa_split4: split4_ptx.get_function("batched_gqa_attention_split4_fp16")?,
            gqa_combine: split_ptx.get_function("batched_gqa_attn_combine_fp16")?,
            sigmoid_gate: gate_ptx.get_function("batched_sigmoid_gate_fp16")?,
            embed: embed_ptx.get_function("batched_embed_lookup_fp16")?,
            argmax: argmax_ptx.get_function("batched_argmax_direct_f16")?,
            conv_state_prefill: pf_ptx.get_function("conv_state_prefill_fp16")?,
            quant_gemv: QuantGemv::new(gpu)?,
        })
    }

    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Maximum rows (prompt tokens) in one prefill wave.
    pub fn prefill_rows(&self) -> usize {
        self.rows
    }

    /// Device pointer of a prefill conv-state row (decode-ready state of the
    /// chunk's last token). Only valid after `forward_step` with `prefill`.
    pub fn prefill_state_ptr(&self, layer: usize, row: usize) -> u64 {
        self.scratch.conv_pool.inner().cu_deviceptr()
            + (((layer * self.rows + row) * HIDDEN * 6 * 2) as u64)
    }

    /// Score one prefill chunk with the GEMM attention path.
    ///
    /// `rows` consecutive tokens of one request, with absolute positions
    /// `slen_min - 1 ..= slen_max - 1`, all read the same KV cache. Keys
    /// `t_lo .. slen_max` are scored, where `t_lo` is the earliest key any row
    /// may attend to (0 for full layers, `slen_min - window` for sliding
    /// ones); the softmax kernel applies each row's own causal and window
    /// bounds inside that range and zeroes everything else.
    ///
    /// Per KV head: one QK GEMM (f32 scores), one fused mask+softmax pass
    /// (f16 weights + per-column normaliser), one PV GEMM whose transposed
    /// output is the packed `[head][row][dim]` layout. A single gate+scatter
    /// pass then applies `sigmoid(gate) / sum` and writes the row-major
    /// context, so this replaces both the attention and gate launches.
    #[allow(clippy::too_many_arguments)]
    fn gemm_attention_chunk(
        &mut self,
        stream: &CuStream,
        seg_start: usize,
        rows: usize,
        slen_min: u32,
        slen_max: u32,
        window: u32,
        k_base: u64,
        v_base: u64,
        d_q: u64,
        d_gate: u64,
        d_ctx: u64,
    ) -> Result<()> {
        let t_lo = if window > 0 {
            slen_min.saturating_sub(window)
        } else {
            0
        };
        let keys = (slen_max - t_lo) as usize;
        if keys > self.max_attn_keys {
            return Err(anyhow!(
                "attention chunk needs {keys} keys, scratch holds {}",
                self.max_attn_keys
            ));
        }
        let n = 4 * rows;
        let mut n_u = n as u32;
        let mut keys_u = keys as u32;
        let mut rows_u = rows as u32;
        let mut hd = HEAD_DIM as u32;
        let mut nq = NUM_Q_HEADS as u32;
        let mut t_lo_u = t_lo;
        let mut window_u = window;
        let mut slen_ptr = self.scratch.seq_lens.inner_mut().cu_deviceptr()
            + (seg_start * 4) as u64;

        // Q regrouped to [head][row][dim] so each KV head's four query heads
        // are one contiguous GEMM operand.
        let qpack = self.scratch.attn_qpack.inner_mut().cu_deviceptr();
        let mut q_in = d_q + (seg_start * HIDDEN * 2) as u64;
        let mut q_out = qpack;
        let mut args_pack: [*mut c_void; 5] = [
            &mut q_in as *mut _ as *mut c_void,
            &mut q_out as *mut _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut rows_u as *mut _ as *mut c_void,
        ];
        unsafe {
            launch_kernel_on_stream(
                &self.attn_pack_q,
                (rows_u, 1, 1),
                (256, 1, 1),
                0,
                stream,
                &mut args_pack,
            )?;
        }

        let mut scores = self.scratch.attn_scores.inner_mut().cu_deviceptr();
        let mut p = self.scratch.attn_p.inner_mut().cu_deviceptr();
        let mut attn_tmp = self.scratch.attn_tmp.inner_mut().cu_deviceptr();
        let sums_base = self.scratch.attn_sums.inner_mut().cu_deviceptr();
        let kv_dim = KV_DIM as i64;
        let hd_i = HEAD_DIM as i64;
        let keys_i = keys as i64;

        for kvh in 0..NUM_KV_HEADS {
            // QK: scores[4*rows][keys] = Q[4*rows][128] * K[keys][128]^T.
            // K is a strided view of the interleaved slab (one head of four).
            let a = k_base
                + (kvh * HEAD_DIM * 2) as u64
                + (t_lo as u64 * KV_DIM as u64 * 2);
            let b = qpack + (kvh * n * HEAD_DIM * 2) as u64;
            self.lt.gemm_strided(
                stream,
                keys,
                HEAD_DIM,
                n,
                kv_dim,
                hd_i,
                keys_i,
                true,
                true,
                1024,
                a,
                b,
                scores,
            )?;

            let mut sums_k = sums_base + (kvh * 4 * rows * 4) as u64;
            let mut sc = 0.0883883476f32;
            let mut args_sm: [*mut c_void; 9] = [
                &mut scores as *mut _ as *mut c_void,
                &mut p as *mut _ as *mut c_void,
                &mut sums_k as *mut _ as *mut c_void,
                &mut slen_ptr as *mut _ as *mut c_void,
                &mut rows_u as *mut _ as *mut c_void,
                &mut keys_u as *mut _ as *mut c_void,
                &mut t_lo_u as *mut _ as *mut c_void,
                &mut window_u as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
            ];
            unsafe {
                launch_kernel_on_stream(
                    &self.attn_softmax,
                    (rows_u, 4, 1),
                    (128, 1, 1),
                    0,
                    stream,
                    &mut args_sm,
                )?;
            }

            // PV: out[128][4*rows] = V[keys][128]^T * P[4*rows][keys]^T.
            // V is read untransposed (`trans_a = false`), which makes the
            // output already the [head][row][dim] the scatter pass expects.
            let a = v_base
                + (kvh * HEAD_DIM * 2) as u64
                + (t_lo as u64 * KV_DIM as u64 * 2);
            let c = attn_tmp + (kvh * n * HEAD_DIM * 2) as u64;
            self.lt.gemm_strided(
                stream, HEAD_DIM, keys, n, kv_dim, keys_i, hd_i, false, false, 1024, a, p, c,
            )?;
        }

        // Gate + normalise + transpose to the row-major context.
        let mut ctx_ptr = d_ctx + (seg_start * HIDDEN * 2) as u64;
        let mut gate_ptr = d_gate + (seg_start * NUM_Q_HEADS * 2) as u64;
        let mut tmp_ptr = attn_tmp;
        let mut sums_ptr = sums_base;
        let mut args_g: [*mut c_void; 7] = [
            &mut ctx_ptr as *mut _ as *mut c_void,
            &mut tmp_ptr as *mut _ as *mut c_void,
            &mut gate_ptr as *mut _ as *mut c_void,
            &mut sums_ptr as *mut _ as *mut c_void,
            &mut rows_u as *mut _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        unsafe {
            launch_kernel_on_stream(
                &self.attn_gate_scatter,
                (nq, rows_u, 1),
                (hd, 1, 1),
                0,
                stream,
                &mut args_g,
            )?;
        }
        let _ = n_u;
        Ok(())
    }

    /// Run one transformer step for `ctxs`.
    ///
    /// `tokens[i]` is embedded for sequence `i`; `pos_ids[i]` is used for RoPE
    /// and the KV cache write. When `sample` is true the final RMSNorm, LM head
    /// and argmax are executed and the next tokens are returned.
    /// If `custom_embeddings` is provided, those embeddings are used directly
    /// instead of looking up `tokens` in `embed_table`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_step<C: BatchSequence>(
        &mut self,
        gpu: &GpuContext,
        layers: &mut [FullLfm3DecoderLayer],
        is_conv: &[bool],
        ctxs: &[C],
        tokens: &[u32],
        pos_ids: &[u32],
        embed_table: &DeviceBuffer<f16>,
        emb_norm: &DeviceBuffer<f16>,
        sample: bool,
        prefill: Option<&PrefillMeta>,
    ) -> Result<Vec<u32>> {
        self.forward_step_with_embeds(
            gpu,
            layers,
            is_conv,
            ctxs,
            tokens,
            pos_ids,
            embed_table,
            emb_norm,
            sample,
            prefill,
            None,
        )
    }

    /// Run one transformer step with optional custom input embeddings (e.g. multimodal vision tokens).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_step_with_embeds<C: BatchSequence>(
        &mut self,
        gpu: &GpuContext,
        layers: &mut [FullLfm3DecoderLayer],
        is_conv: &[bool],
        ctxs: &[C],
        tokens: &[u32],
        pos_ids: &[u32],
        embed_table: &DeviceBuffer<f16>,
        emb_norm: &DeviceBuffer<f16>,
        sample: bool,
        prefill: Option<&PrefillMeta>,
        custom_embeddings: Option<&DeviceBuffer<f16>>,
    ) -> Result<Vec<u32>> {
        let b = ctxs.len();
        if b == 0 || b > self.rows {
            return Err(anyhow!("invalid batch size {b}"));
        }
        let b_u32 = b as u32;
        let stream = gpu.compute_stream();

        // Optional per-phase device timing. Seven event pairs per layer:
        // operator norm, qkv/in_proj gemm, attention setup / conv prep,
        // attention / conv kernel, out-proj gemm, ffn gemms, post-ffn norm.
        // Opt-in so the serving path pays nothing (an event record is ~1 us,
        // but 170 of them per wave would still distort a 0.1 ms step).
        let prof = std::env::var("OXIDE_PROFILE_PHASES").is_ok();
        let mut prof_ev: Vec<CudaEvent> = if prof {
            (0..1 + layers.len() * 7)
                .map(|_| gpu.context().new_event().expect("prof event"))
                .collect()
        } else {
            Vec::new()
        };
        let mut mark = 0usize;
        macro_rules! phase {
            () => {{
                if prof {
                    prof_ev[mark].record(stream).expect("prof event record");
                    mark += 1;
                }
            }};
        }

        // Prefill waves address up to `rows` rows, so their metadata must be
        // padded to the device buffer length (and `copy_from_slice`
        // synchronizes, which is noise inside a multi-wave prefill). Decode
        // only touches `b` rows and must stay synchronous-free: a stream
        // synchronize per metadata array costs a round trip per token.
        let mb = self.rows;
        let async_meta = prefill.is_none() && sample;
        let pad = if async_meta { b } else { mb };
        let mut tokens_pad = vec![0u32; pad];
        let mut pos_pad = vec![0u32; pad];
        let mut seq_pad = vec![0u32; pad];
        tokens_pad[..b].copy_from_slice(tokens);
        pos_pad[..b].copy_from_slice(pos_ids);
        for i in 0..b {
            seq_pad[i] = pos_ids[i] + 1;
        }
        if async_meta {
            upload_async(&self.scratch.tokens_in, &tokens_pad, stream);
            upload_async(&self.scratch.pos_ids, &pos_pad, stream);
            upload_async(&self.scratch.seq_lens, &seq_pad, stream);
        } else {
            self.scratch.tokens_in.copy_from_slice(gpu, &tokens_pad)?;
            self.scratch.pos_ids.copy_from_slice(gpu, &pos_pad)?;
            self.scratch.seq_lens.copy_from_slice(gpu, &seq_pad)?;
        }

        // Longest key range in this batch; decides whether decode attention
        // splits its key loop across blocks.
        let max_slen = seq_pad[..b].iter().copied().max().unwrap_or(0);
        // Prefill waves already fill the GPU with one block per row; decode
        // batches (one row per sequence) do not, so they split keys as well.
        let decode_step = prefill.is_none();

        // Build all per-layer pointer tables in one shot (uploaded once per step).
        let tbl = if async_meta { 24 * b } else { 24 * mb };
        let mut k_host = vec![0u64; tbl];
        let mut v_host = vec![0u64; tbl];
        let mut conv_host = vec![0u64; tbl];
        for l in 0..layers.len() {
            if is_conv[l] {
                for (i, ctx) in ctxs.iter().enumerate() {
                    conv_host[l * b + i] = match prefill {
                        Some(_) => {
                            let base = self.scratch.conv_pool.inner().cu_deviceptr();
                            base + (((l * mb + i) * HIDDEN * 6 * 2) as u64)
                        }
                        None => ctx.conv_state_ptr(l),
                    };
                }
            } else {
                for (i, ctx) in ctxs.iter().enumerate() {
                    k_host[l * b + i] = ctx.k_cache_ptr(l);
                    v_host[l * b + i] = ctx.v_cache_ptr(l);
                }
            }
        }
        if async_meta {
            upload_async(&self.scratch.k_ptrs, &k_host, stream);
            upload_async(&self.scratch.v_ptrs, &v_host, stream);
            upload_async(&self.scratch.conv_ptrs, &conv_host, stream);
        } else {
            self.scratch.k_ptrs.copy_from_slice(gpu, &k_host)?;
            self.scratch.v_ptrs.copy_from_slice(gpu, &v_host)?;
            self.scratch.conv_ptrs.copy_from_slice(gpu, &conv_host)?;
        }
        if let Some(pf) = prefill {
            let mut carry_pad = vec![0u64; 24 * mb];
            let mut origin_pad = vec![0u32; mb];
            let mut pos_pad_local = vec![0u32; mb];
            carry_pad[..24 * b].copy_from_slice(&pf.carry_ptrs[..24 * b]);
            origin_pad[..b].copy_from_slice(&pf.origins[..b]);
            pos_pad_local[..b].copy_from_slice(&pf.local_pos[..b]);
            self.scratch.pf_carry.copy_from_slice(gpu, &carry_pad)?;
            self.scratch.pf_origin.copy_from_slice(gpu, &origin_pad)?;
            self.scratch.pf_pos.copy_from_slice(gpu, &pos_pad_local)?;
        }

        let mut h_u32 = HIDDEN as u32;
        let mut b_arg = b_u32;
        phase!();

        // 1. Batched embedding lookup or custom embedding copy.
        if let Some(embeds) = custom_embeddings {
            let dst = self.scratch.hidden.inner_mut().cu_deviceptr();
            let src = embeds.inner().cu_deviceptr();
            unsafe {
                cuMemcpyDtoDAsync_v2(dst, src, b * HIDDEN * 2, stream.cu_stream() as *mut c_void);
            }
        } else {
            let mut d_hidden = fptr(&self.scratch.hidden);
            let mut d_tbl = embed_table.inner().cu_deviceptr();
            let mut d_tok_in = self.scratch.tokens_in.inner().cu_deviceptr();
            let mut h_arg = h_u32;
            let mut b_tmp = b_arg;
            let mut args_embed: [*mut c_void; 5] = [
                &mut d_hidden as *mut _ as *mut c_void,
                &mut d_tbl as *mut _ as *mut c_void,
                &mut d_tok_in as *mut _ as *mut c_void,
                &mut h_arg as *mut _ as *mut c_void,
                &mut b_tmp as *mut _ as *mut c_void,
            ];
            unsafe {
                launch_kernel_on_stream(&self.embed, (b_u32, 1, 1), (256, 1, 1), 0, stream, &mut args_embed)?;
            }
        }

        // GEMM prefill attention: score each request chunk with tensor cores.
        // Chunks are scored one at a time because a chunk's key count is fixed
        // by its own position in its request, never by how the wave was
        // packed, which is what keeps the numerics batch-shape invariant.
        let gemm_attn_segs: Option<Vec<(usize, usize)>> = match prefill {
            Some(pf)
                if std::env::var("OXIDE_GEMM_ATTN").map(|v| v != "0").unwrap_or(true)
                    && self.lt.supports_f32_c() =>
            {
                // `origins[row]` is the wave row where the request's chunk
                // starts, and chunks never interleave, so equal origins form
                // one contiguous run.
                let mut segs: Vec<(usize, usize)> = Vec::new();
                let mut start = 0usize;
                for i in 1..=b {
                    if i == b || pf.origins[i] != pf.origins[start] {
                        segs.push((start, i - start));
                        start = i;
                    }
                }
                // Positions must be contiguous and ascending, otherwise the
                // shared key range cannot cover every row.
                let ok = segs.iter().all(|(s, n)| {
                    *n >= GEMM_ATTN_MIN_ROWS
                        && (0..*n).all(|r| pos_ids[s + r] == pos_ids[*s] + r as u32)
                });
                if ok {
                    Some(segs)
                } else {
                    None
                }
            }
            _ => None,
        };

        for (l_idx, layer) in layers.iter_mut().enumerate() {
            // Operator RMSNorm: hidden -> norm
            launch_rms(
                &self.rms_norm,
                fptr_mut(&mut self.scratch.norm),
                fptr(&self.scratch.hidden),
                fptr(&layer.operator_norm),
                b_u32,
                stream,
            )?;
            phase!();

            if is_conv[l_idx] {
                let LayerType::Conv(conv) = &layer.inner_layer else {
                    return Err(anyhow!("layer {l_idx} expected conv"));
                };
                // in_proj: [6144, 2048] x [2048, B]
                let d_x = self.scratch.norm.inner().cu_deviceptr();
                let x_len = b_u32 as u64 * HIDDEN as u64;
                let packed_on = std::env::var("OXIDE_DISABLE_PACKED").is_err();
                if packed_on && conv.in_packed.is_some() && conv.quant_type.is_some() {
                    let (ip, qt) = (conv.in_packed.as_ref().unwrap(), conv.quant_type.unwrap());
                    static PRINTED: std::sync::Once = std::sync::Once::new();
                    PRINTED.call_once(|| eprintln!("[quant-gemv] packed conv in_proj active"));
                    let y_len = b_u32 as u64 * 3 * HIDDEN as u64;
                    let d_bcx = self.scratch.bcx.inner_mut().cu_deviceptr();
                    self.gemv_dispatch(
                        stream,
                        qt,
                        (ip.inner().cu_deviceptr(), ip.inner().len() as u64),
                        (d_x, x_len),
                        (d_bcx, y_len),
                        HIDDEN as u32,
                        (3 * HIDDEN) as u32,
                        b_u32,
                    )?;
                } else {
                    let d_w = conv.in_proj.inner().cu_deviceptr();
                    let d_y = self.scratch.bcx.inner_mut().cu_deviceptr();
                    self.lt.gemm(stream, 3 * HIDDEN, HIDDEN, b, d_w, d_x, d_y)?;
                }
                phase!();

                let mut d_y = fptr_mut(&mut self.scratch.conv_y);
                let mut d_ptrs = self.scratch.conv_ptrs.inner_mut().cu_deviceptr()
                    + (l_idx * b * 8) as u64;
                let mut d_bcx = fptr(&self.scratch.bcx);
                let mut d_cw = conv.conv_weight.inner().cu_deviceptr();
                let mut k_len = 7u32;
                let mut bs = b_u32;
                if prefill.is_some() {
                    // Build this layer's per-token conv-state rows from the
                    // wave's B*x history (carry for pre-chunk positions).
                    let mut d_pool = self.scratch.conv_pool.inner_mut().cu_deviceptr()
                        + (l_idx * mb * HIDDEN * 6 * 2) as u64;
                    let mut d_carry = self.scratch.pf_carry.inner().cu_deviceptr()
                        + (l_idx * b * 8) as u64;
                    let mut d_origin = self.scratch.pf_origin.inner().cu_deviceptr();
                    let mut d_lpos = self.scratch.pf_pos.inner().cu_deviceptr();
                    let mut d_bcx_fill = fptr(&self.scratch.bcx);
                    let mut hs_fill = HIDDEN as u32;
                    let mut kl_fill = k_len;
                    let mut bs_fill = b_u32;
                    let mut row_stride = (HIDDEN * 6) as u32;
                    let mut bcx_row_bytes = (3 * HIDDEN * 2) as u32;
                    let mut x_off_bytes = (2 * HIDDEN * 2) as u32; // X is the third in_proj chunk
                    let mut args_fill: [*mut c_void; 11] = [
                        &mut d_pool as *mut _ as *mut c_void,
                        &mut d_carry as *mut _ as *mut c_void,
                        &mut d_origin as *mut _ as *mut c_void,
                        &mut d_lpos as *mut _ as *mut c_void,
                        &mut d_bcx_fill as *mut _ as *mut c_void,
                        &mut hs_fill as *mut _ as *mut c_void,
                        &mut kl_fill as *mut _ as *mut c_void,
                        &mut bs_fill as *mut _ as *mut c_void,
                        &mut row_stride as *mut _ as *mut c_void,
                        &mut bcx_row_bytes as *mut _ as *mut c_void,
                        &mut x_off_bytes as *mut _ as *mut c_void,
                    ];
                    unsafe {
                        launch_kernel_on_stream(
                            &self.conv_state_prefill,
                            (16, b_u32, 1),
                            (128, 1, 1),
                            0,
                            stream,
                            &mut args_fill,
                        )?;
                    }
                }
                phase!();
                let mut args_conv: [*mut c_void; 7] = [
                    &mut d_y as *mut _ as *mut c_void,
                    &mut d_ptrs as *mut _ as *mut c_void,
                    &mut d_bcx as *mut _ as *mut c_void,
                    &mut d_cw as *mut _ as *mut c_void,
                    &mut h_u32 as *mut _ as *mut c_void,
                    &mut k_len as *mut _ as *mut c_void,
                    &mut bs as *mut _ as *mut c_void,
                ];
                unsafe {
                    launch_kernel_on_stream(&self.conv, (16, b_u32, 1), (128, 1, 1), 0, stream, &mut args_conv)?;
                }
                phase!();

                // out_proj: [2048, 2048] x [2048, B]
                let d_x = self.scratch.conv_y.inner().cu_deviceptr();
                let x_len = b_u32 as u64 * HIDDEN as u64;
                let packed_on = std::env::var("OXIDE_DISABLE_PACKED").is_err();
                if packed_on && conv.out_packed.is_some() && conv.quant_type.is_some() {
                    let (op, qt) = (conv.out_packed.as_ref().unwrap(), conv.quant_type.unwrap());
                    static PRINTED2: std::sync::Once = std::sync::Once::new();
                    PRINTED2.call_once(|| eprintln!("[quant-gemv] packed conv out_proj active"));
                    let y_len = b_u32 as u64 * HIDDEN as u64;
                    let d_layer = self.scratch.layer.inner_mut().cu_deviceptr();
                    self.gemv_dispatch(
                        stream,
                        qt,
                        (op.inner().cu_deviceptr(), op.inner().len() as u64),
                        (d_x, x_len),
                        (d_layer, y_len),
                        HIDDEN as u32,
                        HIDDEN as u32,
                        b_u32,
                    )?;
                } else {
                    let d_w = conv.out_proj.inner().cu_deviceptr();
                    let d_y = self.scratch.layer.inner_mut().cu_deviceptr();
                    self.lt.gemm(stream, HIDDEN, HIDDEN, b, d_w, d_x, d_y)?;
                }
                phase!();
            } else {
                let LayerType::Attention(attn) = &layer.inner_layer else {
                    return Err(anyhow!("layer {l_idx} expected attention"));
                };

                // Q/K/V/gate projections.
                let d_x = self.scratch.norm.inner().cu_deviceptr();
                let x_len = b_u32 as u64 * HIDDEN as u64;
                let d_q = self.scratch.q.inner_mut().cu_deviceptr();
                let d_k = self.scratch.k.inner_mut().cu_deviceptr();
                let d_v = self.scratch.v.inner_mut().cu_deviceptr();
                let d_g = self.scratch.gate.inner_mut().cu_deviceptr();
                let packed_on = std::env::var("OXIDE_DISABLE_PACKED").is_err();
                if packed_on
                    && attn.q_packed.is_some()
                    && attn.k_packed.is_some()
                    && attn.v_packed.is_some()
                    && attn.gate_packed.is_some()
                    && attn.quant_type.is_some()
                {
                    let qp = attn.q_packed.as_ref().unwrap();
                    let kp = attn.k_packed.as_ref().unwrap();
                    let vp = attn.v_packed.as_ref().unwrap();
                    let gp = attn.gate_packed.as_ref().unwrap();
                    let qt = attn.quant_type.unwrap();
                    static PRINTED3: std::sync::Once = std::sync::Once::new();
                    PRINTED3.call_once(|| eprintln!("[quant-gemv] packed attention projections active"));
                    let q_len = b_u32 as u64 * HIDDEN as u64;
                    self.gemv_dispatch(
                        stream,
                        qt,
                        (qp.inner().cu_deviceptr(), qp.inner().len() as u64),
                        (d_x, x_len),
                        (d_q, q_len),
                        HIDDEN as u32,
                        HIDDEN as u32,
                        b_u32,
                    )?;
                    let kv_len = b_u32 as u64 * KV_DIM as u64;
                    self.gemv_dispatch(
                        stream,
                        qt,
                        (kp.inner().cu_deviceptr(), kp.inner().len() as u64),
                        (d_x, x_len),
                        (d_k, kv_len),
                        HIDDEN as u32,
                        KV_DIM as u32,
                        b_u32,
                    )?;
                    self.gemv_dispatch(
                        stream,
                        qt,
                        (vp.inner().cu_deviceptr(), vp.inner().len() as u64),
                        (d_x, x_len),
                        (d_v, kv_len),
                        HIDDEN as u32,
                        KV_DIM as u32,
                        b_u32,
                    )?;
                    let gate_len = b_u32 as u64 * NUM_Q_HEADS as u64;
                    self.gemv_dispatch(
                        stream,
                        qt,
                        (gp.inner().cu_deviceptr(), gp.inner().len() as u64),
                        (d_x, x_len),
                        (d_g, gate_len),
                        HIDDEN as u32,
                        NUM_Q_HEADS as u32,
                        b_u32,
                    )?;
                } else {
                    let d_qw = attn.q_proj.inner().cu_deviceptr();
                    self.lt.gemm(stream, HIDDEN, HIDDEN, b, d_qw, d_x, d_q)?;
                    let d_kw = attn.k_proj.inner().cu_deviceptr();
                    self.lt.gemm(stream, KV_DIM, HIDDEN, b, d_kw, d_x, d_k)?;
                    let d_vw = attn.v_proj.inner().cu_deviceptr();
                    self.lt.gemm(stream, KV_DIM, HIDDEN, b, d_vw, d_x, d_v)?;
                    let d_gw = attn.gate_proj.inner().cu_deviceptr();
                    self.lt.gemm(stream, NUM_Q_HEADS, HIDDEN, b, d_gw, d_x, d_g)?;
                }
                phase!();

                let mut eps = 1e-5f32;
                let mut nq = NUM_Q_HEADS as u32;
                let mut nkv = NUM_KV_HEADS as u32;
                let mut hd = HEAD_DIM as u32;
                let mut d_q = fptr_mut(&mut self.scratch.q);
                let mut d_k = fptr_mut(&mut self.scratch.k);
                let mut d_v = fptr_mut(&mut self.scratch.v);
                let mut d_gate = fptr_mut(&mut self.scratch.gate);
                let mut d_ctx = fptr_mut(&mut self.scratch.ctx);
                let mut d_qw = attn.q_norm.inner().cu_deviceptr();
                let mut d_kw = attn.k_norm.inner().cu_deviceptr();

                // Per-head Q/K RMSNorm.
                let mut args_qn: [*mut c_void; 6] = [
                    &mut d_q as *mut _ as *mut c_void,
                    &mut d_qw as *mut _ as *mut c_void,
                    &mut eps as *mut _ as *mut c_void,
                    &mut nq as *mut _ as *mut c_void,
                    &mut hd as *mut _ as *mut c_void,
                    &mut b_arg as *mut _ as *mut c_void,
                ];
                let mut args_kn: [*mut c_void; 6] = [
                    &mut d_k as *mut _ as *mut c_void,
                    &mut d_kw as *mut _ as *mut c_void,
                    &mut eps as *mut _ as *mut c_void,
                    &mut nkv as *mut _ as *mut c_void,
                    &mut hd as *mut _ as *mut c_void,
                    &mut b_arg as *mut _ as *mut c_void,
                ];
                unsafe {
                    launch_kernel_on_stream(&self.qk_norm, (nq, b_u32, 1), (32, 1, 1), 0, stream, &mut args_qn)?;
                    launch_kernel_on_stream(&self.qk_norm, (nkv, b_u32, 1), (32, 1, 1), 0, stream, &mut args_kn)?;
                }

                // RoPE with per-sequence positions.
                let rope = layer.rope.as_ref().ok_or_else(|| anyhow!("missing rope"))?;
                let mut d_cos = rope.cos_table.inner().cu_deviceptr();
                let mut d_sin = rope.sin_table.inner().cu_deviceptr();
                let mut d_pos = self.scratch.pos_ids.inner().cu_deviceptr();
                let mut args_qr: [*mut c_void; 7] = [
                    &mut d_q as *mut _ as *mut c_void,
                    &mut d_cos as *mut _ as *mut c_void,
                    &mut d_sin as *mut _ as *mut c_void,
                    &mut d_pos as *mut _ as *mut c_void,
                    &mut nq as *mut _ as *mut c_void,
                    &mut hd as *mut _ as *mut c_void,
                    &mut b_arg as *mut _ as *mut c_void,
                ];
                let mut args_kr: [*mut c_void; 7] = [
                    &mut d_k as *mut _ as *mut c_void,
                    &mut d_cos as *mut _ as *mut c_void,
                    &mut d_sin as *mut _ as *mut c_void,
                    &mut d_pos as *mut _ as *mut c_void,
                    &mut nkv as *mut _ as *mut c_void,
                    &mut hd as *mut _ as *mut c_void,
                    &mut b_arg as *mut _ as *mut c_void,
                ];
                unsafe {
                    launch_kernel_on_stream(&self.rope, (nq, b_u32, 1), (hd, 1, 1), 0, stream, &mut args_qr)?;
                    launch_kernel_on_stream(&self.rope, (nkv, b_u32, 1), (hd, 1, 1), 0, stream, &mut args_kr)?;
                }

                // KV cache writes.
                let mut d_kp = self.scratch.k_ptrs.inner_mut().cu_deviceptr()
                    + (l_idx * b * 8) as u64;
                let mut d_vp = self.scratch.v_ptrs.inner_mut().cu_deviceptr()
                    + (l_idx * b * 8) as u64;
                let mut kv_dim = KV_DIM as u32;
                let mut args_kc: [*mut c_void; 5] = [
                    &mut d_kp as *mut _ as *mut c_void,
                    &mut d_k as *mut _ as *mut c_void,
                    &mut d_pos as *mut _ as *mut c_void,
                    &mut kv_dim as *mut _ as *mut c_void,
                    &mut b_arg as *mut _ as *mut c_void,
                ];
                let mut args_vc: [*mut c_void; 5] = [
                    &mut d_vp as *mut _ as *mut c_void,
                    &mut d_v as *mut _ as *mut c_void,
                    &mut d_pos as *mut _ as *mut c_void,
                    &mut kv_dim as *mut _ as *mut c_void,
                    &mut b_arg as *mut _ as *mut c_void,
                ];
                unsafe {
                    launch_kernel_on_stream(&self.kv_copy, (b_u32, 1, 1), (64, 1, 1), 0, stream, &mut args_kc)?;
                    launch_kernel_on_stream(&self.kv_copy, (b_u32, 1, 1), (64, 1, 1), 0, stream, &mut args_vc)?;
                }
                phase!();

                // GQA attention. Sliding layers carry their 256-token window
                // in the rope table; full layers pass 0 (unmasked).
                let mut d_sl = self.scratch.seq_lens.inner_mut().cu_deviceptr();
                let mut window_u32 = rope.window;
                let mut args_gqa: [*mut c_void; 10] = [
                    &mut d_ctx as *mut _ as *mut c_void,
                    &mut d_q as *mut _ as *mut c_void,
                    &mut d_kp as *mut _ as *mut c_void,
                    &mut d_vp as *mut _ as *mut c_void,
                    &mut d_sl as *mut _ as *mut c_void,
                    &mut nq as *mut _ as *mut c_void,
                    &mut nkv as *mut _ as *mut c_void,
                    &mut hd as *mut _ as *mut c_void,
                    &mut b_arg as *mut _ as *mut c_void,
                    &mut window_u32 as *mut _ as *mut c_void,
                ];
                let eff = if rope.window > 0 {
                    max_slen.min(rope.window) as usize
                } else {
                    max_slen as usize
                };
                let nchunks = eff
                    .div_ceil(ATTN_CHUNK_KEYS)
                    .clamp(1, self.max_attn_chunks) as u32;
                // Tensor-core prefill attention: QK and PV as cuBLASLt
                // GEMMs over strided views of the KV slab, with one fused
                // mask+softmax pass and one gate+scatter pass per chunk.
                let mut gemm_done = false;
                if let Some(segs) = gemm_attn_segs.as_ref() {
                    // Every row of a chunk must address the same KV cache.
                    let mut cache_ok = true;
                    for (s0, n) in segs.iter() {
                        let kb = ctxs[*s0].k_cache_ptr(l_idx);
                        let vb = ctxs[*s0].v_cache_ptr(l_idx);
                        for r in 1..*n {
                            if ctxs[s0 + r].k_cache_ptr(l_idx) != kb
                                || ctxs[s0 + r].v_cache_ptr(l_idx) != vb
                            {
                                cache_ok = false;
                            }
                        }
                    }
                    if cache_ok {
                        for (s0, n) in segs.iter() {
                            let slen_min = pos_ids[*s0] + 1;
                            let slen_max = pos_ids[s0 + n - 1] + 1;
                            self.gemm_attention_chunk(
                                stream,
                                *s0,
                                *n,
                                slen_min,
                                slen_max,
                                rope.window,
                                ctxs[*s0].k_cache_ptr(l_idx),
                                ctxs[*s0].v_cache_ptr(l_idx),
                                d_q,
                                d_g,
                                d_ctx,
                            )?;
                        }
                        gemm_done = true;
                        if std::env::var("OXIDE_ATTN_DEBUG").ok().as_deref() == Some("1")
                            && ATTN_DEBUG_LEFT
                                .fetch_update(
                                    std::sync::atomic::Ordering::Relaxed,
                                    std::sync::atomic::Ordering::Relaxed,
                                    |v| v.checked_sub(1),
                                )
                                .is_ok()
                        {
                            // Run the single-pass kernel over the same wave
                            // into the logits scratch and diff on the host.
                            let mut alt = self.scratch.logits.inner_mut().cu_deviceptr();
                            let mut qq = d_q;
                            let mut kp = d_kp;
                            let mut vp = d_vp;
                            let mut sll = self.scratch.seq_lens.inner_mut().cu_deviceptr();
                            let mut nnq = nq;
                            let mut nnkv = nkv;
                            let mut hhd = hd;
                            let mut bb = b_arg;
                            let mut ww = window_u32;
                            let mut aa: [*mut c_void; 10] = [
                                &mut alt as *mut _ as *mut c_void,
                                &mut qq as *mut _ as *mut c_void,
                                &mut kp as *mut _ as *mut c_void,
                                &mut vp as *mut _ as *mut c_void,
                                &mut sll as *mut _ as *mut c_void,
                                &mut nnq as *mut _ as *mut c_void,
                                &mut nnkv as *mut _ as *mut c_void,
                                &mut hhd as *mut _ as *mut c_void,
                                &mut bb as *mut _ as *mut c_void,
                                &mut ww as *mut _ as *mut c_void,
                            ];
                            unsafe {
                                launch_kernel_on_stream(
                                    &self.gqa,
                                    (nnkv, bb, 1),
                                    (hhd, 1, 1),
                                    0,
                                    stream,
                                    &mut aa,
                                )?;
                            }
                            stream.synchronize()?;
                            let got = self.scratch.ctx.inner().to_host_vec(stream)?;
                            let altv = self.scratch.logits.inner().to_host_vec(stream)?;
                            let gate = self.scratch.gate.inner().to_host_vec(stream)?;
                            let rows_dbg = b;
                            let mut worst = (0usize, 0usize, 0.0f32);
                            let mut per_head = vec![0f32; NUM_Q_HEADS];
                            let mut scale = 0f32;
                            for r in 0..rows_dbg {
                                for h in 0..NUM_Q_HEADS {
                                    let g =
                                        1.0 / (1.0 + (-gate[r * NUM_Q_HEADS + h].to_f32()).exp());
                                    for d in 0..HEAD_DIM {
                                        let i = (r * NUM_Q_HEADS + h) * HEAD_DIM + d;
                                        let want = altv[i].to_f32() * g;
                                        let e = (got[i].to_f32() - want).abs();
                                        scale = scale.max(want.abs());
                                        per_head[h] = per_head[h].max(e);
                                        if e > worst.2 {
                                            worst = (r, h, e);
                                        }
                                    }
                                }
                            }
                            println!(
                                "[attn-debug] layer {l_idx} window {window_u32} rows {rows_dbg}: max |gemm-online| {:.5} at row {} head {} (ctx scale {:.3})",
                                worst.2, worst.0, worst.1, scale
                            );
                            println!("[attn-debug] per-head max: {:?}", per_head);
                        }
                    }
                }

                if !gemm_done {
                    if decode_step && nchunks > 1 {
                        // Decode: split the key range across blocks so the keys of
                        // one sequence are not scanned by a handful of blocks.
                        let mut chunk_keys = ATTN_CHUNK_KEYS as u32;
                        let mut d_acc = self.scratch.attn_part_acc.inner_mut().cu_deviceptr();
                        let mut d_stats = self.scratch.attn_part_stats.inner_mut().cu_deviceptr();
                        let mut nch = nchunks;
                        let mut args_split: [*mut c_void; 13] = [
                            &mut d_q as *mut _ as *mut c_void,
                            &mut d_kp as *mut _ as *mut c_void,
                            &mut d_vp as *mut _ as *mut c_void,
                            &mut d_sl as *mut _ as *mut c_void,
                            &mut d_acc as *mut _ as *mut c_void,
                            &mut d_stats as *mut _ as *mut c_void,
                            &mut nq as *mut _ as *mut c_void,
                            &mut nkv as *mut _ as *mut c_void,
                            &mut hd as *mut _ as *mut c_void,
                            &mut b_arg as *mut _ as *mut c_void,
                            &mut window_u32 as *mut _ as *mut c_void,
                            &mut chunk_keys as *mut _ as *mut c_void,
                            &mut nch as *mut _ as *mut c_void,
                        ];
                        let split_fn = if std::env::var("OXIDE_ATTN_SPLIT4")
                            .map(|v| v != "0")
                            .unwrap_or(true)
                        {
                            &self.gqa_split4
                        } else {
                            &self.gqa_split
                        };
                        unsafe {
                            launch_kernel_on_stream(
                                split_fn,
                                (nkv, b_u32, nchunks),
                                (hd, 1, 1),
                                0,
                                stream,
                                &mut args_split,
                            )?;
                            let mut d_ctxc = d_ctx;
                            let mut n_heads = nq;
                            let mut args_comb: [*mut c_void; 7] = [
                                &mut d_ctxc as *mut _ as *mut c_void,
                                &mut d_acc as *mut _ as *mut c_void,
                                &mut d_stats as *mut _ as *mut c_void,
                                &mut n_heads as *mut _ as *mut c_void,
                                &mut hd as *mut _ as *mut c_void,
                                &mut b_arg as *mut _ as *mut c_void,
                                &mut nch as *mut _ as *mut c_void,
                            ];
                            launch_kernel_on_stream(
                                &self.gqa_combine,
                                (nq, b_u32, 1),
                                (32, 1, 1),
                                0,
                                stream,
                                &mut args_comb,
                            )?;
                        }
                    } else {
                        unsafe {
                            // One block per (KV head, row); its four warps cover the
                            // four query heads that share that KV head.
                            let nkv = NUM_KV_HEADS as u32;
                            launch_kernel_on_stream(&self.gqa, (nkv, b_u32, 1), (hd, 1, 1), 0, stream, &mut args_gqa)?;
                        }
                    }
                }

                phase!();

                // Sigmoid gating. The GEMM path already folded it (and the
                // softmax normaliser) into its gate+scatter pass.
                if !gemm_done {
                    let mut args_gate: [*mut c_void; 5] = [
                        &mut d_ctx as *mut _ as *mut c_void,
                        &mut d_gate as *mut _ as *mut c_void,
                        &mut nq as *mut _ as *mut c_void,
                        &mut hd as *mut _ as *mut c_void,
                        &mut b_arg as *mut _ as *mut c_void,
                    ];
                    unsafe {
                        launch_kernel_on_stream(&self.sigmoid_gate, (nq, b_u32, 1), (hd, 1, 1), 0, stream, &mut args_gate)?;
                    }
                }

                // o_proj: [2048, 2048] x [2048, B]
                let d_x = self.scratch.ctx.inner().cu_deviceptr();
                let x_len = b_u32 as u64 * HIDDEN as u64;
                let packed_on = std::env::var("OXIDE_DISABLE_PACKED").is_err();
                if packed_on && attn.o_packed.is_some() && attn.quant_type.is_some() {
                    let (op, qt) = (attn.o_packed.as_ref().unwrap(), attn.quant_type.unwrap());
                    static PRINTED4: std::sync::Once = std::sync::Once::new();
                    PRINTED4.call_once(|| eprintln!("[quant-gemv] packed o_proj active"));
                    let y_len = b_u32 as u64 * HIDDEN as u64;
                    let d_layer = self.scratch.layer.inner_mut().cu_deviceptr();
                    self.gemv_dispatch(
                        stream,
                        qt,
                        (op.inner().cu_deviceptr(), op.inner().len() as u64),
                        (d_x, x_len),
                        (d_layer, y_len),
                        HIDDEN as u32,
                        HIDDEN as u32,
                        b_u32,
                    )?;
                } else {
                    let d_w = attn.o_proj.inner().cu_deviceptr();
                    let d_y = self.scratch.layer.inner_mut().cu_deviceptr();
                    self.lt.gemm(stream, HIDDEN, HIDDEN, b, d_w, d_x, d_y)?;
                }
                phase!();
            }

            // Post-operator norm (in place), then residual add.
            launch_rms(
                &self.rms_norm,
                fptr_mut(&mut self.scratch.layer),
                fptr(&self.scratch.layer),
                fptr(&layer.post_operator_norm),
                b_u32,
                stream,
            )?;
            self.residual_add(&self.scratch.hidden, &self.scratch.layer, b_u32, stream)?;

            // FFN norm.
            launch_rms(
                &self.rms_norm,
                fptr_mut(&mut self.scratch.norm),
                fptr(&self.scratch.hidden),
                fptr(&layer.ffn_norm),
                b_u32,
                stream,
            )?;

            // FFN: W1/W3 -> SwiGLU -> W2.
            let d_x = self.scratch.norm.inner().cu_deviceptr();
            let x_len = b_u32 as u64 * HIDDEN as u64;

            let packed_disabled = std::env::var("OXIDE_DISABLE_PACKED").is_ok();
            let packed_ready = !packed_disabled
                && layer.ffn.w1_packed.is_some()
                && layer.ffn.w3_packed.is_some()
                && layer.ffn.w2_packed.is_some()
                && layer.ffn.quant_type.is_some();
            let ffn_int8 = std::env::var("OXIDE_MMVQ_INT8_FFN").map(|v| v != "0").unwrap_or(true);
            let int8_ready = packed_ready
                && ffn_int8
                && b_u32 == 1
                && oxide_infer_kernels::quant_gemv::int8_mmvq_enabled()
                && oxide_infer_kernels::quant_gemv::int8_mmvq_supported(layer.ffn.quant_type.unwrap());
            if int8_ready {
                let qt = layer.ffn.quant_type.unwrap();
                static PRINTED_I8: std::sync::Once = std::sync::Once::new();
                PRINTED_I8.call_once(|| eprintln!("[quant-gemv] int8 MMVQ FFN path active (type {:?})", qt));
                // Int8 MMVQ path: activations are quantized to q8_1 once per
                // stage (k = HIDDEN for the gate/up GEMM, k = INTER for the
                // down GEMM), then consumed by dp4a matvec kernels - the same
                // kernel llama.cpp launches at batch = 1 on sm_86.
                let act_ptr = self.scratch.ffn_act.inner_mut().cu_deviceptr();
                let act_len = b_u32 as u64 * INTER as u64;
                let q8h_ptr = self.scratch.q8_1_hidden.inner_mut().cu_deviceptr();
                let q8h_len = b_u32 as u64 * HIDDEN as u64 / 32 * 36;
                self.quant_gemv.quantize_q8_1(
                    stream,
                    (d_x, x_len),
                    (q8h_ptr, q8h_len),
                    HIDDEN as u32,
                    b_u32,
                )?;
                self.quant_gemv.mmvq_dual_swiglu_q4_0(
                    stream,
                    (layer.ffn.w1_packed.as_ref().unwrap().inner().cu_deviceptr(),
                     layer.ffn.w1_packed.as_ref().unwrap().inner().len() as u64),
                    (layer.ffn.w3_packed.as_ref().unwrap().inner().cu_deviceptr(),
                     layer.ffn.w3_packed.as_ref().unwrap().inner().len() as u64),
                    (q8h_ptr, q8h_len),
                    (act_ptr, act_len),
                    HIDDEN as u32,
                    INTER as u32,
                    b_u32,
                )?;
                let q8i_ptr = self.scratch.q8_1_inter.inner_mut().cu_deviceptr();
                let q8i_len = b_u32 as u64 * INTER as u64 / 32 * 36;
                self.quant_gemv.quantize_q8_1(
                    stream,
                    (act_ptr, act_len),
                    (q8i_ptr, q8i_len),
                    INTER as u32,
                    b_u32,
                )?;
                let out_ptr = self.scratch.ffn_out.inner_mut().cu_deviceptr();
                let out_len = b_u32 as u64 * HIDDEN as u64;
                self.quant_gemv.mmvq_gemv_q4_0(
                    stream,
                    (layer.ffn.w2_packed.as_ref().unwrap().inner().cu_deviceptr(),
                     layer.ffn.w2_packed.as_ref().unwrap().inner().len() as u64),
                    (q8i_ptr, q8i_len),
                    (out_ptr, out_len),
                    INTER as u32,
                    HIDDEN as u32,
                    b_u32,
                )?;
            } else if packed_ready {
                let qt = layer.ffn.quant_type.unwrap();
                static PRINTED: std::sync::Once = std::sync::Once::new();
                PRINTED.call_once(|| eprintln!("[quant-gemv] packed FFN path active (type {:?})", qt));
                // Packed path: fused dual GEMV + SwiGLU straight from packed
                // ggml weights in VRAM.
                let act_ptr = self.scratch.ffn_act.inner_mut().cu_deviceptr();
                let act_len = b_u32 as u64 * INTER as u64;
                self.quant_gemv.dual_swiglu(
                    stream,
                    qt,
                    (layer.ffn.w1_packed.as_ref().unwrap().inner().cu_deviceptr(),
                     layer.ffn.w1_packed.as_ref().unwrap().inner().len() as u64),
                    (layer.ffn.w3_packed.as_ref().unwrap().inner().cu_deviceptr(),
                     layer.ffn.w3_packed.as_ref().unwrap().inner().len() as u64),
                    (d_x, x_len),
                    (act_ptr, act_len),
                    HIDDEN as u32,
                    INTER as u32,
                    b_u32,
                )?;
                let out_ptr = self.scratch.ffn_out.inner_mut().cu_deviceptr();
                let out_len = b_u32 as u64 * HIDDEN as u64;
                self.gemv_dispatch(
                    stream,
                    qt,
                    (layer.ffn.w2_packed.as_ref().unwrap().inner().cu_deviceptr(),
                     layer.ffn.w2_packed.as_ref().unwrap().inner().len() as u64),
                    (act_ptr, act_len),
                    (out_ptr, out_len),
                    INTER as u32,
                    HIDDEN as u32,
                    b_u32,
                )?;
            } else {
                let d_w1 = layer.ffn.w1_gate.inner().cu_deviceptr();
                let d_gate = self.scratch.ffn_gate.inner_mut().cu_deviceptr();
                self.lt.gemm(stream, INTER, HIDDEN, b, d_w1, d_x, d_gate)?;
                let d_w3 = layer.ffn.w3_up.inner().cu_deviceptr();
                let d_up = self.scratch.ffn_up.inner_mut().cu_deviceptr();
                self.lt.gemm(stream, INTER, HIDDEN, b, d_w3, d_x, d_up)?;

                let mut d_act = fptr_mut(&mut self.scratch.ffn_act);
                let mut d_fg = fptr(&self.scratch.ffn_gate);
                let mut d_fu = fptr(&self.scratch.ffn_up);
                let mut inter = INTER as u32;
                let mut args_sw: [*mut c_void; 5] = [
                    &mut d_act as *mut _ as *mut c_void,
                    &mut d_fg as *mut _ as *mut c_void,
                    &mut d_fu as *mut _ as *mut c_void,
                    &mut inter as *mut _ as *mut c_void,
                    &mut b_arg as *mut _ as *mut c_void,
                ];
                unsafe {
                    launch_kernel_on_stream(&self.swiglu, (30, b_u32, 1), (256, 1, 1), 0, stream, &mut args_sw)?;
                }
                let d_w2 = layer.ffn.w2_down.inner().cu_deviceptr();
                let d_act = self.scratch.ffn_act.inner().cu_deviceptr();
                let d_out = self.scratch.ffn_out.inner_mut().cu_deviceptr();
                self.lt.gemm(stream, HIDDEN, INTER, b, d_w2, d_act, d_out)?;
            }
            phase!();

            // Post-FFN norm (in place) + residual add.
            launch_rms(
                &self.rms_norm,
                fptr_mut(&mut self.scratch.ffn_out),
                fptr(&self.scratch.ffn_out),
                fptr(&layer.post_ffn_norm),
                b_u32,
                stream,
            )?;
            self.residual_add(&self.scratch.hidden, &self.scratch.ffn_out, b_u32, stream)?;
            phase!();
        }
        debug_assert!(!prof || mark == 1 + layers.len() * 7);

        if !sample {
            // Persist this position's hidden state into each sequence's
            // context buffer. After a sequence's LAST prefill position this
            // holds the final prompt hidden, which the post-prefill
            // `sample_only` step consumes (see `sample_only`).
            let s_hidden = self.scratch.hidden.inner().cu_deviceptr();
            for (i, ctx) in ctxs.iter().enumerate() {
                let dst = ctx.hidden_ptr();
                let src = s_hidden + (i * HIDDEN * 2) as u64;
                unsafe {
                    cuMemcpyDtoDAsync_v2(dst, src, HIDDEN * 2, stream.cu_stream() as *mut c_void);
                }
            }
            if prof {
                report_phases(&prof_ev[..mark], layers.len(), b, is_conv);
            }
            return Ok(Vec::new());
        }

        // Final RMSNorm + LM head + batched argmax.
        let mut out = self.sample_tail(gpu, b_u32, embed_table, emb_norm, stream)?;

        if prof {
            report_phases(&prof_ev[..mark], layers.len(), b, is_conv);
        }
        out.truncate(b);
        Ok(out)
    }

    /// Final RMSNorm + LM head + batched argmax over `scratch.hidden`.
    /// Callers prepare `scratch.hidden` themselves: either from a full
    /// 24-layer forward (`forward_step`) or by copying a context's
    /// persisted prefill hidden (`sample_only`).
    fn sample_tail(
        &mut self,
        gpu: &GpuContext,
        b_u32: u32,
        embed_table: &DeviceBuffer<f16>,
        emb_norm: &DeviceBuffer<f16>,
        stream: &CuStream,
    ) -> Result<Vec<u32>> {
        launch_rms(
            &self.rms_norm,
            fptr_mut(&mut self.scratch.norm),
            fptr(&self.scratch.hidden),
            emb_norm.inner().cu_deviceptr(),
            b_u32,
            stream,
        )?;

        let d_tbl = embed_table.inner().cu_deviceptr();
        let d_x = self.scratch.norm.inner().cu_deviceptr();
        let d_logits = self.scratch.logits.inner_mut().cu_deviceptr();
        self.lt.gemm(stream, VOCAB, HIDDEN, b_u32 as usize, d_tbl, d_x, d_logits)?;

        let mut d_out = self.scratch.tokens_out.inner_mut().cu_deviceptr();
        let mut d_logits = self.scratch.logits.inner().cu_deviceptr();
        let mut vocab = VOCAB as u32;
        let mut args_am: [*mut c_void; 3] = [
            &mut d_out as *mut _ as *mut c_void,
            &mut d_logits as *mut _ as *mut c_void,
            &mut vocab as *mut _ as *mut c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&self.argmax, (b_u32, 1, 1), (256, 1, 1), 0, stream, &mut args_am)?;
        }

        let mut out = self
            .scratch
            .tokens_out
            .inner()
            .to_host_vec(stream)
            .map_err(|e| anyhow!("token download: {e}"))?;
        Ok(out)
    }

    /// Sample the next token directly from a sequence's persisted prefill
    /// hidden state (`ctx.hidden_ptr()`), without running any transformer
    /// layer. Used right after prefill: the last prompt position's hidden
    /// already produced the next-token distribution.
    pub fn sample_only<C: BatchSequence>(
        &mut self,
        gpu: &GpuContext,
        ctxs: &[C],
        embed_table: &DeviceBuffer<f16>,
        emb_norm: &DeviceBuffer<f16>,
    ) -> Result<Vec<u32>> {
        let b = ctxs.len();
        if b == 0 || b > self.max_batch {
            return Err(anyhow!("invalid sample_only batch size {b}"));
        }
        let b_u32 = b as u32;
        let stream = gpu.compute_stream();

        // Copy each context's persisted hidden into the engine scratch so
        // the shared sampling tail can run over it.
        let s_hidden = self.scratch.hidden.inner_mut().cu_deviceptr();
        for (i, ctx) in ctxs.iter().enumerate() {
            let src = ctx.hidden_ptr();
            let dst = s_hidden + (i * HIDDEN * 2) as u64;
            unsafe {
                cuMemcpyDtoDAsync_v2(dst, src, HIDDEN * 2, stream.cu_stream() as *mut c_void);
            }
        }

        let mut out = self.sample_tail(gpu, b_u32, embed_table, emb_norm, stream)?;
        out.truncate(b);
        Ok(out)
    }

    fn residual_add(
        &self,
        residual: &DeviceBuffer<f16>,
        update: &DeviceBuffer<f16>,
        b: u32,
        stream: &cuda_core::Stream,
    ) -> Result<()> {
        let mut d_res = residual.inner().cu_deviceptr();
        let mut d_upd = update.inner().cu_deviceptr();
        let mut h = HIDDEN as u32;
        let mut bs = b;
        let mut args: [*mut c_void; 4] = [
            &mut d_res as *mut _ as *mut c_void,
            &mut d_upd as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&self.residual_add, (1, b, 1), (256, 1, 1), 0, stream, &mut args)?;
        }
        Ok(())
    }
}

fn launch_rms(
    func: &Function,
    out: u64,
    input: u64,
    weight: u64,
    b: u32,
    stream: &cuda_core::Stream,
) -> Result<()> {
    let mut d_out = out;
    let mut d_in = input;
    let mut d_w = weight;
    let mut eps = 1e-5f32;
    let mut h = HIDDEN as u32;
    let mut bs = b;
    let mut args: [*mut c_void; 6] = [
        &mut d_out as *mut _ as *mut c_void,
        &mut d_in as *mut _ as *mut c_void,
        &mut d_w as *mut _ as *mut c_void,
        &mut eps as *mut _ as *mut c_void,
        &mut h as *mut _ as *mut c_void,
        &mut bs as *mut _ as *mut c_void,
    ];
    unsafe {
        launch_kernel_on_stream(func, (1, b, 1), (256, 1, 1), 0, stream, &mut args)?;
    }
    Ok(())
}
