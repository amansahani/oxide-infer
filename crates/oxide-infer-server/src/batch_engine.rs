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
    BATCHED_ARGMAX_DIRECT_PTX, BATCHED_CONV_PTR_PTX, BATCHED_EMBED_LOOKUP_PTX,
    BATCHED_GQA_ATTENTION_PTX, BATCHED_KV_COPY_PTX, BATCHED_QK_RMSNORM_PTX, BATCHED_ROPE_PTX,
    BATCHED_SIGMOID_GATE_PTX, CONV_STATE_PREFILL_PTX,
};
use oxide_infer_kernels::PtxModule;
use oxide_infer_models::{FullLfm3DecoderLayer, LayerType};
use cuda_core::{launch_kernel_on_stream, Function, Stream as CuStream};
use std::ffi::c_void;

use crate::lt_gemm::LtGemm;

extern "C" {
    /// CUDA driver D2D copy, declared in server.rs too; duplicated here so
    /// the engine can persist/restore per-context hidden states.
    pub fn cuMemcpyDtoDAsync_v2(dst: u64, src: u64, byte_count: usize, stream: *mut c_void) -> i32;
}

pub const HIDDEN: usize = 2048;
pub const INTER: usize = 7680;
pub const KV_DIM: usize = 512;
pub const NUM_Q_HEADS: usize = 16;
pub const NUM_KV_HEADS: usize = 4;
pub const HEAD_DIM: usize = 128;
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
    bcx: DeviceBuffer<f16>,
    conv_y: DeviceBuffer<f16>,
    logits: DeviceBuffer<f16>,
    tokens_in: DeviceBuffer<u32>,
    tokens_out: DeviceBuffer<u32>,
    pos_ids: DeviceBuffer<u32>,
    seq_lens: DeviceBuffer<u32>,
    k_ptrs: DeviceBuffer<u64>,
    v_ptrs: DeviceBuffer<u64>,
    conv_ptrs: DeviceBuffer<u64>,
    /// Per-layer slices of prefill conv-state rows. One slice per model layer
    /// (only the conv layers are used), each with `max_batch` rows of
    /// `HIDDEN * 6` f16. Reused by every prefill wave.
    conv_pool: DeviceBuffer<f16>,
    pf_carry: DeviceBuffer<u64>,
    pf_origin: DeviceBuffer<u32>,
    pf_pos: DeviceBuffer<u32>,
}

/// Token-batched prefill metadata for one wave.
///
/// `carry_ptrs` uses the same `[layer * max_batch + row]` layout as the
/// per-layer pointer tables and must hold each request's persistent conv
/// state pointer for that layer. `origins[row]` is the wave row where the
/// request's chunk begins (used to locate window tokens inside `bcx`) and
/// `local_pos[row]` is the row's position within its chunk.
pub struct PrefillMeta {
    pub carry_ptrs: Vec<u64>,
    pub origins: Vec<u32>,
    pub local_pos: Vec<u32>,
}

/// Batched transformer engine. All scratch buffers are allocated once for the
/// maximum supported batch size.
pub struct BatchEngine {
    scratch: BatchScratch,
    max_batch: usize,
    lt: LtGemm,
    rms_norm: Function,
    residual_add: Function,
    swiglu: Function,
    conv: Function,
    qk_norm: Function,
    rope: Function,
    kv_copy: Function,
    gqa: Function,
    sigmoid_gate: Function,
    embed: Function,
    argmax: Function,
    conv_state_prefill: Function,
}

fn fptr(buf: &DeviceBuffer<f16>) -> u64 {
    buf.inner().cu_deviceptr()
}

fn fptr_mut(buf: &mut DeviceBuffer<f16>) -> u64 {
    buf.inner_mut().cu_deviceptr()
}

impl BatchEngine {
    pub fn new(gpu: &GpuContext, max_batch: usize) -> Result<Self> {
        let b = max_batch;
        let alloc_f16 = |n: usize| -> Result<DeviceBuffer<f16>> {
            DeviceBuffer::<f16>::zeroed(gpu, n).map_err(|e| anyhow!("alloc: {e}"))
        };

        let norm_ptx = PtxModule::load_from_ptx(gpu, oxide_infer_models::RESIDUAL_ADD_PTX)
            .map_err(|e| anyhow!("norm PTX: {e}"))?;
        let attn_ptx = PtxModule::load_from_ptx(gpu, BATCHED_QK_RMSNORM_PTX)
            .map_err(|e| anyhow!("qk_norm PTX: {e}"))?;
        let rope_ptx = PtxModule::load_from_ptx(gpu, BATCHED_ROPE_PTX)
            .map_err(|e| anyhow!("rope PTX: {e}"))?;
        let kv_ptx = PtxModule::load_from_ptx(gpu, BATCHED_KV_COPY_PTX)
            .map_err(|e| anyhow!("kv_copy PTX: {e}"))?;
        let gqa_ptx = PtxModule::load_from_ptx(gpu, BATCHED_GQA_ATTENTION_PTX)
            .map_err(|e| anyhow!("gqa PTX: {e}"))?;
        let gate_ptx = PtxModule::load_from_ptx(gpu, BATCHED_SIGMOID_GATE_PTX)
            .map_err(|e| anyhow!("sigmoid gate PTX: {e}"))?;
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
            bcx: alloc_f16(b * 3 * HIDDEN)?,
            conv_y: alloc_f16(b * HIDDEN)?,
            logits: alloc_f16(b * VOCAB)?,
            tokens_in: DeviceBuffer::<u32>::zeroed(gpu, b).map_err(|e| anyhow!("alloc: {e}"))?,
            tokens_out: DeviceBuffer::<u32>::zeroed(gpu, b).map_err(|e| anyhow!("alloc: {e}"))?,
            pos_ids: DeviceBuffer::<u32>::zeroed(gpu, b).map_err(|e| anyhow!("alloc: {e}"))?,
            seq_lens: DeviceBuffer::<u32>::zeroed(gpu, b).map_err(|e| anyhow!("alloc: {e}"))?,
            k_ptrs: DeviceBuffer::<u64>::zeroed(gpu, 24 * b).map_err(|e| anyhow!("alloc: {e}"))?,
            v_ptrs: DeviceBuffer::<u64>::zeroed(gpu, 24 * b).map_err(|e| anyhow!("alloc: {e}"))?,
            conv_ptrs: DeviceBuffer::<u64>::zeroed(gpu, 24 * b).map_err(|e| anyhow!("alloc: {e}"))?,
            conv_pool: alloc_f16(24 * b * HIDDEN * 6)?,
            pf_carry: DeviceBuffer::<u64>::zeroed(gpu, 24 * b).map_err(|e| anyhow!("alloc: {e}"))?,
            pf_origin: DeviceBuffer::<u32>::zeroed(gpu, b).map_err(|e| anyhow!("alloc: {e}"))?,
            pf_pos: DeviceBuffer::<u32>::zeroed(gpu, b).map_err(|e| anyhow!("alloc: {e}"))?,
        };

        let lt = LtGemm::new(gpu, max_batch)?;
        Ok(Self {
            scratch,
            max_batch,
            lt,
            rms_norm: norm_ptx.get_function("batched_rms_norm_vec8_fp16")?,
            residual_add: norm_ptx.get_function("batched_residual_add_fp16")?,
            swiglu: norm_ptx.get_function("batched_swiglu_separate_fp16")?,
            conv: conv_ptx.get_function("batched_fused_causal_conv1d_gating_ptr_fp16")?,
            qk_norm: attn_ptx.get_function("batched_qk_rmsnorm_fp16")?,
            rope: rope_ptx.get_function("batched_rope_fp16")?,
            kv_copy: kv_ptx.get_function("batched_kv_copy_fp16")?,
            gqa: gqa_ptx.get_function("batched_gqa_attention_fp16")?,
            sigmoid_gate: gate_ptx.get_function("batched_sigmoid_gate_fp16")?,
            embed: embed_ptx.get_function("batched_embed_lookup_fp16")?,
            argmax: argmax_ptx.get_function("batched_argmax_direct_f16")?,
            conv_state_prefill: pf_ptx.get_function("conv_state_prefill_fp16")?,
        })
    }

    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Device pointer of a prefill conv-state row (decode-ready state of the
    /// chunk's last token). Only valid after `forward_step` with `prefill`.
    pub fn prefill_state_ptr(&self, layer: usize, row: usize) -> u64 {
        self.scratch.conv_pool.inner().cu_deviceptr()
            + (((layer * self.max_batch + row) * HIDDEN * 6 * 2) as u64)
    }

    /// Run one transformer step for `ctxs`.
    ///
    /// `tokens[i]` is embedded for sequence `i`; `pos_ids[i]` is used for RoPE
    /// and the KV cache write. When `sample` is true the final RMSNorm, LM head
    /// and argmax are executed and the next tokens are returned.
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
        let b = ctxs.len();
        if b == 0 || b > self.max_batch {
            return Err(anyhow!("invalid batch size {b}"));
        }
        let b_u32 = b as u32;
        let stream = gpu.compute_stream();

        // Device buffers are allocated for the maximum batch size; pad the
        // tiny metadata arrays so the copy lengths always match.
        let mb = self.max_batch;
        let mut tokens_pad = vec![0u32; mb];
        let mut pos_pad = vec![0u32; mb];
        let mut seq_pad = vec![0u32; mb];
        tokens_pad[..b].copy_from_slice(tokens);
        pos_pad[..b].copy_from_slice(pos_ids);
        for i in 0..b {
            seq_pad[i] = pos_ids[i] + 1;
        }
        self.scratch.tokens_in.copy_from_slice(gpu, &tokens_pad)?;
        self.scratch.pos_ids.copy_from_slice(gpu, &pos_pad)?;
        self.scratch.seq_lens.copy_from_slice(gpu, &seq_pad)?;

        // Build all per-layer pointer tables in one shot (uploaded once per step).
        let mut k_host = vec![0u64; 24 * mb];
        let mut v_host = vec![0u64; 24 * mb];
        let mut conv_host = vec![0u64; 24 * mb];
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
        self.scratch.k_ptrs.copy_from_slice(gpu, &k_host)?;
        self.scratch.v_ptrs.copy_from_slice(gpu, &v_host)?;
        self.scratch.conv_ptrs.copy_from_slice(gpu, &conv_host)?;
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

        // 1. Batched embedding lookup.
        let mut d_hidden = fptr(&self.scratch.hidden);
        let mut d_tbl = embed_table.inner().cu_deviceptr();
        let mut d_tok_in = self.scratch.tokens_in.inner().cu_deviceptr();
        let mut h_u32 = HIDDEN as u32;
        let mut b_arg = b_u32;
        let mut args_embed: [*mut c_void; 5] = [
            &mut d_hidden as *mut _ as *mut c_void,
            &mut d_tbl as *mut _ as *mut c_void,
            &mut d_tok_in as *mut _ as *mut c_void,
            &mut h_u32 as *mut _ as *mut c_void,
            &mut b_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&self.embed, (b_u32, 1, 1), (256, 1, 1), 0, stream, &mut args_embed)?;
        }

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

            if is_conv[l_idx] {
                let LayerType::Conv(conv) = &layer.inner_layer else {
                    return Err(anyhow!("layer {l_idx} expected conv"));
                };
                // in_proj: [6144, 2048] x [2048, B]
                let d_w = conv.in_proj.inner().cu_deviceptr();
                let d_x = self.scratch.norm.inner().cu_deviceptr();
                let d_y = self.scratch.bcx.inner_mut().cu_deviceptr();
                self.lt.gemm(stream, 3 * HIDDEN, HIDDEN, b, d_w, d_x, d_y)?;

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

                // out_proj: [2048, 2048] x [2048, B]
                let d_w = conv.out_proj.inner().cu_deviceptr();
                let d_x = self.scratch.conv_y.inner().cu_deviceptr();
                let d_y = self.scratch.layer.inner_mut().cu_deviceptr();
                self.lt.gemm(stream, HIDDEN, HIDDEN, b, d_w, d_x, d_y)?;
            } else {
                let LayerType::Attention(attn) = &layer.inner_layer else {
                    return Err(anyhow!("layer {l_idx} expected attention"));
                };

                // Q/K/V/gate projections.
                let d_x = self.scratch.norm.inner().cu_deviceptr();
                let d_qw = attn.q_proj.inner().cu_deviceptr();
                let d_q = self.scratch.q.inner_mut().cu_deviceptr();
                self.lt.gemm(stream, HIDDEN, HIDDEN, b, d_qw, d_x, d_q)?;
                let d_kw = attn.k_proj.inner().cu_deviceptr();
                let d_k = self.scratch.k.inner_mut().cu_deviceptr();
                self.lt.gemm(stream, KV_DIM, HIDDEN, b, d_kw, d_x, d_k)?;
                let d_vw = attn.v_proj.inner().cu_deviceptr();
                let d_v = self.scratch.v.inner_mut().cu_deviceptr();
                self.lt.gemm(stream, KV_DIM, HIDDEN, b, d_vw, d_x, d_v)?;
                let d_gw = attn.gate_proj.inner().cu_deviceptr();
                let d_g = self.scratch.gate.inner_mut().cu_deviceptr();
                self.lt.gemm(stream, NUM_Q_HEADS, HIDDEN, b, d_gw, d_x, d_g)?;

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

                // GQA attention.
                let mut d_sl = self.scratch.seq_lens.inner_mut().cu_deviceptr();
                let mut args_gqa: [*mut c_void; 9] = [
                    &mut d_ctx as *mut _ as *mut c_void,
                    &mut d_q as *mut _ as *mut c_void,
                    &mut d_kp as *mut _ as *mut c_void,
                    &mut d_vp as *mut _ as *mut c_void,
                    &mut d_sl as *mut _ as *mut c_void,
                    &mut nq as *mut _ as *mut c_void,
                    &mut nkv as *mut _ as *mut c_void,
                    &mut hd as *mut _ as *mut c_void,
                    &mut b_arg as *mut _ as *mut c_void,
                ];
                unsafe {
                    launch_kernel_on_stream(&self.gqa, (nq, b_u32, 1), (hd, 1, 1), 0, stream, &mut args_gqa)?;
                }

                // Sigmoid gating.
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

                // o_proj: [2048, 2048] x [2048, B]
                let d_w = attn.o_proj.inner().cu_deviceptr();
                let d_x = self.scratch.ctx.inner().cu_deviceptr();
                let d_y = self.scratch.layer.inner_mut().cu_deviceptr();
                self.lt.gemm(stream, HIDDEN, HIDDEN, b, d_w, d_x, d_y)?;
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
        }

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
            return Ok(Vec::new());
        }

        // Final RMSNorm + LM head + batched argmax.
        let mut out = self.sample_tail(gpu, b_u32, embed_table, emb_norm, stream)?;

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
