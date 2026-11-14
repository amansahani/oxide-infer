# Pure-Rust Migration Roadmap & Multi-Architecture Support

Goal: eliminate every handwritten inline-PTX kernel in the engine, replacing
them with cuda-oxide Rust kernels, and make the engine run on **all NVIDIA
architectures** (Volta through Hopper, Blackwell and Rubin), with dynamic,
per-checkpoint quant-variant selection.

**Status:** planning. Baseline inventory committed below. No code migrated yet.

---

## 1. Inventory of what exists today (measured 2026-11-14)

91 inline-PTX kernels embedded as strings, all targeting `sm_80` (one `sm_86`),
all loaded through `cuda-core`'s `PtxModule::load_from_ptx`:

| file | kernels | bytes | notes |
|---|---|---|---|
| oxide-infer-server/src/bin/server.rs | 2 | 112,641 | argmax + another; test binary only |
| oxide-infer-kernels/src/batched_attn.rs | 18 | 90,698 | flash-attn online/softmax/pack/scatter/GQA |
| oxide-infer-kernels/src/gemv.rs | 1 | 59,197 | generic GEMV variants |
| oxide-infer-models/src/full_layer.rs | 9 | 67,766 | fused residual/norm/swiglu |
| oxide-infer-kernels/src/gemm.rs | 4 | 24,050 | strided GEMM helpers |
| oxide-infer-models/src/attention_layer.rs | 5 | 28,086 | GQA attention |
| oxide-infer-models/src/lm_head.rs | 5 | 41,101 | tied LM head + argmax |
| oxide-infer-kernels/src/vision.rs | 14 | 45,816 | vision tower (conv, attention, RoPE…) |
| oxide-infer-models/src/conv_layer.rs | 1 | 14,399 | shortconv |
| oxide-infer-kernels/src/conv.rs | 1 | 7,671 | conv pointer tables |
| oxide-infer-models/src/projector.rs | 2 | 8,678 | vision projector |
| oxide-infer-models/src/rope.rs | 1 | 7,716 | rotary embeddings |
| oxide-infer-kernels/src/activations.rs | 2 | 6,520 | silu/swiglu |
| oxide-infer-kernels/src/norm.rs | 1 | 5,639 | RMSNorm |
| oxide-infer-kernels/src/attention.rs | 1 | 3,515 | small attention helper |
| oxide-infer-models/src/embedding.rs | 1 | 3,877 | embedding lookup |
| oxide-infer-models/src/gqa_engine.rs | 1 | 8,636 | GQA online attention |
| oxide-infer-server/src/bin/debug_rust_l0_end.rs | 1 | 7,731 | debug tool |

Model-layer kernels (attention_layer, lm_head, full_layer, conv_layer, rope,
projector, embedding, gqa_engine) are duplicated conceptually by the
server-layer equivalents; consolidating to `oxide-infer-kernels` is part of the
migration.

Engineered kernels already in the cuda-oxide crates:
- `kernels/quant-gemv` — packed GEMV / fused swiglu for all 14 quant types,
  plus int8 MMVQ path (the decode hot path), 3,238 lines PTX committed as
  `kernels/quant-gemv/gemv.ptx`, verified byte-exact against the
  dequantized reference.

---

## 2. Support matrix we are designing for

PTX is forward-compatible: an sm_80 PTX module JIT-compiles on any later GPU
(Ampere → Hopper → Blackwell → Rubin). So **correctness on all architectures
comes for free by shipping PTX**; **performance** requires per-arch codepaths.

| family | CC | features we use/target |
|---|---|---|
| Volta | sm_70 | none (drop or best-effort) |
| Turing | sm_75 | drop or best-effort |
| Ampere | sm_80/sm_86 | current target; keep as baseline |
| Hopper | sm_90/sm_90a | TMA, WGMMA (`wgmma_mma_async`), cluster launch |
| Blackwell DC | sm_100a/sm_103a | tcgen05, CLC, TMEM, 2-CTA MMA |
| Blackwell gaming/Spark | sm_120/sm_121 | mma.sync-style, no tcgen05/TMEM |
| Rubin | sm_120-class+family | **unknown ID**; ships with PTX and JIT; perf path added when silicon samples exist |

Architecture knobs we must respect:
- `compute_100f` / `compute_120f` family-specific targets (CUDA 12.9+) for
  forward compatibility *within* a family.
- `sm_90a` vs `sm_90` — `a` variants enable WGMMA.
- PTX embedding for JIT on unknown future silicon.

---

## 3. Migration architecture

### 3.1 One crate, one artifact, arch matrix at load time

`kernels/oxide-kernels/` (new cuda-oxide crate):
- every kernel is a `#[kernel]` function, monomorphized per quant type
- build with `cargo oxide build --arch sm_80` etc. → `oxide_kernels_sm80.ptx`,
  `oxide_kernels_sm90.ptx`, `oxide_kernels_sm100a.ptx`, `oxide_kernels_sm120.ptx`
- artifacts **committed** (same model as quant-gemv today); no build-time
  cuda-oxide dependency in oxide-infer-kernels
- runtime loader picks the **highest supported PTX** for the detected device;
  falls back to the lowest PTX for JIT on unknown silicon

### 3.2 Dynamic quant variant (monomorphized runtime dispatch)

Dispatch table built at model-load time:
```
match (quant_type, kernel_family) {
    (Q4_0, Gemv)        => gpu::kernels::gemv_q4_0,
    (Q4_K, Gemv)        => gpu::kernels::gemv_q4_k,
    ...14 quant types x {gemv, dual_swiglu, mmvq}
}
```
- dispatch is a single `match` in `oxide-infer-kernels/src/quant_gemv.rs`
  (already the pattern today); extended to all 14 types for the *int8 MMVQ*
  path and all 14 for the f32 packed path; K types remain HGEMM-gated until
  their MMVQ kernels beat cuBLASLt on measured hardware (scientific gate).
- model config (`__quantized__`) decides which kernel set is loaded; engine
  never recompiles.

### 3.3 Verification per phase (same gates as the packed-path work)

Every ported kernel must clear, on this machine:
1. bitwise vs current PTX on the same synthetic input (rel-max ≤ 4e-4 or
   byte-equal output),
2. full greedy decode at temp=0 on LFM3-VL-2B-Q4_0-GGML, byte-identical or
   diverging only at near-ties (same heuristic),
3. batch-shape invariance (b=1 vs b=4),
4. phases timing no slower than the PTX baseline (target: ≥ parity).

Loader/dispatch changes additionally need:
5. `OXIDE_DISABLE_PACKED=1` fallback path still works,
6. `cargo build --release --bin server` and `cargo test` pass.

## 4. Phases

### Status (2026-11-14)
- **Phase 0 DONE** - `kernels/oxide-kernels/` crate + multi-arch committed
  PTX (sm_80/90/100/103/110/120/121) + `OxideKernels` loader (device SM
  detection, committed per-arch artifact, sm_80 as JIT fallback for
  unknown silicon incl. Rubin) + `OXIDE_PURE_RUST=1` engine gate.
- **Phase 1 DONE** - 6/6 utility kernels ported and verified:
  rms_norm, residual_add, swiglu, rope, embedding, argmax. Engine
  dispatches through the cuda-oxide path when OXIDE_PURE_RUST=1; PTX
  fallback unchanged. Full fidelity battery 14/14 byte-identical,
  phases parity, France=Paris.
- **Phase 2 DONE for the serving path** - decode hot-path GEMMs dispatch
  through the int8 MMVQ kernels (pure-Rust, committed PTX) for Q4_0
  with HGEMM fallback. K-types gated on/off honestly by measurement.
- **Phase 3 DONE** - attention family ported: batched_gqa_attention_online,
  batched_gqa_attention_split, batched_gqa_attention_split4,
  batched_gqa_attn_combine, batched_qk_rmsnorm_fp16,
  batched_sigmoid_gate_fp16, batched_kv_copy_fp16. Engine dispatches
  through them under OXIDE_PURE_RUST=1. Full fidelity battery parity
  with the PTX path: 7/14 byte-identical, remaining 7 near-tie argmax
  flips accepted per the fidelity policy.
- **Phase 4 DONE** - causal conv1d + gating ported. Phases parity
  (4.1 ms decode). nb==1 no longer forces the scalar-PTX prefill
  path; all prefill/decode now flow through the batched engine,
  with OXIDE_SCALAR_PREFILL=1 as the escape hatch. Verified
  byte-identical to before on the fidelity battery.

### Status (2026-11-14)

- **Phase 0 — multi-arch build + loader matrix** (unblocks everything):
  `kernels/oxide-kernels` skeleton, arch matrix build script, `oxide-infer-kernels`
  loader change, baseline benchmarks for all arch PTXs (JIT on this GPU).
- **Phase 1 — utility kernels** (template + proof): rms_norm, residual_add,
  silu/swiglu, rope, embedding, argmax. ~4 files, ~40 KB PTX → 0.
- **Phase 2 — GEMV/GEMM**: lm_head, strided GEMM helpers, packed GEMV for all
  14 quant types (dispatch only, kernels already exist).
- **Phase 3 — attention**: batched_attn (18 kernels) + gqa_engine,
  attention_layer consolidation.
- **Phase 4 — vision**: vision.rs (14 kernels), projector, embedding lookup.
- **Phase 5 — arch-specific tuning** *only if profiling justifies it*: WGMMA/TMA
  for Hopper, tcgen05 for sm_100a, dp4a→mma for sm_120; Rubin stays JIT/PTX until
  silicon.
- **Phase 6 — polish**: delete remaining PTX strings, single-artifact loading,
  CI matrix (jit-on-sm_86, sm_90/sm_100/sm_120 CI runners when available).

Estimated effort: Phase 0+1 ≈ one focused session; 2–4 ≈ days each; 5 uncertain.

## 5. Scientific methodology (unchanged from the packed-path campaign)

- every change gated by the fidelity battery (14 prompts) + batch-invariance +
  phases timing,
- no brute-force sweeps; measure, attribute, fix root causes,
- packed int8 path remains default on sm_86; per-arch dispatch decisions made
  *after* measurements on each arch.
