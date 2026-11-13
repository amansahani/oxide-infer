# quant-gemv

Production quantized GEMV kernels for oxide-infer, written in cuda-oxide
(pure Rust → PTX). The engine embeds the committed `gemv.ptx` via
`include_str!`; no build-time cuda-oxide dependency is required.

## Regenerating gemv.ptx

```sh
<cuda-oxide checkout>/cuda-oxide/target/debug/cargo-oxide build .
cp gemv_q.ptx gemv.ptx   # name follows the package
```

## Kernels

### f32 packed path (f16 activations, f32 FMA)
- `fused_dual_gemv_swiglu_q<Q: QuantVariant>` — gate/up GEMVs (W1, W3) on
  packed ggml weights, `act = up * gate / (1 + ex2(-gate*log2e))` (matches
  oxide's SWIGLU_FP16_PTX formula), f16 out. grid = (INTER, B, 1).
- `gemv_q<Q: QuantVariant>` — plain GEMV on packed weights (used for the
  W2 down projection), f16 out. grid = (N, B, 1).

### int8 MMVQ path (q8_1-quantized activations, dp4a) - llama.cpp's sm_86 batch=1 kernel
- `quantize_q8_1_f16` — activations -> ggml q8_1 blocks (f16 d, f16 s = raw
  f32 block sum, 32 x i8). Butterfly reduce; verified byte-exact vs host mirror.
- `mmvq_dual_swiglu_q4_0` / `mmvq_q4_0` — Q4_0 weights x q8_1 activations with
  `dp4a.s32`. grid = (N, B, 1), 128 threads (4 warps/row), cross-warp f32
  reduction tree. Measured ~323 GB/s vs ~232 GB/s for the f32 packed path
  (2.1x), ~90% of RTX 3060 DRAM peak.
- Dispatch: batch == 1, quant type Q4_0, `OXIDE_DISABLE_INT8_MMVQ=1` or
  `OXIDE_MMVQ_INT8_FFN=0` / `OXIDE_MMVQ_INT8_PROJ=0` restores the f32 packed path.

Batch invariance: each (row, batch-element) pair is an independent CTA with
an independent f32 reduction tree; results do not depend on batch size.
