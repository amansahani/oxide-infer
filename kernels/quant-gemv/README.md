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

- `fused_dual_gemv_swiglu_q<Q: QuantVariant>` — gate/up GEMVs (W1, W3) on
  packed ggml weights, `act = up * gate / (1 + ex2(-gate*log2e))` (matches
  oxide's SWIGLU_FP16_PTX formula), f16 out. grid = (INTER, B, 1).
- `gemv_q<Q: QuantVariant>` — plain GEMV on packed weights (used for the
  W2 down projection), f16 out. grid = (N, B, 1).

Batch invariance: each (row, batch-element) pair is an independent CTA with
an independent f32 reduction tree; results do not depend on batch size.
