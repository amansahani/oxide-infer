# Pilots

Experiments validating NVIDIA's `cuda-oxide` (pure Rust -> PTX) as a replacement
for oxide-infer's embedded hand-written PTX. Standalone crates (own workspaces),
pinned to the cuda-oxide toolchain via each crate's `rust-toolchain.toml`.

Build/run with cargo-oxide built from the sibling cuda-oxide checkout:

```sh
<cuda-oxide>/cuda-oxide/target/debug/cargo-oxide run .
```

- `cuda-oxide-rms-norm` — `residual_add` (f16x2) and `rms_norm_fp16` (port of
  RMS_NORM_FP16_PTX); bit-fidelity vs host reference.
- `cuda-oxide-quant-gemv` — quantized GEMV, **one generic kernel monomorphized
  per quant variant** (Q4_0, Q8_0), the cuda-oxide analog of llama.cpp's MMQ /
  vLLM's Marlin. Proves per-variant PTX entry points coexist in one module and
  each matches the host dequantize-then-dot reference.
