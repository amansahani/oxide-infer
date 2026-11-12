# cuda-oxide quantized GEMV pilot

One generic `#[kernel]` monomorphized per quant variant — the cuda-oxide
analog of llama.cpp's templated MMQ/`mul_mat_q_case` or vLLM's Marlin
instantiation matrix. Host-side dispatch is a `match` over variants calling
`module.gemv_q::<Q4_0>`, `module.gemv_q::<Q8_0>`, …; each instantiation gets a
distinct `gemv_q_TID_<hash>` PTX entry point.

## Status

- [x] All 8 simple ggml types: `q1_0`, `q2_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q8_1`
- [ ] K-types (`q2_k`…`q8_k`): not yet. Blocked on a ggml-verified host dequantizer —
      our K dequantizers have not been validated against a reference (K-quantized
      models produce garbage output), and `q2_k`'s device formula is only approximate.
      Porting them now would faithfully replicate wrong math.

## What is validated

For every variant: host → device → kernel → host round-trip on sm_86, with the
device dot product compared against an f64 host reference computed from the
packed bytes using the same dequantize formula as `oxide-infer-core::quant`.
Synthesized weights (deterministic xorshift bytes + f16 scale/min headers) —
both sides dequantize the same bytes, which is exactly the parity property the
on-the-fly dequant path in the engine needs.

K=256 f16 elements per row, 3 rows, block_dim=128, one CTA per row. Runtime
dispatch (match over a `QuantVariant` tag → monomorphized launcher call) is
trivially expressible but is left to the engine wiring milestone.

## Run

```sh
<cuda-oxide checkout>/cuda-oxide/target/debug/cargo-oxide run .
```

Host-side quick check without CUDA toolkit:

```sh
RUSTC_BOOTSTRAP=1 cargo +1.99.1 check --release
```
