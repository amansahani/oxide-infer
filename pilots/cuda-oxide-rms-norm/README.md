# cuda-oxide pilots

Scratch crate(s) validating NVIDIA's `cuda-oxide` (pure Rust -> PTX compiler) as a
replacement for oxide-infer's embedded hand-written PTX.

Built and run with the **v0.4.0** cargo-oxide from the sibling cuda-oxide checkout:

```sh
/home/tomi/Desktop/rust_projects/cuda-oxide/cuda-oxide/target/debug/cargo-oxide run .
```

(The globally installed `cargo-oxide` on maintainers' machines may be older and
use an incompatible cache layout; build it from the checkout instead, or install it
with `cargo install --path <checkout>/cuda-oxide/crates/cargo-oxide`.)

## pilot_rms_norm (src/main.rs)

- `residual_add` kernel: packed `f16x2` add, `add.rn.f16x2`
- `rms_norm_fp16` kernel: port of `RMS_NORM_FP16_PTX` (crates/oxide-infer-kernels/src/norm.rs)
  - shared-memory per-warp reduction tree (deterministic ordered sum)
  - f32 accumulation, `rsqrt.approx.f32`, packed f16x2 I/O
  - max 2 ulp vs f64/f32 reference (f16 envelope), on sm_86
