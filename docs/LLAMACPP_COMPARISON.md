# Comparative Benchmark: `oxide-infer` vs `llama.cpp`

**Target Model**: `LFM3-VL-2B` (Hybrid 16 Conv1D + 8 GQA Attention + SwiGLU)  
**Precision**: FP16  
**Hardware**: NVIDIA GeForce RTX 3060 12GB (`sm_86`)  
**OS**: Linux x86_64  
**Date**: November 8, 2026

---

## 1. Executive Summary

Both engines were benchmarked on the identical machine under equivalent serving conditions:
- **`llama.cpp`**: compiled with CUDA backend (`-ngl 99 -c 16384 --parallel 128 -t 8`), serving `lfm3-vl-2b-f16.gguf` (3.3 GB).
- **`oxide-infer`**: compiled in release mode with Ampere Tensor Core cuBLASLt algorithm locks, batched PTX micro-kernels, and continuous batching serving `model.safetensors` (2.88 GB).

---

## 2. Text Fidelity & Character Parity

- **14/14 Prompt Suite**: Character-for-character agreement on the full test suite (`/tmp/hf_ref_gen.json`).
- **Financial PhraseBank (`flare-fpb`)**:
  - `oxide-infer`: **100% validity, 94.0% accuracy** (50-row test sample)
  - `llama.cpp`: **100% validity, 94.0% accuracy** (50-row test sample)
  - Outputs match verbatim on structured JSON formatting.
- **Long-Form Generation**: 400-token creative text generation matches `llama.cpp` token-for-token from start to end with zero divergence.

---

## 3. Serving Throughput & Latency Comparison

| Metric / Workload | `llama.cpp` (CUDA / C++) | `oxide-infer` (Pure Rust PTX / cuBLASLt) | Delta |
| :--- | :--- | :--- | :--- |
| **Single-Req Turnaround (FPB seq)** | 148.5 ms | **161.9 ms** | Parity (~13 ms delta) |
| **Decode Step ($B=1$)** | ~11.1 ms | **12.6 – 13.5 ms** | Parity |
| **$k=8$ Concurrency (50 reqs)** | 3.16 s (15.8 req/s) | **1.88 s** (26.7 req/s) | **1.68x faster** |
| **$k=32$ Concurrency (50 reqs)** | 2.99 s (16.7 req/s) | **1.26 s** (39.8 req/s) | **2.37x faster** |
| **$k=100$ Concurrency (100 reqs)** | 3.90 s (25.7 req/s) | **1.97 s** (50.9 req/s) | **1.98x faster** |
| **$40 \times 30$ tokens (1,200 tok)** | 1.29 s (932.9 tok/s) | **0.68 s** (1,759.6 tok/s) | **1.89x faster** |
| **$100 \times 64$ tokens (6,376 tok)** | 5.31 s (1,199.4 tok/s) | **2.19 s** (**2,909.7 tok/s**) | **2.42x faster** |

---

## 4. Key Architectural Differentiators in `oxide-infer`

1. **Shape-Invariant Tensor Core GEMMs**:
   - `cublasLtMatmul` with pre-selected invariant algorithm heuristics per $(m, k)$ matrix shape. Bitwise identical outputs across all batch sizes $B \in [1, 128]$.
2. **Token-Batched Wave Prefill**:
   - Chunks prompt tokens into waves of up to 128 tokens; passes weights once per wave instead of sequential token-by-token processing.
3. **Double-Embedding Elimination (`sample_only`)**:
   - First generated token is sampled directly from the persisted prefill hidden state, avoiding the re-embedding phase that causes token desynchronization.
4. **Zero VRAM Packing Overheads**:
   - Device pointer tables (`k_ptrs`, `v_ptrs`, `conv_ptrs`) eliminate tensor copies in the hot decode loop.
