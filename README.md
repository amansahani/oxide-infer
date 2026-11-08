# oxide-infer

High-performance inference engine for hybrid language models (Conv1D + GQA Attention + SwiGLU MLP) written in pure Rust using direct CUDA driver bindings (`cuda-oxide`), custom PTX micro-kernels, and Ampere Tensor Core cuBLASLt algorithm-locked GEMMs.

Currently focused on serving **Liquid AI's `LFM3-VL-2B`** on NVIDIA Ampere GPUs (`sm_86` / RTX 3060 12GB).

---

## Highlights

- **Pure Rust System**: No Python, PyTorch, or C++ runtime in the critical inference path. All GPU kernels compile directly from inline PTX or link to CUDA driver libraries (`libcuda.so`, `libcublasLt.so`).
- **Continuous Unified Batching**: Fully dynamic request scheduling supporting up to 128 concurrent sequences in a single batched engine pass.
- **Token-Batched Wave Prefill**: Prefills incoming multi-turn prompts as row batches (up to 128 tokens per wave) with persistent 6-tap conv carry state across chunk boundaries. Prefill latency dropped from ~880 ms to ~11–15 ms.
- **Bitwise Shape-Invariant cuBLASLt GEMM**: Locked heuristics per `(m, k)` layer projection ensuring bitwise-identical outputs across all batch sizes ($B=1..128$) with FP32 accumulation.
- **Pure-Rust Multimodal Vision Pipeline**: Native support for image comprehension (`LFM3-VL-2B`) via pure Rust image planar normalization, $(gh/m, gw/m, m, m)$ patch unfolding, and GPU MultiModalProjector with zero external Python or C++ dependencies.
- **Direct Pointer-Table Addressing**: Avoids GPU tensor concatenation, reallocations, or memory copies in the hot decode loop by passing device pointer tables (`k_ptrs`, `v_ptrs`, `conv_ptrs`) directly into PTX kernels.

---

## Architecture

`LFM3-VL-2B` is a 24-layer hybrid architecture:
- **16 Causal Conv1D layers**: $K=7$ channel-wise causal convolution with 6 historical state slots per channel, fused input gating ($B \cdot x$), and output gating ($C \cdot y$).
- **8 GQA Attention layers**: 16 query heads, 4 KV heads ($D=128$), per-head RMSNorm, RoPE, and per-head sigmoid gating before $O$-projection.
- **24 SwiGLU FFN blocks**: Hidden dimension 2048, intermediate dimension 7680 ($\text{silu}(W_1 x) \odot W_3 x \to W_2$).

```
Prompt / Tokens In
      │
      ▼
Embedding Table Lookup (PTX)
      │
┌─────┴──────────────────────────────────────────────────────┐
│  For Layer 0 .. 23:                                        │
│                                                            │
│   Operator RMSNorm (PTX)                                   │
│            │                                               │
│            ├─► If Conv Layer (16/24):                      │
│            │     in_proj (cuBLASLt 6144x2048)              │
│            │     conv_state_prefill / batched_conv (PTX)   │
│            │     out_proj (cuBLASLt 2048x2048)             │
│            │                                               │
│            └─► If Attention Layer (8/24):                  │
│                  Q/K/V/Gate projections (cuBLASLt)         │
│                  Per-head Q/K RMSNorm + RoPE (PTX)         │
│                  D2D Vectorized KV Copy (PTX)              │
│                  Batched GQA Attention + Sigmoid Gate (PTX)│
│                  out_proj (cuBLASLt 2048x2048)             │
│            │                                               │
│   Post-Operator Norm + Residual Add (PTX)                  │
│   FFN RMSNorm (PTX)                                        │
│   FFN W1 Gate + W3 Up GEMMs (cuBLASLt 7680x2048)           │
│   Batched SwiGLU Activation (PTX)                          │
│   FFN W2 Down GEMM (cuBLASLt 2048x7680)                    │
│   Post-FFN Norm + Residual Add (PTX)                       │
└─────┬──────────────────────────────────────────────────────┘
      │
Final Embedding RMSNorm (PTX)
      │
LM Head Projection (cuBLASLt 128000x2048)
      │
GPU-Side Parallel Argmax (PTX)
      │
Generated Token Out
```

---

## Performance (NVIDIA RTX 3060 12GB)

Evaluated against `llama.cpp` (`b3900+`, `-ngl 99 --parallel 128 -t 8`) on the identical machine with FP16 weights:

### Concurrency & Serving Throughput

| Workload Scenario | `oxide-infer` (:8090) | `llama.cpp` (:8081) | Advantage |
| :--- | :--- | :--- | :--- |
| **FPB $k=8$ concurrent (50 reqs)** | **1.88 s** (26.7 req/s) | 3.16 s (15.8 req/s) | **1.68x faster** |
| **FPB $k=32$ concurrent (50 reqs)** | **1.26 s** (39.8 req/s) | 2.99 s (16.7 req/s) | **2.37x faster** |
| **FPB $k=100$ concurrent (100 reqs)** | **1.97 s** (50.9 req/s) | 3.90 s (25.7 req/s) | **1.98x faster** |
| **$40 \times 30$ tokens (1,200 tok)** | **0.68 s** (1,759.6 tok/s) | 1.29 s (932.9 tok/s) | **1.89x faster** |
| **$100 \times 64$ tokens (6,376 tok)** | **2.19 s** (**2,909.7 tok/s**) | 5.31 s (1,199.4 tok/s) | **2.42x faster** |

### Output Fidelity Verification

- **Hugging Face 14-Prompt Reference Suite**: 14 / 14 exact character matches.
- **Financial PhraseBank Benchmark (`flare-fpb`)**: 100.0% JSON formatting validity, 94.0% accuracy (identical to `llama.cpp`).
- **Long-Form Text Decoding**: 400-token creative generation produces token-for-token identical output to `llama.cpp`.

---

## Quickstart

### Prerequisites

- Linux x86_64
- NVIDIA GPU (Ampere `sm_80` / `sm_86` or newer)
- CUDA Toolkit installed with `nvcc`, `libcuda.so`, and `libcublasLt.so` available in system paths
- Rust 1.80+ (`cargo`)

### Building

```bash
cargo build --release --bin server
```

### Running the OpenAI-Compatible Server

```bash
# Set model path (directory containing model.safetensors and tokenizer.json)
export OXIDE_MODEL="/path/to/LFM3-VL-2B"
export OXIDE_HOST="0.0.0.0"
export OXIDE_PORT="8090"

./target/release/server
```

The server exposes standard endpoints:
- `GET  /health`
- `POST /v1/chat/completions`

### Example Request

```bash
curl http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain general relativity in simple terms."}
    ],
    "max_tokens": 64,
    "temperature": 0.0
  }'
```

---

## Repository Structure

```
├── crates/
│   ├── oxide-infer-core/      # Memory buffers, GPU context, safetensors loader
│   ├── oxide-infer-kernels/   # PTX micro-kernels (RMSNorm, GQA, RoPE, Conv1D, SwiGLU)
│   ├── oxide-infer-models/    # Layer glue, AttentionLayer, ConvLayer, FfnLayer
│   └── oxide-infer-server/    # Async server, continuous batch engine, cuBLASLt GEMM
├── docs/                      # Architectural deep dives, benchmarks, and comparison logs
└── Cargo.toml
```

---

## License

Apache-2.0 / MIT
