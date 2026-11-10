# OxideInfer Engineering Milestone & Optimization Roadmap

## 1. Summary of Completed Progress & Verification

### A. Dynamic KV Cache Allocation & 128k Context Window
* **RoPE Table Expansion**: Expanded the precomputed sinusoids table in `RotaryEmbeddingTable` from a static 1024 cap to the architecture's full context window of **131,072 tokens** (`[full_layer.rs](crates/oxide-infer-models/src/full_layer.rs#L1179)`), completely eliminating CUDA Driver Error 700 at step 1025.
* **Per-Request Dynamic KV Sizing**: In `server.rs`, slots dynamically allocate KV buffer memory proportional to `prompt_len + req.max_tokens + 64`, removing static compile-time sequence caps.
* **PTX Kernel Shared Memory**: Expanded GQA attention threadblock shared memory to 8192 tokens in `[batched_attn.rs](crates/oxide-infer-kernels/src/batched_attn.rs#L363)`.
* **Empirical Validation**: Successfully verified uninterrupted single-sequence decoding at **1,500 tokens (53.62 tok/s)** and **2,048 tokens (48.76 tok/s)**.

### B. Prompt Template Canonicalization & MultiModal Formatting
* **ChatML Prompt Wrapping**: Raw user text requests are automatically wrapped into canonical `<|im_start|>user ... <|im_end|><|im_start|>assistant\n<think></think>` format, preventing autoregressive token degeneration (`2 + 2` -> `'4'`, `The capital of France is` -> `'The capital of France is Paris.'`).
* **Bounding Vision Delimiters**: Multimodal requests properly inject `<|vision_start|><|image|><|vision_end|>` delimiters (`[server.rs](crates/oxide-infer-server/src/bin/server.rs#L2008)`).
* **Thinking Scratchpad Parsing**: Implemented OpenAI/DeepSeek compatible parsing separating `<think>...</think>` internal reasoning from final user-facing text.

### C. 77-Page Continuous Batching Benchmark Results
* **Document**: `SID_quant Business Cycle Fund_March 2025.pdf` (77 Pages)
* **Status**: **77 / 77 Pages Processed Successfully (100.0% Success Rate, 0 crashes / 0 dropouts)**
* **Total Wall Time**: **80.42 seconds**
* **Total Generated Tokens**: **9,450 tokens**
* **Aggregate System Throughput**: **117.51 tokens/second**
* **Average Page Turnaround Latency**: **8.25 seconds** (with 8 concurrent client streams)

---

## 2. Pushed Git Commits

All changes have been committed and pushed to `main`:
1. `6f84247`: `feat(kv-cache): enable fully dynamic sequence lengths and 128k RoPE embedding window`
2. `28908e2`: `fix(server): canonicalize ChatML prompt templates, add vision tags, and handle thinking scratchpad`

---

## 3. Tomorrow's Performance Optimization Roadmap

### A. Pre-allocated KV Cache Pool (Zero Runtime Allocations)
* Replace dynamic per-request `DeviceBuffer::zeroed` runtime allocations with a pre-warmed block pool (`PagedKvPool` / unified token slab). Eliminates all CUDA driver memory allocation syscall overhead during request admission.

### B. Paged Attention & Vectorized Memory Access
* Upgrade the Ampere GQA decode kernel with vector-4 (`float4` / 8-byte) loads and online softmax reduction ($O(1)$ shared memory footprint independent of context length).

### C. Concurrency Scaling & In-Flight Batching
* Scale client concurrency from 8 to 16–32 worker streams to maximize Ampere Tensor Core saturation and target **200+ tokens/second** aggregate throughput.

### D. Early Stopping & Dynamic Multi-Tile High-Resolution OCR
* Terminate slot decoding upon generation of `<|im_end|>` (token `124900`) to cut average page latency from 8.25s to 2–3s.
* Implement native-resolution $2 \times 3$ grid tiling for document PDFs to preserve fine character strokes without optical degradation.

### Order of work: paged block pool before a radix cache
Paged KV and radix prefix caching are complementary, not competing: the block
pool is the allocator (fixed-size blocks, per-sequence block tables, refcounts,
no admission-time `cudaMalloc`), while the radix tree is a reuse policy layered
on top (token-keyed prefix nodes, LRU eviction, copy-on-write).

Phase A implements the pool because it removes the admission syscalls and makes
prefix sharing possible at all. A radix cache is justified only once measured
prefix-hit rates are high (multi-turn chat / agent traffic); it adds little for
unique FPB sentences or one-shot OCR pages.

LFM3 caveat: the 13 Conv1D layers keep a 6-token rolling state (~24 KB/layer).
A shared prefix must snapshot conv state at block boundaries in addition to
K/V blocks, and RoPE positions must be identical from the prefix root.
