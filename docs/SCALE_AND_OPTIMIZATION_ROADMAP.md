# Scale & High-Throughput Roadmap: Achieving >2,900 tok/s in Pure Rust `cuda-oxide`

**Engine**: `oxide-infer`  
**Current Milestone**: Achieved **2,909.7 tok/s** aggregate serving rate on NVIDIA RTX 3060 (12GB) across 100 concurrent requests, outperforming `llama.cpp` by 2.42x and matching Hugging Face token-level output fidelity.

---

## 1. What Has Been Achieved

```
+-----------------------------------------------------------------------------------------+
| Milestone                                   | Result                                   |
+---------------------------------------------+------------------------------------------+
| 1. Pure Rust & Direct PTX Kernels           | 100% Rust + cuda-oxide driver bindings   |
| 2. Shape-Invariant cuBLASLt GEMM            | Bitwise invariant outputs across B=1..128|
| 3. Token-Batched Wave Prefill               | Prefill drop from 884 ms to ~11-15 ms    |
| 4. Single-Stream Parity with llama.cpp      | 161.9 ms vs 148.5 ms (FPB turnaround)    |
| 5. High-Concurrency Scale (k=100)           | 2.19 s for 6,376 tokens (2,909 tok/s)    |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Core Architectural Pillars

1. **Shape-Invariant Matrix Engine**:
   - `LtGemm` locks in a single high-performance cuBLASLt algorithm per `(m, k)` layer projection.
   - Prevents reduction-order jitter across different batch sizes without memory padding.
2. **Unified Continuous Batching**:
   - Dynamic scheduling admits incoming requests seamlessly into the active sequence pool.
   - Eliminates fixed-slot starvation and stalls.
3. **Token-Batched Prefill with Conv Carry**:
   - Wave prefill allows prompts from multiple requests to execute in tensor-core matrix GEMMs.
   - `conv_state_prefill_fp16` bridges the 6-token historical conv window across chunk boundaries.
4. **Pointer-Table Addressed KV & Conv Buffers**:
   - Eliminates tensor concatenation/splitting overheads on GPU.

---

## 3. Next High-Performance Milestones

1. **Paged KV Cache Pool Integration**:
   - Connect `crates/oxide-infer-core/src/paged_pool.rs` into the batched GQA kernel for non-contiguous physical page allocation to support even longer contexts (>4,096 tokens) with zero fragmentation.
2. **FlashAttention-Style Tiled GQA Kernel**:
   - Implement online softmax tiling across sequence length in the batched attention kernel to reduce shared memory pressure on very long context lengths ($S > 2048$).
3. **CUDA Graph Capture for Decode Step**:
   - For constant active batch size runs, capture the 24-layer decode kernel dispatch into a single hardware graph (`cuGraphLaunch`), reducing CPU driver dispatch overhead to near zero.
