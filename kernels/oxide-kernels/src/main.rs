//! Production utility kernels for oxide-infer, written in cuda-oxide
//! (pure Rust -> PTX). The engine embeds the committed `oxide_kernels.ptx`
//! via `include_str!`; no build-time cuda-oxide dependency is required.
//!
//! Batch invariance: one CTA per (row, batch-element); per-CTA deterministic
//! f32 reduction trees.

#![feature(f16)]

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use half::f16;
use cuda_device::cuda_module;

#[cuda_module]
mod kernels {
    use cuda_device::{DisjointSlice, SharedArray, convert, float, kernel, thread, warp};

    const HIDDEN_TILE: usize = 2048; // 256 threads x 8 elements

    /// warp-level sum of f32 (down shuffle 16,8,4,2,1)
    #[inline(always)]
    // All-reduce: every lane ends with the warp's total. Mirrors the PTX
    // pattern (five shuffle.down steps, then shfl.idx broadcast from lane 0).
    fn warp_tree_sum(val: f32) -> f32 {
        let mut val = val;
        val += warp::shuffle_down_f32_sync(0xffffffff, val, 16);
        val += warp::shuffle_down_f32_sync(0xffffffff, val, 8);
        val += warp::shuffle_down_f32_sync(0xffffffff, val, 4);
        val += warp::shuffle_down_f32_sync(0xffffffff, val, 2);
        val += warp::shuffle_down_f32_sync(0xffffffff, val, 1);
        warp::shuffle_f32_sync(0xffffffff, val, 0)
    }

    /// RMSNorm: out[b*n + i] = f16(f32(x[b*n + i]) * rsqrt(mean(x^2) + eps) * f32(w[i]))
    /// Launch: grid (1, B, 1), block (256, 1, 1). `h_size` must be a multiple of 2048.
    #[kernel]
    pub fn batched_rms_norm_vec8_fp16(
        mut out: DisjointSlice<u16>,
        mut input: DisjointSlice<u16>,
        mut weight: DisjointSlice<u16>,
        eps: f32,
        h_size: u32,
        b_size: u32,
    ) {
        static mut S_WARP_SUMS: SharedArray<f32, 8> = SharedArray::UNINIT;
        static mut S_INV_RMS: SharedArray<f32, 1> = SharedArray::UNINIT;

        let b = thread::blockIdx_y() as usize;
        if b >= b_size as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_id = tid >> 5;
        let h = h_size as usize;
        let x_off = b * h;

        // pass 1: sum of squares
        let mut sum_sq = 0.0f32;
        let mut k = tid * 8;
        while k < h {
            let xptr = input.as_mut_ptr().wrapping_add(x_off + k) as *const u32;
            let mut i = 0usize;
            while i < 4 {
                let (lo, hi) = convert::cvt_f32x2_f16x2(unsafe { *xptr.add(i) });
                sum_sq = float::fma_rn_f32(lo, lo, sum_sq);
                sum_sq = float::fma_rn_f32(hi, hi, sum_sq);
                i += 1;
            }
            k += 256 * 8;
        }

        sum_sq = warp_tree_sum(sum_sq);
        if lane == 0 {
            unsafe { S_WARP_SUMS[warp_id] = sum_sq; }
        }
        thread::sync_threads();

        // warp 0: combine the per-warp sums
        if warp_id == 0 {
            let mut v = if tid < 8 { unsafe { S_WARP_SUMS[tid] } } else { 0.0f32 };
            v += warp::shuffle_down_f32_sync(0xffffffff, v, 4);
            v += warp::shuffle_down_f32_sync(0xffffffff, v, 2);
            v += warp::shuffle_down_f32_sync(0xffffffff, v, 1);
            if tid == 0 {
                let mean_sq = v / (h as f32);
                let mean_sq = mean_sq + eps;
                let inv_rms = float::rsqrt_approx_f32(mean_sq);
                unsafe {
                    S_INV_RMS[0] = inv_rms;
                }
            }
        }
        thread::sync_threads();
        let inv_rms = unsafe { S_INV_RMS[0] };

        // pass 2: write normalized x * w
        let mut k = tid * 8;
        while k < h {
            let xptr = input.as_mut_ptr().wrapping_add(x_off + k) as *const u32;
            let wptr = weight.as_mut_ptr().wrapping_add(k) as *const u32;
            let outptr = out.as_mut_ptr().wrapping_add(x_off + k) as *mut u32;
            let mut i = 0usize;
            while i < 4 {
                let (xlo, xhi) = convert::cvt_f32x2_f16x2(unsafe { *xptr.add(i) });
                let (wlo, whi) = convert::cvt_f32x2_f16x2(unsafe { *wptr.add(i) });
                let olo = xlo * inv_rms * wlo;
                let ohi = xhi * inv_rms * whi;
                let packed = convert::cvt_f16x2_f32(olo, ohi);
                unsafe { *outptr.add(i) = packed; }
                i += 1;
            }
            k += 256 * 8;
        }
    }

    /// SwiGLU: act[i] = f16(f32(up[i]) * silu(f32(gate[i]))) with silu(g) = g/(1+ex2(-g*log2e))
    /// Launch: grid (ceil(n/256), 1, 1), block (256, 1, 1).
    #[kernel]
    pub fn batched_swiglu_separate_fp16(
        mut act: DisjointSlice<u16>,
        gate: &[u16],
        up: &[u16],
        n: u32,
    ) {
        let tid = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
        if tid >= n as usize {
            return;
        }
        let g = gate[tid];
        let u = up[tid];
        let gf = f16::from_bits(g) as f32;
        let uf = f16::from_bits(u) as f32;
        let ex = float::ex2_approx_f32(gf * -1.4426950218889634f32);
        let silu = gf / (1.0 + ex);
        let v = uf * silu;
        let packed = convert::cvt_f16x2_f32(v, v);
        unsafe { *act.get_unchecked_mut(tid) = (packed & 0xffff) as u16; }
    }

    /// RoPE applied to a [batch, heads, dim] tensor; cos/sin are
    /// [max_seq_len, dim] f16 tables. Same formula as the embedded PTX:
    ///   out[i] = x[i]*cos[pid*dim + i] - x[i+half]*sin[i]     (i < half)
    ///   out[i] = x[i]*cos[pid*dim + i] + x[i-half]*sin[i]     (i >= half)
    /// Launch: grid (heads, B, 1), block (dim, 1, 1).
    #[kernel]
    pub fn batched_rope_fp16(
        mut x_io: DisjointSlice<u16>,
        cos_table: &[u16],
        sin_table: &[u16],
        pos_ids: &[u32],
        num_heads: u32,
        head_dim: u32,
        batch_size: u32,
    ) {
        let head_idx = thread::blockIdx_x() as usize;
        if head_idx >= num_heads as usize {
            return;
        }
        let b = thread::blockIdx_y() as usize;
        if b >= batch_size as usize {
            return;
        }
        let dim = thread::threadIdx_x() as usize;
        let hd = head_dim as usize;
        if dim >= hd {
            return;
        }
        let half = hd / 2;
        let other = if dim < half { dim + half } else { dim - half };
        let pid = pos_ids[b] as usize;

        let x_off = (b * num_heads as usize + head_idx) * hd + dim;
        let other_off = (b * num_heads as usize + head_idx) * hd + other;

        let x = f16::from_bits(unsafe { *(x_io.as_mut_ptr() as *const u16).add(x_off) }) as f32;
        let xo = f16::from_bits(unsafe { *(x_io.as_mut_ptr() as *const u16).add(other_off) }) as f32;
        let c = f16::from_bits(cos_table[pid * hd + dim]) as f32;
        let s = f16::from_bits(sin_table[pid * hd + dim]) as f32;
        let xo = if dim < half { -xo } else { xo };
        let out = x * c + xo * s;
        let packed = convert::cvt_f16x2_f32(out, out);
        unsafe { *x_io.get_unchecked_mut(x_off) = (packed & 0xffff) as u16; }
    }

    /// Argmax over the vocab dimension, one block per batch element.
    /// 256 threads per block iterate vocab in strides of 256, warp-tree
    /// reduce, then a single 8-lane reduce across warps. Deterministic:
    /// on ties, the lower index wins (matches the PTX `>` semantics).
    #[kernel]
    pub fn batched_argmax_direct_f16(
        mut tokens_out: DisjointSlice<u32>,
        logits_in: &[u16],
        vocab_size: u32,
    ) {
        static mut S_WARP_MAX: SharedArray<f32, 8> = SharedArray::UNINIT;
        static mut S_WARP_IDX: SharedArray<u32, 8> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let v = vocab_size as usize;
        let base = b * v;

        let mut max_val = -1.0e30f32;
        let mut best_idx = 0u32;
        let mut idx = tid;
        while idx < v {
            let f = f16::from_bits(unsafe { *logits_in.as_ptr().add(base + idx) }) as f32;
            if f > max_val {
                max_val = f;
                best_idx = idx as u32;
            }
            idx += 256;
        }

        let mut off = 16u32;
        while off > 0 {
            let ov = warp::shuffle_down_f32_sync(0xffffffff, max_val, off);
            let oi = warp::shuffle_down_sync(0xffffffff, best_idx, off);
            if ov > max_val || (ov == max_val && oi < best_idx) {
                max_val = ov;
                best_idx = oi;
            }
            off >>= 1;
        }

        let lane = tid & 31;
        let warp_id = tid >> 5;
        if lane == 0 {
            unsafe {
                S_WARP_MAX[warp_id] = max_val;
                S_WARP_IDX[warp_id] = best_idx;
            }
        }
        thread::sync_threads();

        if tid < 8 {
            let mut m = unsafe { S_WARP_MAX[tid] };
            let mut bi = unsafe { S_WARP_IDX[tid] };
            let mut off = 4u32;
            while off > 0 {
                let ov = warp::shuffle_down_f32_sync(0xffffffff, m, off);
                let oi = warp::shuffle_down_sync(0xffffffff, bi, off);
                if ov > m || (ov == m && oi < bi) {
                    m = ov;
                    bi = oi;
                }
                off >>= 1;
            }
            if tid == 0 {
                unsafe { *tokens_out.get_unchecked_mut(b) = bi; }
            }
        }
    }

    /// Split-K attention 4-keys-per-iteration variant (pass 1 of 2).
    /// Same result as `batched_gqa_attention_split_fp16`; unrolled for
    /// throughput. Requires head_dim = 128 (lane owns dims 4*l..4*l+3).
    #[kernel]
    pub fn batched_gqa_attention_split4_fp16(
        q_in: &[u16],
        k_ptrs: &[u64],
        v_ptrs: &[u64],
        seq_lens: &[u32],
        mut partial_acc: DisjointSlice<f32>,
        mut partial_stats: DisjointSlice<f32>,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        batch_size: u32,
        window: u32,
        chunk_keys: u32,
        nchunks: u32,
    ) {
        let b = thread::blockIdx_y() as usize;
        if b >= batch_size as usize {
            return;
        }
        let kvh = thread::blockIdx_x() as usize;
        if kvh >= num_kv_heads as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp = tid >> 5;
        let hd = head_dim as usize;
        let nq = num_q_heads as usize;
        let nkv = num_kv_heads as usize;
        let head = (kvh << 2) + warp;

        let laneoff = lane * 4;
        let q_off = (b * nq + head) * hd + laneoff;
        let q0 = f16::from_bits(unsafe { *q_in.as_ptr().add(q_off) }) as f32;
        let q1 = f16::from_bits(unsafe { *q_in.as_ptr().add(q_off + 1) }) as f32;
        let q2 = f16::from_bits(unsafe { *q_in.as_ptr().add(q_off + 2) }) as f32;
        let q3 = f16::from_bits(unsafe { *q_in.as_ptr().add(q_off + 3) }) as f32;

        let k_ptr = unsafe { *k_ptrs.get_unchecked(b) } as *const u16;
        let v_ptr = unsafe { *v_ptrs.get_unchecked(b) } as *const u16;
        let slen = unsafe { *seq_lens.get_unchecked(b) } as usize;

        let mut wstart = 0usize;
        let win = window as usize;
        if win > 0 && slen > win {
            wstart = slen - win;
        }
        let c = thread::blockIdx_z() as usize;
        let chunk = chunk_keys as usize;
        let t_start = wstart + c * chunk;
        let mut t_end = t_start + chunk;
        if t_end > slen {
            t_end = slen;
        }

        let mut m = -1.0e30f32;
        let mut s = 0.0f32;
        let mut a0 = 0.0f32;
        let mut a1 = 0.0f32;
        let mut a2 = 0.0f32;
        let mut a3 = 0.0f32;

        let mut t = t_start;
        while t < t_end {
            let kvbase = (t * nkv + kvh) * hd + laneoff;
            let has1 = t + 1 < t_end;
            let has2 = t + 2 < t_end;
            let has3 = t + 3 < t_end;

            let k0p = kvbase;
            let stride = nkv * hd;
            let kv1 = k0p + stride;
            let kv2 = kv1 + stride;
            let kv3 = kv2 + stride;

            let h0 = unsafe { k_ptr.add(kvbase).read() };
            let h1 = unsafe { k_ptr.add(kvbase + 1).read() };
            let h2 = unsafe { k_ptr.add(kvbase + 2).read() };
            let h3 = unsafe { k_ptr.add(kvbase + 3).read() };
            let mut h4 = 0u16;
            let mut h5 = 0u16;
            let mut h6 = 0u16;
            let mut h7 = 0u16;
            let mut h8 = 0u16;
            let mut h9 = 0u16;
            let mut h10 = 0u16;
            let mut h11 = 0u16;
            let mut h12 = 0u16;
            let mut h13 = 0u16;
            let mut h14 = 0u16;
            let mut h15 = 0u16;
            if has1 {
                h4 = unsafe { k_ptr.add(kv1).read() };
                h5 = unsafe { k_ptr.add(kv1 + 1).read() };
                h6 = unsafe { k_ptr.add(kv1 + 2).read() };
                h7 = unsafe { k_ptr.add(kv1 + 3).read() };
            }
            if has2 {
                h8 = unsafe { k_ptr.add(kv2).read() };
                h9 = unsafe { k_ptr.add(kv2 + 1).read() };
                h10 = unsafe { k_ptr.add(kv2 + 2).read() };
                h11 = unsafe { k_ptr.add(kv2 + 3).read() };
            }
            if has3 {
                h12 = unsafe { k_ptr.add(kv3).read() };
                h13 = unsafe { k_ptr.add(kv3 + 1).read() };
                h14 = unsafe { k_ptr.add(kv3 + 2).read() };
                h15 = unsafe { k_ptr.add(kv3 + 3).read() };
            }

            let mut d0 = q0 * (f16::from_bits(h0) as f32);
            d0 = float::fma_rn_f32(q1, f16::from_bits(h1) as f32, d0);
            d0 = float::fma_rn_f32(q2, f16::from_bits(h2) as f32, d0);
            d0 = float::fma_rn_f32(q3, f16::from_bits(h3) as f32, d0);
            let mut d1 = q0 * (f16::from_bits(h4) as f32);
            d1 = float::fma_rn_f32(q1, f16::from_bits(h5) as f32, d1);
            d1 = float::fma_rn_f32(q2, f16::from_bits(h6) as f32, d1);
            d1 = float::fma_rn_f32(q3, f16::from_bits(h7) as f32, d1);
            let mut d2 = q0 * (f16::from_bits(h8) as f32);
            d2 = float::fma_rn_f32(q1, f16::from_bits(h9) as f32, d2);
            d2 = float::fma_rn_f32(q2, f16::from_bits(h10) as f32, d2);
            d2 = float::fma_rn_f32(q3, f16::from_bits(h11) as f32, d2);
            let mut d3 = q0 * (f16::from_bits(h12) as f32);
            d3 = float::fma_rn_f32(q1, f16::from_bits(h13) as f32, d3);
            d3 = float::fma_rn_f32(q2, f16::from_bits(h14) as f32, d3);
            d3 = float::fma_rn_f32(q3, f16::from_bits(h15) as f32, d3);

            d0 = warp_tree_sum(d0);
            d1 = warp_tree_sum(d1);
            d2 = warp_tree_sum(d2);
            d3 = warp_tree_sum(d3);

            let s0 = d0 * 0.0883883476f32;
            let s1 = d1 * 0.0883883476f32;
            let s2 = d2 * 0.0883883476f32;
            let s3 = d3 * 0.0883883476f32;

            // key 0
            if s0 > m {
                let corr = float::ex2_approx_f32((m - s0) * 1.442695041f32);
                a0 *= corr; a1 *= corr; a2 *= corr; a3 *= corr; s *= corr; m = s0;
            }
            let w0 = float::ex2_approx_f32((s0 - m) * 1.442695041f32);
            s += w0;
            let v0 = f16::from_bits(unsafe { *v_ptr.add(kvbase) }) as f32;
            let v1 = f16::from_bits(unsafe { *v_ptr.add(kvbase + 1) }) as f32;
            let v2 = f16::from_bits(unsafe { *v_ptr.add(kvbase + 2) }) as f32;
            let v3 = f16::from_bits(unsafe { *v_ptr.add(kvbase + 3) }) as f32;
            a0 = float::fma_rn_f32(w0, v0, a0);
            a1 = float::fma_rn_f32(w0, v1, a1);
            a2 = float::fma_rn_f32(w0, v2, a2);
            a3 = float::fma_rn_f32(w0, v3, a3);

            // key 1
            if has1 {
                if s1 > m {
                    let corr = float::ex2_approx_f32((m - s1) * 1.442695041f32);
                    a0 *= corr; a1 *= corr; a2 *= corr; a3 *= corr; s *= corr; m = s1;
                }
                let w1 = float::ex2_approx_f32((s1 - m) * 1.442695041f32);
                s += w1;
                let v0 = f16::from_bits(unsafe { *v_ptr.add(kv1) }) as f32;
                let v1 = f16::from_bits(unsafe { *v_ptr.add(kv1 + 1) }) as f32;
                let v2 = f16::from_bits(unsafe { *v_ptr.add(kv1 + 2) }) as f32;
                let v3 = f16::from_bits(unsafe { *v_ptr.add(kv1 + 3) }) as f32;
                a0 = float::fma_rn_f32(w1, v0, a0);
                a1 = float::fma_rn_f32(w1, v1, a1);
                a2 = float::fma_rn_f32(w1, v2, a2);
                a3 = float::fma_rn_f32(w1, v3, a3);
            }

            // key 2
            if has2 {
                if s2 > m {
                    let corr = float::ex2_approx_f32((m - s2) * 1.442695041f32);
                    a0 *= corr; a1 *= corr; a2 *= corr; a3 *= corr; s *= corr; m = s2;
                }
                let w2 = float::ex2_approx_f32((s2 - m) * 1.442695041f32);
                s += w2;
                let v0 = f16::from_bits(unsafe { *v_ptr.add(kv2) }) as f32;
                let v1 = f16::from_bits(unsafe { *v_ptr.add(kv2 + 1) }) as f32;
                let v2 = f16::from_bits(unsafe { *v_ptr.add(kv2 + 2) }) as f32;
                let v3 = f16::from_bits(unsafe { *v_ptr.add(kv2 + 3) }) as f32;
                a0 = float::fma_rn_f32(w2, v0, a0);
                a1 = float::fma_rn_f32(w2, v1, a1);
                a2 = float::fma_rn_f32(w2, v2, a2);
                a3 = float::fma_rn_f32(w2, v3, a3);
            }

            // key 3
            if has3 {
                if s3 > m {
                    let corr = float::ex2_approx_f32((m - s3) * 1.442695041f32);
                    a0 *= corr; a1 *= corr; a2 *= corr; a3 *= corr; s *= corr; m = s3;
                }
                let w3 = float::ex2_approx_f32((s3 - m) * 1.442695041f32);
                s += w3;
                let v0 = f16::from_bits(unsafe { *v_ptr.add(kv3) }) as f32;
                let v1 = f16::from_bits(unsafe { *v_ptr.add(kv3 + 1) }) as f32;
                let v2 = f16::from_bits(unsafe { *v_ptr.add(kv3 + 2) }) as f32;
                let v3 = f16::from_bits(unsafe { *v_ptr.add(kv3 + 3) }) as f32;
                a0 = float::fma_rn_f32(w3, v0, a0);
                a1 = float::fma_rn_f32(w3, v1, a1);
                a2 = float::fma_rn_f32(w3, v2, a2);
                a3 = float::fma_rn_f32(w3, v3, a3);
            }

            t += 4;
        }

        let pidx = (b * nchunks as usize + c) * nq + head;

        if lane == 0 {
            let st_base = pidx * 2;
            unsafe {
                *partial_stats.get_unchecked_mut(st_base) = m;
                *partial_stats.get_unchecked_mut(st_base + 1) = s;
            }
        }
        let acc_off = pidx * hd + lane * 4;
        unsafe {
            *partial_acc.get_unchecked_mut(acc_off) = a0;
            *partial_acc.get_unchecked_mut(acc_off + 1) = a1;
            *partial_acc.get_unchecked_mut(acc_off + 2) = a2;
            *partial_acc.get_unchecked_mut(acc_off + 3) = a3;
        }
    }

    /// Split-K attention: pass 1 of 2. Batched partial (m, s, acc) per
    /// (row, chunk, head) slot. Same math as the online kernel, but the
    /// block scans only [ctaid.z*chunk_keys, +chunk_keys) intersected with
    /// [wstart, slen).
    #[kernel]
    pub fn batched_gqa_attention_split_fp16(
        q_in: &[u16],
        k_ptrs: &[u64],
        v_ptrs: &[u64],
        seq_lens: &[u32],
        mut partial_acc: DisjointSlice<f32>,
        mut partial_stats: DisjointSlice<f32>,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        batch_size: u32,
        window: u32,
        chunk_keys: u32,
        nchunks: u32,
    ) {
        let b = thread::blockIdx_y() as usize;
        if b >= batch_size as usize {
            return;
        }
        let kvh = thread::blockIdx_x() as usize;
        if kvh >= num_kv_heads as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp = tid >> 5;
        let hd = head_dim as usize;
        let nq = num_q_heads as usize;
        let nkv = num_kv_heads as usize;
        let head = (kvh << 2) + warp;

        let laneoff = lane * 4;
        let q_off = (b * nq + head) * hd + laneoff;
        let q0 = f16::from_bits(unsafe { *q_in.as_ptr().add(q_off) }) as f32;
        let q1 = f16::from_bits(unsafe { *q_in.as_ptr().add(q_off + 1) }) as f32;
        let q2 = f16::from_bits(unsafe { *q_in.as_ptr().add(q_off + 2) }) as f32;
        let q3 = f16::from_bits(unsafe { *q_in.as_ptr().add(q_off + 3) }) as f32;

        let k_ptr = unsafe { *k_ptrs.get_unchecked(b) } as *const u16;
        let v_ptr = unsafe { *v_ptrs.get_unchecked(b) } as *const u16;
        let slen = unsafe { *seq_lens.get_unchecked(b) } as usize;

        let mut wstart = 0usize;
        let win = window as usize;
        if win > 0 && slen > win {
            wstart = slen - win;
        }
        let c = thread::blockIdx_z() as usize;
        let chunk = chunk_keys as usize;
        let t_start = wstart + c * chunk;
        let mut t_end = t_start + chunk;
        if t_end > slen {
            t_end = slen;
        }

        let mut m = -1.0e30f32;
        let mut s = 0.0f32;
        let mut a0 = 0.0f32;
        let mut a1 = 0.0f32;
        let mut a2 = 0.0f32;
        let mut a3 = 0.0f32;

        let mut t = t_start;
        while t < t_end {
            let kvoff = (t * nkv + kvh) * hd + laneoff;
            let k0 = f16::from_bits(unsafe { *k_ptr.add(kvoff) }) as f32;
            let k1 = f16::from_bits(unsafe { *k_ptr.add(kvoff + 1) }) as f32;
            let k2 = f16::from_bits(unsafe { *k_ptr.add(kvoff + 2) }) as f32;
            let k3 = f16::from_bits(unsafe { *k_ptr.add(kvoff + 3) }) as f32;

            let mut dot = q0 * k0;
            dot = float::fma_rn_f32(q1, k1, dot);
            dot = float::fma_rn_f32(q2, k2, dot);
            dot = float::fma_rn_f32(q3, k3, dot);
            dot = warp_tree_sum(dot);

            let sc = dot * 0.0883883476f32;

            if sc > m {
                let corr = float::ex2_approx_f32((m - sc) * 1.442695041f32);
                a0 *= corr;
                a1 *= corr;
                a2 *= corr;
                a3 *= corr;
                s *= corr;
                m = sc;
            }
            let w = float::ex2_approx_f32((sc - m) * 1.442695041f32);
            s += w;

            let v0 = f16::from_bits(unsafe { *v_ptr.add(kvoff) }) as f32;
            let v1 = f16::from_bits(unsafe { *v_ptr.add(kvoff + 1) }) as f32;
            let v2 = f16::from_bits(unsafe { *v_ptr.add(kvoff + 2) }) as f32;
            let v3 = f16::from_bits(unsafe { *v_ptr.add(kvoff + 3) }) as f32;

            a0 = float::fma_rn_f32(w, v0, a0);
            a1 = float::fma_rn_f32(w, v1, a1);
            a2 = float::fma_rn_f32(w, v2, a2);
            a3 = float::fma_rn_f32(w, v3, a3);

            t += 1;
        }

        let pidx = (b * nchunks as usize + c) * nq + head;

        if lane == 0 {
            let st_base = pidx * 2;
            unsafe {
                *partial_stats.get_unchecked_mut(st_base) = m;
                *partial_stats.get_unchecked_mut(st_base + 1) = s;
            }
        }

        // Partial layout: ((row * nchunks + chunk) * nq + head), 128 f32/head.
        let acc_off = pidx * hd + lane * 4;
        unsafe {
            *partial_acc.get_unchecked_mut(acc_off) = a0;
            *partial_acc.get_unchecked_mut(acc_off + 1) = a1;
            *partial_acc.get_unchecked_mut(acc_off + 2) = a2;
            *partial_acc.get_unchecked_mut(acc_off + 3) = a3;
        }
    }

    /// Combine (pass 2): merge nchunks partials per (row, head) into the
    /// context output row. One warp (32 lanes) per block; lane l owns
    /// dims 4l..4l+3.
    #[kernel]
    pub fn batched_gqa_attn_combine_fp16(
        mut context_out: DisjointSlice<u16>,
        partial_acc: &[f32],
        partial_stats: &[f32],
        num_q_heads: u32,
        head_dim: u32,
        batch_size: u32,
        nchunks: u32,
    ) {
        let b = thread::blockIdx_y() as usize;
        if b >= batch_size as usize {
            return;
        }
        let head = thread::blockIdx_x() as usize;
        if head >= num_q_heads as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let hd = head_dim as usize;
        let nq = num_q_heads as usize;

        let mut m = -1.0e30f32;
        let mut s = 0.0f32;
        let mut a0 = 0.0f32;
        let mut a1 = 0.0f32;
        let mut a2 = 0.0f32;
        let mut a3 = 0.0f32;

        let mut c = 0usize;
        let nch = nchunks as usize;
        while c < nch {
            let pidx = (b * nch + c) * nq + head;
            let st = pidx * 2;
            let mc = partial_stats[st];
            let sc = partial_stats[st + 1];

            if mc > m {
                let corr = float::ex2_approx_f32((m - mc) * 1.442695041f32);
                a0 *= corr;
                a1 *= corr;
                a2 *= corr;
                a3 *= corr;
                s *= corr;
                m = mc;
            }
            let w = float::ex2_approx_f32((mc - m) * 1.442695041f32);
            let wsc = w * sc;
            s += wsc;

            let acc = pidx * hd + lane * 4;
            let p0v = partial_acc[acc];
            let p1v = partial_acc[acc + 1];
            let p2v = partial_acc[acc + 2];
            let p3v = partial_acc[acc + 3];
            a0 = float::fma_rn_f32(w, p0v, a0);
            a1 = float::fma_rn_f32(w, p1v, a1);
            a2 = float::fma_rn_f32(w, p2v, a2);
            a3 = float::fma_rn_f32(w, p3v, a3);

            c += 1;
        }

        let denom = 1.0f32 / s;
        let out_base = (b * nq + head) * hd + lane * 4;
        let out_ptr = context_out.as_mut_ptr().wrapping_add(out_base);
        let p0 = convert::cvt_f16x2_f32(a0 * denom, a1 * denom);
        let p1 = convert::cvt_f16x2_f32(a2 * denom, a3 * denom);
        unsafe { out_ptr.add(0).write(p0 as u16) };
        unsafe { out_ptr.add(1).write((p0 >> 16) as u16) };
        unsafe { out_ptr.add(2).write(p1 as u16) };
        unsafe { out_ptr.add(3).write((p1 >> 16) as u16) };
    }

    /// GQA attention with online softmax for decode (head_dim, 128).
    /// Grid (num_kv_heads, B, 1), block (128, 1, 1); warp w -> q_head = kv_head*4 + w.
    /// Each lane holds 4 contiguous f16 of the q/k/v row. Same math as the PTX.
    #[kernel]
    pub fn batched_gqa_attention_online_fp16(
        mut context_out: DisjointSlice<u16>,
        q_in: &[u16],
        k_ptrs: &[u64],
        v_ptrs: &[u64],
        seq_lens: &[u32],
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        batch_size: u32,
        window: u32,
    ) {
        let b = thread::blockIdx_y() as usize;
        if b >= batch_size as usize {
            return;
        }
        let kvh = thread::blockIdx_x() as usize;
        if kvh >= num_kv_heads as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let warp = tid >> 5;
        let hd = head_dim as usize;
        let nq = num_q_heads as usize;
        let nkv = num_kv_heads as usize;
        let head = (kvh << 2) + warp;

        let laneoff = lane * 4;
        let q_off = ((b * nq + head) * hd + laneoff) * 1;
        let q0 = f16::from_bits(unsafe { *q_in.as_ptr().add(q_off) }) as f32;
        let q1 = f16::from_bits(unsafe { *q_in.as_ptr().add(q_off + 1) }) as f32;
        let q2 = f16::from_bits(unsafe { *q_in.as_ptr().add(q_off + 2) }) as f32;
        let q3 = f16::from_bits(unsafe { *q_in.as_ptr().add(q_off + 3) }) as f32;

        let k_ptr = unsafe { *k_ptrs.get_unchecked(b) } as *const u16;
        let v_ptr = unsafe { *v_ptrs.get_unchecked(b) } as *const u16;
        let slen = unsafe { *seq_lens.get_unchecked(b) } as usize;

        let mut t_start = 0usize;
        let win = window as usize;
        if win > 0 && slen > win {
            t_start = slen - win;
        }

        let mut m = -1.0e30f32;
        let mut s = 0.0f32;
        let mut a0 = 0.0f32;
        let mut a1 = 0.0f32;
        let mut a2 = 0.0f32;
        let mut a3 = 0.0f32;

        let mut t = t_start;
        while t < slen {
            let kvoff = (t * nkv + kvh) * hd + laneoff;
            let k0 = f16::from_bits(unsafe { *k_ptr.add(kvoff) }) as f32;
            let k1 = f16::from_bits(unsafe { *k_ptr.add(kvoff + 1) }) as f32;
            let k2 = f16::from_bits(unsafe { *k_ptr.add(kvoff + 2) }) as f32;
            let k3 = f16::from_bits(unsafe { *k_ptr.add(kvoff + 3) }) as f32;

            let mut dot = q0 * k0;
            dot = float::fma_rn_f32(q1, k1, dot);
            dot = float::fma_rn_f32(q2, k2, dot);
            dot = float::fma_rn_f32(q3, k3, dot);
            dot = warp_tree_sum(dot);

            let sc = dot * 0.0883883476f32;

            if sc > m {
                let corr = float::ex2_approx_f32((m - sc) * 1.442695041f32);
                a0 *= corr;
                a1 *= corr;
                a2 *= corr;
                a3 *= corr;
                s *= corr;
                m = sc;
            }
            let w = float::ex2_approx_f32((sc - m) * 1.442695041f32);
            s += w;

            let v0 = f16::from_bits(unsafe { *v_ptr.add(kvoff) }) as f32;
            let v1 = f16::from_bits(unsafe { *v_ptr.add(kvoff + 1) }) as f32;
            let v2 = f16::from_bits(unsafe { *v_ptr.add(kvoff + 2) }) as f32;
            let v3 = f16::from_bits(unsafe { *v_ptr.add(kvoff + 3) }) as f32;

            a0 = float::fma_rn_f32(w, v0, a0);
            a1 = float::fma_rn_f32(w, v1, a1);
            a2 = float::fma_rn_f32(w, v2, a2);
            a3 = float::fma_rn_f32(w, v3, a3);

            t += 1;
        }

        let inv_s = 1.0f32 / s;
        let out_off = (b * nq + head) * hd + laneoff;
        let out_ptr = context_out.as_mut_ptr().wrapping_add(out_off);
        let p0 = convert::cvt_f16x2_f32(a0 * inv_s, a1 * inv_s);
        let p1 = convert::cvt_f16x2_f32(a2 * inv_s, a3 * inv_s);
        unsafe { out_ptr.add(0).write(p0 as u16) };
        unsafe { out_ptr.add(1).write((p0 >> 16) as u16) };
        unsafe { out_ptr.add(2).write(p1 as u16) };
        unsafe { out_ptr.add(3).write((p1 >> 16) as u16) };
    }

    /// Per-head RMSNorm over [batch, num_heads, head_dim].
    /// One warp per (head, batch); each lane strides over head_dim.
    /// Matches DIV.APPROX + RSQRT.APPROX.FTZ semantics of the PTX.
    #[kernel]
    pub fn batched_qk_rmsnorm_fp16(
        mut x_io: DisjointSlice<u16>,
        mut weight: DisjointSlice<u16>,
        eps: f32,
        num_heads: u32,
        head_dim: u32,
        batch_size: u32,
    ) {
        let head_idx = thread::blockIdx_x() as usize;
        if head_idx >= num_heads as usize {
            return;
        }
        let b = thread::blockIdx_y() as usize;
        if b >= batch_size as usize {
            return;
        }
        let lane = thread::threadIdx_x() as usize;
        let hd = head_dim as usize;
        let base = (b * num_heads as usize + head_idx) * hd;

        let mut fsum = 0.0f32;
        let mut dim = lane;
        while dim < hd {
            let v = f16::from_bits(unsafe { *(x_io.as_mut_ptr() as *const u16).add(base + dim) }) as f32;
            fsum = float::fma_rn_f32(v, v, fsum);
            dim += 32;
        }
        fsum = warp_tree_sum(fsum);
        let mean = fsum / (hd as f32);
        let inv = float::rsqrt_approx_f32(mean + eps);

        dim = lane;
        while dim < hd {
            let off = base + dim;
            let x = f16::from_bits(unsafe { *(x_io.as_mut_ptr() as *const u16).add(off) }) as f32;
            let w = f16::from_bits(unsafe { *(weight.as_mut_ptr() as *const u16).add(dim) }) as f32;
            let packed = convert::cvt_f16x2_f32(x * inv * w, x * inv * w);
            unsafe { *x_io.get_unchecked_mut(off) = (packed & 0xffff) as u16; }
            dim += 32;
        }
    }

    /// Sigmoid gate applied per (head, batch): ctx[...] *= sigmoid(gate[...]).
    /// sig = rcp(1 + ex2(-gate * log2e)). Grid (heads, B, 1), block (dim, 1, 1).
    #[kernel]
    pub fn batched_sigmoid_gate_fp16(
        mut ctx_io: DisjointSlice<u16>,
        gate_in: &[u16],
        num_heads: u32,
        head_dim: u32,
        batch_size: u32,
    ) {
        let head_idx = thread::blockIdx_x() as usize;
        if head_idx >= num_heads as usize {
            return;
        }
        let b = thread::blockIdx_y() as usize;
        if b >= batch_size as usize {
            return;
        }
        let dim = thread::threadIdx_x() as usize;
        let hd = head_dim as usize;
        if dim >= hd {
            return;
        }
        let stride = hd;
        let batch_off = b * num_heads as usize * hd;
        let gate_off = b * num_heads as usize + head_idx;

        let mut ctx = ctx_io.as_mut_ptr().wrapping_add(batch_off + head_idx * hd);
        let gate = f16::from_bits(unsafe { *gate_in.as_ptr().add(gate_off) }) as f32;
        let neg = gate * -1.442695041f32;
        let ex = float::ex2_approx_f32(neg);
        let sig = 1.0f32 / (ex + 1.0f32);

        let mut d = dim;
        while d < hd {
            let v = f16::from_bits(unsafe { *(ctx as *const u16).add(d) }) as f32;
            let packed = convert::cvt_f16x2_f32(v * sig, v * sig);
            unsafe { *ctx.wrapping_add(d) = (packed & 0xffff) as u16; }
            d += stride;
        }
    }

    /// Copy a [batch, kv_dim] K or V block into per-sequence KV caches
    /// addressed through a device pointer table.
    /// Launch: grid (B, 1, 1), block (64, 1, 1). kv_dim % 8 == 0.
    #[kernel]
    pub fn batched_kv_copy_fp16(
        dst_ptrs: &[u64],
        src: &[u16],
        pos_ids: &[u32],
        kv_dim: u32,
        batch_size: u32,
    ) {
        let b = thread::blockIdx_x() as usize;
        if b >= batch_size as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let idx = tid * 8;
        let kv = kv_dim as usize;
        if idx >= kv {
            return;
        }
        let dst = dst_ptrs[b] as *mut u16;
        let pid = unsafe { *pos_ids.get_unchecked(b) } as usize;
        let src_base = b * kv;
        let dst_base = pid * kv;
        let sptr = src.as_ptr().wrapping_add(src_base + idx) as *const u32;
        let dptr = dst.wrapping_add(dst_base + idx) as *mut u32;
        let mut i = 0usize;
        while i < 4 {
            let v = unsafe { sptr.add(i).read() };
            unsafe { dptr.add(i).write(v) };
            i += 1;
        }
    }

    /// Embedding lookup: out[b*dim + i] = f16(table[tokens[b]*dim + i])
    /// Launch: grid (B, 1, 1), block (256, 1, 1). dim % 8 == 0.
    #[kernel]
    pub fn embedding_lookup_fp16(
        mut out: DisjointSlice<u16>,
        table: &[u16],
        tokens: &[u32],
        dim: u32,
        batch_size: u32,
    ) {
        let b = thread::blockIdx_x() as usize;
        if b >= batch_size as usize {
            return;
        }
        let token = unsafe { *tokens.get_unchecked(b) } as usize;
        let d = dim as usize;
        let base = token * d;
        let tid = thread::threadIdx_x() as usize;
        let out_base = b * d;
        let stride = 256 * 8;
        let mut k = tid * 8;
        while k < d {
            let dptr = table.as_ptr() as *const u32;
            let optr = out.as_mut_ptr().wrapping_add(out_base + k) as *mut u32;
            let mut i = 0usize;
            while i < 4 {
                let v = unsafe { dptr.wrapping_add((base + k) / 2 + i).read() };
                unsafe { optr.add(i).write(v) };
                i += 1;
            }
            k += stride;
        }
    }

    /// Residual add: hidden[i] = f16(f32(hidden[i]) + f32(ffn_out[i]))
    /// Launch: grid (1, B, 1), block (256, 1, 1). `h_size` must be a multiple of 8.
    #[kernel]
    pub fn residual_add_fp16(
        mut hidden: DisjointSlice<u16>,
        mut ffn_out: DisjointSlice<u16>,
        h_size: u32,
        b_size: u32,
    ) {
        let b = thread::blockIdx_y() as usize;
        if b >= b_size as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let h = h_size as usize;
        let stride = 256 * 8;
        let mut k = tid * 8;
        while k < h {
            let hp = hidden.as_mut_ptr().wrapping_add(b * h + k) as *mut u32;
            let fp = ffn_out.as_mut_ptr().wrapping_add(b * h + k) as *const u32;
            let mut i = 0usize;
            while i < 4 {
                let (hlo, hhi) = convert::cvt_f32x2_f16x2(unsafe { *hp.add(i) });
                let (flo, fhi) = convert::cvt_f32x2_f16x2(unsafe { *fp.add(i) });
                let plo = convert::cvt_f16x2_f32(hlo + flo, hhi + fhi);
                unsafe { *hp.add(i) = plo; }
                i += 1;
            }
            k += stride;
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    // quick smoke test on random data vs host reference
    const HIDDEN: usize = 2048;
    const B: usize = 2;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let x: Vec<f32> = (0..B * HIDDEN).map(|_| rng.f32(-1.5, 1.5)).collect();
    let w: Vec<f32> = (0..HIDDEN).map(|_| rng.f32(-1.5, 1.5)).collect();
    let x_host: Vec<u16> = x.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    let w_host: Vec<u16> = w.iter().map(|&v| f16::from_f32(v).to_bits()).collect();

    let mut x_dev = DeviceBuffer::from_host(&stream, &x_host).unwrap();
    let mut w_dev = DeviceBuffer::from_host(&stream, &w_host).unwrap();
    let mut y_dev = DeviceBuffer::<u16>::zeroed(&stream, B * HIDDEN).unwrap();
    let cfg = LaunchConfig { grid_dim: (1, B as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        module.batched_rms_norm_vec8_fp16(&stream, cfg, &mut y_dev, &mut x_dev, &mut w_dev,
            1e-5f32, HIDDEN as u32, B as u32).expect("launch");
    }
    stream.synchronize().unwrap();
    let y = y_dev.to_host_vec(&stream).unwrap();

    // host reference (matches kernel arithmetic)
    let eps = 1e-5f32;
    let mut max_rel = 0.0f64;
    for b in 0..B {
        let base = b * HIDDEN;
        let mut ss = 0.0f32;
        for i in 0..HIDDEN {
            let v = f16::from_f32((x[base + i])).to_f64() as f32; // f16-rounded
            ss += v * v;
        }
        let inv = 1.0 / (ss / HIDDEN as f32 + eps).sqrt();
        for i in 0..HIDDEN {
            let xv = f16::from_f32(x[base + i]).to_f64() as f32;
            let wv = f16::from_f32(w[i]).to_f64() as f32;
            let want = (xv * inv * wv) as f64;
            let got = f16::from_bits(y[base + i]).to_f64();
            max_rel = max_rel.max((got - want).abs() / (1.0 + want.abs()));
        }
    }
    println!("rms_norm rel = {max_rel:.6} {}", if max_rel < 2e-3 { "ok" } else { "FAIL" });

    // residual add smoke test
    let mut h_dev = DeviceBuffer::from_host(&stream, &x_host).unwrap();
    let ffn: Vec<f32> = (0..B * HIDDEN).map(|i| (i as f32 * 0.27 + 0.9).cos()).collect();
    let ffn_host: Vec<u16> = ffn.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    let mut f_dev = DeviceBuffer::from_host(&stream, &ffn_host).unwrap();
    unsafe {
        module.residual_add_fp16(&stream, cfg, &mut h_dev, &mut f_dev, HIDDEN as u32, B as u32).expect("launch");
    }
    stream.synchronize().unwrap();
    let h = h_dev.to_host_vec(&stream).unwrap();
    let max_add = (0..B).map(|b| {
        (0..HIDDEN).map(|i| {
            let hl = f16::from_f32(x[b * HIDDEN + i]).to_f64();
            let fl = f16::from_f32(ffn[b * HIDDEN + i]).to_f64();
            f16::from_bits(h[b * HIDDEN + i]).to_f64() - (hl + fl)
        }).fold(0.0f64, |a, v| a.max(v.abs()))
    }).fold(0.0f64, |a, v| a.max(v));
    println!("residual_add max err = {max_add:.6} {}", if max_add < 2e-3 { "ok" } else { "FAIL" });

    // ---- smoke test swiglu ----
    let n = 1024usize;
    let gate: Vec<f32> = (0..n).map(|i| (i as f32 * 0.31 + 0.2).sin() * 3.0).collect();
    let up: Vec<f32> = (0..n).map(|i| (i as f32 * 0.17 + 0.7).cos() * 2.0).collect();
    let gate_host: Vec<u16> = gate.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    let up_host: Vec<u16> = up.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    let mut act_dev = DeviceBuffer::<u16>::zeroed(&stream, n).unwrap();
    let gate_dev = DeviceBuffer::from_host(&stream, &gate_host).unwrap();
    let up_dev = DeviceBuffer::from_host(&stream, &up_host).unwrap();
    let cfg2 = LaunchConfig { grid_dim: ((n as u32 + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        module.batched_swiglu_separate_fp16(&stream, cfg2, &mut act_dev, &gate_dev, &up_dev, n as u32)
            .expect("swiglu launch");
    }
    stream.synchronize().unwrap();
    let act = act_dev.to_host_vec(&stream).unwrap();
    let mut max_swiglu = 0.0f64;
    for i in 0..n {
        let g = f16::from_f32(gate[i]).to_f64();
        let u = f16::from_f32(up[i]).to_f64();
        let want = u * g / (1.0 + (-g * 1.4426950208889634f64).exp2());
        let got = f16::from_bits(act[i]).to_f64();
        max_swiglu = max_swiglu.max((got - want).abs());
    }
    println!("swiglu max err = {max_swiglu:.6} ");

    // ---- smoke test rope ----
    const HD: usize = 128;
    const B2: usize = 2;
    const NH: usize = 4;
    let x2: Vec<f32> = (0..B2 * NH * HD).map(|i| (i as f32 * 0.13 + 0.4).sin()).collect();
    let x2_host: Vec<u16> = x2.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    // cos/sin table [max_seq_len=8, HD]
    const MAXSEQ: usize = 8;
    // build cos/sin host arrays directly with duplicated halves
    let mut cos_v = Vec::with_capacity(MAXSEQ * HD);
    let mut sin_v = Vec::with_capacity(MAXSEQ * HD);
    for pos in 0..MAXSEQ {
        let mut freqs = Vec::with_capacity(HD);
        for d in 0..HD / 2 {
            freqs.push(pos as f32 * 1.0f32 / 10000f32.powf((2 * d) as f32 / HD as f32));
        }
        for d in 0..HD / 2 {
            freqs.push(pos as f32 * 1.0f32 / 10000f32.powf((2 * d) as f32 / HD as f32));
        }
        for &f in &freqs {
            cos_v.push(f16::from_f32(f.cos()).to_bits());
            sin_v.push(f16::from_f32(f.sin()).to_bits());
        }
    }
    let cos_dev = DeviceBuffer::from_host(&stream, &cos_v).unwrap();
    let sin_dev = DeviceBuffer::from_host(&stream, &sin_v).unwrap();
    let mut x2_dev = DeviceBuffer::<u16>::from_host(&stream, &x2_host).unwrap();
    let pos_dev = DeviceBuffer::from_host(&stream, &[1u32, 3u32]).unwrap();
    let cfg3 = LaunchConfig { grid_dim: (NH as u32, B2 as u32, 1), block_dim: (HD as u32, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        module.batched_rope_fp16(&stream, cfg3, &mut x2_dev, &cos_dev, &sin_dev, &pos_dev, NH as u32, HD as u32, B2 as u32)
            .expect("rope launch");
    }
    stream.synchronize().unwrap();
    let xo = x2_dev.to_host_vec(&stream).unwrap();
    // host reference
    let mut x_ref = x2.clone();
    for b in 0..B2 {
        let pos = [1usize, 3usize][b];
        for h in 0..NH {
            for i in 0..HD / 2 {
                let f = 1.0f32 / 10000f32.powf(2.0f32 * i as f32 / HD as f32);
                let c = (pos as f32 * f).cos();
                let s = (pos as f32 * f).sin();
                let xi = x_ref[(b * NH + h) * HD + i];
                let xj = x_ref[(b * NH + h) * HD + i + HD / 2];
                x_ref[(b * NH + h) * HD + i] = xi * c - xj * s;
                x_ref[(b * NH + h) * HD + i + HD / 2] = xj * c + xi * s;
            }
        }
    }
    let mut max_rope = 0.0f64;
    for i in 0..B2 * NH * HD {
        let got = f16::from_bits(xo[i]).to_f64();
        let want = f16::from_f32(x_ref[i] as f32).to_f64();
        max_rope = max_rope.max((got - want).abs() / (1.0 + want.abs()));
    }
    println!("rope rel = {max_rope:.6} {}", if max_rope < 2e-3 { "ok" } else { "FAIL" });

    // ---- smoke test embedding lookup ----
    const DIM: usize = 64;
    const VOCAB: usize = 32;
    let table: Vec<f32> = (0..VOCAB * DIM).map(|i| (i as f32 * 0.17 + 0.3).sin()).collect();
    let table_host: Vec<u16> = table.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    let tokens_host: Vec<u32> = vec![3u32, 17u32];
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, 2 * DIM).unwrap();
    let table_dev = DeviceBuffer::from_host(&stream, &table_host).unwrap();
    let tokens_dev = DeviceBuffer::from_host(&stream, &tokens_host).unwrap();
    let cfg4 = LaunchConfig { grid_dim: (2, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        module.embedding_lookup_fp16(&stream, cfg4, &mut out_dev, &table_dev, &tokens_dev, DIM as u32, 2)
            .expect("embed launch");
    }
    stream.synchronize().unwrap();
    let out = out_dev.to_host_vec(&stream).unwrap();
    let mut max_emb = 0.0f64;
    for b in 0..2usize {
        let tok = tokens_host[b] as usize;
        for i in 0..DIM {
            let want = f16::from_f32(table[tok * DIM + i]).to_f64();
            let got = f16::from_bits(out[b * DIM + i]).to_f64();
            max_emb = max_emb.max((got - want).abs());
        }
    }
    println!("embedding max err = {max_emb:.6} {}", if max_emb < 2e-3 { "ok" } else { "FAIL" });

    // ---- smoke test argmax ----
    const VOC: usize = 2000;
    let logits: Vec<f32> = (0..VOC).map(|i| (i as f32 * 0.13 + 0.5).sin() * 100.0).collect();
    let logits_host: Vec<u16> = logits.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    let logits_dev = DeviceBuffer::from_host(&stream, &logits_host).unwrap();
    let mut tok_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    let cfg5 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    // batch 0: row = logits, batch 1: row = logits with last element spiked
    let logits2: Vec<u16> = {
        let mut v = logits.clone();
        v[VOC - 1] = 10_000.0;
        v.iter().map(|&x| f16::from_f32(x).to_bits()).collect()
    };
    let logits2_dev = DeviceBuffer::from_host(&stream, &logits2).unwrap();
    let mut tok2_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    unsafe {
        module.batched_argmax_direct_f16(&stream, cfg5, &mut tok_dev, &logits_dev, VOC as u32)
            .expect("argmax launch");
    }
    unsafe {
        module.batched_argmax_direct_f16(&stream, cfg5, &mut tok2_dev, &logits2_dev, VOC as u32)
            .expect("argmax2 launch");
    }
    stream.synchronize().unwrap();
    let tok = tok_dev.to_host_vec(&stream).unwrap();
    let tok2 = tok2_dev.to_host_vec(&stream).unwrap();
    let mut want0 = 0usize;
    for i in 1..VOC {
        if logits[i] > logits[want0] { want0 = i; }
    }
    let ok = tok[0] as usize == want0 && tok2[0] as usize == VOC - 1;
    println!("argmax batch0 want={want0} got={} | batch1 spiked got={} {}", tok[0], tok2[0], if ok { "ok" } else { "FAIL" });
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f32(&mut self, lo: f32, hi: f32) -> f32 {
        let t = (self.next() as f32 / u64::MAX as f32) as f32;
        lo + t * (hi - lo)
    }
}
