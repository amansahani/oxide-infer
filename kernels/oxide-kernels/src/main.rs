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
    fn warp_tree_sum(val: f32) -> f32 {
        let mut val = val;
        val += warp::shuffle_down_f32_sync(0xffffffff, val, 16);
        val += warp::shuffle_down_f32_sync(0xffffffff, val, 8);
        val += warp::shuffle_down_f32_sync(0xffffffff, val, 4);
        val += warp::shuffle_down_f32_sync(0xffffffff, val, 2);
        val += warp::shuffle_down_f32_sync(0xffffffff, val, 1);
        val
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
