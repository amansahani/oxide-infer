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
