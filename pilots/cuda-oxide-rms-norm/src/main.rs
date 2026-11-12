//! Pilot 2: RMSNorm (f16 weights, f32 accumulate) in cuda-oxide.
//!
//! Faithful port of oxide-infer's RMS_NORM_FP16_PTX (crates/oxide-infer-kernels/src/norm.rs)
//! to a `#[kernel]` Rust function. Differences are deliberate and documented:
//!
//! - The PTX reduces warps with a shared-memory `atom.shared.add.f32` in
//!   arbitrary order. This port uses a fixed-order per-warp slot sum, which is
//!   deterministic across runs. Both are f32 sums; results differ at most by
//!   a few ulps pre-rsqrt.
//! - Writes use `get_unchecked_mut`: uniqueness comes from the striding
//!   algorithm (each thread owns disjoint words of each row), bounds are
//!   validated by the loop condition.
//! - f16 data is carried as packed u32 words, exactly like the PTX.

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use half::f16;

#[cuda_module]
mod kernels {
    use cuda_device::{DisjointSlice, SharedArray, convert, float, kernel, thread, warp};

    /// c[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * w
    ///
    /// input/output: words of 2 packed f16 values per u32.
    /// weight: same packed layout, one row shared by all rows of input.
    /// hidden_size is given in f16 elements (matches the PTX parameter).
    #[kernel]
    pub fn rms_norm_fp16(
        input: &[u32],
        weight: &[u32],
        mut output: DisjointSlice<u32>,
        epsilon: f32,
        hidden_size: u32,
    ) {
        static mut S_WARP_SUMS: SharedArray<f32, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let ntid = thread::blockDim_x();
        let row = thread::blockIdx_x() as usize;
        let hw = (hidden_size / 2) as usize;
        let row_base = row * hw;

        // Phase 1: sum of squares, f32 accumulate (mirrors PTX fma.rn.f32).
        let mut sum_sq = 0.0f32;
        let mut i = tid as usize;
        while i < hw {
            let (lo, hi) = convert::cvt_f32x2_f16x2(input[row_base + i]);
            sum_sq = float::fma_rn_f32(lo, lo, sum_sq);
            sum_sq = float::fma_rn_f32(hi, hi, sum_sq);
            i += ntid as usize;
        }

        // Phase 2: warp down-shuffle reduction (PTX shfl.sync.down.b32 x5).
        let mut val = sum_sq;
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 16);
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 8);
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 4);
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 2);
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 1);

        let lane = warp::lane_id();
        let warp_id = warp::warp_id() as usize;
        if lane == 0 {
            // SAFETY: 32 slots, warp_id < 32 for any blockDim <= 1024.
            unsafe { S_WARP_SUMS[warp_id] = val; }
        }
        thread::sync_threads();

        // Phase 3: deterministic ordered sum across warps (PTX used
        // atom.shared.add.f32 in arbitrary order; we don't).
        let num_warps = (ntid as usize).div_ceil(32);
        let mut total = 0.0f32;
        for w in 0..num_warps {
            // SAFETY: w < num_warps <= 32 and every warp wrote its slot.
            total += unsafe { S_WARP_SUMS[w] };
        }

        let mean_sq = total / hidden_size as f32 + epsilon;
        let inv_rms = float::rsqrt_approx_f32(mean_sq);

        // Phase 4: normalize + scale + round to packed f16x2.
        let mut i = tid as usize;
        while i < hw {
            let (x0, x1) = convert::cvt_f32x2_f16x2(input[row_base + i]);
            let (w0, w1) = convert::cvt_f32x2_f16x2(weight[i]);
            let packed = convert::cvt_f16x2_f32(inv_rms * x0 * w0, inv_rms * x1 * w1);
            // SAFETY:
            // - Bounds: row < gridDim.y-implicit, i < hw by the loop condition.
            // - Uniqueness: within a row each word is written by exactly one
            //   thread (striding by ntid); rows are disjoint word ranges.
            unsafe { *output.get_unchecked_mut(row_base + i) = packed; }
            i += ntid as usize;
        }
    }
}

use cuda_device::cuda_module;

const ROWS: usize = 4;
const HIDDEN: usize = 2048; // f16 elements; matches LFM3 hidden

fn pack(lo: f16, hi: f16) -> u32 {
    (lo.to_bits() as u32) | ((hi.to_bits() as u32) << 16)
}
fn lo16(w: u32) -> f16 {
    f16::from_bits(w as u16)
}
fn hi16(w: u32) -> f16 {
    f16::from_bits((w >> 16) as u16)
}

fn main() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let n_words = ROWS * HIDDEN / 2;
    let hw = HIDDEN / 2;

    let a_f32: Vec<f32> = (0..ROWS * HIDDEN)
        .map(|i| (i as f32 * 0.0017).sin() * 2.0 - 1.0)
        .collect();
    let w_f32: Vec<f32> = (0..HIDDEN)
        .map(|i| 1.0 + (i as f32 * 0.0031).cos() * 0.5)
        .collect();

    let in_host: Vec<u32> = (0..n_words)
        .map(|w| pack(f16::from_f32(a_f32[2 * w]), f16::from_f32(a_f32[2 * w + 1])))
        .collect();
    let wt_host: Vec<u32> = (0..hw)
        .map(|w| pack(f16::from_f32(w_f32[2 * w]), f16::from_f32(w_f32[2 * w + 1])))
        .collect();

    let in_dev = DeviceBuffer::from_host(&stream, &in_host).unwrap();
    let wt_dev = DeviceBuffer::from_host(&stream, &wt_host).unwrap();
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, n_words).unwrap();

    let epsilon = 1e-6f32;

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    unsafe {
        module.rms_norm_fp16(
            &stream,
            LaunchConfig {
                grid_dim: (ROWS as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            &in_dev,
            &wt_dev,
            &mut out_dev,
            epsilon,
            HIDDEN as u32,
        )
    }
    .expect("Kernel launch failed");

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    // Host reference: same algorithm, f64 ground truth then compare f16 bits.
    let mut max_ulp_diff = 0i32;
    let mut mismatch = 0usize;
    for r in 0..ROWS {
        let base = r * HIDDEN;
        let mut sum_sq = 0.0f64;
        for j in 0..HIDDEN {
            let x = a_f32[base + j] as f64;
            sum_sq += x * x;
        }
        let inv_rms = 1.0 / (sum_sq / HIDDEN as f64 + epsilon as f64).sqrt();
        for j in 0..HIDDEN {
            let got = if j % 2 == 0 {
                lo16(out_host[r * hw + j / 2])
            } else {
                hi16(out_host[r * hw + j / 2])
            };
            let reference = f16::from_f32((inv_rms * a_f32[base + j] as f64 * w_f32[j] as f64) as f32);
            let gb = got.to_bits() as i32;
            let rb = reference.to_bits() as i32;
            // Ignore sign-flip on zero.
            let d = (gb - rb).abs();
            if d != 0 && !(got == f16::from_f32(0.0) && reference == f16::from_f32(0.0)) {
                mismatch += 1;
                max_ulp_diff = max_ulp_diff.max(d);
            }
        }
    }

    // Also compare against a strict f32 sequential host reference.
    let mut seq_max_ulp = 0i32;
    for r in 0..ROWS {
        let base = r * HIDDEN;
        let mut sum_sq = 0.0f32;
        for j in 0..HIDDEN {
            let x = a_f32[base + j];
            sum_sq = x.mul_add(x, sum_sq);
        }
        let mean_sq = sum_sq / HIDDEN as f32 + epsilon;
        let inv_rms = 1.0 / mean_sq.sqrt(); // host f32 sqrt, not rsqrt.approx
        for j in 0..HIDDEN {
            let got = if j % 2 == 0 {
                lo16(out_host[r * hw + j / 2])
            } else {
                hi16(out_host[r * hw + j / 2])
            };
            let reference = f16::from_f32(inv_rms * a_f32[base + j] * w_f32[j]);
            let d = (got.to_bits() as i32 - reference.to_bits() as i32).abs();
            seq_max_ulp = seq_max_ulp.max(d);
        }
    }

    println!(
        "rms_norm_fp16: {} rows x {} hidden | bit-mismatches vs f64: {} | max ulp diff vs f64: {} | max ulp diff vs f32-seq: {}",
        ROWS, HIDDEN, mismatch, max_ulp_diff, seq_max_ulp
    );
    if max_ulp_diff <= 2 && seq_max_ulp <= 2 {
        println!("pilot OK: within 2 ulp (rsqrt.approx + reordering envelope)");
    } else {
        eprintln!("pilot FAILED: max ulp diffs {max_ulp_diff}/{seq_max_ulp}");
        std::process::exit(1);
    }
}
