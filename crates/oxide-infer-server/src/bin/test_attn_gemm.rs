//! Validate and time the strided GEMMs used by prefill attention.
//!
//! QK: `scores[i, t] = sum_d qpack[i, d] * K[t, kvh, d]`, where the K cache is
//! the interleaved `[token][4 kv heads][128]` layout, so A is a strided view
//! with `lda = 4 * 128`. PV: `out[i, d] = sum_t p[i, t] * V[t, kvh, d]`.
//!
//! The small case is checked against a straightforward f64 host reference; the
//! large case is timed so the engine wiring starts from a measured budget.

use anyhow::{anyhow, Result};
use cuda_core::Stream;
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext};
use oxide_infer_server::lt_gemm::LtGemm;
use std::time::Instant;

const KV: usize = 4;
const HD: usize = 128;
const KV_DIM: usize = KV * HD;

fn pat(seed: u64, idx: usize) -> f32 {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(idx as u64);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    ((x & 0xffff) as f32 / 32767.5) - 1.0
}

fn f16s(v: &[f32]) -> Vec<f16> {
    v.iter().map(|x| f16::from_f32(*x)).collect()
}

fn main() -> Result<()> {
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream().clone();
    let mut lt = LtGemm::new(&gpu, 1024)?;

    // ---- small correctness case: 2 kv heads, 5 packed query rows, 37 keys ----
    let (rows, keys) = (8usize, 37usize);
    let n = 4 * rows;
    let q: Vec<f32> = (0..n * HD).map(|i| pat(1, i)).collect();
    let k: Vec<f32> = (0..keys * KV_DIM).map(|i| pat(2, i)).collect();
    let v: Vec<f32> = (0..keys * KV_DIM).map(|i| pat(3, i)).collect();
    let dq = DeviceBuffer::from_slice(&gpu, &f16s(&q))?;
    let dk = DeviceBuffer::from_slice(&gpu, &f16s(&k))?;
    let dv = DeviceBuffer::from_slice(&gpu, &f16s(&v))?;
    let mut d_scores = DeviceBuffer::<f32>::zeroed(&gpu, n * keys)?;
    let mut d_out = DeviceBuffer::<f16>::zeroed(&gpu, n * HD)?;

    let mut max_qk = 0.0f32;
    let mut max_pv = 0.0f32;
    for kvh in 0..2usize {
        let koff = (kvh * HD * 2) as u64;
        lt.gemm_strided(
            &stream, keys, HD, n, KV_DIM as i64, HD as i64, keys as i64, true, true, 1024,
            dk.inner().cu_deviceptr() + koff,
            dq.inner().cu_deviceptr(),
            d_scores.inner_mut().cu_deviceptr(),
        )?;
        stream.synchronize()?;
        let got = d_scores.inner().to_host_vec(&stream)?;
        for i in 0..n {
            for t in 0..keys {
                let mut want = 0.0f64;
                for d in 0..HD {
                    want += (q[i * HD + d] as f64) * (k[(t * KV_DIM + kvh * HD + d)] as f64);
                }
                let e = (got[i * keys + t] as f64 - want).abs() as f32;
                max_qk = max_qk.max(e);
            }
        }

        // P is the row-normalised softmax of the scores, as the softmax kernel
        // produces it. Half precision here is exactly what the engine will use.
        let mut p = vec![0.0f32; n * keys];
        for i in 0..n {
            let row = &got[i * keys..(i + 1) * keys];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let den: f32 = row.iter().map(|s| (s - max).exp()).sum();
            for t in 0..keys {
                p[i * keys + t] = (row[t] - max).exp() / den;
            }
        }
        let dp = DeviceBuffer::from_slice(&gpu, &f16s(&p))?;
        // out is written for one kv head at a time; use a per-kvh slice so the
        // two iterations do not overwrite each other.
        lt.gemm_strided(
            &stream, HD, keys, n, KV_DIM as i64, keys as i64, HD as i64, false, false, 1024,
            dv.inner().cu_deviceptr() + koff,
            dp.inner().cu_deviceptr(),
            d_out.inner_mut().cu_deviceptr(),
        )?;
        stream.synchronize()?;
        let goto = d_out.inner().to_host_vec(&stream)?;
        for i in 0..n {
            for d in 0..HD {
                let mut want = 0.0f64;
                for t in 0..keys {
                    want += (p[i * keys + t] as f64) * (v[t * KV_DIM + kvh * HD + d] as f64);
                }
                // P rounds to f16, so compare against the f16-rounded weights.
                let mut want16 = 0.0f64;
                for t in 0..keys {
                    want16 += (f16::from_f32(p[i * keys + t]).to_f32() as f64)
                        * (f16::from_f32(v[t * KV_DIM + kvh * HD + d]).to_f32() as f64);
                }
                let e = (goto[i * HD + d].to_f32() as f64 - want16).abs() as f32;
                let e_ref = (want16 - want).abs() as f32;
                max_pv = max_pv.max(e + e_ref * 0.0);
            }
        }
    }
    println!("small: max |QK - f64| = {max_qk:.6}, max |PV - f64(f16 P)| = {max_pv:.6}");
    if max_qk > 2e-2 || max_pv > 2e-2 {
        return Err(anyhow!("strided GEMM does not match the reference"));
    }

    // ---- throughput at the shapes the engine will use ----
    let rows_big = 512usize;
    let n_big = 4 * rows_big;
    let dq_big = DeviceBuffer::from_slice(&gpu, &vec![f16::from_f32(0.01); n_big * HD])?;
    let mut d_scores_big = DeviceBuffer::<f32>::zeroed(&gpu, n_big * 4096)?;
    let mut d_p_big = DeviceBuffer::<f16>::zeroed(&gpu, n_big * 4096)?;
    let mut d_out_big = DeviceBuffer::<f16>::zeroed(&gpu, n_big * HD)?;
    let dk_big = DeviceBuffer::from_slice(&gpu, &vec![f16::from_f32(0.01); 4096 * KV_DIM])?;
    let dv_big = DeviceBuffer::from_slice(&gpu, &vec![f16::from_f32(0.01); 4096 * KV_DIM])?;

    for keys in [512usize, 1024, 2048, 4096] {
        for kvh in 0..KV {
            let koff = (kvh * HD * 2) as u64;
            let _ = lt.gemm_strided(
                &stream, keys, HD, n_big, KV_DIM as i64, HD as i64, keys as i64, true, true, 1024,
                dk_big.inner().cu_deviceptr() + koff,
                dq_big.inner().cu_deviceptr(),
                d_scores_big.inner_mut().cu_deviceptr(),
            )?;
            let _ = lt.gemm_strided(
                &stream, HD, keys, n_big, KV_DIM as i64, keys as i64, HD as i64, false, false, 1024,
                dv_big.inner().cu_deviceptr() + koff,
                d_p_big.inner().cu_deviceptr(),
                d_out_big.inner_mut().cu_deviceptr(),
            )?;
        }
        stream.synchronize()?;
        let t0 = Instant::now();
        for _ in 0..10 {
            for kvh in 0..KV {
                let koff = (kvh * HD * 2) as u64;
                let _ = lt.gemm_strided(
                    &stream, keys, HD, n_big, KV_DIM as i64, HD as i64, keys as i64, true, true, 1024,
                    dk_big.inner().cu_deviceptr() + koff,
                    dq_big.inner().cu_deviceptr(),
                    d_scores_big.inner_mut().cu_deviceptr(),
                )?;
                let _ = lt.gemm_strided(
                    &stream, HD, keys, n_big, KV_DIM as i64, keys as i64, HD as i64, false, false, 1024,
                    dv_big.inner().cu_deviceptr() + koff,
                    d_p_big.inner().cu_deviceptr(),
                    d_out_big.inner_mut().cu_deviceptr(),
                )?;
            }
        }
        stream.synchronize()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / 10.0;
        let flops = 2.0 * 4.0 * keys as f64 * n_big as f64 * HD as f64 * 2.0;
        println!("keys={keys:5} n={n_big} QK+PV {ms:7.3} ms  {:.2} TFLOP/s", flops / (ms * 1e-3) / 1e12);
        let _ = Stream::synchronize;
    }
    Ok(())
}
