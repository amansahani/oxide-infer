//! Unit check for the sliding-window mask in `batched_gqa_attention_fp16`.
//!
//! Runs the real PTX kernel over a synthetic 300-token KV cache and compares
//! against an f64 host reference. The boundary matters: LFM3/llama mask a key
//! when `p_query - p_key >= window`, so for a query at position 299 a window
//! of 256 keeps keys 44..=299 (256 keys) and drops key 43 exactly.

use anyhow::{anyhow, Result};
use cuda_core::launch_kernel_on_stream;
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext};
use oxide_infer_kernels::batched_attn::{BATCHED_GQA_ATTENTION_ONLINE_PTX, BATCHED_GQA_ATTENTION_PTX};
use oxide_infer_kernels::PtxModule;
use std::ffi::c_void;

const NQ: usize = 16;
const NKV: usize = 4;
const HD: usize = 128;
const SEQ: usize = 300;
const SCALE: f64 = 0.0883883476; // 1 / sqrt(128), matching the kernel

fn pat(seed: u64, idx: usize) -> f32 {
    // Cheap deterministic pseudo-random in [-1, 1].
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(idx as u64);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    ((x & 0xffff) as f32 / 32767.5) - 1.0
}

fn reference(q: &[f32], k: &[f32], v: &[f32], window: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; NQ * HD];
    let p = (SEQ - 1) as i64;
    for h in 0..NQ {
        let kvh = h / 4;
        let mut scores = vec![f64::NEG_INFINITY; SEQ];
        for t in 0..SEQ {
            let dist = p - t as i64;
            if window > 0 && dist >= window as i64 {
                continue;
            }
            let mut dot = 0.0f64;
            for d in 0..HD {
                dot += (q[h * HD + d] as f64) * (k[(t * NKV + kvh) * HD + d] as f64);
            }
            scores[t] = dot * SCALE;
        }
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0.0f64;
        let mut w = vec![0.0f64; SEQ];
        for t in 0..SEQ {
            if scores[t].is_finite() {
                w[t] = (scores[t] - max).exp();
                denom += w[t];
            }
        }
        for d in 0..HD {
            let mut acc = 0.0f64;
            for t in 0..SEQ {
                if w[t] != 0.0 {
                    acc += w[t] * (v[(t * NKV + kvh) * HD + d] as f64);
                }
            }
            out[h * HD + d] = (acc / denom) as f32;
        }
    }
    out
}

fn main() -> Result<()> {
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();
    let scan = PtxModule::load_from_ptx(&gpu, BATCHED_GQA_ATTENTION_PTX)
        .map_err(|e| anyhow!("gqa PTX: {e}"))?;
    let scan_fn = scan.get_function("batched_gqa_attention_fp16")?;
    let online = PtxModule::load_from_ptx(&gpu, BATCHED_GQA_ATTENTION_ONLINE_PTX)
        .map_err(|e| anyhow!("gqa PTX: {e}"))?;
    let online_fn = online.get_function("batched_gqa_attention_online_fp16")?;

    let q: Vec<f32> = (0..NQ * HD).map(|i| pat(1, i)).collect();
    let k: Vec<f32> = (0..SEQ * NKV * HD).map(|i| pat(2, i) * 2.0).collect();
    let v: Vec<f32> = (0..SEQ * NKV * HD).map(|i| pat(3, i)).collect();

    let d_q = DeviceBuffer::from_slice(&gpu, &q.iter().map(|x| f16::from_f32(*x)).collect::<Vec<_>>())?;
    let d_k = DeviceBuffer::from_slice(&gpu, &k.iter().map(|x| f16::from_f32(*x)).collect::<Vec<_>>())?;
    let d_v = DeviceBuffer::from_slice(&gpu, &v.iter().map(|x| f16::from_f32(*x)).collect::<Vec<_>>())?;
    let mut d_ctx = DeviceBuffer::<f16>::zeroed(&gpu, NQ * HD)?;
    let d_sl = DeviceBuffer::from_slice(&gpu, &[SEQ as u32])?;
    let d_kp = DeviceBuffer::from_slice(&gpu, &[d_k.inner().cu_deviceptr()])?;
    let d_vp = DeviceBuffer::from_slice(&gpu, &[d_v.inner().cu_deviceptr()])?;

    let mut failed = false;
    for (name, gqa, grid_x) in [("scan", &scan_fn, NQ), ("online", &online_fn, NKV)] {
    for window in [0u32, 512, 300, 257, 256, 128] {
        let mut d_ctxp = d_ctx.inner_mut().cu_deviceptr();
        let mut d_qp = d_q.inner().cu_deviceptr();
        let mut d_kpp = d_kp.inner().cu_deviceptr();
        let mut d_vpp = d_vp.inner().cu_deviceptr();
        let mut d_slp = d_sl.inner().cu_deviceptr();
        let (mut nq, mut nkv, mut hd) = (NQ as u32, NKV as u32, HD as u32);
        let mut b = 1u32;
        let mut w = window;
        let mut args: [*mut c_void; 10] = [
            &mut d_ctxp as *mut _ as *mut c_void,
            &mut d_qp as *mut _ as *mut c_void,
            &mut d_kpp as *mut _ as *mut c_void,
            &mut d_vpp as *mut _ as *mut c_void,
            &mut d_slp as *mut _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut w as *mut _ as *mut c_void,
        ];
        unsafe {
            launch_kernel_on_stream(gqa, (grid_x as u32, 1, 1), (HD as u32, 1, 1), 0, stream, &mut args)?;
        }
        stream.synchronize()?;

        let got = d_ctx.inner().to_host_vec(stream)?;
        let want = reference(&q, &k, &v, window);
        let mut max_diff = 0.0f32;
        for (g, e) in got.iter().zip(want.iter()) {
            max_diff = max_diff.max((g.to_f32() - e).abs());
        }
        let kept = if window == 0 { SEQ } else { (window as usize).min(SEQ) };
        let ok = max_diff < 2e-3;
        failed |= !ok;
        println!(
            "{name:6} window={window:4} kept={kept:3} max|ctx-ref|={max_diff:.6} {}",
            if ok { "OK" } else { "FAIL" }
        );
        failed |= !ok;
    }
    }
    if failed {
        return Err(anyhow!("sliding-window attention check failed"));
    }
    Ok(())
}
