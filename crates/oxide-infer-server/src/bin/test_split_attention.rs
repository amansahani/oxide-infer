//! Gate for split-K decode attention (`split` + `combine`) against an f64
//! host reference and against the single-pass online kernel.
//!
//! Rows have different sequence lengths (300/180/64/300) so chunk bounds and
//! the window intersection are exercised on partial chunks, and each row has
//! its own queries and K/V cache. Run for windows 0 and 256.

use anyhow::{anyhow, Result};
use cuda_core::launch_kernel_on_stream;
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext};
use oxide_infer_kernels::batched_attn::{
    BATCHED_GQA_ATTENTION_ONLINE_PTX, BATCHED_GQA_ATTENTION_SPLIT_PTX,
};
use oxide_infer_kernels::PtxModule;
use std::ffi::c_void;

const NQ: usize = 16;
const NKV: usize = 4;
const HD: usize = 128;
const MAX_SEQ: usize = 300;
const ROWS: [usize; 4] = [300, 180, 64, 300];
const CHUNK_KEYS: u32 = 64;
const NCHUNKS: usize = 5; // ceil(MAX_SEQ / CHUNK_KEYS)
const SCALE: f64 = 0.0883883476;

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

/// Full softmax over the keys kept by `window`, in f64.
fn reference(q: &[f32], k: &[f32], v: &[f32], slen: usize, window: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; NQ * HD];
    let p = (slen - 1) as i64;
    for h in 0..NQ {
        let kvh = h / 4;
        let mut scores = vec![f64::NEG_INFINITY; slen];
        for t in 0..slen {
            if window > 0 && p - t as i64 >= window as i64 {
                continue;
            }
            let mut dot = 0.0f64;
            for d in 0..HD {
                dot += (q[h * HD + d] as f64) * (k[(t * NKV + kvh) * HD + d] as f64);
            }
            scores[t] = dot * SCALE;
        }
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = scores
            .iter()
            .map(|s| if s.is_finite() { (s - max).exp() } else { 0.0 })
            .collect();
        let den: f64 = w.iter().sum();
        for d in 0..HD {
            let mut acc = 0.0f64;
            for t in 0..slen {
                acc += w[t] * (v[(t * NKV + kvh) * HD + d] as f64);
            }
            out[h * HD + d] = (acc / den) as f32;
        }
    }
    out
}

fn max_diff(a: &[f16], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x.to_f32() - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() -> Result<()> {
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();

    let online_mod = PtxModule::load_from_ptx(&gpu, BATCHED_GQA_ATTENTION_ONLINE_PTX)
        .map_err(|e| anyhow!("online PTX: {e}"))?;
    let online = online_mod.get_function("batched_gqa_attention_online_fp16")?;
    let split_mod = PtxModule::load_from_ptx(&gpu, BATCHED_GQA_ATTENTION_SPLIT_PTX)
        .map_err(|e| anyhow!("split PTX: {e}"))?;
    let split = split_mod.get_function("batched_gqa_attention_split_fp16")?;
    let combine = split_mod.get_function("batched_gqa_attn_combine_fp16")?;

    let nrows = ROWS.len();

    // Per-row queries, K/V caches and scalar views shared by both paths.
    let qh: Vec<f32> = (0..nrows * NQ * HD).map(|i| pat(1, i)).collect();
    let d_qall = DeviceBuffer::from_slice(&gpu, &f16s(&qh))?;
    let mut qrows: Vec<DeviceBuffer<f16>> = Vec::new();
    let mut kv: Vec<(Vec<f32>, Vec<f32>, DeviceBuffer<f16>, DeviceBuffer<f16>)> = Vec::new();
    for (r, &_slen) in ROWS.iter().enumerate() {
        qrows.push(DeviceBuffer::from_slice(
            &gpu,
            &f16s(&qh[r * NQ * HD..(r + 1) * NQ * HD]),
        )?);
        let k: Vec<f32> = (0..MAX_SEQ * NKV * HD).map(|i| pat(2 + r as u64, i) * 2.0).collect();
        let v: Vec<f32> = (0..MAX_SEQ * NKV * HD).map(|i| pat(9 + r as u64, i)).collect();
        let dk = DeviceBuffer::from_slice(&gpu, &f16s(&k))?;
        let dv = DeviceBuffer::from_slice(&gpu, &f16s(&v))?;
        kv.push((k, v, dk, dv));
    }

    let d_kp = DeviceBuffer::<u64>::from_slice(
        &gpu,
        &kv.iter().map(|c| c.2.inner().cu_deviceptr()).collect::<Vec<_>>(),
    )?;
    let d_vp = DeviceBuffer::<u64>::from_slice(
        &gpu,
        &kv.iter().map(|c| c.3.inner().cu_deviceptr()).collect::<Vec<_>>(),
    )?;
    let d_sl = DeviceBuffer::<u32>::from_slice(&gpu, &ROWS.iter().map(|s| *s as u32).collect::<Vec<_>>())?;

    let mut d_acc = DeviceBuffer::<f32>::zeroed(&gpu, nrows * NCHUNKS * NQ * HD)?;
    let mut d_stats = DeviceBuffer::<f32>::zeroed(&gpu, nrows * NCHUNKS * NQ * 2)?;
    let mut d_ctx = DeviceBuffer::<f16>::zeroed(&gpu, NQ * HD)?;
    let mut d_occ = DeviceBuffer::<f16>::zeroed(&gpu, nrows * NQ * HD)?;

    let mut failed = false;
    for window in [0u32, 256] {
        // --- single-pass online kernel, one row at a time ---
        let mut online_out = Vec::new();
        for r in 0..nrows {
            let d_k1 = DeviceBuffer::<u64>::from_slice(&gpu, &[kv[r].2.inner().cu_deviceptr()])?;
            let d_v1 = DeviceBuffer::<u64>::from_slice(&gpu, &[kv[r].3.inner().cu_deviceptr()])?;
            let d_s1 = DeviceBuffer::<u32>::from_slice(&gpu, &[ROWS[r] as u32])?;
            let mut d_ctxp = d_ctx.inner_mut().cu_deviceptr();
            let mut d_qp = qrows[r].inner().cu_deviceptr();
            let mut d_kpp = d_k1.inner().cu_deviceptr();
            let mut d_vpp = d_v1.inner().cu_deviceptr();
            let mut d_slp = d_s1.inner().cu_deviceptr();
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
                launch_kernel_on_stream(&online, (NKV as u32, 1, 1), (HD as u32, 1, 1), 0, stream, &mut args)?;
            }
            stream.synchronize()?;
            online_out.push(d_ctx.inner().to_host_vec(stream)?);
        }

        // --- split + combine over all rows at once ---
        let nchunks = NCHUNKS as u32;
        {
            let mut d_qp = d_qall.inner().cu_deviceptr();
            let mut d_kpp = d_kp.inner().cu_deviceptr();
            let mut d_vpp = d_vp.inner().cu_deviceptr();
            let mut d_slp = d_sl.inner().cu_deviceptr();
            let mut d_accp = d_acc.inner_mut().cu_deviceptr();
            let mut d_stp = d_stats.inner_mut().cu_deviceptr();
            let (mut nq, mut nkv, mut hd) = (NQ as u32, NKV as u32, HD as u32);
            let mut b = nrows as u32;
            let mut w = window;
            let mut ck = CHUNK_KEYS;
            let mut nc = nchunks;
            let mut args: [*mut c_void; 13] = [
                &mut d_qp as *mut _ as *mut c_void,
                &mut d_kpp as *mut _ as *mut c_void,
                &mut d_vpp as *mut _ as *mut c_void,
                &mut d_slp as *mut _ as *mut c_void,
                &mut d_accp as *mut _ as *mut c_void,
                &mut d_stp as *mut _ as *mut c_void,
                &mut nq as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut b as *mut _ as *mut c_void,
                &mut w as *mut _ as *mut c_void,
                &mut ck as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            unsafe {
                launch_kernel_on_stream(&split, (NKV as u32, b, nc), (HD as u32, 1, 1), 0, stream, &mut args)?;
            }
        }
        {
            let mut d_ctxp = d_occ.inner_mut().cu_deviceptr();
            let mut d_accp = d_acc.inner_mut().cu_deviceptr();
            let mut d_stp = d_stats.inner_mut().cu_deviceptr();
            let (mut nq, mut hd) = (NQ as u32, HD as u32);
            let mut b = nrows as u32;
            let mut nc = nchunks;
            let mut args: [*mut c_void; 7] = [
                &mut d_ctxp as *mut _ as *mut c_void,
                &mut d_accp as *mut _ as *mut c_void,
                &mut d_stp as *mut _ as *mut c_void,
                &mut nq as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut b as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            unsafe {
                launch_kernel_on_stream(&combine, (nq, b, 1), (32, 1, 1), 0, stream, &mut args)?;
            }
        }
        stream.synchronize()?;
        let occ = d_occ.inner().to_host_vec(stream)?;

        for (r, &slen) in ROWS.iter().enumerate() {
            let want = reference(&qh[r * NQ * HD..(r + 1) * NQ * HD], &kv[r].0, &kv[r].1, slen, window);
            let got_occ = &occ[r * NQ * HD..(r + 1) * NQ * HD];
            let d_ref = max_diff(got_occ, &want);
            let d_online = max_diff(&online_out[r], &want);
            let d_cross = got_occ
                .iter()
                .zip(online_out[r].iter())
                .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
                .fold(0.0f32, f32::max);
            let ok = d_ref < 3e-3 && d_cross < 3e-3 && d_online < 3e-3;
            failed |= !ok;
            println!(
                "window={window:4} row={r} slen={slen:3} split-vs-f64={d_ref:.6} online-vs-f64={d_online:.6} split-vs-online={d_cross:.6} {}",
                if ok { "OK" } else { "FAIL" }
            );
        }
    }

    if failed {
        return Err(anyhow!("split-K attention check failed"));
    }
    Ok(())
}
