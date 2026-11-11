//! Gate for the GEMM prefill attention path.
//!
//! Mirrors the engine's per-chunk launch sequence (pack Q, QK GEMM, fused
//! mask+softmax, PV GEMM, gate+scatter) on two synthetic chunks that differ
//! in row count and start position, so the shared key range, the earliest
//! allowed key (`t_lo`) and the per-row window are all exercised. Each chunk
//! is compared against the single-pass online kernel (with the gate applied on
//! the host) and spot-checked against an f64 reference. Run for windows 0 and
//! 256, the two window values the model uses.

use anyhow::{anyhow, Result};
use cuda_core::{launch_kernel_on_stream, Stream};
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext};
use oxide_infer_kernels::batched_attn::{
    BATCHED_ATTN_GATE_SCATTER_PTX, BATCHED_ATTN_PACK_Q_PTX, BATCHED_ATTN_SOFTMAX_MASK_PTX,
    BATCHED_GQA_ATTENTION_ONLINE_PTX,
};
use oxide_infer_kernels::PtxModule;
use oxide_infer_server::lt_gemm::LtGemm;
use std::ffi::c_void;

const NQ: usize = 16;
const NKV: usize = 4;
const HD: usize = 128;
const KV_DIM: usize = NKV * HD;
const HIDDEN: usize = NQ * HD;
const SCALE: f64 = 0.0883883476;
const ATTN_SCALE: f32 = 0.0883883476;
const CACHE_LEN: usize = 600;
const ALGO_N: u64 = 1024;

/// One prefill chunk: `rows` tokens starting at absolute position `p0`.
struct Chunk {
    start: usize, // wave row where the chunk starts
    rows: usize,
    p0: u32,
}

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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// f64 reference for the rows of one chunk: `out[row][head][dim]`, already
/// gated, so both kernels are compared on the same final numbers.
#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    c: &Chunk,
    total_rows: usize,
    window: u32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; c.rows * NQ * HD];
    for r in 0..c.rows {
        let row = c.start + r;
        let slen = (c.p0 as usize + r + 1) as i64;
        for h in 0..NQ {
            let kvh = h / 4;
            let mut scores = vec![f64::NEG_INFINITY; slen as usize];
            for t in 0..slen as usize {
                if window > 0 && slen - 1 - t as i64 >= window as i64 {
                    continue;
                }
                let mut dot = 0.0f64;
                for d in 0..HD {
                    dot += (q[(row * NQ + h) * HD + d] as f64)
                        * (k[(t * NKV + kvh) * HD + d] as f64);
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
                for t in 0..slen as usize {
                    acc += w[t] * (v[(t * NKV + kvh) * HD + d] as f64);
                }
                let g = sigmoid(gate[row * NQ + h]);
                out[(r * NQ + h) * HD + d] = (acc / den * g as f64) as f32;
            }
        }
    }
    let _ = total_rows;
    out
}

fn main() -> Result<()> {
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream().clone();
    let mut lt = LtGemm::new(&gpu, 1024)?;

    let pack_mod = PtxModule::load_from_ptx(&gpu, BATCHED_ATTN_PACK_Q_PTX)
        .map_err(|e| anyhow!("pack PTX: {e}"))?;
    let pack_q = pack_mod.get_function("batched_attn_pack_q_fp16")?;
    let sm_mod = PtxModule::load_from_ptx(&gpu, BATCHED_ATTN_SOFTMAX_MASK_PTX)
        .map_err(|e| anyhow!("softmax PTX: {e}"))?;
    let softmax = sm_mod.get_function("batched_attn_softmax_mask_fp16")?;
    let sc_mod = PtxModule::load_from_ptx(&gpu, BATCHED_ATTN_GATE_SCATTER_PTX)
        .map_err(|e| anyhow!("scatter PTX: {e}"))?;
    let scatter = sc_mod.get_function("batched_attn_gate_scatter_fp16")?;
    let on_mod = PtxModule::load_from_ptx(&gpu, BATCHED_GQA_ATTENTION_ONLINE_PTX)
        .map_err(|e| anyhow!("online PTX: {e}"))?;
    let online = on_mod.get_function("batched_gqa_attention_online_fp16")?;

    // Two chunks with different lengths and start positions.
    let chunks = [
        Chunk { start: 0, rows: 96, p0: 0 },
        Chunk { start: 96, rows: 128, p0: 400 },
    ];
    let total_rows: usize = chunks.iter().map(|c| c.rows).sum();

    let qh: Vec<f32> = (0..total_rows * NQ * HD).map(|i| pat(1, i)).collect();
    let d_q = DeviceBuffer::from_slice(&gpu, &f16s(&qh))?;
    let gateh: Vec<f32> = (0..total_rows * NQ)
        .map(|i| pat(7, i) * 2.0)
        .collect();
    let d_gate = DeviceBuffer::from_slice(&gpu, &f16s(&gateh))?;

    // One KV cache per chunk; rows of a chunk share its base (as in the engine).
    let mut kv: Vec<(Vec<f32>, Vec<f32>, DeviceBuffer<f16>, DeviceBuffer<f16>)> = Vec::new();
    for (i, _c) in chunks.iter().enumerate() {
        let k: Vec<f32> = (0..CACHE_LEN * KV_DIM)
            .map(|j| pat(20 + i as u64, j) * 1.5)
            .collect();
        let v: Vec<f32> = (0..CACHE_LEN * KV_DIM)
            .map(|j| pat(40 + i as u64, j))
            .collect();
        let dk = DeviceBuffer::from_slice(&gpu, &f16s(&k))?;
        let dv = DeviceBuffer::from_slice(&gpu, &f16s(&v))?;
        kv.push((k, v, dk, dv));
    }

    let mut k_ptrs = Vec::new();
    let mut v_ptrs = Vec::new();
    let mut seq_lens = Vec::new();
    for (ci, c) in chunks.iter().enumerate() {
        for r in 0..c.rows {
            k_ptrs.push(kv[ci].2.inner().cu_deviceptr());
            v_ptrs.push(kv[ci].3.inner().cu_deviceptr());
            seq_lens.push(c.p0 + r as u32 + 1);
        }
    }
    let d_kp = DeviceBuffer::<u64>::from_slice(&gpu, &k_ptrs)?;
    let d_vp = DeviceBuffer::<u64>::from_slice(&gpu, &v_ptrs)?;
    let d_sl = DeviceBuffer::<u32>::from_slice(&gpu, &seq_lens)?;

    let max_keys = CACHE_LEN;
    let mut d_qpack = DeviceBuffer::<f16>::zeroed(&gpu, total_rows * NQ * HD)?;
    let mut d_tmp = DeviceBuffer::<f16>::zeroed(&gpu, total_rows * NQ * HD)?;
    let mut d_scores = DeviceBuffer::<f32>::zeroed(&gpu, 4 * total_rows * max_keys)?;
    let mut d_p = DeviceBuffer::<f16>::zeroed(&gpu, 4 * total_rows * max_keys)?;
    let mut d_sums = DeviceBuffer::<f32>::zeroed(&gpu, total_rows * NQ)?;

    // Online-kernel reference (gate applied on the host).
    let mut d_ctx_on = DeviceBuffer::<f16>::zeroed(&gpu, total_rows * HIDDEN)?;

    let mut failed = false;
    for window in [0u32, 256u32] {
        // ---- online kernel over the whole wave ----
        {
            let mut ctx = d_ctx_on.inner_mut().cu_deviceptr();
            let mut q = d_q.inner().cu_deviceptr();
            let mut kp = d_kp.inner().cu_deviceptr();
            let mut vp = d_vp.inner().cu_deviceptr();
            let mut sl = d_sl.inner().cu_deviceptr();
            let (mut nq, mut nkv, mut hd) = (NQ as u32, NKV as u32, HD as u32);
            let mut b = total_rows as u32;
            let mut w = window;
            let mut args: [*mut c_void; 10] = [
                &mut ctx as *mut _ as *mut c_void,
                &mut q as *mut _ as *mut c_void,
                &mut kp as *mut _ as *mut c_void,
                &mut vp as *mut _ as *mut c_void,
                &mut sl as *mut _ as *mut c_void,
                &mut nq as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut b as *mut _ as *mut c_void,
                &mut w as *mut _ as *mut c_void,
            ];
            unsafe {
                launch_kernel_on_stream(&online, (NKV as u32, b, 1), (HD as u32, 1, 1), 0, &stream, &mut args)?;
            }
            stream.synchronize()?;
        }
        let online_ctx = d_ctx_on.inner().to_host_vec(&stream)?;
        let mut online_gated = vec![0f32; total_rows * NQ * HD];
        for r in 0..total_rows {
            for h in 0..NQ {
                let g = sigmoid(gateh[r * NQ + h]);
                for d in 0..HD {
                    let i = (r * NQ + h) * HD + d;
                    online_gated[i] = online_ctx[i].to_f32() * g;
                }
            }
        }

        // ---- GEMM path, one chunk at a time (engine launch order) ----
        let mut d_ctx_gemm = DeviceBuffer::<f16>::zeroed(&gpu, total_rows * HIDDEN)?;
        for (ci, c) in chunks.iter().enumerate() {
            let t_lo = if window > 0 {
                (c.p0 + 1).saturating_sub(window)
            } else {
                0
            };
            let slen_max = c.p0 + c.rows as u32;
            let keys = (slen_max - t_lo) as usize;
            let n = 4 * c.rows;
            let mut keys_u = keys as u32;
            let mut rows_u = c.rows as u32;
            let mut hd = HD as u32;
            let mut nq = NQ as u32;
            let mut t_lo_u = t_lo;
            let mut window_u = window;

            // pack Q
            let mut q_in = d_q.inner().cu_deviceptr() + (c.start * HIDDEN * 2) as u64;
            let mut q_out = d_qpack.inner_mut().cu_deviceptr();
            let mut args: [*mut c_void; 5] = [
                &mut q_in as *mut _ as *mut c_void,
                &mut q_out as *mut _ as *mut c_void,
                &mut nq as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut rows_u as *mut _ as *mut c_void,
            ];
            unsafe {
                launch_kernel_on_stream(&pack_q, (rows_u, 1, 1), (256, 1, 1), 0, &stream, &mut args)?;
            }

            let mut scores = d_scores.inner_mut().cu_deviceptr();
            let mut p = d_p.inner_mut().cu_deviceptr();
            let mut tmp = d_tmp.inner_mut().cu_deviceptr();
            let sums_base = d_sums.inner_mut().cu_deviceptr();
            let mut slen_ptr = d_sl.inner().cu_deviceptr() + (c.start * 4) as u64;

            for kvh in 0..NKV {
                let a = kv[ci].2.inner().cu_deviceptr()
                    + (kvh * HD * 2) as u64
                    + (t_lo as u64 * KV_DIM as u64 * 2);
                let b = q_out + (kvh * n * HD * 2) as u64;
                lt.gemm_strided(
                    &stream, keys, HD, n, KV_DIM as i64, HD as i64, keys as i64, true, true,
                    ALGO_N, a, b, scores,
                )?;

                let mut sums_k = sums_base + (kvh * 4 * c.rows * 4) as u64;
                let mut sc = ATTN_SCALE;
                let mut args: [*mut c_void; 9] = [
                    &mut scores as *mut _ as *mut c_void,
                    &mut p as *mut _ as *mut c_void,
                    &mut sums_k as *mut _ as *mut c_void,
                    &mut slen_ptr as *mut _ as *mut c_void,
                    &mut rows_u as *mut _ as *mut c_void,
                    &mut keys_u as *mut _ as *mut c_void,
                    &mut t_lo_u as *mut _ as *mut c_void,
                    &mut window_u as *mut _ as *mut c_void,
                    &mut sc as *mut _ as *mut c_void,
                ];
                unsafe {
                    launch_kernel_on_stream(
                        &softmax,
                        (rows_u, 4, 1),
                        (128, 1, 1),
                        0,
                        &stream,
                        &mut args,
                    )?;
                }

                let a = kv[ci].3.inner().cu_deviceptr()
                    + (kvh * HD * 2) as u64
                    + (t_lo as u64 * KV_DIM as u64 * 2);
                let dst = tmp + (kvh * n * HD * 2) as u64;
                lt.gemm_strided(
                    &stream, HD, keys, n, KV_DIM as i64, keys as i64, HD as i64, false, false,
                    ALGO_N, a, p, dst,
                )?;
            }

            let mut ctx_ptr = d_ctx_gemm.inner_mut().cu_deviceptr() + (c.start * HIDDEN * 2) as u64;
            let mut gate_ptr = d_gate.inner().cu_deviceptr() + (c.start * NQ * 2) as u64;
            let mut tmp_ptr = tmp;
            let mut sums_ptr = sums_base;
            let mut args: [*mut c_void; 7] = [
                &mut ctx_ptr as *mut _ as *mut c_void,
                &mut tmp_ptr as *mut _ as *mut c_void,
                &mut gate_ptr as *mut _ as *mut c_void,
                &mut sums_ptr as *mut _ as *mut c_void,
                &mut rows_u as *mut _ as *mut c_void,
                &mut nq as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
            ];
            unsafe {
                launch_kernel_on_stream(&scatter, (nq, rows_u, 1), (hd, 1, 1), 0, &stream, &mut args)?;
            }
        }
        stream.synchronize()?;
        let gemm_ctx = d_ctx_gemm.inner().to_host_vec(&stream)?;

        // ---- compare ----
        let mut max_online = 0f32;
        let mut worst = (0usize, 0usize);
        for r in 0..total_rows {
            for h in 0..NQ {
                for d in 0..HD {
                    let i = (r * NQ + h) * HD + d;
                    let e = (gemm_ctx[i].to_f32() - online_gated[i]).abs();
                    if e > max_online {
                        max_online = e;
                        worst = (r, h);
                    }
                }
            }
        }

        // f64 reference on the first and last row of each chunk, compared
        // against both kernels so a mismatch is attributed to one side.
        let mut max_ref = 0f32;
        for (ci, c) in chunks.iter().enumerate() {
            for r in [0usize, c.rows - 1] {
                let row = c.start + r;
                let mut q1 = vec![0f32; NQ * HD];
                q1.copy_from_slice(&qh[row * NQ * HD..(row + 1) * NQ * HD]);
                let want = reference(
                    &q1,
                    &kv[ci].0,
                    &kv[ci].1,
                    &gateh[row * NQ..(row + 1) * NQ],
                    &Chunk { start: 0, rows: 1, p0: c.p0 + r as u32 },
                    1,
                    window,
                );
                let mut e_gemm = 0f32;
                let mut e_on = 0f32;
                for h in 0..NQ {
                    for d in 0..HD {
                        let got = gemm_ctx[(row * NQ + h) * HD + d].to_f32();
                        let on = online_gated[(row * NQ + h) * HD + d];
                        e_gemm = e_gemm.max((got - want[h * HD + d]).abs());
                        e_on = e_on.max((on - want[h * HD + d]).abs());
                    }
                }
                println!(
                    "  chunk {ci} row {r:3} (abs {row:3}): gemm-vs-f64 {e_gemm:.5}  online-vs-f64 {e_on:.5}"
                );
                max_ref = max_ref.max(e_gemm);
                let _ = e_on;
            }
        }

        println!(
            "window={window:3}: GEMM vs online(max) {max_online:.5} at row {} head {}  vs f64(max) {max_ref:.5}",
            worst.0, worst.1
        );
        if max_online > 8e-3 || max_ref > 2e-2 {
            failed = true;
        }
    }
    // Stage diagnostics (`OXIDE_DIAG=1`): run chunk 0 (window 0) one stage at
    // a time and compare scores, f16 weights, the per-column sum and the PV
    // output against host references, so a mismatch names the broken stage.
    if std::env::var("OXIDE_DIAG").is_ok() {
        let c = &chunks[0];
        let (rows, n, keys) = (c.rows, 4 * c.rows, c.rows);
        let (mut keys_u, mut rows_u, mut hd, mut nq) =
            (keys as u32, rows as u32, HD as u32, NQ as u32);
        let mut q_in = d_q.inner().cu_deviceptr();
        let mut q_out = d_qpack.inner_mut().cu_deviceptr();
        let mut args: [*mut c_void; 5] = [
            &mut q_in as *mut _ as *mut c_void,
            &mut q_out as *mut _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut rows_u as *mut _ as *mut c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&pack_q, (rows_u, 1, 1), (256, 1, 1), 0, &stream, &mut args)?;
        }
        stream.synchronize()?;
        {
            let qp = d_qpack.inner().to_host_vec(&stream)?;
            let mut e = 0f32;
            let mut worst = (0usize, 0usize);
            for head in 0..NQ {
                for row in 0..rows {
                    for d in 0..HD {
                        let got = qp[(head * rows + row) * HD + d].to_f32();
                        let want = f16::from_f32(qh[(row * NQ + head) * HD + d]).to_f32();
                        if (got - want).abs() > e {
                            e = (got - want).abs();
                            worst = (head, row);
                        }
                    }
                }
            }
            println!("  pack_q vs f16(q): max err {e:.3e} at head {} row {}", worst.0, worst.1);
        }
        let q16: Vec<f32> = qh[..rows * NQ * HD]
            .iter()
            .map(|v| f16::from_f32(*v).to_f32())
            .collect();
        let k16: Vec<f32> = kv[0].0.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
        let v16: Vec<f32> = kv[0].1.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
        let mut kv_err = vec![0f32; NKV];
        let mut pv_err = vec![0f32; NKV];
        for (kvh, (ke, pv_e)) in kv_err.iter_mut().zip(pv_err.iter_mut()).enumerate() {
            let a = kv[0].2.inner().cu_deviceptr() + (kvh * HD * 2) as u64;
            let b = q_out + (kvh * n * HD * 2) as u64;
            lt.gemm_strided(
                &stream, keys, HD, n, KV_DIM as i64, HD as i64, keys as i64, true, true, ALGO_N,
                a, b, d_scores.inner_mut().cu_deviceptr(),
            )?;
            let mut scores = d_scores.inner_mut().cu_deviceptr();
            let mut p = d_p.inner_mut().cu_deviceptr();
            let mut sums_k = d_sums.inner_mut().cu_deviceptr() + (kvh * 4 * rows * 4) as u64;
            let mut slen_ptr = d_sl.inner().cu_deviceptr();
            let mut t_lo_u = 0u32;
            let mut window_u = 0u32;
            let mut sc = ATTN_SCALE;
            let mut args: [*mut c_void; 9] = [
                &mut scores as *mut _ as *mut c_void,
                &mut p as *mut _ as *mut c_void,
                &mut sums_k as *mut _ as *mut c_void,
                &mut slen_ptr as *mut _ as *mut c_void,
                &mut rows_u as *mut _ as *mut c_void,
                &mut keys_u as *mut _ as *mut c_void,
                &mut t_lo_u as *mut _ as *mut c_void,
                &mut window_u as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
            ];
            unsafe {
                launch_kernel_on_stream(
                    &softmax, (rows_u, 4, 1), (128, 1, 1), 0, &stream, &mut args,
                )?;
            }
            let va = kv[0].3.inner().cu_deviceptr() + (kvh * HD * 2) as u64;
            lt.gemm_strided(
                &stream, HD, keys, n, KV_DIM as i64, keys as i64, HD as i64, false, false, ALGO_N,
                va, d_p.inner().cu_deviceptr(),
                d_tmp.inner_mut().cu_deviceptr() + (kvh * n * HD * 2) as u64,
            )?;
            stream.synchronize()?;

            let sh = d_scores.inner().to_host_vec(&stream)?;
            let ph = d_p.inner().to_host_vec(&stream)?;
            let th = d_tmp.inner().to_host_vec(&stream)?;
            let sums = d_sums.inner().to_host_vec(&stream)?;
            for &hl in &[0usize, 3] {
                for &row in &[1usize, 32, rows - 1] {
                    let col = hl * rows + row;
                    let gh = kvh * 4 + hl;
                    let qbase = (row * NQ + gh) * HD;
                    let mut raw = vec![0f32; keys];
                    for (t, hv) in raw.iter_mut().enumerate() {
                        let mut acc = 0f64;
                        for d in 0..HD {
                            acc += q16[qbase + d] as f64 * k16[t * KV_DIM + kvh * HD + d] as f64;
                        }
                        *hv = acc as f32;
                    }
                    let mut e_sc = 0f32;
                    for t in 0..keys {
                        // Scores are the raw dots; the kernel scales in softmax.
                        e_sc = e_sc.max((sh[col * keys + t] - raw[t]).abs());
                    }
                    let max = raw[..=row]
                        .iter()
                        .map(|v| v * ATTN_SCALE)
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut e_p = 0f32;
                    let mut den = 0f32;
                    for (t, rv) in raw.iter().enumerate() {
                        let want = if t <= row { (rv * ATTN_SCALE - max).exp() } else { 0.0 };
                        den += want;
                        e_p = e_p.max((ph[col * keys + t].to_f32() - want).abs());
                    }
                    let e_sum = (sums[kvh * 4 * rows + col] - den).abs() / den.max(1e-9);
                    let mut e_pv = 0f32;
                    for d in 0..HD {
                        let mut acc = 0f64;
                        for t in 0..keys {
                            acc += ph[col * keys + t].to_f32() as f64
                                * v16[t * KV_DIM + kvh * HD + d] as f64;
                        }
                        let got = th[kvh * n * HD + col * HD + d].to_f32();
                        e_pv = e_pv.max((got - acc as f32).abs());
                    }
                    *ke = ke.max(e_sc);
                    *pv_e = pv_e.max(e_pv);
                    if hl == 3 && row == rows - 1 {
                        println!(
                            "  stage kvh {kvh}: scores {e_sc:.2e}  p {e_p:.2e}  sum {e_sum:.2e}  pv {e_pv:.2e} (last column; maxima below)"
                        );
                    }
                }
            }
        }
        for kvh in 0..NKV {
            println!(
                "  stage max kvh {kvh}: scores {:.2e}  pv {:.2e}",
                kv_err[kvh], pv_err[kvh]
            );
        }
    }

    if failed {
        return Err(anyhow!("GEMM attention path does not match the reference"));
    }
    println!("ok");
    Ok(())
}
