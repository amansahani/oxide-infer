//! Microbenchmark for the batch-invariant cuBLASLt path.
//!
//! Times one attention layer's and one conv layer's GEMM stack at several
//! batch sizes, so prefill/decode cost can be attributed to GEMM vs attention.

use anyhow::Result;
use cuda_core::Stream;
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext};
use oxide_infer_server::lt_gemm::LtGemm;
use std::time::Instant;

const HIDDEN: usize = 2048;
const INTER: usize = 7680;
const KV_DIM: usize = 512;
const GATE: usize = 16;

struct Shape {
    name: &'static str,
    m: usize,
    k: usize,
    reps: usize,
}

fn bench(
    lt: &mut LtGemm,
    stream: &Stream,
    weight: &DeviceBuffer<f16>,
    act: &DeviceBuffer<f16>,
    out: &mut DeviceBuffer<f16>,
    s: &Shape,
    n: usize,
) -> Result<f64> {
    for _ in 0..3 {
        lt.gemm(stream, s.m, s.k, n, weight.inner().cu_deviceptr(), act.inner().cu_deviceptr(), out.inner_mut().cu_deviceptr())?;
    }
    stream.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..s.reps {
        lt.gemm(stream, s.m, s.k, n, weight.inner().cu_deviceptr(), act.inner().cu_deviceptr(), out.inner_mut().cu_deviceptr())?;
    }
    stream.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() / s.reps as f64)
}

fn candidates(gpu: &GpuContext) -> Result<()> {
    let lt = LtGemm::new(gpu, 1024)?;
    let ns = [1usize, 8, 32, 128, 512];
    println!("algorithm candidates per weight shape (median ms, 30 reps), col0 diff = invariance");
    for &(name, m, k) in &[
        ("q   2048x2048", 2048usize, 2048usize),
        ("k    512x2048", 512, 2048),
        ("v    512x2048", 512, 2048),
        ("gate  16x2048", 16, 2048),
        ("o   2048x2048", 2048, 2048),
        ("ffn 7680x2048", 7680, 2048),
        ("dn  2048x7680", 2048, 7680),
        ("qkv 3088x2048", 3088, 2048),
        ("ffn 15360x2048", 15360, 2048),
        ("conv 6144x2048", 6144, 2048),
        ("lm 128000x2048", 128000, 2048),
    ] {
        let cands = lt.bench_candidates(gpu, m, k, &ns, 8, 30)?;
        println!("{name}:");
        for c in &cands {
            let t: Vec<String> = c.times_ms.iter().map(|x| format!("{x:7.3}")).collect();
            println!(
                "  idx {}{} waves {:5.1} ws {:7} col0diff {:.1e} | n=1 {} n=8 {} n=32 {} n=128 {} n=512 {}",
                c.index,
                if c.index == 0 { " *" } else { "  " },
                c.waves,
                c.workspace,
                c.col0_diff,
                t[0], t[1], t[2], t[3], t[4]
            );
        }
    }
    Ok(())
}

/// Does one concatenated weight matrix beat the separate projections?
///
/// The decode step issues four small GEMMs per attention layer (q, k, v,
/// gate) and two for the ffn gate/up pair. Each small call pays its own
/// launch and ramp, so the same bytes in one wider GEMM could be cheaper.
/// This measures exactly that, interleaved in one process with event timing.
///
/// Output layout of a column-major GEMM is [m][n], so the concatenated
/// result is each projection's rows contiguous per column, which is what the
/// consumers would need to index with a row stride.
fn fuse(gpu: &GpuContext) -> Result<()> {
    let stream = gpu.compute_stream().clone();
    let mut lt = LtGemm::new(gpu, 1024)?;
    let ev0 = gpu.context().new_event()?;
    let ev1 = gpu.context().new_event()?;
    let zeros = |n: usize| vec![f16::from_f32(0.01); n];

    let max_n = 512usize;
    let x = DeviceBuffer::<f16>::from_slice(gpu, &zeros(HIDDEN * max_n))?;

    let qw = DeviceBuffer::<f16>::from_slice(gpu, &zeros(2048 * 2048))?;
    let kw = DeviceBuffer::<f16>::from_slice(gpu, &zeros(512 * 2048))?;
    let vw = DeviceBuffer::<f16>::from_slice(gpu, &zeros(512 * 2048))?;
    let gw = DeviceBuffer::<f16>::from_slice(gpu, &zeros(16 * 2048))?;
    let qkvw = DeviceBuffer::<f16>::from_slice(gpu, &zeros(3088 * 2048))?;
    let mut qo = DeviceBuffer::<f16>::zeroed(gpu, 2048 * max_n)?;
    let mut ko = DeviceBuffer::<f16>::zeroed(gpu, 512 * max_n)?;
    let mut vo = DeviceBuffer::<f16>::zeroed(gpu, 512 * max_n)?;
    let mut go = DeviceBuffer::<f16>::zeroed(gpu, 16 * max_n)?;
    let mut qkvo = DeviceBuffer::<f16>::zeroed(gpu, 3088 * max_n)?;

    let upw = DeviceBuffer::<f16>::from_slice(gpu, &zeros(INTER * HIDDEN))?;
    let dnw = DeviceBuffer::<f16>::from_slice(gpu, &zeros(INTER * HIDDEN))?;
    let fuw = DeviceBuffer::<f16>::from_slice(gpu, &zeros(2 * INTER * HIDDEN))?;
    let mut upo = DeviceBuffer::<f16>::zeroed(gpu, INTER * max_n)?;
    let mut dno = DeviceBuffer::<f16>::zeroed(gpu, INTER * max_n)?;
    let mut fuo = DeviceBuffer::<f16>::zeroed(gpu, 2 * INTER * max_n)?;

    for &n in &[1usize, 8, 32, 512] {
        let mut sep = |lt: &mut LtGemm| -> Result<()> {
            lt.gemm(&stream, HIDDEN, HIDDEN, n, qw.inner().cu_deviceptr(), x.inner().cu_deviceptr(), qo.inner_mut().cu_deviceptr())?;
            lt.gemm(&stream, KV_DIM, HIDDEN, n, kw.inner().cu_deviceptr(), x.inner().cu_deviceptr(), ko.inner_mut().cu_deviceptr())?;
            lt.gemm(&stream, KV_DIM, HIDDEN, n, vw.inner().cu_deviceptr(), x.inner().cu_deviceptr(), vo.inner_mut().cu_deviceptr())?;
            lt.gemm(&stream, GATE, HIDDEN, n, gw.inner().cu_deviceptr(), x.inner().cu_deviceptr(), go.inner_mut().cu_deviceptr())?;
            Ok(())
        };
        let mut one = |lt: &mut LtGemm| -> Result<()> {
            lt.gemm(&stream, 3088, HIDDEN, n, qkvw.inner().cu_deviceptr(), x.inner().cu_deviceptr(), qkvo.inner_mut().cu_deviceptr())?;
            Ok(())
        };
        let mut ffn_sep = |lt: &mut LtGemm| -> Result<()> {
            lt.gemm(&stream, INTER, HIDDEN, n, upw.inner().cu_deviceptr(), x.inner().cu_deviceptr(), upo.inner_mut().cu_deviceptr())?;
            lt.gemm(&stream, INTER, HIDDEN, n, dnw.inner().cu_deviceptr(), x.inner().cu_deviceptr(), dno.inner_mut().cu_deviceptr())?;
            Ok(())
        };
        let mut ffn_one = |lt: &mut LtGemm| -> Result<()> {
            lt.gemm(&stream, 2 * INTER, HIDDEN, n, fuw.inner().cu_deviceptr(), x.inner().cu_deviceptr(), fuo.inner_mut().cu_deviceptr())?;
            Ok(())
        };
        for _ in 0..3 {
            sep(&mut lt)?;
            one(&mut lt)?;
            ffn_sep(&mut lt)?;
            ffn_one(&mut lt)?;
        }
        stream.synchronize()?;
        let mut fns: [&mut dyn FnMut(&mut LtGemm) -> Result<()>; 4] =
            [&mut sep, &mut one, &mut ffn_sep, &mut ffn_one];
        // Per-call span: an event pair around one call, so it includes any
        // host-side gap before the kernel starts.
        let mut t = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for _ in 0..40 {
            for (i, f) in fns.iter_mut().enumerate() {
                ev0.record(&stream)?;
                f(&mut lt)?;
                ev1.record(&stream)?;
                ev1.synchronize()?;
                t[i].push(ev0.elapsed_time(&ev1)? as f64);
            }
        }
        // Amortised device cost: ten calls per event pair, so the host cost is
        // divided by ten. If this is much smaller, the per-call span is
        // host-launch-bound rather than bandwidth-bound.
        const BATCH: usize = 10;
        let mut b = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for _ in 0..10 {
            for (i, f) in fns.iter_mut().enumerate() {
                ev0.record(&stream)?;
                for _ in 0..BATCH {
                    f(&mut lt)?;
                }
                ev1.record(&stream)?;
                ev1.synchronize()?;
                b[i].push(ev0.elapsed_time(&ev1)? as f64 / BATCH as f64);
            }
        }
        for v in t.iter_mut().chain(b.iter_mut()) {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        let med = |v: &Vec<f64>| v[v.len() / 2];
        println!(
            "n={n:4}  qkv 4 calls {:6.3} (dev {:6.3}) | fused {:6.3} (dev {:6.3}) | delta {:+.3}   ffn 2 calls {:6.3} (dev {:6.3}) | fused {:6.3} (dev {:6.3}) | delta {:+.3}",
            med(&t[0]), med(&b[0]), med(&t[1]), med(&b[1]), med(&t[1]) - med(&t[0]),
            med(&t[2]), med(&b[2]), med(&t[3]), med(&b[3]), med(&t[3]) - med(&t[2])
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let gpu = GpuContext::init(0)?;
    if std::env::args().any(|a| a == "--candidates") {
        return candidates(&gpu);
    }
    if std::env::args().any(|a| a == "--fuse") {
        return fuse(&gpu);
    }
    let stream = gpu.compute_stream().clone();
    let mut lt = LtGemm::new(&gpu, 1024)?;

    let attn = [
        Shape { name: "q   2048x2048", m: HIDDEN, k: HIDDEN, reps: 20 },
        Shape { name: "k    512x2048", m: KV_DIM, k: HIDDEN, reps: 20 },
        Shape { name: "v    512x2048", m: KV_DIM, k: HIDDEN, reps: 20 },
        Shape { name: "gate  16x2048", m: GATE, k: HIDDEN, reps: 20 },
        Shape { name: "o   2048x2048", m: HIDDEN, k: HIDDEN, reps: 20 },
        Shape { name: "ffn 7680x2048", m: INTER, k: HIDDEN, reps: 20 },
        Shape { name: "ffn 7680x2048", m: INTER, k: HIDDEN, reps: 20 },
        Shape { name: "dn  2048x7680", m: HIDDEN, k: INTER, reps: 20 },
    ];
    let conv = [
        Shape { name: "in  6144x2048", m: 3 * HIDDEN, k: HIDDEN, reps: 20 },
        Shape { name: "o   2048x2048", m: HIDDEN, k: HIDDEN, reps: 20 },
        Shape { name: "ffn 7680x2048", m: INTER, k: HIDDEN, reps: 20 },
        Shape { name: "ffn 7680x2048", m: INTER, k: HIDDEN, reps: 20 },
        Shape { name: "dn  2048x7680", m: HIDDEN, k: INTER, reps: 20 },
    ];

    let max_k = 7680;
    let act = DeviceBuffer::<f16>::from_slice(&gpu, &vec![f16::from_f32(0.01); 1024 * max_k])?;

    for (label, shapes) in [("attention layer", &attn[..]), ("conv layer", &conv[..])] {
        for n in [1usize, 8, 32, 128, 512] {
            let mut total = 0.0f64;
            let mut flops = 0.0f64;
            let mut detail = String::new();
            for s in shapes {
                let weight = DeviceBuffer::<f16>::from_slice(&gpu, &vec![f16::from_f32(0.01); s.m * s.k])?;
                let mut out = DeviceBuffer::<f16>::zeroed(&gpu, s.m * n)?;
                let t = bench(&mut lt, &stream, &weight, &act, &mut out, s, n)?;
                total += t;
                flops += 2.0 * s.m as f64 * s.k as f64 * n as f64;
                detail.push_str(&format!("  {} {:.3}ms", s.name, t * 1e3));
            }
            println!(
                "{} n={:4}  total {:7.3} ms  {:6.2} TFLOP/s{}",
                label, n, total * 1e3,
                flops / total / 1e12,
                if n == 512 { &detail } else { "" }
            );
        }
    }

    // Attention-shape GEMMs used by the prefill flash path: Q x K^T and P x V.
    // Both are tall-skinny (K = 128 or N = 128), which is the regime where
    // tensor-core efficiency is least certain.
    let attn_shapes = [
        Shape { name: "QK  keys x 128 ", m: 2000, k: 128, reps: 50 },
        Shape { name: "PV  128 x keys ", m: 128, k: 2000, reps: 50 },
    ];
    for n in [512usize, 2048] {
        let mut total = 0.0f64;
        let mut flops = 0.0f64;
        let mut detail = String::new();
        for s in &attn_shapes {
            let weight = DeviceBuffer::<f16>::from_slice(&gpu, &vec![f16::from_f32(0.01); s.m * s.k])?;
            let mut out = DeviceBuffer::<f16>::zeroed(&gpu, s.m * n)?;
            let t = bench(&mut lt, &stream, &weight, &act, &mut out, s, n)?;
            total += t;
            flops += 2.0 * s.m as f64 * s.k as f64 * n as f64;
            detail.push_str(&format!("  {} {:.3}ms", s.name, t * 1e3));
        }
        println!(
            "attn gemm        n={:4}  total {:7.3} ms  {:6.2} TFLOP/s{}",
            n, total * 1e3, flops / total / 1e12, detail
        );
    }
    Ok(())
}
