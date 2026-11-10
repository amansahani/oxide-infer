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

fn main() -> Result<()> {
    let gpu = GpuContext::init(0)?;
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
    Ok(())
}
