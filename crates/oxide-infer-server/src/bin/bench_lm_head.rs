//! Time the production tied LM-head kernel (128k vocab) at a single token.
//!
//! The kernel runs once per generated token and reads the whole embedding
//! table, so it is a pure bandwidth consumer. Timing is a CUDA-event pair on
//! the compute stream, which measures the kernel itself: wall-clock timing
//! around the launch reported 2.5-2.7 ms on this card for a kernel that takes
//! 1.55 ms, because the host-side launch and sync dominate a single 500 MiB
//! read.

use anyhow::Result;
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext};
use oxide_infer_models::TiedLmHead;

fn main() -> Result<()> {
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream().clone();
    let vocab = 128_000usize;
    let hidden = 2048usize;
    let lm = TiedLmHead::new(&gpu, vocab, hidden)?;
    let table = DeviceBuffer::<f16>::zeroed(&gpu, vocab * hidden)?;
    let x = DeviceBuffer::<f16>::zeroed(&gpu, hidden)?;
    let mut logits = DeviceBuffer::<f32>::zeroed(&gpu, vocab)?;
    let ev0 = gpu.context().new_event()?;
    let ev1 = gpu.context().new_event()?;

    for _ in 0..5 {
        lm.compute_logits(&gpu, &x, &table, &mut logits)?;
    }
    stream.synchronize()?;

    let reps = 60;
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        ev0.record(&stream)?;
        lm.compute_logits(&gpu, &x, &table, &mut logits)?;
        ev1.record(&stream)?;
        ev1.synchronize()?;
        times.push(ev0.elapsed_time(&ev1)? as f64);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let t = times[times.len() / 2];
    let iqr = times[(3 * times.len()) / 4] - times[times.len() / 4];
    let bytes = (vocab * hidden * 2) as f64;
    println!(
        "lm_head: {t:.3} ms (IQR {iqr:.3})  {:.1} GB/s  reads {:.0} MiB/token",
        bytes / (t * 1e-3) / 1e9,
        bytes / (1024.0 * 1024.0) as f64
    );
    Ok(())
}
