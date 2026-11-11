//! Validate and time the decode GEMV against a host reference.
//!
//! Correctness: every output element is checked against an f64 dot product of
//! the f16-rounded inputs, for a range of `(m, k)` shapes and column counts.
//! Timing: bytes streamed (weights + activations + output) divided by the
//! measured kernel time, so the kernel can be compared with the ~360 GB/s
//! DRAM ceiling of the target card and with the cuBLASLt path it replaces.
//!
//! Outcome: competitive at n=1, hopeless beyond the first tile (n>=17). The
//! engine does not use this kernel; see the status note in `gemv.rs`.

use anyhow::{anyhow, Result};
use cuda_core::{launch_kernel_on_stream, Function, Stream};
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext};
use oxide_infer_kernels::{PtxModule, GEMV_FP16_PTX};
use std::ffi::c_void;
use std::time::Instant;

fn pat(seed: u64, idx: usize) -> f32 {
    let mut x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(idx as u64);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    ((x & 0xffff) as f32 / 32767.5) - 1.0
}

fn f16s(v: &[f32]) -> Vec<f16> {
    v.iter().map(|x| f16::from_f32(*x)).collect()
}

#[allow(clippy::too_many_arguments)]
fn launch(
    func: &Function,
    stream: &Stream,
    m: usize,
    k: usize,
    n: usize,
    a: u64,
    x: u64,
    c: u64,
) -> Result<()> {
    let (mut a, mut x, mut c) = (a, x, c);
    let (mut mm, mut kk, mut nn) = (m as u32, k as u32, n as u32);
    let mut args: [*mut c_void; 6] = [
        &mut a as *mut _ as *mut c_void,
        &mut x as *mut _ as *mut c_void,
        &mut c as *mut _ as *mut c_void,
        &mut mm as *mut _ as *mut c_void,
        &mut kk as *mut _ as *mut c_void,
        &mut nn as *mut _ as *mut c_void,
    ];
    let grid = ((m + 3) / 4) as u32;
    unsafe {
        launch_kernel_on_stream(func, (grid, 1, 1), (128, 1, 1), 0, stream, &mut args)
            .map_err(|e| anyhow!("gemv launch: {e}"))?;
    }
    Ok(())
}

fn check(
    gpu: &GpuContext,
    func: &Function,
    stream: &Stream,
    m: usize,
    k: usize,
    n: usize,
) -> Result<f32> {
    let a: Vec<f32> = (0..m * k).map(|i| pat(1, i) * 0.05).collect();
    let x: Vec<f32> = (0..n * k).map(|i| pat(2, i) * 2.0).collect();
    let d_a = DeviceBuffer::from_slice(gpu, &f16s(&a))?;
    let d_x = DeviceBuffer::from_slice(gpu, &f16s(&x))?;
    let mut d_c = DeviceBuffer::<f16>::zeroed(gpu, m * n)?;
    launch(
        func,
        stream,
        m,
        k,
        n,
        d_a.inner().cu_deviceptr(),
        d_x.inner().cu_deviceptr(),
        d_c.inner_mut().cu_deviceptr(),
    )?;
    stream.synchronize()?;
    let got = d_c.inner().to_host_vec(stream)?;
    let a16 = f16s(&a);
    let x16 = f16s(&x);
    let mut worst = (0f32, 0f32);
    for j in 0..n {
        for row in 0..m {
            let mut acc = 0f64;
            for i in 0..k {
                acc += a16[row * k + i].to_f64() * x16[j * k + i].to_f64();
            }
            let e = (got[j * m + row].to_f32() - acc as f32).abs();
            worst = (worst.0.max(e), worst.1.max(acc.abs() as f32));
        }
    }
    Ok(worst.0 / worst.1.max(1e-9))
}

fn time_gemv(
    gpu: &GpuContext,
    func: &Function,
    stream: &Stream,
    m: usize,
    k: usize,
    n: usize,
    reps: usize,
) -> Result<f64> {
    let d_a = DeviceBuffer::<f16>::zeroed(gpu, m * k)?;
    let d_x = DeviceBuffer::<f16>::zeroed(gpu, n * k)?;
    let mut d_c = DeviceBuffer::<f16>::zeroed(gpu, m * n)?;
    let (pa, px, pc) = (
        d_a.inner().cu_deviceptr(),
        d_x.inner().cu_deviceptr(),
        d_c.inner_mut().cu_deviceptr(),
    );
    for _ in 0..3 {
        launch(func, stream, m, k, n, pa, px, pc)?;
    }
    stream.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..reps {
        launch(func, stream, m, k, n, pa, px, pc)?;
    }
    stream.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() / reps as f64)
}

fn main() -> Result<()> {
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream().clone();
    let ptx = PtxModule::load_from_ptx(&gpu, GEMV_FP16_PTX)?;
    let func = ptx.get_function("gemv_fp16")?;

    // Correctness, including tile edges (n = 15/16/17) and the scalar tail
    // (k = 2048 + 8 so the 8-element remainder path runs).
    let mut worst = 0f32;
    for &(m, k) in &[
        (16usize, 2048usize),
        (512, 2048),
        (2048, 2048),
        (7680, 2048),
        (2048, 7680),
        (2048, 2056),
    ] {
        for &n in &[1usize, 3, 8, 15, 16, 17, 32] {
            let e = check(&gpu, &func, &stream, m, k, n)?;
            worst = worst.max(e);
            println!("check m={m:5} k={k:5} n={n:2}: rel |c - f64| = {e:.3e}");
        }
    }
    if worst > 3e-3 {
        return Err(anyhow!("gemv relative error {worst} exceeds 3e-3"));
    }

    println!();
    for &(m, k) in &[
        (512usize, 2048usize),
        (2048, 2048),
        (6144, 2048),
        (7680, 2048),
        (128000, 2048),
        (2048, 7680),
    ] {
        for &n in &[1usize, 2, 8, 15, 16, 17, 20, 24, 32, 48, 64, 128] {
            let reps = 40;
            let t = time_gemv(&gpu, &func, &stream, m, k, n, reps)?;
            let bytes = (m * k + n * k + m * n) as f64 * 2.0;
            println!(
                "m={m:5} k={k:5} n={n:3}: {:7.3} ms  {:6.1} GB/s",
                t * 1e3,
                bytes / t / 1e9
            );
        }
    }
    Ok(())
}
