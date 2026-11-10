//! Benchmark for `batched_gqa_attention_fp16` (current scan kernel).
//!
//! Reports effective key-scans per second for representative prefill/decode
//! shapes so the cost can be compared against the tiled implementation.

use anyhow::{anyhow, Result};
use cuda_core::launch_kernel_on_stream;
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext};
use oxide_infer_kernels::batched_attn::{BATCHED_GQA_ATTENTION_ONLINE_PTX, BATCHED_GQA_ATTENTION_PTX};
use oxide_infer_kernels::PtxModule;
use std::ffi::c_void;
use std::time::Instant;

const NQ: usize = 16;
const NKV: usize = 4;
const HD: usize = 128;

fn main() -> Result<()> {
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();
    let which = std::env::var("BENCH_KERNEL").unwrap_or_else(|_| "online".into());
    let (ptx, entry) = if which == "scan" {
        (BATCHED_GQA_ATTENTION_PTX, "batched_gqa_attention_fp16")
    } else {
        (BATCHED_GQA_ATTENTION_ONLINE_PTX, "batched_gqa_attention_online_fp16")
    };
    let module = PtxModule::load_from_ptx(&gpu, ptx)
        .map_err(|e| anyhow!("gqa PTX: {e}"))?;
    let gqa = module.get_function(entry)?;

    for (b, slen) in [(1usize, 512usize), (1, 1000), (1, 2000), (1, 4000), (4, 2000), (8, 2000), (64, 2000), (512, 512), (512, 2000)] {
        let q = DeviceBuffer::<f16>::from_slice(&gpu, &vec![f16::from_f32(0.1); b * NQ * HD])?;
        let k = DeviceBuffer::<f16>::from_slice(&gpu, &vec![f16::from_f32(0.05); slen * NKV * HD])?;
        let v = DeviceBuffer::<f16>::from_slice(&gpu, &vec![f16::from_f32(0.05); slen * NKV * HD])?;
        let mut ctx = DeviceBuffer::<f16>::zeroed(&gpu, b * NQ * HD)?;
        let sl = DeviceBuffer::<u32>::from_slice(&gpu, &vec![slen as u32; b])?;
        let kp = DeviceBuffer::<u64>::from_slice(&gpu, &vec![k.inner().cu_deviceptr(); b])?;
        let vp = DeviceBuffer::<u64>::from_slice(&gpu, &vec![v.inner().cu_deviceptr(); b])?;

        let mut launch = |window: u32| -> Result<()> {
            let mut d_ctxp = ctx.inner_mut().cu_deviceptr();
            let mut d_qp = q.inner().cu_deviceptr();
            let mut d_kpp = kp.inner().cu_deviceptr();
            let mut d_vpp = vp.inner().cu_deviceptr();
            let mut d_slp = sl.inner().cu_deviceptr();
            let (mut nq, mut nkv, mut hd) = (NQ as u32, NKV as u32, HD as u32);
            let mut bb = b as u32;
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
                &mut bb as *mut _ as *mut c_void,
                &mut w as *mut _ as *mut c_void,
            ];
            unsafe {
                let grid = if which == "scan" { (NQ as u32, bb, 1) } else { (NKV as u32, bb, 1) };
                launch_kernel_on_stream(&gqa, grid, (HD as u32, 1, 1), 0, stream, &mut args)?;
            }
            Ok(())
        };

        for w in [0u32, 256] {
            launch(w)?;
            stream.synchronize()?;
            let reps = if b * slen > 2_000_000 { 5 } else { 50 };
            let t0 = Instant::now();
            for _ in 0..reps {
                launch(w)?;
            }
            stream.synchronize()?;
            let dt = t0.elapsed().as_secs_f64() / reps as f64;
            let keys = (b * NQ * slen) as f64;
            println!(
                "{} b={:4} slen={:5} window={:4}  {:8.2} ms  {:7.2} Gkeys/s  {:5.1} TFLOP/s",
                which, b, slen, w, dt * 1e3,
                keys / dt / 1e9,
                keys * 2.0 * HD as f64 / dt / 1e12
            );
        }
    }
    Ok(())
}
