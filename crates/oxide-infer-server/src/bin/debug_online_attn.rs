//! Deterministic diagnostic for the online GQA kernel.
//!
//! With q = k = 1 and v[t] = t the softmax is uniform, so every output element
//! must equal the mean of the kept key positions. That pins down both the lane
//! -> dimension mapping and the window boundary without float noise.

use anyhow::{anyhow, Result};
use cuda_core::launch_kernel_on_stream;
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext};
use oxide_infer_kernels::batched_attn::BATCHED_GQA_ATTENTION_ONLINE_PTX;
use oxide_infer_kernels::PtxModule;
use std::ffi::c_void;

const NQ: usize = 16;
const NKV: usize = 4;
const HD: usize = 128;
const SEQ: usize = 300;

fn main() -> Result<()> {
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();
    let module = PtxModule::load_from_ptx(&gpu, BATCHED_GQA_ATTENTION_ONLINE_PTX)
        .map_err(|e| anyhow!("gqa PTX: {e}"))?;
    let gqa = module.get_function("batched_gqa_attention_online_fp16")?;

    let q = DeviceBuffer::<f16>::from_slice(&gpu, &vec![f16::from_f32(1.0); NQ * HD])?;
    let k = DeviceBuffer::<f16>::from_slice(&gpu, &vec![f16::from_f32(1.0); SEQ * NKV * HD])?;
    let mut vh = vec![f16::from_f32(0.0); SEQ * NKV * HD];
    for t in 0..SEQ {
        for h in 0..NKV * HD {
            vh[t * NKV * HD + h] = f16::from_f32(t as f32);
        }
    }
    let v = DeviceBuffer::<f16>::from_slice(&gpu, &vh)?;
    let mut ctx = DeviceBuffer::<f16>::zeroed(&gpu, NQ * HD)?;
    let sl = DeviceBuffer::<u32>::from_slice(&gpu, &[SEQ as u32])?;
    let kp = DeviceBuffer::<u64>::from_slice(&gpu, &[k.inner().cu_deviceptr()])?;
    let vp = DeviceBuffer::<u64>::from_slice(&gpu, &[v.inner().cu_deviceptr()])?;

    // Non-uniform scores: q/k/v pseudo-random, compared against a host reference.
    {
        let pat = |seed: u64, i: usize| -> f32 {
            let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
            x ^= x >> 33;
            x = x.wrapping_mul(0xff51afd7ed558ccd);
            x ^= x >> 33;
            ((x & 0xffff) as f32 / 32767.5) - 1.0
        };
        let qh: Vec<f32> = (0..NQ * HD).map(|i| pat(1, i)).collect();
        let kh: Vec<f32> = (0..SEQ * NKV * HD).map(|i| pat(2, i) * 2.0).collect();
        let vhh: Vec<f32> = (0..SEQ * NKV * HD).map(|i| pat(3, i)).collect();
        let dq = DeviceBuffer::<f16>::from_slice(&gpu, &qh.iter().map(|x| f16::from_f32(*x)).collect::<Vec<_>>())?;
        let dk = DeviceBuffer::<f16>::from_slice(&gpu, &kh.iter().map(|x| f16::from_f32(*x)).collect::<Vec<_>>())?;
        let dv = DeviceBuffer::<f16>::from_slice(&gpu, &vhh.iter().map(|x| f16::from_f32(*x)).collect::<Vec<_>>())?;
        let kp2 = DeviceBuffer::<u64>::from_slice(&gpu, &[dk.inner().cu_deviceptr()])?;
        let vp2 = DeviceBuffer::<u64>::from_slice(&gpu, &[dv.inner().cu_deviceptr()])?;
        let mut ctx2 = DeviceBuffer::<f16>::zeroed(&gpu, NQ * HD)?;
        let sl2 = DeviceBuffer::<u32>::from_slice(&gpu, &[SEQ as u32])?;
        let mut d_ctxp = ctx2.inner_mut().cu_deviceptr();
        let mut d_qp = dq.inner().cu_deviceptr();
        let mut d_kpp = kp2.inner().cu_deviceptr();
        let mut d_vpp = vp2.inner().cu_deviceptr();
        let mut d_slp = sl2.inner().cu_deviceptr();
        let (mut nq, mut nkv, mut hd) = (NQ as u32, NKV as u32, HD as u32);
        let mut b = 1u32;
        let mut w = 0u32;
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
            launch_kernel_on_stream(&gqa, (NKV as u32, 1, 1), (HD as u32, 1, 1), 0, stream, &mut args)?;
        }
        stream.synchronize()?;
        let got = ctx2.inner().to_host_vec(stream)?;
        // host reference for head 0
        let kvh = 0usize;
        let mut sc = vec![f64::NEG_INFINITY; SEQ];
        for t in 0..SEQ {
            let mut d = 0.0f64;
            for i in 0..HD {
                d += qh[i] as f64 * kh[(t * NKV + kvh) * HD + i] as f64;
            }
            sc[t] = d * 0.0883883476;
        }
        let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ws: Vec<f64> = sc.iter().map(|x| (x - mx).exp()).collect();
        let den: f64 = ws.iter().sum();
        let mut want = vec![0.0f32; HD];
        for i in 0..HD {
            let mut acc = 0.0f64;
            for t in 0..SEQ {
                acc += ws[t] * vhh[(t * NKV + kvh) * HD + i] as f64;
            }
            want[i] = (acc / den) as f32;
        }
        for i in 0..8 {
            println!("  head0 dim{i}: kernel {:.6} ref {:.6}", got[i].to_f32(), want[i]);
        }
        // Is the kernel's head 0 a permutation of the reference?
        let mut matches = 0;
        for i in 0..HD {
            if want.iter().any(|w| (w - got[i].to_f32()).abs() < 1e-3) {
                matches += 1;
            }
        }
        println!("  head0 dims matching some reference dim: {matches}/{HD}");
    }

    for window in [0u32, 256] {
        let expect = if window == 0 {
            (SEQ - 1) as f32 / 2.0
        } else {
            (SEQ - window as usize + SEQ - 1) as f32 / 2.0
        };
        let mut d_ctxp = ctx.inner_mut().cu_deviceptr();
        let mut d_qp = q.inner().cu_deviceptr();
        let mut d_kpp = kp.inner().cu_deviceptr();
        let mut d_vpp = vp.inner().cu_deviceptr();
        let mut d_slp = sl.inner().cu_deviceptr();
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
            launch_kernel_on_stream(&gqa, (NKV as u32, 1, 1), (HD as u32, 1, 1), 0, stream, &mut args)?;
        }
        stream.synchronize()?;
        let out = ctx.inner().to_host_vec(stream)?;
        let mut bad = 0;
        let mut worst = 0.0f32;
        for (i, x) in out.iter().enumerate() {
            let d = (x.to_f32() - expect).abs();
            if d > 0.05 {
                if bad < 8 {
                    println!("  head {} dim {} = {} (want {})", i / HD, i % HD, x.to_f32(), expect);
                }
                bad += 1;
            }
            worst = worst.max(d);
        }
        println!(
            "online window={window:4} expect={expect:.2} bad={bad}/{} worst={worst:.4}",
            NQ * HD
        );
    }
    Ok(())
}
