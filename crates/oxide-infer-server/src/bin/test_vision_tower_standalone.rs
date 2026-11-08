use anyhow::Result;
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext, SafeTensorsLoader};
use oxide_infer_models::VisionTower;

fn main() -> Result<()> {
    println!("Testing standalone VisionTower::forward on GPU...");
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();

    let loader = SafeTensorsLoader::open("/mnt/hdd/models/LFM3-VL-2B/model.safetensors")?;
    println!("Loading VisionTower weights...");
    let mut vision_tower = VisionTower::load_from_safetensors(&gpu, &loader)?;

    // Load pixel_values reference: [256, 1536]
    let pix_bytes = std::fs::read("/tmp/vis_ref_pixel_values.npy")?;
    let header_len = 128usize;
    let pix_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(
            pix_bytes[header_len..].as_ptr() as *const f32,
            (pix_bytes.len() - header_len) / 4,
        )
    };
    let pix_f16: Vec<f16> = pix_slice.iter().map(|&x| f16::from_f32(x)).collect();
    let d_patches = DeviceBuffer::from_slice(&gpu, &pix_f16)?;

    let mut d_projected = DeviceBuffer::<f16>::zeroed(&gpu, 64 * 2048)?;

    println!("Executing VisionTower::forward (14 layers + projector)...");
    let t0 = std::time::Instant::now();
    vision_tower.forward(&gpu, &d_patches, &mut d_projected)?;
    stream.synchronize()?;
    let dt = t0.elapsed();
    println!("VisionTower forward completed in {:.2} ms", dt.as_secs_f64() * 1000.0);

    // Read back projected tokens and compare against PyTorch reference
    let proj_host = d_projected.inner().to_host_vec(stream)?;
    let ref_proj_bytes = std::fs::read("/tmp/vis_ref_projected.npy")?;
    let ref_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(
            ref_proj_bytes[header_len..].as_ptr() as *const f32,
            (ref_proj_bytes.len() - header_len) / 4,
        )
    };

    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut count_above_005 = 0;
    for (i, &val) in proj_host.iter().enumerate() {
        let diff = (val.to_f32() - ref_slice[i]).abs();
        sum_diff += diff as f64;
        if diff > 0.05 {
            count_above_005 += 1;
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }
    let mean_diff = sum_diff / proj_host.len() as f64;
    println!("Projector output count: {} floats", proj_host.len());
    println!("Max difference vs PyTorch official reference: {:.6}", max_diff);
    println!("Mean difference vs PyTorch official reference: {:.6}", mean_diff);
    println!("Count with diff > 0.05: {} / {} ({:.4}%)", count_above_005, proj_host.len(), count_above_005 as f64 / proj_host.len() as f64 * 100.0);
    assert!(mean_diff < 0.01 && max_diff <= 0.05, "Diff exceeds tolerance!");
    println!("✅ FULL VISION TOWER + PROJECTOR PARITY CONFIRMED!");

    Ok(())
}
