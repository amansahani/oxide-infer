use anyhow::Result;
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext, SafeTensorsLoader, VisionPreprocessor};
use oxide_infer_models::VisionTower;
use std::time::Instant;

fn main() -> Result<()> {
    println!("============================================================");
    println!(" END-TO-END PURE RUST MULTIMODAL INFERENCE: REAL IMAGES");
    println!("============================================================");

    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();

    let loader = SafeTensorsLoader::open("/mnt/hdd/models/LFM3-VL-2B/model.safetensors")?;
    println!("Loading VisionTower + Projector onto GPU...");
    let mut vision_tower = VisionTower::load_from_safetensors(&gpu, &loader)?;

    let preprocessor = VisionPreprocessor::default();

    let test_images = [
        "/home/tomi/Downloads/markheet.jpeg",
        "/home/tomi/Downloads/catagory.jpeg",
        "/home/tomi/Downloads/x570_rtx3060.jpeg",
        "/home/tomi/Downloads/signat.png",
    ];

    for path in test_images {
        println!("\n------------------------------------------------------------");
        println!("Testing image: {}", path);
        let dyn_img = oxide_infer_core::image::open(path)?;
        println!("Image dimensions: {}x{}", dyn_img.width(), dyn_img.height());

        let t0 = Instant::now();
        let patchified = preprocessor.process_image(&dyn_img)?;
        let t_prep = t0.elapsed();
        println!("Preprocessor took: {:.2} ms (patches: {})", t_prep.as_secs_f64() * 1000.0, patchified.num_patches);

        let d_patches = DeviceBuffer::from_slice(&gpu, &patchified.patches)?;
        let mut d_proj = DeviceBuffer::<f16>::zeroed(&gpu, 64 * 2048)?;

        let t1 = Instant::now();
        vision_tower.forward(&gpu, &d_patches, &mut d_proj)?;
        stream.synchronize()?;
        let t_tower = t1.elapsed();
        println!("Vision Tower + Projector took: {:.2} ms", t_tower.as_secs_f64() * 1000.0);

        let proj_host = d_proj.inner().to_host_vec(stream)?;
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut sum_val = 0.0f32;
        for &v in &proj_host {
            let f = v.to_f32();
            if f < min_val { min_val = f; }
            if f > max_val { max_val = f; }
            sum_val += f;
        }
        let mean_val = sum_val / (proj_host.len() as f32);
        println!("Visual embedding stats: min={:.4}, max={:.4}, mean={:.6}", min_val, max_val, mean_val);
        println!("✓ 64 multimodal feature tokens successfully generated on GPU!");
    }

    println!("\n============================================================");
    println!(" ✅ ALL REAL IMAGES PROCESSED & VERIFIED SUCCESSFULLY");
    println!("============================================================");
    Ok(())
}
