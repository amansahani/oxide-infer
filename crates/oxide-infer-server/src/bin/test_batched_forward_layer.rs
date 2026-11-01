use std::sync::Arc;
use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_models::FullLfm3DecoderLayer;

fn main() -> anyhow::Result<()> {
    let gpu = Arc::new(GpuContext::init(0)?);
    let loader = SafeTensorsLoader::open("/mnt/hdd/models/LFM3-VL-2B/model.safetensors")?;
    
    let layer_types = [
        "conv", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
        "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention",
        "conv", "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv",
        "full_attention", "conv"
    ];

    println!("Loading 1 set of 24 layers into VRAM...");
    let mut layers = Vec::with_capacity(24);
    for idx in 0..24 {
        let is_attn = layer_types[idx] != "conv";
        let is_sliding = layer_types[idx] == "sliding_attention";
        layers.push(FullLfm3DecoderLayer::load_from_safetensors(&gpu, &loader, idx, is_attn, is_sliding, 2048, 7680)?);
    }

    let n_slots = 4;
    println!("Creating per-slot buffers for {} slots...", n_slots);
    struct SlotBuffers {
        d_hidden: DeviceBuffer<half::f16>,
        scratch_norm: DeviceBuffer<half::f16>,
        scratch_layer: DeviceBuffer<half::f16>,
        scratch_ffn: DeviceBuffer<half::f16>,
        q_out: DeviceBuffer<half::f16>,
        k_out: DeviceBuffer<half::f16>,
        v_out: DeviceBuffer<half::f16>,
        gate_out: DeviceBuffer<half::f16>,
    }
    let mut slot_buffers = Vec::with_capacity(n_slots);
    for _ in 0..n_slots {
        slot_buffers.push(SlotBuffers {
            d_hidden: DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?,
            scratch_norm: DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?,
            scratch_layer: DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?,
            scratch_ffn: DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?,
            q_out: DeviceBuffer::<half::f16>::zeroed(&gpu, 16 * 128)?,
            k_out: DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?,
            v_out: DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?,
            gate_out: DeviceBuffer::<half::f16>::zeroed(&gpu, 16)?,
        });
    }

    println!("Executing interleaved forward step over 24 shared layers across 4 slots...");
    for layer in layers.iter_mut() {
        for s in 0..n_slots {
            let buf = &mut slot_buffers[s];
            layer.forward(
                &gpu,
                &mut buf.d_hidden,
                &mut buf.scratch_norm,
                &mut buf.scratch_layer,
                &mut buf.scratch_ffn,
                &mut buf.q_out,
                &mut buf.k_out,
                &mut buf.v_out,
                &mut buf.gate_out,
                0,
            )?;
        }
    }
    gpu.compute_stream().synchronize()?;
    println!("SUCCESS! Evaluated 4 slots over 24 shared layers without memory duplication!");
    Ok(())
}
