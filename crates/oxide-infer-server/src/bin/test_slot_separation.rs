use std::sync::Arc;
use oxide_infer_core::{GpuContext, DeviceBuffer};

fn main() -> anyhow::Result<()> {
    let gpu = Arc::new(GpuContext::init(0)?);
    let n_slots = 4;
    let hidden_size = 2048;
    let k_len = 7;
    let max_seq_len = 1024;
    let kv_dim = 4 * 128; // 512

    println!("Allocating dynamic state for {} slots across 24 layers...", n_slots);
    
    // Per slot: 16 conv layers (state_cache: 2048 * 7 * 2 bytes = 28 KB)
    // 8 attention layers (k_cache: 1024 * 512 * 2 = 1 MB, v_cache: 1 MB)
    // Total per slot: 16 * 28KB + 8 * 2MB = ~16.4 MB!
    let mut all_slots = Vec::new();
    for s in 0..n_slots {
        let mut slot_conv_states = Vec::with_capacity(16);
        for _ in 0..16 {
            slot_conv_states.push(DeviceBuffer::<half::f16>::zeroed(&gpu, hidden_size * k_len)?);
        }
        let mut slot_kv_caches = Vec::with_capacity(8);
        for _ in 0..8 {
            let kc = DeviceBuffer::<half::f16>::zeroed(&gpu, max_seq_len * kv_dim)?;
            let vc = DeviceBuffer::<half::f16>::zeroed(&gpu, max_seq_len * kv_dim)?;
            slot_kv_caches.push((kc, vc));
        }
        all_slots.push((slot_conv_states, slot_kv_caches));
        println!("  Slot {} dynamic state allocated!", s);
    }
    println!("SUCCESS: 4 Slots dynamic state fully allocated! Total state size: ~65 MB!");
    Ok(())
}
