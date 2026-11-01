use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_models::Lfm3AttentionLayer;

#[test]
fn test_lfm3_attention_layer_forward_real_weights() {
    let gpu = GpuContext::init(0).expect("Failed to init GPU");
    let loader = SafeTensorsLoader::open("/mnt/hdd/models/LFM3-VL-2B/model.safetensors")
        .expect("Failed to open safetensors");

    println!("Loading Lfm3AttentionLayer layer 2 (first attention layer)...");
    let mut layer = Lfm3AttentionLayer::load_from_safetensors(
        &gpu,
        &loader,
        2,    // Layer 2
        2048, // hidden_size
        16,   // num_q_heads
        4,    // num_kv_heads
        128,  // head_dim
    ).expect("Failed to load attention layer");

    let stream = gpu.compute_stream();

    // Create a mock normed input vector [2048]
    let input_data: Vec<half::f16> = (0..2048).map(|i| half::f16::from_f32((i as f32 % 5.0) * 0.1)).collect();
    let d_input = DeviceBuffer::from_slice(&gpu, &input_data).unwrap();

    let mut q_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 16 * 128).unwrap();
    let mut k_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128).unwrap();
    let mut v_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128).unwrap();
    let mut gate_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 16).unwrap();

    layer.forward_projections(&gpu, &d_input, &mut q_out, &mut k_out, &mut v_out, &mut gate_out)
        .expect("Forward projections failed");

    stream.synchronize().expect("GPU sync failed");

    let q_host = q_out.inner().to_host_vec(stream).unwrap();
    let k_host = k_out.inner().to_host_vec(stream).unwrap();
    let v_host = v_out.inner().to_host_vec(stream).unwrap();
    let gate_host = gate_out.inner().to_host_vec(stream).unwrap();

    println!("Q projection output[0..5]: {:?}", &q_host[0..5]);
    println!("K projection output[0..5]: {:?}", &k_host[0..5]);
    println!("V projection output[0..5]: {:?}", &v_host[0..5]);
    println!("Per-head gate output[0..5]: {:?}", &gate_host[0..5]);

    assert!(q_host[0].to_f32().abs() > 0.0);
    assert!(k_host[0].to_f32().abs() > 0.0);
    assert!(v_host[0].to_f32().abs() > 0.0);
    assert!(gate_host[0].to_f32().abs() > 0.0);

    // Test output gating and o_proj GEMV (copy q_out into context_buf to simulate computed attention context)
    layer.context_buf.copy_from_slice(&gpu, &q_host).expect("Copy to context_buf");
    let mut final_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048).unwrap();
    layer.forward_gating_and_o_proj(&gpu, &gate_out, &mut final_out)
        .expect("Forward gating and o_proj failed");

    stream.synchronize().expect("GPU sync failed");

    let final_host = final_out.inner().to_host_vec(stream).unwrap();
    println!("Final attention output[0..5]: {:?}", &final_host[0..5]);
    assert!(final_host[0].to_f32().abs() > 0.0);
}
