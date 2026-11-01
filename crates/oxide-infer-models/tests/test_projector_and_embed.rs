use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_models::{MultiModalProjector, EmbeddingTable};

#[test]
fn test_multimodal_projector_and_embedding_real_weights() {
    let gpu = GpuContext::init(0).expect("Failed to initialize GPU context");
    let loader = SafeTensorsLoader::open("/mnt/hdd/models/LFM3-VL-2B/model.safetensors")
        .expect("Failed to open safetensors");

    println!("Testing MultiModalProjector loading...");
    let mut projector = MultiModalProjector::load_from_safetensors(&gpu, &loader)
        .expect("Failed to load MultiModalProjector");

    // Create a dummy merged visual token: 4096 f16 values
    let input_vis_data: Vec<half::f16> = (0..4096).map(|i| half::f16::from_f32((i as f32 % 10.0) * 0.01)).collect();
    let d_input = DeviceBuffer::from_slice(&gpu, &input_vis_data).unwrap();
    let mut d_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048).unwrap();

    projector.project(&gpu, &d_input, &mut d_out).expect("Projector forward failed");
    gpu.compute_stream().synchronize().expect("GPU sync failed");

    let stream = gpu.compute_stream();
    let out_host = d_out.inner().to_host_vec(stream).unwrap();
    println!("MultiModalProjector output[0..5]: {:?}", &out_host[0..5]);
    assert!(out_host[0].to_f32().abs() > 0.0, "Projector output should not be zero: {}", out_host[0].to_f32());

    println!("Testing EmbeddingTable lookup on BF16 weights...");
    let embed_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embed_tokens.weight")
        .expect("Failed to load embed_tokens");
    let host_bf16 = embed_bf16.inner().to_host_vec(stream).unwrap();
    let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();

    let embed_table = EmbeddingTable::new(&gpu, &host_f16, 128000, 2048)
        .expect("Failed to init EmbeddingTable");

    let token_ids = vec![124894u32, 100u32, 2048u32]; // 3 tokens
    let d_tokens = DeviceBuffer::from_slice(&gpu, &token_ids).unwrap();
    let mut d_embeds = DeviceBuffer::<half::f16>::zeroed(&gpu, 3 * 2048).unwrap();

    embed_table.lookup(&gpu, &d_tokens, 3, &mut d_embeds).expect("Embedding lookup failed");
    gpu.compute_stream().synchronize().expect("GPU sync failed");

    let embeds_host = d_embeds.inner().to_host_vec(stream).unwrap();
    println!("Embedding token 0 output[0..5]: {:?}", &embeds_host[0..5]);
    println!("Embedding token 1 output[0..5]: {:?}", &embeds_host[2048..2053]);
    assert!(embeds_host[0].to_f32().abs() > 0.0);
}
