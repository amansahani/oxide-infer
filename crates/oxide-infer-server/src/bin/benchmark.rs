use std::sync::Arc;
use std::time::Instant;
use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_models::{Lfm3AttentionLayer, CausalConvLayer, FfnLayer, MultiModalProjector, EmbeddingTable};

fn main() -> anyhow::Result<()> {
    println!("================================================================================");
    println!("            OXIDE-INFER: PURE RUST CUDA-OXIDE BENCHMARK HARNESS                 ");
    println!("             Target Architecture: LFM3-VL-2B (RTX 3060 12GB)                   ");
    println!("================================================================================\n");

    let gpu = Arc::new(GpuContext::init(0)?);
    let stream = gpu.compute_stream();

    let weights_path = "/mnt/hdd/models/LFM3-VL-2B/model.safetensors";
    println!("[1/6] Opening SafeTensors zero-copy mmap from {}...", weights_path);
    let t0 = Instant::now();
    let loader = SafeTensorsLoader::open(weights_path)?;
    println!("      -> SafeTensors mmap initialized in {:.2?} ms\n", t0.elapsed().as_secs_f64() * 1000.0);

    // -------------------------------------------------------------------------
    // Benchmark 1: Token Embedding Lookup
    // -------------------------------------------------------------------------
    println!("[2/6] Benchmarking Vocabulary Embedding Lookup (128,000 Vocab x 2048 Hidden)...");
    let embed_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embed_tokens.weight")?;
    let host_bf16 = embed_bf16.inner().to_host_vec(stream)?;
    let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let embed_table = EmbeddingTable::new(&gpu, &host_f16, 128000, 2048)?;

    let test_tokens = vec![124894u32, 1500u32, 25000u32, 8912u32];
    let d_tokens = DeviceBuffer::from_slice(&gpu, &test_tokens)?;
    let mut d_embed_out = DeviceBuffer::<half::f16>::zeroed(&gpu, test_tokens.len() * 2048)?;

    // Warmup
    for _ in 0..10 {
        embed_table.lookup(&gpu, &d_tokens, test_tokens.len(), &mut d_embed_out)?;
    }
    stream.synchronize()?;

    let iters = 1000;
    let t_embed = Instant::now();
    for _ in 0..iters {
        embed_table.lookup(&gpu, &d_tokens, test_tokens.len(), &mut d_embed_out)?;
    }
    stream.synchronize()?;
    let elapsed_embed = t_embed.elapsed().as_secs_f64() / iters as f64;
    println!("      -> Embedding Lookup Latency: {:.3} µs / call\n", elapsed_embed * 1e6);

    // -------------------------------------------------------------------------
    // Benchmark 2: 1D Causal Convolution Layer
    // -------------------------------------------------------------------------
    println!("[3/6] Benchmarking 1D Causal Convolution Layer (K=7, Hidden=2048)...");
    let mut conv_layer = CausalConvLayer::load_from_safetensors(&gpu, &loader, 0, 2048, 7)?;
    let d_hidden = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut d_conv_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;

    // Warmup
    for _ in 0..10 {
        conv_layer.forward(&gpu, &d_hidden, &mut d_conv_out)?;
    }
    stream.synchronize()?;

    let t_conv = Instant::now();
    for _ in 0..iters {
        conv_layer.forward(&gpu, &d_hidden, &mut d_conv_out)?;
    }
    stream.synchronize()?;
    let elapsed_conv = t_conv.elapsed().as_secs_f64() / iters as f64;
    println!("      -> CausalConv1D Step Latency: {:.3} µs / token ({:.1} kTok/s)\n",
        elapsed_conv * 1e6, (1.0 / elapsed_conv) / 1000.0);

    // -------------------------------------------------------------------------
    // Benchmark 3: SwiGLU Gated Feed-Forward Network
    // -------------------------------------------------------------------------
    println!("[4/6] Benchmarking SwiGLU FFN Layer (Hidden=2048 -> Intermediate=7680 -> 2048)...");
    let mut ffn_layer = FfnLayer::load_from_safetensors(&gpu, &loader, 0, 2048, 7680)?;
    let mut d_ffn_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;

    // Warmup
    for _ in 0..10 {
        ffn_layer.forward(&gpu, &d_hidden, &mut d_ffn_out)?;
    }
    stream.synchronize()?;

    let t_ffn = Instant::now();
    for _ in 0..iters {
        ffn_layer.forward(&gpu, &d_hidden, &mut d_ffn_out)?;
    }
    stream.synchronize()?;
    let elapsed_ffn = t_ffn.elapsed().as_secs_f64() / iters as f64;
    println!("      -> SwiGLU FFN Step Latency: {:.3} µs / token ({:.1} kTok/s)\n",
        elapsed_ffn * 1e6, (1.0 / elapsed_ffn) / 1000.0);

    // -------------------------------------------------------------------------
    // Benchmark 4: Self-Attention Projections + Sigmoid Gating
    // -------------------------------------------------------------------------
    println!("[5/6] Benchmarking Attention Layer 2 (16 Q heads, 4 KV heads, Sigmoid Gating, O-proj)...");
    let mut attn_layer = Lfm3AttentionLayer::load_from_safetensors(&gpu, &loader, 2, 2048, 16, 4, 128)?;

    let mut q_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 16 * 128)?;
    let mut k_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?;
    let mut v_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 4 * 128)?;
    let mut gate_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 16)?;
    let mut attn_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut final_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;

    // Warmup
    for _ in 0..10 {
        attn_layer.forward_projections(&gpu, &d_hidden, &mut q_out, &mut k_out, &mut v_out, &mut gate_out)?;
        attn_layer.forward_output(&gpu, &q_out, &k_out, &v_out, &gate_out, &mut final_out)?;
    }
    stream.synchronize()?;

    let t_attn = Instant::now();
    for _ in 0..iters {
        attn_layer.forward_projections(&gpu, &d_hidden, &mut q_out, &mut k_out, &mut v_out, &mut gate_out)?;
        attn_layer.forward_output(&gpu, &q_out, &k_out, &v_out, &gate_out, &mut final_out)?;
    }
    stream.synchronize()?;
    let elapsed_attn = t_attn.elapsed().as_secs_f64() / iters as f64;
    println!("      -> Attention Forward Step Latency: {:.3} µs / token ({:.1} kTok/s)\n",
        elapsed_attn * 1e6, (1.0 / elapsed_attn) / 1000.0);

    // -------------------------------------------------------------------------
    // Benchmark 5: Multimodal Projector
    // -------------------------------------------------------------------------
    println!("[6/6] Benchmarking Multimodal Projector (4096-dim Image Feature -> GELU -> 2048-dim Token)...");
    let mut projector = MultiModalProjector::load_from_safetensors(&gpu, &loader)?;
    let d_vis = DeviceBuffer::<half::f16>::zeroed(&gpu, 4096)?;
    let mut d_proj_out = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;

    // Warmup
    for _ in 0..10 {
        projector.project(&gpu, &d_vis, &mut d_proj_out)?;
    }
    stream.synchronize()?;

    let t_proj = Instant::now();
    for _ in 0..iters {
        projector.project(&gpu, &d_vis, &mut d_proj_out)?;
    }
    stream.synchronize()?;
    let elapsed_proj = t_proj.elapsed().as_secs_f64() / iters as f64;
    println!("      -> Multimodal Projection Latency: {:.3} µs / patch ({:.1} kPatches/s)\n",
        elapsed_proj * 1e6, (1.0 / elapsed_proj) / 1000.0);

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("================================================================================");
    println!("                       BENCHMARK PERFORMANCE SUMMARY                           ");
    println!("================================================================================");
    let total_layer_latency_conv = elapsed_conv + elapsed_ffn;
    let total_layer_latency_attn = elapsed_attn + elapsed_ffn;
    // 24 layers total: 16 conv layers + 8 attention layers
    let est_full_model_latency = (16.0 * total_layer_latency_conv) + (8.0 * total_layer_latency_attn);
    let est_tokens_per_sec = 1.0 / est_full_model_latency;

    println!("  * Single Conv Layer (Conv1D + SwiGLU FFN):     {:>8.2} µs", total_layer_latency_conv * 1e6);
    println!("  * Single Attention Layer (GQA + Gate + FFN):   {:>8.2} µs", total_layer_latency_attn * 1e6);
    println!("  * Full 24-Layer Model Step (Estimated B=1):    {:>8.2} ms", est_full_model_latency * 1000.0);
    println!("  * Estimated Single-Stream Autoregressive Speed:{:>8.1} tokens/sec", est_tokens_per_sec);
    println!("================================================================================");

    Ok(())
}
