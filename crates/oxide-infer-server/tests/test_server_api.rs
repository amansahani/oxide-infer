use std::sync::Arc;
use tokio::net::TcpListener;
use oxide_infer_core::{GpuContext, PagedKvPoolConfig};
use oxide_infer_server::service::OxideEngineService;
use oxide_infer_server::tokenizer::OxideTokenizer;
use oxide_infer_server::api::create_router;

#[tokio::test]
async fn test_full_server_streaming_and_non_streaming() {
    let gpu = GpuContext::init(0).expect("Failed to init GPU");
    let kv_config = PagedKvPoolConfig {
        block_size: 16,
        num_layers: 2,
        num_heads: 4,
        head_dim: 128,
        num_blocks: 64,
    };
    let tokenizer = OxideTokenizer::from_file("/mnt/hdd/models/LFM3-VL-2B/tokenizer.json")
        .expect("Failed to load tokenizer");

    let engine = Arc::new(OxideEngineService::new(gpu, 4, kv_config, tokenizer).unwrap());
    let app = create_router(engine);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();

    // 1. Health check
    let resp = client.get(format!("http://{}/health", addr))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    // 2. Non-streaming chat completion
    let req_body = serde_json::json!({
        "prompt": "Hello world from pure Rust cuda-oxide",
        "max_tokens": 16,
        "stream": false
    });
    let resp = client.post(format!("http://{}/v1/chat/completions", addr))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["prompt_tokens"].as_u64().unwrap(), 9);

    // 3. Streaming SSE chat completion
    let req_stream_body = serde_json::json!({
        "prompt": "Stream generation test",
        "max_tokens": 5,
        "stream": true
    });
    let stream_resp = client.post(format!("http://{}/v1/chat/completions", addr))
        .json(&req_stream_body)
        .send()
        .await
        .unwrap();
    assert!(stream_resp.status().is_success());
    let body_text = stream_resp.text().await.unwrap();
    println!("SSE Stream received:\n{}", body_text);
    assert!(body_text.contains("data: {\"id\":\"cmpl-oxide-1\""));
    assert!(body_text.contains("[DONE]"));
}
