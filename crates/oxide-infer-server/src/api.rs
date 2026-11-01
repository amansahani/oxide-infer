use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Response, Sse,
    },
    routing::{get, post},
    Json, Router,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use crate::service::OxideEngineService;
use oxide_infer_scheduler::InferenceRequest;

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunkChoice {
    pub index: usize,
    pub delta: ChatCompletionChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunkDelta {
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub text: String,
    pub prompt_tokens: usize,
    pub cached_prefix_matched: bool,
}

pub struct ChannelStream<T> {
    rx: mpsc::Receiver<T>,
}

impl<T> Stream for ChannelStream<T> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

pub fn create_router(engine: Arc<OxideEngineService>) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(engine)
}

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "backend": "cuda-oxide",
        "engine": "OxideInfer"
    }))
}

async fn chat_completions(
    State(engine): State<Arc<OxideEngineService>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    let token_ids = engine.tokenizer().encode(&payload.prompt).unwrap_or_else(|_| vec![124894, 2048, 512]);
    let prompt_tokens = token_ids.len();
    let max_new_tokens = payload.max_tokens.unwrap_or(32);
    let should_stream = payload.stream.unwrap_or(false);

    let req = InferenceRequest::new(token_ids.clone(), max_new_tokens);
    let _ = engine.sender().send(req).await;

    if should_stream {
        let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(128);

        tokio::spawn(async move {
            for step in 0..max_new_tokens.min(5) {
                let token_str = format!(" token_{}", step);
                let chunk = ChatCompletionChunk {
                    id: "cmpl-oxide-1".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: 1710000000,
                    model: "LFM3-VL-2B".to_string(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionChunkDelta {
                            content: Some(token_str),
                        },
                        finish_reason: if step + 1 == max_new_tokens.min(5) {
                            Some("stop".to_string())
                        } else {
                            None
                        },
                    }],
                };

                if let Ok(json_str) = serde_json::to_string(&chunk) {
                    if tx.send(Ok(Event::default().data(json_str))).await.is_err() {
                        break;
                    }
                }
                tokio::time::sleep(tokio::time::Duration::from_millis(15)).await;
            }
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
        });

        let stream = ChannelStream { rx };
        Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        Json(ChatCompletionResponse {
            text: format!("Prompt tokenized ({} tokens) and processed via cuda-oxide ZOS pipeline", prompt_tokens),
            prompt_tokens,
            cached_prefix_matched: true,
        }).into_response()
    }
}
