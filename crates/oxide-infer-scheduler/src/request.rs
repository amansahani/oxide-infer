use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_REQ_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub u64);

impl RequestId {
    pub fn next() -> Self {
        Self(NEXT_REQ_ID.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub id: RequestId,
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub block_ids: Vec<usize>,
    pub generated_tokens: Vec<u32>,
    pub is_finished: bool,
}

impl InferenceRequest {
    pub fn new(prompt_tokens: Vec<u32>, max_new_tokens: usize) -> Self {
        Self {
            id: RequestId::next(),
            prompt_tokens,
            max_new_tokens,
            temperature: 1.0,
            top_p: 1.0,
            block_ids: Vec::new(),
            generated_tokens: Vec::new(),
            is_finished: false,
        }
    }
}
