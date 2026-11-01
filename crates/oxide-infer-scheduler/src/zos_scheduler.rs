use crate::block_manager::{PagedBlockManager, SequenceId};
use crate::radix_tree::RadixTree;
use crate::request::{InferenceRequest, RequestId};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;

/// A planned iteration batch ready for zero-overhead dispatch to the GPU.
#[derive(Debug, Clone)]
pub struct BatchPlan {
    pub iteration_id: u64,
    pub batch_size: usize,
    pub active_requests: Vec<RequestId>,
    /// Flattened block tables for paged attention: [batch_size, max_blocks]
    pub block_table: Vec<usize>,
    /// Per-sequence query lengths
    pub seqlens_q: Vec<i32>,
    /// Per-sequence total context lengths (past KV + current Q)
    pub seqlens_kv: Vec<i32>,
    /// Current input token per sequence
    pub input_tokens: Vec<u32>,
}

/// DeepInfer-Style Zero-Overhead Asynchronous Scheduler (ZOS).
pub struct ZosScheduler {
    radix_tree: Arc<RadixTree>,
    block_manager: Arc<PagedBlockManager>,
    waiting_queue: Mutex<VecDeque<InferenceRequest>>,
    running_requests: Mutex<Vec<InferenceRequest>>,
    iteration_counter: std::sync::atomic::AtomicU64,
}

impl ZosScheduler {
    pub fn new(radix_tree: Arc<RadixTree>, block_manager: Arc<PagedBlockManager>) -> Self {
        Self {
            radix_tree,
            block_manager,
            waiting_queue: Mutex::new(VecDeque::new()),
            running_requests: Mutex::new(Vec::new()),
            iteration_counter: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Submit a new inference request to the scheduler.
    pub fn submit_request(&self, req: InferenceRequest) {
        let mut queue = self.waiting_queue.lock();
        queue.push_back(req);
    }

    /// Plan the next iteration batch (runs on host asynchronously without stalling GPU).
    pub fn plan_next_batch(&self, max_batch_size: usize) -> Option<BatchPlan> {
        let mut running = self.running_requests.lock();
        let mut waiting = self.waiting_queue.lock();

        // Admit waiting requests if capacity allows
        while running.len() < max_batch_size && !waiting.is_empty() {
            if let Some(mut req) = waiting.pop_front() {
                // SGLang prefix match against Radix Tree
                let (_matched_tokens, cached_blocks) = self.radix_tree.match_prefix(&req.prompt_tokens);
                let shared_block_ids: Vec<oxide_infer_core::BlockId> = cached_blocks.iter().map(|&id| oxide_infer_core::BlockId(id)).collect();

                // Allocate physical blocks via PagedBlockManager
                let seq_id = SequenceId(req.id.0);
                if self.block_manager.allocate_for_seq(seq_id, req.prompt_tokens.len(), &shared_block_ids).is_ok() {
                    if let Some(table) = self.block_manager.get_block_table(seq_id) {
                        req.block_ids = table;
                    }
                    running.push(req);
                } else {
                    // Out of memory; re-queue request
                    waiting.push_front(req);
                    break;
                }
            }
        }

        if running.is_empty() {
            return None;
        }

        let iter_id = self.iteration_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let batch_size = running.len();

        let mut active_requests = Vec::with_capacity(batch_size);
        let mut block_table = Vec::new();
        let mut seqlens_q = Vec::with_capacity(batch_size);
        let mut seqlens_kv = Vec::with_capacity(batch_size);
        let mut input_tokens = Vec::with_capacity(batch_size);

        for req in running.iter_mut() {
            let seq_id = SequenceId(req.id.0);
            let current_len = req.prompt_tokens.len() + req.generated_tokens.len();

            // Expand block allocation if token crosses page boundary
            let _ = self.block_manager.append_token(seq_id, current_len);
            if let Some(table) = self.block_manager.get_block_table(seq_id) {
                req.block_ids = table;
            }

            active_requests.push(req.id);
            seqlens_q.push(1); // 1 token per decode step
            seqlens_kv.push(current_len as i32);
            
            // Last generated token or last prompt token
            let next_token = req.generated_tokens.last().copied()
                .unwrap_or_else(|| req.prompt_tokens.last().copied().unwrap_or(0));
            input_tokens.push(next_token);

            // Extend block table
            block_table.extend_from_slice(&req.block_ids);
        }

        Some(BatchPlan {
            iteration_id: iter_id,
            batch_size,
            active_requests,
            block_table,
            seqlens_q,
            seqlens_kv,
            input_tokens,
        })
    }
}
