use std::sync::Arc;
use tokio::sync::mpsc;
use parking_lot::Mutex;
use oxide_infer_core::{DeviceBuffer, GpuContext, PagedKvPool, PagedKvPoolConfig, SafeTensorsLoader};
use oxide_infer_models::{HybridTransformerLayer, ModelEngine};
use oxide_infer_scheduler::{InferenceRequest, PagedBlockManager, RadixTree, ZosScheduler};
use crate::tokenizer::OxideTokenizer;

/// Top-level OxideInfer engine service managing serving channels and scheduler threads.
pub struct OxideEngineService {
    gpu: Arc<GpuContext>,
    kv_pool: Arc<PagedKvPool>,
    block_manager: Arc<PagedBlockManager>,
    scheduler: Arc<ZosScheduler>,
    tokenizer: Arc<OxideTokenizer>,
    model_engine: Arc<Mutex<Option<ModelEngine>>>,
    request_tx: mpsc::Sender<InferenceRequest>,
}

impl OxideEngineService {
    pub fn new(
        gpu: GpuContext,
        max_batch_size: usize,
        kv_config: PagedKvPoolConfig,
        tokenizer: OxideTokenizer,
    ) -> anyhow::Result<Self> {
        let block_size = kv_config.block_size;
        let gpu = Arc::new(gpu);
        let kv_pool = Arc::new(PagedKvPool::new(&gpu, kv_config)?);
        let block_manager = Arc::new(PagedBlockManager::new(kv_pool.clone(), block_size));
        let radix_tree = Arc::new(RadixTree::new());
        let scheduler = Arc::new(ZosScheduler::new(radix_tree, block_manager.clone()));
        let tokenizer = Arc::new(tokenizer);
        let model_engine: Arc<Mutex<Option<ModelEngine>>> = Arc::new(Mutex::new(None));
        let (request_tx, mut request_rx) = mpsc::channel::<InferenceRequest>(1024);

        let sched_clone = Arc::clone(&scheduler);
        let gpu_clone = Arc::clone(&gpu);
        let engine_clone = Arc::clone(&model_engine);

        // Spawn DeepInfer-style asynchronous background scheduling worker (ZOS)
        tokio::spawn(async move {
            loop {
                // Ingest new requests from front-end
                while let Ok(req) = request_rx.try_recv() {
                    sched_clone.submit_request(req);
                }

                // Asynchronously plan next batch without blocking GPU execution
                if let Some(_batch_plan) = sched_clone.plan_next_batch(max_batch_size) {
                    let mut engine_guard = engine_clone.lock();
                    if let Some(ref mut engine) = *engine_guard {
                        // Create dummy hidden state vector for the active tokens in the batch
                        if let Ok(mut hidden) = DeviceBuffer::<half::f16>::zeroed(&gpu_clone, engine.hidden_size) {
                            let _ = engine.forward_step(&gpu_clone, &mut hidden);
                        }
                    }
                }

                tokio::time::sleep(tokio::time::Duration::from_micros(150)).await;
            }
        });

        Ok(Self {
            gpu,
            kv_pool,
            block_manager,
            scheduler,
            tokenizer,
            model_engine,
            request_tx,
        })
    }

    /// Load model weights into the engine.
    pub fn load_model(&self, weights_path: &str, num_layers: usize, hidden_size: usize, intermediate_size: usize, k_len: usize) -> anyhow::Result<()> {
        let loader = SafeTensorsLoader::open(weights_path)?;
        let mut layers = Vec::with_capacity(num_layers);

        for idx in 0..num_layers {
            let layer = HybridTransformerLayer::load_from_safetensors(&self.gpu, &loader, idx, hidden_size, intermediate_size, k_len)?;
            layers.push(layer);
        }

        let mut engine_guard = self.model_engine.lock();
        *engine_guard = Some(ModelEngine::new(layers, hidden_size));
        Ok(())
    }

    pub fn tokenizer(&self) -> &Arc<OxideTokenizer> {
        &self.tokenizer
    }

    pub fn scheduler(&self) -> &Arc<ZosScheduler> {
        &self.scheduler
    }

    pub fn block_manager(&self) -> &Arc<PagedBlockManager> {
        &self.block_manager
    }

    pub fn kv_pool(&self) -> &Arc<PagedKvPool> {
        &self.kv_pool
    }

    pub fn gpu(&self) -> &Arc<GpuContext> {
        &self.gpu
    }

    pub fn sender(&self) -> &mpsc::Sender<InferenceRequest> {
        &self.request_tx
    }
}
