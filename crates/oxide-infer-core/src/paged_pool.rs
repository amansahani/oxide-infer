use anyhow::Result;
use crate::context::GpuContext;
use crate::memory::DeviceBuffer;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

#[derive(Debug, Clone)]
pub struct PagedKvPoolConfig {
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl PagedKvPoolConfig {
    pub fn total_elements_per_block(&self) -> usize {
        // Storing keys and values: 2 * num_layers * block_size * num_heads * head_dim
        2 * self.num_layers * self.block_size * self.num_heads * self.head_dim
    }
}

/// Physical Paged KV Cache Pool in GPU VRAM.
pub struct PagedKvPool {
    config: PagedKvPoolConfig,
    data: DeviceBuffer<half::f16>,
    free_blocks: crossbeam_channel::Receiver<BlockId>,
    free_sender: crossbeam_channel::Sender<BlockId>,
    allocated_count: Arc<AtomicUsize>,
}

impl PagedKvPool {
    pub fn new(gpu: &GpuContext, config: PagedKvPoolConfig) -> Result<Self> {
        let total_elems = config.num_blocks * config.total_elements_per_block();
        let data = DeviceBuffer::<half::f16>::zeroed(gpu, total_elems)?;
        
        let (free_sender, free_blocks) = crossbeam_channel::unbounded();
        for i in 0..config.num_blocks {
            let _ = free_sender.send(BlockId(i));
        }

        Ok(Self {
            config,
            data,
            free_blocks,
            free_sender,
            allocated_count: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub fn allocate_block(&self) -> Option<BlockId> {
        if let Ok(block_id) = self.free_blocks.try_recv() {
            self.allocated_count.fetch_add(1, Ordering::Relaxed);
            Some(block_id)
        } else {
            None
        }
    }

    pub fn free_block(&self, block: BlockId) {
        let _ = self.free_sender.send(block);
        self.allocated_count.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn free_block_count(&self) -> usize {
        self.config.num_blocks.saturating_sub(self.allocated_count.load(Ordering::Relaxed))
    }

    pub fn config(&self) -> &PagedKvPoolConfig {
        &self.config
    }
}
