use anyhow::{Context as AnyhowContext, Result};
use cuda_core::{Context as CudaContext, Device, Stream};
use std::sync::Arc;

/// GPU Execution context managing device allocation and primary CUDA streams.
#[derive(Clone)]
pub struct GpuContext {
    device: Arc<Device>,
    context: Arc<CudaContext>,
    compute_stream: Arc<Stream>,
    dma_stream: Arc<Stream>,
}

impl GpuContext {
    /// Initialize primary CUDA context and dual streams for concurrent execution & async DMA.
    pub fn init(device_ordinal: usize) -> Result<Self> {
        let device = Arc::new(
            Device::new(device_ordinal)
                .with_context(|| format!("Failed to initialize CUDA device ordinal {}", device_ordinal))?,
        );
        let context = CudaContext::new(device_ordinal)
            .with_context(|| format!("Failed to create CUDA context for device {}", device_ordinal))?;
        
        let compute_stream = context
            .new_stream()
            .context("Failed to create compute stream")?;
        let dma_stream = context
            .new_stream()
            .context("Failed to create DMA transfer stream")?;

        Ok(Self {
            device,
            context,
            compute_stream,
            dma_stream,
        })
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    pub fn compute_stream(&self) -> &Arc<Stream> {
        &self.compute_stream
    }

    pub fn dma_stream(&self) -> &Arc<Stream> {
        &self.dma_stream
    }
}
