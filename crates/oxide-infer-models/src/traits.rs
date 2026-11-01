use anyhow::Result;
use oxide_infer_core::{DeviceBuffer, GpuContext};

/// Generic forward execution contract for any neural network architecture.
pub trait ModelArchitecture: Send + Sync {
    /// Forward pass taking input token embeddings or hidden states, returning new hidden states.
    fn forward(
        &self,
        gpu: &GpuContext,
        hidden_states: &mut DeviceBuffer<half::f16>,
        position: usize,
    ) -> Result<()>;
}
