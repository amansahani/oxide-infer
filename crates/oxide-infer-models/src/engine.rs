use anyhow::Result;
use oxide_infer_core::{DeviceBuffer, GpuContext, SafeTensorsLoader};
use crate::hybrid_layer::HybridTransformerLayer;

/// Complete Pipeline Engine for executing sequential layers of the model.
pub struct ModelEngine {
    pub layers: Vec<HybridTransformerLayer>,
    pub hidden_size: usize,
}

impl ModelEngine {
    pub fn new(layers: Vec<HybridTransformerLayer>, hidden_size: usize) -> Self {
        Self { layers, hidden_size }
    }

    /// Forward pass through all layers for a single-token decode iteration.
    pub fn forward_step(
        &mut self,
        gpu: &GpuContext,
        hidden_states: &mut DeviceBuffer<half::f16>,
    ) -> Result<()> {
        for layer in self.layers.iter_mut() {
            layer.forward(gpu, hidden_states)?;
        }
        Ok(())
    }
}
