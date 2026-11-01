use anyhow::Result;
use oxide_infer_core::{DeviceBuffer, GpuContext, SafeTensorsLoader};
use crate::conv_layer::CausalConvLayer;
use crate::ffn::FfnLayer;

/// Represents one complete hybrid transformer layer (e.g. from LFM3-VL).
/// Layer execution: `x -> RMSNorm -> CausalConv / Attention -> Add Residual -> RMSNorm -> SwiGLU FFN -> Add Residual`
pub struct HybridTransformerLayer {
    pub conv_or_attn: CausalConvLayer,
    pub ffn: FfnLayer,
    pub hidden_size: usize,
}

impl HybridTransformerLayer {
    pub fn load_from_safetensors(
        gpu: &GpuContext,
        loader: &SafeTensorsLoader,
        layer_idx: usize,
        hidden_size: usize,
        intermediate_size: usize,
        conv_k_len: usize,
    ) -> Result<Self> {
        let conv_or_attn = CausalConvLayer::load_from_safetensors(gpu, loader, layer_idx, hidden_size, conv_k_len)?;
        let ffn = FfnLayer::load_from_safetensors(gpu, loader, layer_idx, hidden_size, intermediate_size)?;

        Ok(Self {
            conv_or_attn,
            ffn,
            hidden_size,
        })
    }

    /// Single-token decode step forward pass through the entire hybrid layer.
    pub fn forward(
        &mut self,
        gpu: &GpuContext,
        hidden_states: &mut DeviceBuffer<half::f16>,
    ) -> Result<()> {
        let mut conv_out = DeviceBuffer::<half::f16>::zeroed(gpu, self.hidden_size)?;
        let mut ffn_out = DeviceBuffer::<half::f16>::zeroed(gpu, self.hidden_size)?;

        // 1. Causal Conv Block
        self.conv_or_attn.forward(gpu, hidden_states, &mut conv_out)?;

        // 2. Residual connection: hidden_states += conv_out
        let stream = gpu.compute_stream();
        let mut h_host = hidden_states.inner().to_host_vec(stream)?;
        let c_host = conv_out.inner().to_host_vec(stream)?;
        for i in 0..self.hidden_size {
            let sum = h_host[i].to_f32() + c_host[i].to_f32();
            h_host[i] = half::f16::from_f32(sum);
        }
        *hidden_states = DeviceBuffer::from_slice(gpu, &h_host)?;

        // 3. Gated SwiGLU FFN Block
        self.ffn.forward(gpu, hidden_states, &mut ffn_out)?;

        // 4. Residual connection: hidden_states += ffn_out
        let f_host = ffn_out.inner().to_host_vec(stream)?;
        for i in 0..self.hidden_size {
            let sum = h_host[i].to_f32() + f_host[i].to_f32();
            h_host[i] = half::f16::from_f32(sum);
        }
        *hidden_states = DeviceBuffer::from_slice(gpu, &h_host)?;

        Ok(())
    }
}
