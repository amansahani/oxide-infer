use anyhow::{Context as AnyhowContext, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::fs::File;
use std::path::Path;
use std::collections::HashMap;
use crate::context::GpuContext;
use crate::memory::DeviceBuffer;

/// Zero-copy weights loader for SafeTensors files directly into GPU VRAM.
///
/// Supports plain F16/BF16 checkpoints and ggml-quantized checkpoints
/// produced by `quant_weights`. Quantized tensors are dequantized to BF16 on
/// the host during load, so callers always receive BF16 weights and no part
/// of the engine needs to know about the quantization layout.
pub struct SafeTensorsLoader {
    mmap: Mmap,
    quantized: Option<crate::quant::QuantType>,
}

impl SafeTensorsLoader {
    /// Open and memory-map a .safetensors model checkpoint file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open weights file {:?}", path.as_ref()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let quantized = Self::detect_quantized(&mmap);
        Ok(Self { mmap, quantized })
    }

    /// Whether the checkpoint carries a `__quantized__` marker.
    pub fn is_quantized(&self) -> bool {
        self.quantized.is_some()
    }

    /// The quantization type, if any.
    pub fn quant_type(&self) -> Option<crate::quant::QuantType> {
        self.quantized
    }

    fn detect_quantized(mmap: &Mmap) -> Option<crate::quant::QuantType> {
        let mut buf = [0u8; 8];
        if mmap.len() < 16 { return None; }
        buf.copy_from_slice(&mmap[..8]);
        let hdr_len = u64::from_le_bytes(buf) as usize;
        if hdr_len == 0 || 8 + hdr_len > mmap.len() { return None; }
        let hdr = &mmap[8..8 + hdr_len];
        let s = std::str::from_utf8(hdr).ok()?;
        let key = "__quantized__";
        let pos = s.find(key)?;
        let rest = &s[pos + key.len()..];
        // skip the colon and opening quote
        let rest = rest.trim_start_matches(&[':', '"', ' '][..]);
        let end = rest.find('"')?;
        crate::quant::QuantType::from_str(&rest[..end])
    }

    /// Return the raw packed bytes for a U8 tensor without copying or
    /// dequantizing. For quantized checkpoints this is the Q4_0 stream that
    /// `dequantize` produces. Callers use this when they want to dequantize
    /// on the host once (e.g. to build a scratch buffer) or to upload raw
    /// blocks for in-kernel dequantization.
    pub fn load_tensor_packed(
        &self,
        tensor_name: &str,
    ) -> Result<&[u8]> {
        let tensors = SafeTensors::deserialize(&self.mmap)
            .context("Failed to deserialize safetensors header")?;
        let tensor = tensors
            .tensor(tensor_name)
            .with_context(|| format!("Tensor '{}' not found in safetensors file", tensor_name))?;
        Ok(tensor.data())
    }

    /// Load a specific tensor from the safetensors file directly into a GPU `DeviceBuffer`.
    ///
    /// In a quantized checkpoint, U8 tensors are Q4_0 blocks and are
    /// dequantized to F16 on the host before upload; the caller cannot tell
    /// the difference in the returned buffer.
    pub fn load_tensor<T: cuda_core::DeviceCopy + bytemuck::Pod>(
        &self,
        gpu: &GpuContext,
        tensor_name: &str,
    ) -> Result<DeviceBuffer<T>> {
        let tensors = SafeTensors::deserialize(&self.mmap)
            .context("Failed to deserialize safetensors header")?;
        let tensor = tensors
            .tensor(tensor_name)
            .with_context(|| format!("Tensor '{}' not found in safetensors file", tensor_name))?;

        let bytes = tensor.data();
        if let Some(qt) = self.quantized {
            if tensor.dtype() == safetensors::tensor::Dtype::U8 {
                let f16_values = crate::quant::dequantize(bytes, qt);
                let slice: &[T] = bytemuck::cast_slice(&f16_values);
                return DeviceBuffer::from_slice(gpu, slice);
            }
        }
        // Copy to an aligned Vec before casting: mmap byte pointers at odd
        // offsets are not aligned for every Pod type, and cast_slice panics.
        let owned = bytes.to_vec();
        let slice: &[T] = bytemuck::cast_slice(&owned);
        DeviceBuffer::from_slice(gpu, slice)
    }

    /// List all tensor names available in the file.
    pub fn tensor_names(&self) -> Result<Vec<String>> {
        let tensors = SafeTensors::deserialize(&self.mmap)?;
        Ok(tensors.names().into_iter().map(String::from).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_safetensors_mmap_and_gpu_upload() {
        let gpu = GpuContext::init(0).expect("Init GPU");

        // Create a synthetic safetensors file
        let mut tmp_file = NamedTempFile::new().unwrap();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let bytes: &[u8] = bytemuck::cast_slice(&data);

        let mut tensor_map = HashMap::new();
        tensor_map.insert(
            "model.embed_tokens.weight".to_string(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![2, 4],
                bytes,
            ).unwrap(),
        );

        let serialized = safetensors::serialize(&tensor_map, &None).unwrap();
        tmp_file.write_all(&serialized).unwrap();
        tmp_file.flush().unwrap();

        let loader = SafeTensorsLoader::open(tmp_file.path()).expect("Open safetensors");
        let names = loader.tensor_names().unwrap();
        assert_eq!(names, vec!["model.embed_tokens.weight"]);

        let d_tensor = loader.load_tensor::<f32>(&gpu, "model.embed_tokens.weight").unwrap();
        assert_eq!(d_tensor.len(), 8);

        let host_vec = d_tensor.inner().to_host_vec(gpu.compute_stream()).unwrap();
        assert_eq!(host_vec, data);
    }
}
