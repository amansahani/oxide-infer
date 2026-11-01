use anyhow::{Context as AnyhowContext, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::fs::File;
use std::path::Path;
use std::collections::HashMap;
use crate::context::GpuContext;
use crate::memory::DeviceBuffer;

/// Zero-copy weights loader for SafeTensors files directly into GPU VRAM.
pub struct SafeTensorsLoader {
    mmap: Mmap,
}

impl SafeTensorsLoader {
    /// Open and memory-map a .safetensors model checkpoint file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open weights file {:?}", path.as_ref()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap })
    }

    /// Load a specific tensor from the safetensors file directly into a GPU `DeviceBuffer`.
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
        let slice: &[T] = bytemuck::cast_slice(bytes);
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
