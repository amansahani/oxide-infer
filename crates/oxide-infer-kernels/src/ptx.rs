use anyhow::{Context as AnyhowContext, Result};
use cuda_core::{Function, Module};
use std::sync::Arc;
use oxide_infer_core::GpuContext;

/// Manages JIT compiled or AOT PTX modules for cuda-oxide execution.
pub struct PtxModule {
    module: Arc<Module>,
}

impl PtxModule {
    /// Load module from compiled PTX source string.
    pub fn load_from_ptx(gpu: &GpuContext, ptx_source: &str) -> Result<Self> {
        let module = gpu
            .context()
            .load_module_from_ptx_src(ptx_source)
            .context("Failed to load PTX module into CUDA context")?;
        Ok(Self { module })
    }

    /// Retrieve a compiled kernel entry point function.
    pub fn get_function(&self, name: &str) -> Result<Function> {
        self.module
            .load_function(name)
            .with_context(|| format!("Failed to find function '{}' in PTX module", name))
    }

    pub fn inner(&self) -> &Arc<Module> {
        &self.module
    }
}
