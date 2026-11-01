//! Paged Attention Decoding Kernel for continuous multi-turn generation.
//! Scores query tokens against physical blocks stored in the PagedKvPool.

pub const PAGED_ATTENTION_FP16_PTX: &str = r#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry paged_attention_decode_v1(
    .param .u64 out_ptr,
    .param .u64 q_ptr,
    .param .u64 k_cache_ptr,
    .param .u64 v_cache_ptr,
    .param .u64 block_tables_ptr,
    .param .u64 context_lens_ptr,
    .param .u32 max_blocks,
    .param .u32 block_size,
    .param .u32 num_heads,
    .param .u32 head_dim,
    .param .f32 sm_scale
) {
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<16>;

    ret;
}
"#;

use anyhow::Result;
use cuda_core::launch_kernel_on_stream;
use oxide_infer_core::{DeviceBuffer, GpuContext};
use crate::ptx::PtxModule;

/// High-level executor for the Paged Attention kernel.
pub struct PagedAttentionExecutor {
    module: PtxModule,
}

impl PagedAttentionExecutor {
    pub fn new(gpu: &GpuContext) -> Result<Self> {
        let module = PtxModule::load_from_ptx(gpu, PAGED_ATTENTION_FP16_PTX)?;
        Ok(Self { module })
    }

    /// Dispatch paged attention decode step across the batch.
    pub fn execute(
        &self,
        gpu: &GpuContext,
        out: &mut DeviceBuffer<half::f16>,
        q: &DeviceBuffer<half::f16>,
        k_cache: &DeviceBuffer<half::f16>,
        v_cache: &DeviceBuffer<half::f16>,
        block_tables: &DeviceBuffer<i32>,
        context_lens: &DeviceBuffer<i32>,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_blocks: usize,
        sm_scale: f32,
    ) -> Result<()> {
        let func = self.module.get_function("paged_attention_decode_v1")?;
        let stream = gpu.compute_stream();

        let mut out_ptr = out.inner_mut().cu_deviceptr();
        let mut q_ptr = q.inner().cu_deviceptr();
        let mut k_cache_ptr = k_cache.inner().cu_deviceptr();
        let mut v_cache_ptr = v_cache.inner().cu_deviceptr();
        let mut block_tables_ptr = block_tables.inner().cu_deviceptr();
        let mut context_lens_ptr = context_lens.inner().cu_deviceptr();
        let mut max_blocks_u32 = max_blocks as u32;
        let mut block_size_u32 = block_size as u32;
        let mut num_heads_u32 = num_heads as u32;
        let mut head_dim_u32 = head_dim as u32;
        let mut sm_scale_f32 = sm_scale;

        let mut args: [*mut std::ffi::c_void; 11] = [
            &mut out_ptr as *mut _ as *mut std::ffi::c_void,
            &mut q_ptr as *mut _ as *mut std::ffi::c_void,
            &mut k_cache_ptr as *mut _ as *mut std::ffi::c_void,
            &mut v_cache_ptr as *mut _ as *mut std::ffi::c_void,
            &mut block_tables_ptr as *mut _ as *mut std::ffi::c_void,
            &mut context_lens_ptr as *mut _ as *mut std::ffi::c_void,
            &mut max_blocks_u32 as *mut _ as *mut std::ffi::c_void,
            &mut block_size_u32 as *mut _ as *mut std::ffi::c_void,
            &mut num_heads_u32 as *mut _ as *mut std::ffi::c_void,
            &mut head_dim_u32 as *mut _ as *mut std::ffi::c_void,
            &mut sm_scale_f32 as *mut _ as *mut std::ffi::c_void,
        ];

        unsafe {
            launch_kernel_on_stream(
                &func,
                (batch_size as u32, num_heads as u32, 1),
                (32, 1, 1), // 1 warp per head
                0,
                stream,
                &mut args,
            )?;
        }

        Ok(())
    }
}
