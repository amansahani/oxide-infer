//! Multi-architecture cuda-oxide kernels: every PTX artifact is committed
//! to the repo (`kernels/oxide-kernels/*.ptx`, one per target) and the
//! loader picks the artifact matching the device's compute capability,
//! with sm_80 as the JIT fallback for newer unknown silicon.

use anyhow::{Context as AnyhowContext, Result};
use std::sync::OnceLock;

use cuda_core::{Stream, launch_kernel_on_stream};
use std::ffi::c_void;

use crate::PtxModule;
use oxide_infer_core::GpuContext;

pub const OXIDE_KERNELS_PTX_SM80: &str =
    include_str!("../../../kernels/oxide-kernels/oxide_kernels_sm80.ptx");
pub const OXIDE_KERNELS_PTX_SM90: &str =
    include_str!("../../../kernels/oxide-kernels/oxide_kernels_sm_90.ptx");
pub const OXIDE_KERNELS_PTX_SM100: &str =
    include_str!("../../../kernels/oxide-kernels/oxide_kernels_sm_100.ptx");
pub const OXIDE_KERNELS_PTX_SM103: &str =
    include_str!("../../../kernels/oxide-kernels/oxide_kernels_sm_103.ptx");
pub const OXIDE_KERNELS_PTX_SM110: &str =
    include_str!("../../../kernels/oxide-kernels/oxide_kernels_sm_110.ptx");
pub const OXIDE_KERNELS_PTX_SM120: &str =
    include_str!("../../../kernels/oxide-kernels/oxide_kernels_sm_120.ptx");
pub const OXIDE_KERNELS_PTX_SM121: &str =
    include_str!("../../../kernels/oxide-kernels/oxide_kernels_sm_121.ptx");

/// Detect the device's compute capability as `sm_XX` (e.g. `sm_86`,
/// `sm_90`) via `cuda_core`.
pub fn detect_sm(gpu: &GpuContext) -> String {
    static CACHED: OnceLock<String> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            let dev = gpu.device().clone();
            unsafe { cuda_core::get_device_sm_name(dev.cu_device()) }
                .unwrap_or_else(|_| "sm_80".to_string())
        })
        .clone()
}

/// Select the committed PTX artifact for `sm`. Unknown archs fall back to
/// sm_80 (which every later GPU can JIT-compile).
pub fn select_ptx(sm: &str) -> &'static str {
    match sm {
        "sm_90" | "sm_90a" => OXIDE_KERNELS_PTX_SM90,
        "sm_100" | "sm_100a" => OXIDE_KERNELS_PTX_SM100,
        "sm_103" | "sm_103a" => OXIDE_KERNELS_PTX_SM103,
        "sm_110" | "sm_110a" => OXIDE_KERNELS_PTX_SM110,
        "sm_120" | "sm_120a" => OXIDE_KERNELS_PTX_SM120,
        "sm_121" | "sm_121a" => OXIDE_KERNELS_PTX_SM121,
        _ => OXIDE_KERNELS_PTX_SM80,
    }
}

/// Loaded multi-arch kernel module for this process.
pub struct OxideKernels {
    module: PtxModule,
    arch: String,
}

impl OxideKernels {
    pub fn new(gpu: &GpuContext) -> Result<Self> {
        let arch = detect_sm(gpu);
        let module = PtxModule::load_from_ptx(gpu, select_ptx(&arch))
            .with_context(|| format!("oxide-kernels PTX load ({arch})"))?;
        Ok(Self { module, arch })
    }

    /// The PTX arch that was loaded (for logs / diagnostics).
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// Underlying module; callers pull `Function` handles out of it.
    /// The module stays alive as long as `self` lives.
    pub fn module(&self) -> &PtxModule {
        &self.module
    }

    /// Raw-pointer launch of `batched_rms_norm_vec8_fp16`.
    /// Params lower as (ptr, len) per slice; grid (1, B, 1), block (256, 1, 1).
    #[allow(clippy::too_many_arguments)]
    pub fn batched_rms_norm(
        &self,
        stream: &Stream,
        out_ptr: u64, out_len: u64,
        input_ptr: u64, input_len: u64,
        weight_ptr: u64, weight_len: u64,
        eps: f32, h_size: u32, b_size: u32,
    ) -> Result<()> {
        let func = self.module().get_function("batched_rms_norm_vec8_fp16")?;
        let mut out_ptr = out_ptr;
        let mut out_len = out_len;
        let mut input_ptr = input_ptr;
        let mut input_len = input_len;
        let mut weight_ptr = weight_ptr;
        let mut weight_len = weight_len;
        let mut eps = eps;
        let mut h_size = h_size;
        let mut b_size = b_size;
        let mut args: [*mut c_void; 9] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut out_len as *mut _ as *mut c_void,
            &mut input_ptr as *mut _ as *mut c_void,
            &mut input_len as *mut _ as *mut c_void,
            &mut weight_ptr as *mut _ as *mut c_void,
            &mut weight_len as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
            &mut h_size as *mut _ as *mut c_void,
            &mut b_size as *mut _ as *mut c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&func, (1, b_size, 1), (256, 1, 1), 0, stream, &mut args)?;
        }
        Ok(())
    }

    /// Raw-pointer launch of `batched_rope_fp16`.
    /// Params lower as (ptr, len) per slice; grid (grid_heads, B, 1), block (dim, 1, 1).
    #[allow(clippy::too_many_arguments)]
    pub fn batched_rope(
        &self,
        stream: &Stream,
        x_ptr: u64, x_len: u64,
        cos_ptr: u64, cos_len: u64,
        sin_ptr: u64, sin_len: u64,
        pos_ptr: u64, pos_len: u64,
        num_heads: u32, head_dim: u32, b_size: u32, grid_heads: u32,
    ) -> Result<()> {
        let func = self.module().get_function("batched_rope_fp16")?;
        let mut x_ptr = x_ptr;
        let mut x_len = x_len;
        let mut cos_ptr = cos_ptr;
        let mut cos_len = cos_len;
        let mut sin_ptr = sin_ptr;
        let mut sin_len = sin_len;
        let mut pos_ptr = pos_ptr;
        let mut pos_len = pos_len;
        let mut num_heads = num_heads;
        let mut head_dim = head_dim;
        let mut b_size = b_size;
        let mut args: [*mut c_void; 11] = [
            &mut x_ptr as *mut _ as *mut c_void,
            &mut x_len as *mut _ as *mut c_void,
            &mut cos_ptr as *mut _ as *mut c_void,
            &mut cos_len as *mut _ as *mut c_void,
            &mut sin_ptr as *mut _ as *mut c_void,
            &mut sin_len as *mut _ as *mut c_void,
            &mut pos_ptr as *mut _ as *mut c_void,
            &mut pos_len as *mut _ as *mut c_void,
            &mut num_heads as *mut _ as *mut c_void,
            &mut head_dim as *mut _ as *mut c_void,
            &mut b_size as *mut _ as *mut c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&func, (grid_heads, b_size, 1), (head_dim, 1, 1), 0, stream, &mut args)?;
        }
        Ok(())
    }

    /// Raw-pointer launch of `batched_swiglu_separate_fp16`.
    /// out[i] = up[i] * silu(gate[i]); grid (ceil(n/256), 1, 1), block (256, 1, 1).
    #[allow(clippy::too_many_arguments)]
    pub fn batched_swiglu_separate(
        &self,
        stream: &Stream,
        act_ptr: u64, act_len: u64,
        gate_ptr: u64, gate_len: u64,
        up_ptr: u64, up_len: u64,
        n: u32,
    ) -> Result<()> {
        let func = self.module().get_function("batched_swiglu_separate_fp16")?;
        let mut act_ptr = act_ptr;
        let mut act_len = act_len;
        let mut gate_ptr = gate_ptr;
        let mut gate_len = gate_len;
        let mut up_ptr = up_ptr;
        let mut up_len = up_len;
        let mut n = n;
        let mut args: [*mut c_void; 7] = [
            &mut act_ptr as *mut _ as *mut c_void,
            &mut act_len as *mut _ as *mut c_void,
            &mut gate_ptr as *mut _ as *mut c_void,
            &mut gate_len as *mut _ as *mut c_void,
            &mut up_ptr as *mut _ as *mut c_void,
            &mut up_len as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&func, (n.div_ceil(256), 1, 1), (256, 1, 1), 0, stream, &mut args)?;
        }
        Ok(())
    }

    /// Raw-pointer launch of `batched_argmax_direct_f16`.
    /// grid (B, 1, 1), block (256, 1, 1).
    #[allow(clippy::too_many_arguments)]
    pub fn batched_argmax(
        &self,
        stream: &Stream,
        out_ptr: u64, out_len: u64,
        logits_ptr: u64, logits_len: u64,
        vocab_size: u32, b_size: u32,
    ) -> Result<()> {
        let func = self.module().get_function("batched_argmax_direct_f16")?;
        let mut out_ptr = out_ptr;
        let mut out_len = out_len;
        let mut logits_ptr = logits_ptr;
        let mut logits_len = logits_len;
        let mut vocab_size = vocab_size;
        let mut args: [*mut c_void; 5] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut out_len as *mut _ as *mut c_void,
            &mut logits_ptr as *mut _ as *mut c_void,
            &mut logits_len as *mut _ as *mut c_void,
            &mut vocab_size as *mut _ as *mut c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&func, (b_size, 1, 1), (256, 1, 1), 0, stream, &mut args)?;
        }
        Ok(())
    }

    /// Raw-pointer launch of `embedding_lookup_fp16`.
    /// out[b*dim + i] = f16(table[tokens[b]*dim + i]); grid (B, 1, 1), block (256, 1, 1).
    #[allow(clippy::too_many_arguments)]
    pub fn embedding_lookup(
        &self,
        stream: &Stream,
        out_ptr: u64, out_len: u64,
        table_ptr: u64, table_len: u64,
        tokens_ptr: u64, tokens_len: u64,
        dim: u32, b_size: u32,
    ) -> Result<()> {
        let func = self.module().get_function("embedding_lookup_fp16")?;
        let mut out_ptr = out_ptr;
        let mut out_len = out_len;
        let mut table_ptr = table_ptr;
        let mut table_len = table_len;
        let mut tokens_ptr = tokens_ptr;
        let mut tokens_len = tokens_len;
        let mut dim = dim;
        let mut b_size = b_size;
        let mut args: [*mut c_void; 8] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut out_len as *mut _ as *mut c_void,
            &mut table_ptr as *mut _ as *mut c_void,
            &mut table_len as *mut _ as *mut c_void,
            &mut tokens_ptr as *mut _ as *mut c_void,
            &mut tokens_len as *mut _ as *mut c_void,
            &mut dim as *mut _ as *mut c_void,
            &mut b_size as *mut _ as *mut c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&func, (b_size, 1, 1), (256, 1, 1), 0, stream, &mut args)?;
        }
        Ok(())
    }

    /// Raw-pointer launch of `residual_add_fp16`: hidden += ffn_out (f32 add, f16 out).
    #[allow(clippy::too_many_arguments)]
    pub fn residual_add(
        &self,
        stream: &Stream,
        hidden_ptr: u64, hidden_len: u64,
        ffn_ptr: u64, ffn_len: u64,
        h_size: u32, b_size: u32,
    ) -> Result<()> {
        let func = self.module().get_function("residual_add_fp16")?;
        let mut hidden_ptr = hidden_ptr;
        let mut hidden_len = hidden_len;
        let mut ffn_ptr = ffn_ptr;
        let mut ffn_len = ffn_len;
        let mut h_size = h_size;
        let mut b_size = b_size;
        let mut args: [*mut c_void; 6] = [
            &mut hidden_ptr as *mut _ as *mut c_void,
            &mut hidden_len as *mut _ as *mut c_void,
            &mut ffn_ptr as *mut _ as *mut c_void,
            &mut ffn_len as *mut _ as *mut c_void,
            &mut h_size as *mut _ as *mut c_void,
            &mut b_size as *mut _ as *mut c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&func, (1, b_size, 1), (256, 1, 1), 0, stream, &mut args)?;
        }
        Ok(())
    }
}
