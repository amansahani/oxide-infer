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
