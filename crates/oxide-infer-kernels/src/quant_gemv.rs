//! Dispatch for cuda-oxide-compiled packed-weight GEMV kernels.
//!
//! The engine keeps the ggml-packed weight bytes in VRAM for quantized
//! checkpoints (instead of only dequantized f16 copies) and launches these
//! kernels for the hot decode-time GEMVs. Output layout and batch invariance
//! match the rest of the engine: one CTA per (row, batch-element), a
//! deterministic f32 reduction tree per CTA.
//!
//! The PTX is committed at `kernels/quant-gemv/gemv.ptx` (regenerate it with
//! `cargo-oxide run` there). All 8 simple ggml variants are supported;
//! K-types return `false` from `supported::<Q>` and callers must fall back to
//! the dequantize-at-load path.

use anyhow::{bail, Result};
use cuda_core::{launch_kernel_on_stream, Function};
use std::collections::HashMap;
use std::ffi::c_void;

use crate::PtxModule;
use cuda_core::Stream;
use oxide_infer_core::GpuContext;
use oxide_infer_core::quant::QuantType;

/// cuda-oxide-generated packed-weight GEMV kernels (committed artifact).
pub const QUANT_GEMV_PTX: &str = include_str!("../../../kernels/quant-gemv/gemv.ptx");

/// Which GGML quantized types have packed kernels available.
pub fn supported(qt: QuantType) -> bool {
    matches!(
        qt,
        QuantType::Q1_0
            | QuantType::Q2_0
            | QuantType::Q4_0
            | QuantType::Q4_1
            | QuantType::Q5_0
            | QuantType::Q5_1
            | QuantType::Q8_0
            | QuantType::Q8_1
    )
}

fn block_elems(qt: QuantType) -> u32 {
    match qt {
        QuantType::Q1_0 => 128,
        QuantType::Q2_0 => 64,
        _ => 32,
    }
}

fn dual_name(qt: QuantType) -> &'static str {
    match qt {
        QuantType::Q1_0 => "fused_dual_gemv_swiglu_q1_0",
        QuantType::Q2_0 => "fused_dual_gemv_swiglu_q2_0",
        QuantType::Q4_0 => "fused_dual_gemv_swiglu_q4_0",
        QuantType::Q4_1 => "fused_dual_gemv_swiglu_q4_1",
        QuantType::Q5_0 => "fused_dual_gemv_swiglu_q5_0",
        QuantType::Q5_1 => "fused_dual_gemv_swiglu_q5_1",
        QuantType::Q8_0 => "fused_dual_gemv_swiglu_q8_0",
        QuantType::Q8_1 => "fused_dual_gemv_swiglu_q8_1",
        _ => unreachable!("unsupported quant type for packed GEMV"),
    }
}

fn gemv_name(qt: QuantType) -> &'static str {
    match qt {
        QuantType::Q1_0 => "gemv_q1_0",
        QuantType::Q2_0 => "gemv_q2_0",
        QuantType::Q4_0 => "gemv_q4_0",
        QuantType::Q4_1 => "gemv_q4_1",
        QuantType::Q5_0 => "gemv_q5_0",
        QuantType::Q5_1 => "gemv_q5_1",
        QuantType::Q8_0 => "gemv_q8_0",
        QuantType::Q8_1 => "gemv_q8_1",
        _ => unreachable!("unsupported quant type for packed GEMV"),
    }
}

/// Loaded cuda-oxide packed GEMV module, one per engine.
pub struct QuantGemv {
    /// Kept alive so the loaded CUDA module outlives the cached Function handles.
    #[allow(dead_code)]
    module: PtxModule,
    dual: HashMap<u8, Function>,
    plain: HashMap<u8, Function>,
}

impl QuantGemv {
    pub fn new(gpu: &GpuContext) -> Result<Self> {
        let module = PtxModule::load_from_ptx(gpu, QUANT_GEMV_PTX)?;
        let mut dual = HashMap::new();
        let mut plain = HashMap::new();
        for &qt in &[
            QuantType::Q1_0,
            QuantType::Q2_0,
            QuantType::Q4_0,
            QuantType::Q4_1,
            QuantType::Q5_0,
            QuantType::Q5_1,
            QuantType::Q8_0,
            QuantType::Q8_1,
        ] {
            dual.insert(qt as u8, module.get_function(dual_name(qt))?);
            plain.insert(qt as u8, module.get_function(gemv_name(qt))?);
        }
        Ok(Self { module, dual, plain })
    }

    /// Packed dual GEMV (W1, W3) + SwiGLU -> activated_buf.
    /// Layout: x is [b * hidden] f16, act is [b * n_elems] f16.
    #[allow(clippy::too_many_arguments)]
    pub fn dual_swiglu(
        &self,
        stream: &Stream,
        qt: QuantType,
        w1_ptr_and_len: (u64, u64),
        w3_ptr_and_len: (u64, u64),
        x_ptr_and_len: (u64, u64),
        act_ptr_and_len: (u64, u64),
        hidden: u32,
        n_elems: u32,
        b: u32,
    ) -> Result<()> {
        let Some(func) = self.dual.get(&(qt as u8)) else {
            bail!("packed GEMV kernel not available for {:?}", qt);
        };
        let hidden_elems = block_elems(qt);
        let n_blocks = hidden / hidden_elems;

        let (mut w1_ptr, mut w1_len) = w1_ptr_and_len;
        let (mut w3_ptr, mut w3_len) = w3_ptr_and_len;
        let (mut x_ptr, mut x_len) = x_ptr_and_len;
        let (mut act_ptr, mut act_len) = act_ptr_and_len;
        let mut hidden = hidden;
        let mut n_blocks = n_blocks;
        let mut n_elems = n_elems;

        // SAFETY: pointers reference live DeviceBuffers; lengths match the
        // packed layouts; grid covers (n_elems rows, b sequences).
        let mut args: [*mut c_void; 11] = [
            &mut w1_ptr as *mut _ as *mut c_void,
            &mut w1_len as *mut _ as *mut c_void,
            &mut w3_ptr as *mut _ as *mut c_void,
            &mut w3_len as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut x_len as *mut _ as *mut c_void,
            &mut act_ptr as *mut _ as *mut c_void,
            &mut act_len as *mut _ as *mut c_void,
            &mut hidden as *mut _ as *mut c_void,
            &mut n_blocks as *mut _ as *mut c_void,
            &mut n_elems as *mut _ as *mut c_void,
        ];
        unsafe {
            launch_kernel_on_stream(
                func,
                (n_elems, b, 1),
                (128, 1, 1),
                0,
                stream,
                &mut args,
            )?;
        }
        Ok(())
    }

    /// Plain packed GEMV (used for the W2 down projection).
    pub fn gemv(
        &self,
        stream: &Stream,
        qt: QuantType,
        w_ptr_and_len: (u64, u64),
        x_ptr_and_len: (u64, u64),
        y_ptr_and_len: (u64, u64),
        hidden: u32,
        n_elems: u32,
        b: u32,
    ) -> Result<()> {
        let Some(func) = self.plain.get(&(qt as u8)) else {
            bail!("packed GEMV kernel not available for {:?}", qt);
        };
        let block_elems = block_elems(qt);
        let n_blocks = hidden / block_elems;

        let (mut w_ptr, mut w_len) = w_ptr_and_len;
        let (mut x_ptr, mut x_len) = x_ptr_and_len;
        let (mut y_ptr, mut y_len) = y_ptr_and_len;
        let mut hidden = hidden;
        let mut n_blocks = n_blocks;
        let mut n_elems = n_elems;

        // SAFETY: pointers reference live DeviceBuffers.
        let mut args: [*mut c_void; 9] = [
            &mut w_ptr as *mut _ as *mut c_void,
            &mut w_len as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut x_len as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut y_len as *mut _ as *mut c_void,
            &mut hidden as *mut _ as *mut c_void,
            &mut n_blocks as *mut _ as *mut c_void,
            &mut n_elems as *mut _ as *mut c_void,
        ];
        unsafe {
            launch_kernel_on_stream(
                func,
                (n_elems, b, 1),
                (128, 1, 1),
                0,
                stream,
                &mut args,
            )?;
        }
        Ok(())
    }
}
