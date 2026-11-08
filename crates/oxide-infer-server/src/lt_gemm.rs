//! Batch-shape-invariant FP16 GEMM on top of cuBLASLt.
//!
//! `cublasHgemm` selects a different kernel (and therefore a different
//! reduction order) depending on the number of columns `n`, which makes token
//! output depend on how many sequences happen to be in a batch. To make the
//! engine numerically invariant we select one algorithm per `(m, k)` weight
//! shape once at startup and then reuse it for every `n`. The chosen algorithm
//! produces bitwise-identical per-column results for all batch sizes, so no
//! padding is needed and throughput is unaffected.

use anyhow::{anyhow, Result};
use cuda_core::Stream;
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;

pub type CublasStatus = i32;
type Handle = *mut c_void;
type Desc = *mut c_void;

const CUDA_R_16F: i32 = 2;
const CUDA_R_32F: i32 = 0;
const CUBLAS_COMPUTE_16F: i32 = 64;
const CUBLAS_COMPUTE_32F: i32 = 68;
const CUBLAS_COMPUTE_32F_FAST_16F: i32 = 74;
const CUBLAS_OP_T: i32 = 1;

const CUBLASLT_MATMUL_DESC_TRANSA: i32 = 3;
const CUBLASLT_MATRIX_LAYOUT_COLS: i32 = 3;
const CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: i32 = 1;

/// Maximum workspace handed to cuBLASLt. The model's largest heuristic request
/// is well under 1 MiB; 8 MiB leaves headroom for other GPUs/architectures.
const WORKSPACE_BYTES: usize = 8 << 20;

#[repr(C)]
#[derive(Clone, Copy)]
struct LtAlgo {
    data: [u64; 8],
}

#[repr(C)]
struct LtHeuristic {
    algo: LtAlgo,
    workspace_size: usize,
    state: CublasStatus,
    waves_count: f32,
    reserved: [i32; 4],
}

#[link(name = "cublasLt")]
extern "C" {
    fn cublasLtCreate(handle: *mut Handle) -> CublasStatus;
    fn cublasLtMatmulDescCreate(desc: *mut Desc, compute: i32, scale: i32) -> CublasStatus;
    fn cublasLtMatmulDescSetAttribute(
        desc: Desc,
        attr: i32,
        value: *const c_void,
        size: usize,
    ) -> CublasStatus;
    fn cublasLtMatrixLayoutCreate(
        layout: *mut Desc,
        dtype: i32,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> CublasStatus;
    fn cublasLtMatrixLayoutSetAttribute(
        layout: Desc,
        attr: i32,
        value: *const c_void,
        size: usize,
    ) -> CublasStatus;
    fn cublasLtMatmulPreferenceCreate(pref: *mut Desc) -> CublasStatus;
    fn cublasLtMatmulPreferenceSetAttribute(
        pref: Desc,
        attr: i32,
        value: *const c_void,
        size: usize,
    ) -> CublasStatus;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmulAlgoGetHeuristic(
        handle: Handle,
        desc: Desc,
        a: Desc,
        b: Desc,
        c: Desc,
        d: Desc,
        pref: Desc,
        requested: i32,
        results: *mut LtHeuristic,
        count: *mut i32,
    ) -> CublasStatus;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmul(
        handle: Handle,
        desc: Desc,
        alpha: *const c_void,
        a: *const c_void,
        a_desc: Desc,
        b: *const c_void,
        b_desc: Desc,
        beta: *const c_void,
        c: *const c_void,
        c_desc: Desc,
        d: *mut c_void,
        d_desc: Desc,
        algo: *const LtAlgo,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> CublasStatus;
    fn cublasLtMatmulDescDestroy(desc: Desc) -> CublasStatus;
    fn cublasLtMatrixLayoutDestroy(layout: Desc) -> CublasStatus;
    fn cublasLtMatmulPreferenceDestroy(pref: Desc) -> CublasStatus;
    fn cublasLtDestroy(handle: Handle) -> CublasStatus;
}

fn status(code: CublasStatus, what: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(anyhow!("{what} failed with cublasLt status {code}"))
    }
}

struct LtConfig {
    a_desc: Desc,
    b_desc: Desc,
    c_desc: Desc,
    algo: LtAlgo,
    /// Column count the B/C descriptors are currently configured for.
    cols: u64,
}

/// One cuBLASLt handle with a per-shape cache of deterministic algorithms.
pub struct LtGemm {
    handle: Handle,
    op_desc: Desc,
    /// Scale type of alpha/beta: `CUDA_R_32F` for FP32 compute.
    scale_type: i32,
    workspace: DeviceBuffer<u8>,
    workspace_size: usize,
    configs: HashMap<(usize, usize), LtConfig>,
    max_batch: usize,
}

impl LtGemm {
    pub fn new(gpu: &GpuContext, max_batch: usize) -> Result<Self> {
        let mut handle = ptr::null_mut();
        status(unsafe { cublasLtCreate(&mut handle) }, "cublasLtCreate")?;

        // FP16 accumulation is the historical default. FP32 accumulation is
        // available for higher fidelity and may match the reference GEMV path
        // more closely.
        let compute_name = std::env::var("OXIDE_LT_COMPUTE").unwrap_or_else(|_| "32F".into());
        let (compute, scale_type) = match compute_name.as_str() {
            "16F" => (CUBLAS_COMPUTE_16F, CUDA_R_16F),
            "32F" => (CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "32F_FAST_16F" => (CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F),
            other => return Err(anyhow!("unknown OXIDE_LT_COMPUTE value: {other}")),
        };

        let mut op_desc = ptr::null_mut();
        status(
            unsafe { cublasLtMatmulDescCreate(&mut op_desc, compute, scale_type) },
            "cublasLtMatmulDescCreate",
        )?;
        let op_t = CUBLAS_OP_T;
        status(
            unsafe {
                cublasLtMatmulDescSetAttribute(
                    op_desc,
                    CUBLASLT_MATMUL_DESC_TRANSA,
                    &op_t as *const i32 as *const c_void,
                    std::mem::size_of::<i32>(),
                )
            },
            "cublasLtMatmulDescSetAttribute(TRANSA)",
        )?;

        let workspace = DeviceBuffer::<u8>::zeroed(gpu, WORKSPACE_BYTES)
            .map_err(|e| anyhow!("cublasLt workspace alloc: {e}"))?;

        Ok(Self {
            handle,
            op_desc,
            scale_type,
            workspace,
            workspace_size: WORKSPACE_BYTES,
            configs: HashMap::new(),
            max_batch,
        })
    }

    fn create_config(&self, m: usize, k: usize) -> Result<LtConfig> {
        // The algorithm is fixed for this shape, so its tuning point decides
        // both invariance and efficiency. Allow overriding it for benchmarks;
        // the default favours the common small/medium batch range.
        let algo_n = std::env::var("OXIDE_LT_ALGO_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(32)
            .clamp(1, self.max_batch) as u64;
        let mut a_desc = ptr::null_mut();
        let mut b_desc = ptr::null_mut();
        let mut c_desc = ptr::null_mut();
        // A is the row-major [m, k] weight matrix viewed as column-major
        // [k, m]; TRANSA turns it into the m x k operand.
        status(
            unsafe { cublasLtMatrixLayoutCreate(&mut a_desc, CUDA_R_16F, k as u64, m as u64, k as i64) },
            "cublasLtMatrixLayoutCreate(A)",
        )?;
        status(
            unsafe {
                cublasLtMatrixLayoutCreate(
                    &mut b_desc,
                    CUDA_R_16F,
                    k as u64,
                    algo_n,
                    k as i64,
                )
            },
            "cublasLtMatrixLayoutCreate(B)",
        )?;
        status(
            unsafe {
                cublasLtMatrixLayoutCreate(
                    &mut c_desc,
                    CUDA_R_16F,
                    m as u64,
                    algo_n,
                    m as i64,
                )
            },
            "cublasLtMatrixLayoutCreate(C)",
        )?;

        let mut pref = ptr::null_mut();
        status(
            unsafe { cublasLtMatmulPreferenceCreate(&mut pref) },
            "cublasLtMatmulPreferenceCreate",
        )?;
        let ws = self.workspace_size;
        status(
            unsafe {
                cublasLtMatmulPreferenceSetAttribute(
                    pref,
                    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &ws as *const usize as *const c_void,
                    std::mem::size_of::<usize>(),
                )
            },
            "cublasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE)",
        )?;

        // Pick one algorithm per weight shape; the same reduction order is
        // then used for every real batch size.
        let mut result = LtHeuristic {
            algo: LtAlgo { data: [0; 8] },
            workspace_size: 0,
            state: -1,
            waves_count: 0.0,
            reserved: [0; 4],
        };
        let mut count = 0i32;
        let code = unsafe {
            cublasLtMatmulAlgoGetHeuristic(
                self.handle,
                self.op_desc,
                a_desc,
                b_desc,
                c_desc,
                c_desc,
                pref,
                1,
                &mut result,
                &mut count,
            )
        };
        unsafe {
            cublasLtMatmulPreferenceDestroy(pref);
        }
        status(code, "cublasLtMatmulAlgoGetHeuristic")?;
        if count < 1 || result.state != 0 {
            unsafe {
                cublasLtMatrixLayoutDestroy(a_desc);
                cublasLtMatrixLayoutDestroy(b_desc);
                cublasLtMatrixLayoutDestroy(c_desc);
            }
            return Err(anyhow!(
                "no usable cuBLASLt algorithm for m={m} k={k} (state {})",
                result.state
            ));
        }
        if result.workspace_size > self.workspace_size {
            return Err(anyhow!(
                "cuBLASLt algorithm for m={m} k={k} needs {} workspace bytes, have {}",
                result.workspace_size,
                self.workspace_size
            ));
        }

        Ok(LtConfig {
            a_desc,
            b_desc,
            c_desc,
            algo: result.algo,
            cols: algo_n,
        })
    }

    /// Compute `C[m, n] = A[m, k] * B[k, n]` for column-major operands.
    pub fn gemm(
        &mut self,
        stream: &Stream,
        m: usize,
        k: usize,
        n: usize,
        a: u64,
        b: u64,
        c: u64,
    ) -> Result<()> {
        if !self.configs.contains_key(&(m, k)) {
            let config = self.create_config(m, k)?;
            self.configs.insert((m, k), config);
        }
        let config = self.configs.get_mut(&(m, k)).expect("config just inserted");

        if config.cols != n as u64 {
            let cols = n as u64;
            status(
                unsafe {
                    cublasLtMatrixLayoutSetAttribute(
                        config.b_desc,
                        CUBLASLT_MATRIX_LAYOUT_COLS,
                        &cols as *const u64 as *const c_void,
                        std::mem::size_of::<u64>(),
                    )
                },
                "cublasLtMatrixLayoutSetAttribute(B cols)",
            )?;
            status(
                unsafe {
                    cublasLtMatrixLayoutSetAttribute(
                        config.c_desc,
                        CUBLASLT_MATRIX_LAYOUT_COLS,
                        &cols as *const u64 as *const c_void,
                        std::mem::size_of::<u64>(),
                    )
                },
                "cublasLtMatrixLayoutSetAttribute(C cols)",
            )?;
            config.cols = cols;
        }

        let (alpha_f16, beta_f16) = (f16::from_f32(1.0), f16::from_f32(0.0));
        let (alpha_f32, beta_f32) = (1.0f32, 0.0f32);
        let (alpha_ptr, beta_ptr) = if self.scale_type == CUDA_R_32F {
            (
                &alpha_f32 as *const f32 as *const c_void,
                &beta_f32 as *const f32 as *const c_void,
            )
        } else {
            (
                &alpha_f16 as *const f16 as *const c_void,
                &beta_f16 as *const f16 as *const c_void,
            )
        };
        status(
            unsafe {
                cublasLtMatmul(
                    self.handle,
                    self.op_desc,
                    alpha_ptr,
                    a as *const c_void,
                    config.a_desc,
                    b as *const c_void,
                    config.b_desc,
                    beta_ptr,
                    c as *const c_void,
                    config.c_desc,
                    c as *mut c_void,
                    config.c_desc,
                    &config.algo,
                    self.workspace.inner().cu_deviceptr() as *mut c_void,
                    self.workspace_size,
                    stream.cu_stream() as *mut c_void,
                )
            },
            "cublasLtMatmul",
        )
    }
}

impl Drop for LtGemm {
    fn drop(&mut self) {
        unsafe {
            for (_, config) in self.configs.drain() {
                cublasLtMatrixLayoutDestroy(config.a_desc);
                cublasLtMatrixLayoutDestroy(config.b_desc);
                cublasLtMatrixLayoutDestroy(config.c_desc);
            }
            cublasLtMatmulDescDestroy(self.op_desc);
            cublasLtDestroy(self.handle);
        }
    }
}
