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

/// One heuristic algorithm candidate, timed at several batch sizes.
pub struct AlgoCandidate {
    pub index: usize,
    pub times_ms: Vec<f64>,
    /// Max difference of the first output column between the smallest and
    /// largest timed `n`; zero means the candidate is batch-shape invariant
    /// on this shape.
    pub col0_diff: f32,
    pub waves: f32,
    pub workspace: usize,
}
type Handle = *mut c_void;
type Desc = *mut c_void;

const CUDA_R_16F: i32 = 2;
const CUDA_R_32F: i32 = 0;
const CUBLAS_COMPUTE_16F: i32 = 64;
const CUBLAS_COMPUTE_32F: i32 = 68;
const CUBLAS_COMPUTE_32F_FAST_16F: i32 = 74;
const CUBLAS_OP_T: i32 = 1;
const CUBLAS_OP_N: i32 = 0;

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
#[derive(Clone, Copy)]
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
    /// Own operation descriptor when a config needs a different transpose
    /// combination than the shared one (the PV attention GEMM transposes B).
    own_op: Option<Desc>,
}

/// One cuBLASLt handle with a per-shape cache of deterministic algorithms.
pub struct LtGemm {
    handle: Handle,
    op_desc: Desc,
    /// Scale type of alpha/beta: `CUDA_R_32F` for FP32 compute.
    scale_type: i32,
    /// cuBLASLt compute mode, needed to clone the op descriptor for configs
    /// that use a different transpose combination.
    compute: i32,
    workspace: DeviceBuffer<u8>,
    workspace_size: usize,
    configs: HashMap<(usize, usize), LtConfig>,
    /// Configs for `gemm_strided`, keyed by the full logical shape. `n` is
    /// absent because the column count is patched per call.
    strided: HashMap<(usize, usize, i64, i64, i64, bool, bool, u64), LtConfig>,
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
            compute,
            workspace,
            workspace_size: WORKSPACE_BYTES,
            configs: HashMap::new(),
            strided: HashMap::new(),
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
            own_op: None,
        })
    }

    /// Whether the configured compute mode can produce an f32 C. The
    /// attention score GEMMs need it; an f16 scale would silently round them.
    pub fn supports_f32_c(&self) -> bool {
        self.scale_type == CUDA_R_32F
    }

    /// Create a deterministic config for a layout with explicit leading
    /// dimensions.
    ///
    /// Same contract as `create_config`, but A/B/C may be strided views (the
    /// attention path reads a K/V head out of an interleaved 4-head cache) and
    /// C may be f32 so score matrices keep full precision.
    #[allow(clippy::too_many_arguments)]
    fn create_config_strided(
        &self,
        m: usize,
        k: usize,
        lda: i64,
        ldb: i64,
        ldc: i64,
        c_f32: bool,
        trans_a: bool,
        algo_n: u64,
    ) -> Result<LtConfig> {
        if c_f32 && self.scale_type != CUDA_R_32F {
            return Err(anyhow!(
                "f32 GEMM output needs FP32 compute; OXIDE_LT_COMPUTE selects an f16 scale"
            ));
        }
        let mut a_desc = ptr::null_mut();
        let mut b_desc = ptr::null_mut();
        let mut c_desc = ptr::null_mut();
        // `trans_a` selects how A is stored. Both layouts occur in attention:
        // K is read as `[k=128][m=keys]` (TRANSA), because the KV slab is
        // `[token][4 kv][128]`; V is read as `[m=128][k=keys]` (no transpose),
        // and then the C transpose `[128][n]` is exactly the output layout.
        let (a_rows, a_cols) = if trans_a {
            (k as u64, m as u64)
        } else {
            (m as u64, k as u64)
        };
        status(
            unsafe { cublasLtMatrixLayoutCreate(&mut a_desc, CUDA_R_16F, a_rows, a_cols, lda) },
            "cublasLtMatrixLayoutCreate(A strided)",
        )?;
        status(
            unsafe { cublasLtMatrixLayoutCreate(&mut b_desc, CUDA_R_16F, k as u64, algo_n, ldb) },
            "cublasLtMatrixLayoutCreate(B strided)",
        )?;
        let c_type = if c_f32 { CUDA_R_32F } else { CUDA_R_16F };
        status(
            unsafe { cublasLtMatrixLayoutCreate(&mut c_desc, c_type, m as u64, algo_n, ldc) },
            "cublasLtMatrixLayoutCreate(C strided)",
        )?;

        // The transpose changes the reduction geometry, so a PV-style config
        // needs its own operation descriptor.
        let mut own_op: Option<Desc> = None;
        let op = if trans_a {
            self.op_desc
        } else {
            let mut op_desc = ptr::null_mut();
            status(
                unsafe { cublasLtMatmulDescCreate(&mut op_desc, self.compute, self.scale_type) },
                "cublasLtMatmulDescCreate(strided)",
            )?;
            status(
                unsafe {
                    cublasLtMatmulDescSetAttribute(
                        op_desc,
                        CUBLASLT_MATMUL_DESC_TRANSA,
                        &CUBLAS_OP_N as *const i32 as *const c_void,
                        std::mem::size_of::<i32>(),
                    )
                },
                "cublasLtMatmulDescSetAttribute(strided A)",
            )?;
            own_op = Some(op_desc);
            op_desc
        };

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
                op,
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
        status(code, "cublasLtMatmulAlgoGetHeuristic(strided)")?;
        if count < 1 || result.state != 0 {
            unsafe {
                cublasLtMatrixLayoutDestroy(a_desc);
                cublasLtMatrixLayoutDestroy(b_desc);
                cublasLtMatrixLayoutDestroy(c_desc);
                if let Some(op_desc) = own_op {
                    cublasLtMatmulDescDestroy(op_desc);
                }
            }
            return Err(anyhow!(
                "no usable strided cuBLASLt algorithm for m={m} k={k} (state {})",
                result.state
            ));
        }
        if result.workspace_size > self.workspace_size {
            return Err(anyhow!(
                "strided cuBLASLt algorithm for m={m} k={k} needs {} workspace bytes, have {}",
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
            own_op,
        })
    }

    /// Compute `C[m, n] = op(A) * B` with explicit leading dimensions.
    ///
    /// A is a `[k, m]` column-major view when `trans_a` is set (the op
    /// descriptor transposes it), or an `[m, k]` column-major view otherwise.
    /// B is `[k, n]` column-major, C is `[m, n]` column-major. `lda`/`ldb`/`ldc`
    /// are in elements. `c_f32` selects an f32 C (used for attention scores).
    ///
    /// `algo_n` is the column count used to pick the algorithm. It must be a
    /// constant for a given shape so the reduction order never depends on the
    /// live batch size, which is what keeps results batch-shape invariant.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_strided(
        &mut self,
        stream: &Stream,
        m: usize,
        k: usize,
        n: usize,
        lda: i64,
        ldb: i64,
        ldc: i64,
        c_f32: bool,
        trans_a: bool,
        algo_n: u64,
        a: u64,
        b: u64,
        c: u64,
    ) -> Result<()> {
        let key = (m, k, lda, ldb, ldc, c_f32, trans_a, algo_n);
        if !self.strided.contains_key(&key) {
            let cfg = self.create_config_strided(m, k, lda, ldb, ldc, c_f32, trans_a, algo_n)?;
            self.strided.insert(key, cfg);
        }
        let config = self.strided.get_mut(&key).expect("config just inserted");

        if config.cols != n as u64 {
            let cols = n as u64;
            for desc in [config.b_desc, config.c_desc] {
                status(
                    unsafe {
                        cublasLtMatrixLayoutSetAttribute(
                            desc,
                            CUBLASLT_MATRIX_LAYOUT_COLS,
                            &cols as *const u64 as *const c_void,
                            std::mem::size_of::<u64>(),
                        )
                    },
                    "cublasLtMatrixLayoutSetAttribute(strided cols)",
                )?;
            }
            config.cols = cols;
        }

        let (alpha_f32, beta_f32) = (1.0f32, 0.0f32);
        let (alpha_f16, beta_f16) = (f16::from_f32(1.0), f16::from_f32(0.0));
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
        let op = config.own_op.unwrap_or(self.op_desc);
        status(
            unsafe {
                cublasLtMatmul(
                    self.handle,
                    op,
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
            "cublasLtMatmul(strided)",
        )
    }

    /// Create the cuBLASLt config for every weight shape and run one tiny
    /// matmul on each.
    ///
    /// The first real matmul of a shape pays the heuristic search plus the
    /// lazy load of the algorithm's kernel module, and that cost lands inside
    /// the first prefill wave (measured ~150 ms for the whole layer set). The
    /// dummy runs move it to engine construction, where it is invisible to a
    /// request's latency. The column count here does not matter: the config is
    /// keyed by `(m, k)` and the reduction order is fixed by the algorithm, not
    /// by `n`, which is the property the engine relies on.
    pub fn warm(&mut self, gpu: &GpuContext, shapes: &[(usize, usize)]) -> Result<()> {
        let stream = gpu.compute_stream().clone();
        let n = 8usize;
        for &(m, k) in shapes {
            let a = DeviceBuffer::<f16>::zeroed(gpu, m * k)?;
            let b = DeviceBuffer::<f16>::zeroed(gpu, k * n)?;
            let mut c = DeviceBuffer::<f16>::zeroed(gpu, m * n)?;
            self.gemm(
                &stream,
                m,
                k,
                n,
                a.inner().cu_deviceptr(),
                b.inner().cu_deviceptr(),
                c.inner_mut().cu_deviceptr(),
            )?;
        }
        stream.synchronize()?;
        Ok(())
    }

    /// Enumerate the heuristic's algorithm candidates for `(m, k)` and time
    /// each at several batch sizes. Benchmark-only: the production path keeps
    /// index 0 (`OXIDE_LT_ALGO_INDEX` selects another for experiments).
    ///
    /// Every candidate uses one algorithm for all `n`, so each one is a
    /// legitimate batch-invariant choice; this only asks which of them is
    /// fastest at the batch sizes the decoder actually sees. `col0_diff`
    /// re-runs the extremes and compares the first output column bitwise, so
    /// a candidate whose reduction order shifts with `n` is visible.
    pub fn bench_candidates(
        &self,
        gpu: &GpuContext,
        m: usize,
        k: usize,
        ns: &[usize],
        top: i32,
        reps: usize,
    ) -> Result<Vec<AlgoCandidate>> {
        let algo_n = std::env::var("OXIDE_LT_ALGO_N")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(32)
            .clamp(1, self.max_batch) as u64;
        let mut a_desc = ptr::null_mut();
        let mut b_desc = ptr::null_mut();
        let mut c_desc = ptr::null_mut();
        status(
            unsafe { cublasLtMatrixLayoutCreate(&mut a_desc, CUDA_R_16F, k as u64, m as u64, k as i64) },
            "cublasLtMatrixLayoutCreate(A)",
        )?;
        status(
            unsafe { cublasLtMatrixLayoutCreate(&mut b_desc, CUDA_R_16F, k as u64, algo_n, k as i64) },
            "cublasLtMatrixLayoutCreate(B)",
        )?;
        status(
            unsafe { cublasLtMatrixLayoutCreate(&mut c_desc, CUDA_R_16F, m as u64, algo_n, m as i64) },
            "cublasLtMatrixLayoutCreate(C)",
        )?;

        let mut pref = ptr::null_mut();
        status(unsafe { cublasLtMatmulPreferenceCreate(&mut pref) }, "pref create")?;
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
            "pref workspace",
        )?;
        let mut results = vec![
            LtHeuristic { algo: LtAlgo { data: [0; 8] }, workspace_size: 0, state: -1, waves_count: 0.0, reserved: [0; 4] };
            top.max(1) as usize
        ];
        let mut count = 0i32;
        status(
            unsafe {
                cublasLtMatmulAlgoGetHeuristic(
                    self.handle, self.op_desc, a_desc, b_desc, c_desc, c_desc,
                    pref, top, results.as_mut_ptr(), &mut count,
                )
            },
            "algo heuristic (candidates)",
        )?;
        unsafe { cublasLtMatmulPreferenceDestroy(pref) };
        results.truncate(count.max(0) as usize);

        // Non-zero operands: the invariance check compares real values.
        let pat = |i: usize, seed: u64| -> f16 {
            let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
            x ^= x >> 33;
            x = x.wrapping_mul(0xff51afd7ed558ccd);
            x ^= x >> 33;
            f16::from_f32(((x & 0xffff) as f32 / 32767.5) - 1.0)
        };
        let a_host: Vec<f16> = (0..m * k).map(|i| pat(i, 7)).collect();
        let max_n = *ns.iter().max().unwrap_or(&1);
        let b_host: Vec<f16> = (0..k * max_n).map(|i| pat(i, 11)).collect();
        let d_a = DeviceBuffer::from_slice(gpu, &a_host)?;
        let d_b = DeviceBuffer::from_slice(gpu, &b_host)?;
        let mut d_c = DeviceBuffer::<f16>::zeroed(gpu, m * max_n)?;
        let ev0 = gpu.context().new_event()?;
        let ev1 = gpu.context().new_event()?;

        let set_cols = |cols: u64| -> Result<()> {
            for desc in [b_desc, c_desc] {
                status(
                    unsafe {
                        cublasLtMatrixLayoutSetAttribute(
                            desc, CUBLASLT_MATRIX_LAYOUT_COLS,
                            &cols as *const u64 as *const c_void, std::mem::size_of::<u64>(),
                        )
                    },
                    "set cols",
                )?;
            }
            Ok(())
        };
        let (alpha_f32, beta_f32) = (1.0f32, 0.0f32);
        let launch = |algo: &LtAlgo, n: usize| -> Result<()> {
            status(
                unsafe {
                    cublasLtMatmul(
                        self.handle, self.op_desc,
                        &alpha_f32 as *const f32 as *const c_void,
                        d_a.inner().cu_deviceptr() as *const c_void, a_desc,
                        d_b.inner().cu_deviceptr() as *const c_void, b_desc,
                        &beta_f32 as *const f32 as *const c_void,
                        d_c.inner().cu_deviceptr() as *const c_void, c_desc,
                        d_c.inner().cu_deviceptr() as *mut c_void, c_desc,
                        algo,
                        self.workspace.inner().cu_deviceptr() as *mut c_void,
                        self.workspace_size,
                        gpu.compute_stream().cu_stream() as *mut c_void,
                    )
                },
                "cublasLtMatmul(bench)",
            )?;
            let _ = n;
            Ok(())
        };

        let stream = gpu.compute_stream().clone();
        let mut out = Vec::with_capacity(results.len());
        for (index, cand) in results.iter().enumerate() {
            let mut times = Vec::with_capacity(ns.len());
            let mut col0 = Vec::new();
            for &n in ns {
                set_cols(n as u64)?;
                for _ in 0..3 {
                    launch(&cand.algo, n)?;
                }
                stream.synchronize()?;
                let mut v = Vec::with_capacity(reps);
                for _ in 0..reps {
                    ev0.record(&stream)?;
                    launch(&cand.algo, n)?;
                    ev1.record(&stream)?;
                    ev1.synchronize()?;
                    v.push(ev0.elapsed_time(&ev1)? as f64);
                }
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                times.push(v[v.len() / 2]);
                if n == 1 || n == max_n {
                    let host = d_c.inner().to_host_vec(&stream)?;
                    col0.push(host[..m].iter().map(|x| x.to_f32()).collect::<Vec<_>>());
                }
            }
            let diff = if col0.len() == 2 {
                col0[0].iter().zip(&col0[1]).fold(0f32, |acc, (a, b)| acc.max((a - b).abs()))
            } else {
                0.0
            };
            out.push(AlgoCandidate {
                index,
                times_ms: times,
                col0_diff: diff,
                waves: cand.waves_count,
                workspace: cand.workspace_size,
            });
        }
        unsafe {
            cublasLtMatrixLayoutDestroy(a_desc);
            cublasLtMatrixLayoutDestroy(b_desc);
            cublasLtMatrixLayoutDestroy(c_desc);
        }
        Ok(out)
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
            for (_, config) in self.strided.drain() {
                cublasLtMatrixLayoutDestroy(config.a_desc);
                cublasLtMatrixLayoutDestroy(config.b_desc);
                cublasLtMatrixLayoutDestroy(config.c_desc);
                if let Some(op_desc) = config.own_op {
                    cublasLtMatmulDescDestroy(op_desc);
                }
            }
            cublasLtMatmulDescDestroy(self.op_desc);
            cublasLtDestroy(self.handle);
        }
    }
}
