use std::ffi::c_void;
use half::f16;
use anyhow::{Result, bail};

pub type CublasHandle = *mut c_void;
pub type CublasStatus = i32;

pub const CUBLAS_OP_N: i32 = 0;
pub const CUBLAS_OP_T: i32 = 1;

#[link(name = "cublas")]
extern "C" {
    pub fn cublasCreate_v2(handle: *mut CublasHandle) -> CublasStatus;
    pub fn cublasDestroy_v2(handle: CublasHandle) -> CublasStatus;
    pub fn cublasSetStream_v2(handle: CublasHandle, stream: *mut c_void) -> CublasStatus;
    pub fn cublasHgemm(
        handle: CublasHandle,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f16,
        A: *const c_void,
        lda: i32,
        B: *const c_void,
        ldb: i32,
        beta: *const f16,
        C: *mut c_void,
        ldc: i32,
    ) -> CublasStatus;
}

/// Computes row-major matrix multiplication C = X * W^T
/// X is [M, K] row-major (leading dimension K)
/// W is [N, K] row-major (leading dimension K)
/// C is [M, N] row-major (leading dimension N)
#[inline]
pub unsafe fn cublas_matmul_rowmajor(
    handle: CublasHandle,
    m: i32, // rows of X and C
    n: i32, // cols of C, rows of W
    k: i32, // cols of X and cols of W
    d_x: *const c_void, // [m, k]
    d_w: *const c_void, // [n, k] (weight matrix)
    d_c: *mut c_void,   // [m, n]
) -> Result<()> {
    let alpha = f16::ONE;
    let beta = f16::ZERO;
    let status = cublasHgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        d_w,
        k,
        d_x,
        k,
        &beta,
        d_c,
        n,
    );
    if status != 0 {
        bail!("cublasHgemm error status: {}", status);
    }
    Ok(())
}
