//! Production quantized GEMV kernels for oxide-infer.
//!
//! Two kernels, monomorphized per ggml quant variant:
//!   - fused_dual_gemv_swiglu_q<Q>: W1/W3 packed weights + activation x
//!     -> act = up * gate / (1 + ex2(gate * -log2e)), f16 out.
//!     grid (INTER, B, 1), one CTA per (row, batch-element).
//!   - gemv_q<Q>: plain packed-weight GEMV -> f32 accumulate, f16 out.
//!     grid (N, B, 1).
//!
//! Batch invariance: independent CTAs, deterministic f32 reduction trees.

#![feature(f16)]

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use half::f16;
use cuda_device::cuda_module;

#[cuda_module]
mod kernels {
    use cuda_device::{DisjointSlice, SharedArray, convert, float, kernel, thread, warp};

    pub trait QuantVariant {
        const BLOCK_BYTES: usize;
        const BLOCK_ELEMS: usize;

        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32;
    }

    /// q1_0 (ggml): 128-elem, f16 d + 16 bitmap bytes (LSB first). w = bit ? d : -d.
    pub struct Q1_0;
    impl QuantVariant for Q1_0 {
        const BLOCK_BYTES: usize = 18;
        const BLOCK_ELEMS: usize = 128;
        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d = f16::from_bits(u16::from_le_bytes([packed[off], packed[off + 1]])) as f32;
            let mut acc = 0.0f32;
            let mut j = 0usize;
            while j < 128 {
                let byte = packed[off + 2 + j / 8];
                let bit = (byte >> (j % 8)) & 1;
                let xj = f16::from_bits(x[x_base + j]) as f32;
                let w = if bit != 0 { d } else { -d };
                acc = float::fma_rn_f32(w, xj, acc);
                j += 1;
            }
            acc
        }
    }

    /// q2_0 (ggml): 64-elem, f16 d + 16 2-bit bytes. w = (q - 1) * d.
    pub struct Q2_0;
    impl QuantVariant for Q2_0 {
        const BLOCK_BYTES: usize = 18;
        const BLOCK_ELEMS: usize = 64;
        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d = f16::from_bits(u16::from_le_bytes([packed[off], packed[off + 1]])) as f32;
            let mut acc = 0.0f32;
            let mut j = 0usize;
            while j < 64 {
                let byte = packed[off + 2 + j / 4];
                let q = (byte >> ((j % 4) * 2)) & 0x03;
                let xj = f16::from_bits(x[x_base + j]) as f32;
                acc = float::fma_rn_f32((q as f32 - 1.0) * d, xj, acc);
                j += 1;
            }
            acc
        }
    }

    /// Q4_0 (ggml): 32-elem, f16 d + 16 nibble bytes. w = d*(q - 8).
    pub struct Q4_0;
    impl QuantVariant for Q4_0 {
        const BLOCK_BYTES: usize = 18;
        const BLOCK_ELEMS: usize = 32;
        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d = f16::from_bits(u16::from_le_bytes([packed[off], packed[off + 1]])) as f32;
            let mut acc = 0.0f32;
            let mut i = 0usize;
            while i < 16 {
                let byte = packed[off + 2 + i];
                let x0 = f16::from_bits(x[x_base + i]) as f32;
                acc = float::fma_rn_f32(d * ((byte & 0x0F) as f32 - 8.0), x0, acc);
                i += 1;
            }
            let mut i = 0usize;
            while i < 16 {
                let byte = packed[off + 2 + i];
                let x1 = f16::from_bits(x[x_base + 16 + i]) as f32;
                acc = float::fma_rn_f32(d * (((byte >> 4) & 0x0F) as f32 - 8.0), x1, acc);
                i += 1;
            }
            acc
        }
    }

    /// Q4_1 (ggml): 32-elem, f16 d, f16 m, 16 nibble bytes. w = d*q + m.
    pub struct Q4_1;
    impl QuantVariant for Q4_1 {
        const BLOCK_BYTES: usize = 20;
        const BLOCK_ELEMS: usize = 32;
        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d = f16::from_bits(u16::from_le_bytes([packed[off], packed[off + 1]])) as f32;
            let m = f16::from_bits(u16::from_le_bytes([packed[off + 2], packed[off + 3]])) as f32;
            let mut acc = 0.0f32;
            let mut i = 0usize;
            while i < 16 {
                let byte = packed[off + 4 + i];
                let x0 = f16::from_bits(x[x_base + i]) as f32;
                acc = float::fma_rn_f32(d * (byte & 0x0F) as f32 + m, x0, acc);
                i += 1;
            }
            let mut i = 0usize;
            while i < 16 {
                let byte = packed[off + 4 + i];
                let x1 = f16::from_bits(x[x_base + 16 + i]) as f32;
                acc = float::fma_rn_f32(d * ((byte >> 4) & 0x0F) as f32 + m, x1, acc);
                i += 1;
            }
            acc
        }
    }

    /// Q5_0 (ggml): 32-elem, f16 d, u32 qh, 16 nibble bytes. w = d*(q | qh<<4 - 16).
    pub struct Q5_0;
    impl QuantVariant for Q5_0 {
        const BLOCK_BYTES: usize = 22;
        const BLOCK_ELEMS: usize = 32;
        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d = f16::from_bits(u16::from_le_bytes([packed[off], packed[off + 1]])) as f32;
            let qh = u32::from_le_bytes([packed[off + 2], packed[off + 3], packed[off + 4], packed[off + 5]]);
            let mut acc = 0.0f32;
            let mut j = 0usize;
            while j < 16 {
                let xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
                let byte = packed[off + 6 + j];
                let w0 = (((byte & 0x0F) as u32 | xh_0) as f32 - 16.0) * d;
                let xi0 = f16::from_bits(x[x_base + j]) as f32;
                acc = float::fma_rn_f32(w0, xi0, acc);
                j += 1;
            }
            let mut j = 0usize;
            while j < 16 {
                let xh_1 = ((qh >> (j + 12)) & 0x10) << 0;
                let byte = packed[off + 6 + j];
                let w1 = ((((byte >> 4) as u32) | xh_1) as f32 - 16.0) * d;
                let xi1 = f16::from_bits(x[x_base + 16 + j]) as f32;
                acc = float::fma_rn_f32(w1, xi1, acc);
                j += 1;
            }
            acc
        }
    }

    /// Q5_1 (ggml): 32-elem, f16 d, f16 m, u32 qh, 16 nibble bytes. w = d*q + m.
    pub struct Q5_1;
    impl QuantVariant for Q5_1 {
        const BLOCK_BYTES: usize = 24;
        const BLOCK_ELEMS: usize = 32;
        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d = f16::from_bits(u16::from_le_bytes([packed[off], packed[off + 1]])) as f32;
            let m = f16::from_bits(u16::from_le_bytes([packed[off + 2], packed[off + 3]])) as f32;
            let qh = u32::from_le_bytes([packed[off + 4], packed[off + 5], packed[off + 6], packed[off + 7]]);
            let mut acc = 0.0f32;
            let mut j = 0usize;
            while j < 16 {
                let xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
                let byte = packed[off + 8 + j];
                let w0 = ((byte & 0x0F) as u32 | xh_0) as f32 * d + m;
                let xi0 = f16::from_bits(x[x_base + j]) as f32;
                acc = float::fma_rn_f32(w0, xi0, acc);
                j += 1;
            }
            let mut j = 0usize;
            while j < 16 {
                let xh_1 = ((qh >> (j + 12)) & 0x10) << 0;
                let byte = packed[off + 8 + j];
                let w1 = ((((byte >> 4) as u32) | xh_1) as f32) * d + m;
                let xi1 = f16::from_bits(x[x_base + 16 + j]) as f32;
                acc = float::fma_rn_f32(w1, xi1, acc);
                j += 1;
            }
            acc
        }
    }

    /// Q8_0 (ggml): 32-elem, f16 d + 32 int8. w = d * q.
    pub struct Q8_0;
    impl QuantVariant for Q8_0 {
        const BLOCK_BYTES: usize = 34;
        const BLOCK_ELEMS: usize = 32;
        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d = f16::from_bits(u16::from_le_bytes([packed[off], packed[off + 1]])) as f32;
            let mut acc = 0.0f32;
            let mut i = 0usize;
            while i < 32 {
                let q = packed[off + 2 + i] as i8 as f32;
                let xi = f16::from_bits(x[x_base + i]) as f32;
                acc = float::fma_rn_f32(d * q, xi, acc);
                i += 1;
            }
            acc
        }
    }

    /// Q8_1 (ggml): 32-elem, f16 d, f16 s (ignored), 32 int8. w = d * q.
    pub struct Q8_1;
    impl QuantVariant for Q8_1 {
        const BLOCK_BYTES: usize = 36;
        const BLOCK_ELEMS: usize = 32;
        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d = f16::from_bits(u16::from_le_bytes([packed[off], packed[off + 1]])) as f32;
            let mut acc = 0.0f32;
            let mut i = 0usize;
            while i < 32 {
                let q = packed[off + 4 + i] as i8 as f32;
                let xi = f16::from_bits(x[x_base + i]) as f32;
                acc = float::fma_rn_f32(d * q, xi, acc);
                i += 1;
            }
            acc
        }
    }

    const WARP_SUMS: usize = 32;

    fn warp_tree_sum(mut val: f32) -> f32 {
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 16);
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 8);
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 4);
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 2);
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 1);
        val
    }

    /// Fused dual GEMV (W1, W3) + SwiGLU activation.
    /// grid (INTER, B, 1); block 128 threads.
    /// act[b*n + row] = f16(up(row) * silu(gate(row))).
    pub fn fused_dual_swiglu_impl<Q: QuantVariant>(
        w1: &[u8],
        w3: &[u8],
        x: &[u16],
        mut act: DisjointSlice<u16>,
        hidden: u32,
        n_blocks: u32,
        n_elems: u32,
    ) {
        static mut S_SUMS: SharedArray<f32, WARP_SUMS> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        let b_idx = thread::blockIdx_y() as usize;
        let tid = thread::threadIdx_x() as usize;
        let ntid = thread::blockDim_x() as usize;
        let nb = n_blocks as usize;
        let row_off = row * nb * Q::BLOCK_BYTES;
        let x_base = b_idx * hidden as usize;

        let mut acc1 = 0.0f32;
        let mut acc3 = 0.0f32;
        let mut b = tid;
        while b < nb {
            let off = row_off + b * Q::BLOCK_BYTES;
            let xb = x_base + b * Q::BLOCK_ELEMS;
            acc1 += Q::dot_block(w1, off, x, xb);
            acc3 += Q::dot_block(w3, off, x, xb);
            b += ntid;
        }

        acc1 = warp_tree_sum(acc1);
        acc3 = warp_tree_sum(acc3);

        let lane = warp::lane_id();
        let warp_id = warp::warp_id() as usize;
        let num_warps = ntid.div_ceil(32);

        if lane == 0 {
            // SAFETY: 32 slots, warp_id < 32.
            unsafe { S_SUMS[warp_id] = acc1; }
        }
        thread::sync_threads();
        let mut gate = 0.0f32;
        for w in 0..num_warps {
            // SAFETY: w < num_warps <= 32.
            gate += unsafe { S_SUMS[w] };
        }

        thread::sync_threads();
        if lane == 0 {
            // SAFETY: 32 slots, warp_id < 32.
            unsafe { S_SUMS[warp_id] = acc3; }
        }
        thread::sync_threads();
        let mut up = 0.0f32;
        for w in 0..num_warps {
            // SAFETY: w < num_warps <= 32.
            up += unsafe { S_SUMS[w] };
        }

        if tid == 0 {
            let ex = float::ex2_approx_f32(gate * -1.4426950208889634f32);
            let value = up * (gate / (1.0 + ex));
            let packed32 = convert::cvt_f16x2_f32(value, 0.0);
            // SAFETY: one writer per (row, b) pair; rows/batch disjoint.
            unsafe { *act.get_unchecked_mut(b_idx * n_elems as usize + row) = packed32 as u16; }
        }
    }

    /// Plain GEMV over packed weights (used for the W2 down projection).
    /// grid (N, B, 1); block 128 threads.
    pub fn gemv_impl<Q: QuantVariant>(
        w: &[u8],
        x: &[u16],
        mut y: DisjointSlice<u16>,
        hidden: u32,
        n_blocks: u32,
        n_elems: u32,
    ) {
        static mut S_SUMS: SharedArray<f32, WARP_SUMS> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        let b_idx = thread::blockIdx_y() as usize;
        let tid = thread::threadIdx_x() as usize;
        let ntid = thread::blockDim_x() as usize;
        let nb = n_blocks as usize;
        let row_off = row * nb * Q::BLOCK_BYTES;
        let x_base = b_idx * hidden as usize;

        let mut acc = 0.0f32;
        let mut b = tid;
        while b < nb {
            let off = row_off + b * Q::BLOCK_BYTES;
            acc += Q::dot_block(w, off, x, x_base + b * Q::BLOCK_ELEMS);
            b += ntid;
        }

        acc = warp_tree_sum(acc);
        let warp_id = warp::warp_id() as usize;
        let num_warps = ntid.div_ceil(32);
        if warp::lane_id() == 0 {
            // SAFETY: 32 slots, warp_id < 32.
            unsafe { S_SUMS[warp_id] = acc; }
        }
        thread::sync_threads();
        let mut total = 0.0f32;
        for w in 0..num_warps {
            // SAFETY: w < num_warps <= 32.
            total += unsafe { S_SUMS[w] };
        }

        if tid == 0 {
            let packed32 = convert::cvt_f16x2_f32(total, 0.0);
            // SAFETY: one writer per (row, b) pair.
            unsafe { *y.get_unchecked_mut(b_idx * n_elems as usize + row) = packed32 as u16; }
        }
    }
#[kernel]
pub fn fused_dual_gemv_swiglu_q1_0(w1: &[u8], w3: &[u8], x: &[u16], act: DisjointSlice<u16>, hidden: u32, n_blocks: u32, n_elems: u32) {
    fused_dual_swiglu_impl::<Q1_0>(w1, w3, x, act, hidden, n_blocks, n_elems)
}
#[kernel]
pub fn fused_dual_gemv_swiglu_q2_0(w1: &[u8], w3: &[u8], x: &[u16], act: DisjointSlice<u16>, hidden: u32, n_blocks: u32, n_elems: u32) {
    fused_dual_swiglu_impl::<Q2_0>(w1, w3, x, act, hidden, n_blocks, n_elems)
}
#[kernel]
pub fn fused_dual_gemv_swiglu_q4_0(w1: &[u8], w3: &[u8], x: &[u16], act: DisjointSlice<u16>, hidden: u32, n_blocks: u32, n_elems: u32) {
    fused_dual_swiglu_impl::<Q4_0>(w1, w3, x, act, hidden, n_blocks, n_elems)
}
#[kernel]
pub fn fused_dual_gemv_swiglu_q4_1(w1: &[u8], w3: &[u8], x: &[u16], act: DisjointSlice<u16>, hidden: u32, n_blocks: u32, n_elems: u32) {
    fused_dual_swiglu_impl::<Q4_1>(w1, w3, x, act, hidden, n_blocks, n_elems)
}
#[kernel]
pub fn fused_dual_gemv_swiglu_q5_0(w1: &[u8], w3: &[u8], x: &[u16], act: DisjointSlice<u16>, hidden: u32, n_blocks: u32, n_elems: u32) {
    fused_dual_swiglu_impl::<Q5_0>(w1, w3, x, act, hidden, n_blocks, n_elems)
}
#[kernel]
pub fn fused_dual_gemv_swiglu_q5_1(w1: &[u8], w3: &[u8], x: &[u16], act: DisjointSlice<u16>, hidden: u32, n_blocks: u32, n_elems: u32) {
    fused_dual_swiglu_impl::<Q5_1>(w1, w3, x, act, hidden, n_blocks, n_elems)
}
#[kernel]
pub fn fused_dual_gemv_swiglu_q8_0(w1: &[u8], w3: &[u8], x: &[u16], act: DisjointSlice<u16>, hidden: u32, n_blocks: u32, n_elems: u32) {
    fused_dual_swiglu_impl::<Q8_0>(w1, w3, x, act, hidden, n_blocks, n_elems)
}
#[kernel]
pub fn fused_dual_gemv_swiglu_q8_1(w1: &[u8], w3: &[u8], x: &[u16], act: DisjointSlice<u16>, hidden: u32, n_blocks: u32, n_elems: u32) {
    fused_dual_swiglu_impl::<Q8_1>(w1, w3, x, act, hidden, n_blocks, n_elems)
}
#[kernel]
pub fn gemv_q1_0(w: &[u8], x: &[u16], y: DisjointSlice<u16>, hidden: u32, n_blocks: u32, n_elems: u32) {
    gemv_impl::<Q1_0>(w, x, y, hidden, n_blocks, n_elems)
}
#[kernel]
pub fn gemv_q2_0(w: &[u8], x: &[u16], y: DisjointSlice<u16>, hidden: u32, n_blocks: u32, n_elems: u32) {
    gemv_impl::<Q2_0>(w, x, y, hidden, n_blocks, n_elems)
}
#[kernel]
pub fn gemv_q4_0(w: &[u8], x: &[u16], y: DisjointSlice<u16>, hidden: u32, n_blocks: u32, n_elems: u32) {
    gemv_impl::<Q4_0>(w, x, y, hidden, n_blocks, n_elems)
}
#[kernel]
pub fn gemv_q4_1(w: &[u8], x: &[u16], y: DisjointSlice<u16>, hidden: u32, n_blocks: u32, n_elems: u32) {
    gemv_impl::<Q4_1>(w, x, y, hidden, n_blocks, n_elems)
}
#[kernel]
pub fn gemv_q5_0(w: &[u8], x: &[u16], y: DisjointSlice<u16>, hidden: u32, n_blocks: u32, n_elems: u32) {
    gemv_impl::<Q5_0>(w, x, y, hidden, n_blocks, n_elems)
}
#[kernel]
pub fn gemv_q5_1(w: &[u8], x: &[u16], y: DisjointSlice<u16>, hidden: u32, n_blocks: u32, n_elems: u32) {
    gemv_impl::<Q5_1>(w, x, y, hidden, n_blocks, n_elems)
}
#[kernel]
pub fn gemv_q8_0(w: &[u8], x: &[u16], y: DisjointSlice<u16>, hidden: u32, n_blocks: u32, n_elems: u32) {
    gemv_impl::<Q8_0>(w, x, y, hidden, n_blocks, n_elems)
}
#[kernel]
pub fn gemv_q8_1(w: &[u8], x: &[u16], y: DisjointSlice<u16>, hidden: u32, n_blocks: u32, n_elems: u32) {
    gemv_impl::<Q8_1>(w, x, y, hidden, n_blocks, n_elems)
}

}


fn deq_q1_0(p: &[u8]) -> Vec<f64> {
    let mut o = Vec::new();
    for b in 0..p.len() / 18 {
        let off = b * 18;
        let d = f16::from_bits(u16::from_le_bytes([p[off], p[off + 1]])).to_f64();
        for j in 0..128 { let byte = p[off + 2 + j / 8]; o.push(if (byte >> (j % 8)) & 1 != 0 { d } else { -d }); }
    }
    o
}
fn deq_q2_0(p: &[u8]) -> Vec<f64> {
    let mut o = Vec::new();
    for b in 0..p.len() / 18 {
        let off = b * 18;
        let d = f16::from_bits(u16::from_le_bytes([p[off], p[off + 1]])).to_f64();
        for j in 0..64 { let byte = p[off + 2 + j / 4]; o.push((((byte >> ((j % 4) * 2)) & 0x03) as f64 - 1.0) * d); }
    }
    o
}
fn deq_q4_0(p: &[u8]) -> Vec<f64> {
    let mut o = Vec::new();
    for b in 0..p.len() / 18 {
        let off = b * 18;
        let d = f16::from_bits(u16::from_le_bytes([p[off], p[off + 1]])).to_f64();
        for i in 0..16 { o.push(d * ((p[off + 2 + i] & 0x0F) as f64 - 8.0)); }
        for i in 0..16 { o.push(d * (((p[off + 2 + i] >> 4) & 0x0F) as f64 - 8.0)); }
    }
    o
}
fn deq_q4_1(p: &[u8]) -> Vec<f64> {
    let mut o = Vec::new();
    for b in 0..p.len() / 20 {
        let off = b * 20;
        let d = f16::from_bits(u16::from_le_bytes([p[off], p[off + 1]])).to_f64();
        let m = f16::from_bits(u16::from_le_bytes([p[off + 2], p[off + 3]])).to_f64();
        for i in 0..16 { o.push(d * (p[off + 4 + i] & 0x0F) as f64 + m); }
        for i in 0..16 { o.push(d * ((p[off + 4 + i] >> 4) & 0x0F) as f64 + m); }
    }
    o
}
fn deq_q5_0(p: &[u8]) -> Vec<f64> {
    let mut o = Vec::new();
    for b in 0..p.len() / 22 {
        let off = b * 22;
        let d = f16::from_bits(u16::from_le_bytes([p[off], p[off + 1]])).to_f64();
        let qh = u32::from_le_bytes([p[off+2], p[off+3], p[off+4], p[off+5]]);
        for j in 0..16 { let xh = ((qh >> (j + 0)) << 4) & 0x10; o.push((((p[off+6+j] & 0x0F) as u32 | xh) as f64 - 16.0) * d); }
        for j in 0..16 { let xh = (qh >> (j + 12)) & 0x10; o.push((((p[off+6+j] >> 4) as u32 | xh) as f64 - 16.0) * d); }
    }
    o
}
fn deq_q5_1(p: &[u8]) -> Vec<f64> {
    let mut o = Vec::new();
    for b in 0..p.len() / 24 {
        let off = b * 24;
        let d = f16::from_bits(u16::from_le_bytes([p[off], p[off + 1]])).to_f64();
        let m = f16::from_bits(u16::from_le_bytes([p[off + 2], p[off + 3]])).to_f64();
        let qh = u32::from_le_bytes([p[off+4], p[off+5], p[off+6], p[off+7]]);
        for j in 0..16 { let xh = ((qh >> (j + 0)) << 4) & 0x10; o.push(((p[off+8+j] & 0x0F) as u32 | xh) as f64 * d + m); }
        for j in 0..16 { let xh = (qh >> (j + 12)) & 0x10; o.push((((p[off+8+j] >> 4) as u32 | xh) as f64) * d + m); }
    }
    o
}
fn deq_q8_0(p: &[u8]) -> Vec<f64> {
    let mut o = Vec::new();
    for b in 0..p.len() / 34 {
        let off = b * 34;
        let d = f16::from_bits(u16::from_le_bytes([p[off], p[off + 1]])).to_f64();
        for i in 0..32 { o.push(p[off + 2 + i] as i8 as f64 * d); }
    }
    o
}
fn deq_q8_1(p: &[u8]) -> Vec<f64> {
    let mut o = Vec::new();
    for b in 0..p.len() / 36 {
        let off = b * 36;
        let d = f16::from_bits(u16::from_le_bytes([p[off], p[off + 1]])).to_f64();
        for i in 0..32 { o.push(p[off + 4 + i] as i8 as f64 * d); }
    }
    o
}

const HIDDEN: usize = 256;
const INTER: usize = 128;
const B: usize = 2;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.next() % 10000) as f32 / 10000.0 * (hi - lo)
    }
}

fn to_bits(v: f32) -> u16 { f16::from_f32(v).to_bits() }

fn f16_to_f64(w: u16) -> f64 { f16::from_bits(w).to_f64() }

struct Rng2(u64);
impl Rng2 {
    fn next(&mut self) -> u64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.0 >> 33 }
}

macro_rules! case {
    ($name:literal, $Q:ty, $elems:expr, $quant:path, $deq:path, $fused:ident, $gemv:ident, $stream:ident, $module:ident, $worst:ident, $checks:ident) => {{
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ ($name.len() as u64) << 32 ^ 0x12345);
        let w1: Vec<f32> = (0..INTER * HIDDEN).map(|_| rng.f32(-1.5, 1.5)).collect();
        let w3: Vec<f32> = (0..INTER * HIDDEN).map(|_| rng.f32(-1.5, 1.5)).collect();
        let w2: Vec<f32> = (0..HIDDEN * INTER).map(|_| rng.f32(-1.5, 1.5)).collect();
        let w1p = $quant(&w1);
        let w3p = $quant(&w3);
        let w2p = $quant(&w2);
        let x1_f32: Vec<f32> = (0..B * HIDDEN).map(|i| (i as f32 * 0.317 + 0.5).sin()).collect();
        let z_f32: Vec<f32> = (0..B * INTER).map(|i| (i as f32 * 0.71 + 1.0).cos()).collect();
        // The kernels consume f16 x; the reference must use the same rounded values.
        let x1_f64: Vec<f64> = x1_f32.iter().map(|&v| f16::from_f32(v).to_f64()).collect();
        let z_f64: Vec<f64> = z_f32.iter().map(|&v| f16::from_f32(v).to_f64()).collect();
        let x1_host: Vec<u16> = x1_f32.iter().map(|&v| to_bits(v)).collect();
        let z_host: Vec<u16> = z_f32.iter().map(|&v| to_bits(v)).collect();

        let w1_dev = DeviceBuffer::from_host(&$stream, &w1p).unwrap();
        let w3_dev = DeviceBuffer::from_host(&$stream, &w3p).unwrap();
        let w2_dev = DeviceBuffer::from_host(&$stream, &w2p).unwrap();
        let x1_dev = DeviceBuffer::from_host(&$stream, &x1_host).unwrap();
        let z_dev = DeviceBuffer::from_host(&$stream, &z_host).unwrap();
        let mut act_dev = DeviceBuffer::<u16>::zeroed(&$stream, B * INTER).unwrap();
        let mut out_dev = DeviceBuffer::<u16>::zeroed(&$stream, B * HIDDEN).unwrap();

        let cfg = LaunchConfig { grid_dim: (INTER as u32, B as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            $module.$fused(
                &$stream, cfg, &w1_dev, &w3_dev, &x1_dev, &mut act_dev,
                HIDDEN as u32, (HIDDEN / $elems) as u32, INTER as u32,
            )
        }
        .expect(concat!($name, " dual launch failed"));

        let cfg = LaunchConfig { grid_dim: (HIDDEN as u32, B as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            $module.$gemv(
                &$stream, cfg, &w2_dev, &z_dev, &mut out_dev,
                INTER as u32, (INTER / $elems) as u32, HIDDEN as u32,
            )
        }
        .expect(concat!($name, " gemv launch failed"));

        let act = act_dev.to_host_vec(&$stream).unwrap();
        let out = out_dev.to_host_vec(&$stream).unwrap();

        let wd1 = $deq(&w1p);
        let wd3 = $deq(&w3p);
        let wd2 = $deq(&w2p);

        let silu = |g: f64| g / (1.0 + (-g * 1.4426950208889634f64).exp2());
        let mut worst_act = 0.0f64;
        let mut worst_out = 0.0f64;
        let mut worst_act_rel = 0.0f64;
        let mut worst_out_rel = 0.0f64;
        for b in 0..B {
            for r in 0..INTER {
                let mut g = 0.0f64;
                let mut u = 0.0f64;
                for k in 0..HIDDEN {
                    g += wd1[r * HIDDEN + k] * x1_f64[b * HIDDEN + k];
                    u += wd3[r * HIDDEN + k] * x1_f64[b * HIDDEN + k];
                }
                let reference = silu(g) * u;
                let got = f16_to_f64(act[b * INTER + r]);
                worst_act = worst_act.max((got - reference).abs());
                worst_act_rel = worst_act_rel.max((got - reference).abs() / (1.0 + reference.abs()));
            }
            for r in 0..HIDDEN {
                let mut acc = 0.0f64;
                for k in 0..INTER {
                    acc += wd2[r * INTER + k] * z_f64[b * INTER + k];
                }
                let got = f16_to_f64(out[b * HIDDEN + r]);
                worst_out = worst_out.max((got - acc).abs());
                worst_out_rel = worst_out_rel.max((got - acc).abs() / (1.0 + acc.abs()));
            }
        }
        let ok = worst_act_rel < 1e-2 && worst_out_rel < 1e-2;
        $checks += 1;
        if !ok { $worst += 1; }
        println!("{:>5}: swiglu rel = {:.6} | gemv rel = {:.6} {}", $name, worst_act_rel, worst_out_rel, if ok { "ok" } else { "FAIL" });
    }};
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    let mut worst = 0usize;
    let mut checks = 0usize;
    use oxide_infer_core::quant::*;
    case!("q1_0", kernels::Q1_0, 128, quantize_q1_0, deq_q1_0, fused_dual_gemv_swiglu_q1_0, gemv_q1_0, stream, module, worst, checks);
    case!("q2_0", kernels::Q2_0, 64, quantize_q2_0, deq_q2_0, fused_dual_gemv_swiglu_q2_0, gemv_q2_0, stream, module, worst, checks);
    case!("q4_0", kernels::Q4_0, 32, quantize_q4_0, deq_q4_0, fused_dual_gemv_swiglu_q4_0, gemv_q4_0, stream, module, worst, checks);
    case!("q4_1", kernels::Q4_1, 32, quantize_q4_1, deq_q4_1, fused_dual_gemv_swiglu_q4_1, gemv_q4_1, stream, module, worst, checks);
    case!("q5_0", kernels::Q5_0, 32, quantize_q5_0, deq_q5_0, fused_dual_gemv_swiglu_q5_0, gemv_q5_0, stream, module, worst, checks);
    case!("q5_1", kernels::Q5_1, 32, quantize_q5_1, deq_q5_1, fused_dual_gemv_swiglu_q5_1, gemv_q5_1, stream, module, worst, checks);
    case!("q8_0", kernels::Q8_0, 32, quantize_q8_0, deq_q8_0, fused_dual_gemv_swiglu_q8_0, gemv_q8_0, stream, module, worst, checks);
    case!("q8_1", kernels::Q8_1, 32, quantize_q8_1, deq_q8_1, fused_dual_gemv_swiglu_q8_1, gemv_q8_1, stream, module, worst, checks);

    if worst == 0 {
        println!("all {checks} variant(s) pass: packed kernels match host f64 reference");
    } else {
        eprintln!("{worst} variant(s) FAILED");
        std::process::exit(1);
    }
}
