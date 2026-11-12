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
        /// Whether this variant has a fused dual-dot implementation.
        const FUSED_DUAL: bool = false;
        /// Index into the fused dual dot family (const-generic static dispatch).
        const FUSED_IDX: usize = usize::MAX;

        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32;
    }

    /// q1_0 (ggml): 128-elem, f16 d + 16 bitmap bytes (LSB first). w = bit ? d : -d.
    pub struct Q1_0;
    impl QuantVariant for Q1_0 {
        const FUSED_IDX: usize = 0;
        const BLOCK_BYTES: usize = 18;
        const BLOCK_ELEMS: usize = 128;
        const FUSED_DUAL: bool = true;
        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d = f16::from_bits(u16::from_le_bytes([unsafe { *packed.get_unchecked(off) }, unsafe { *packed.get_unchecked(off + 1) }])) as f32;
            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            let mut acc2 = 0.0f32;
            let mut acc3 = 0.0f32;
            let mut w_idx = 0usize;
            while w_idx < 8 {
                let q = u16::from_le_bytes([unsafe { *packed.get_unchecked(off + 2 + w_idx * 2) }, unsafe { *packed.get_unchecked(off + 3 + w_idx * 2) }]);
                let lo = (q & 0x00FF) as u8;
                let hi = (q >> 8) as u8;
                let mut bit = 0usize;
                while bit < 16 {
                    let byte = if bit < 8 { lo } else { hi };
                    let b = (byte >> (bit % 8)) & 1;
                    let xj = x_unchecked(x, x_base + w_idx * 16 + bit);
                    let w = if b != 0 { d } else { -d };
                    let acc = match bit % 4 {
                        0 => &mut acc0,
                        1 => &mut acc1,
                        2 => &mut acc2,
                        _ => &mut acc3,
                    };
                    *acc = float::fma_rn_f32(w, xj, *acc);
                    bit += 1;
                }
                w_idx += 1;
            }
            (acc0 + acc1) + (acc2 + acc3)
        }
    }

    /// q2_0 (ggml): 64-elem, f16 d + 16 2-bit bytes. w = (q - 1) * d.
    pub struct Q2_0;
    impl QuantVariant for Q2_0 {
        const FUSED_IDX: usize = 1;
        const BLOCK_BYTES: usize = 18;
        const BLOCK_ELEMS: usize = 64;
        const FUSED_DUAL: bool = true;
        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d = f16::from_bits(u16::from_le_bytes([unsafe { *packed.get_unchecked(off) }, unsafe { *packed.get_unchecked(off + 1) }])) as f32;
            let mut acc = 0.0f32;
            let mut i = 0usize;
            while i < 16 {
                let q = u16::from_le_bytes([unsafe { *packed.get_unchecked(off + 2 + i) }, unsafe { *packed.get_unchecked(off + 3 + i) }]);
                let bytes = [(q & 0x00FF) as u8, (q >> 8) as u8];
                for bi in 0..2 {
                    let byte = bytes[bi];
                    let elem_base = (i + bi) * 4;
                    let mut k = 0usize;
                    while k < 4 {
                        let v = ((byte >> (k * 2)) & 0x03) as f32 - 1.0;
                        let xj = x_unchecked(x, x_base + elem_base + k);
                        acc = float::fma_rn_f32(v * d, xj, acc);
                        k += 1;
                    }
                }
                i += 2;
            }
            acc
        }
    }

    /// Q4_0 (ggml): 32-elem, f16 d + 16 nibble bytes. w = d*(q - 8).
    pub struct Q4_0;
    impl QuantVariant for Q4_0 {
        const FUSED_IDX: usize = 2;
        const BLOCK_BYTES: usize = 18;
        const BLOCK_ELEMS: usize = 32;
        const FUSED_DUAL: bool = true;
        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d = f16::from_bits(u16::from_le_bytes([unsafe { *packed.get_unchecked(off) }, unsafe { *packed.get_unchecked(off + 1) }])) as f32;
            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            let mut acc2 = 0.0f32;
            let mut acc3 = 0.0f32;
            let mut i = 0usize;
            while i < 16 {
                let q = u16::from_le_bytes([unsafe { *packed.get_unchecked(off + 2 + i) }, unsafe { *packed.get_unchecked(off + 3 + i) }]);
                let w_i = ((q & 0x000F) as f32 - 8.0) * d;
                let w_i1 = (((q >> 8) & 0x000F) as f32 - 8.0) * d;
                let w_i16 = (((q >> 4) & 0x000F) as f32 - 8.0) * d;
                let w_i17 = (((q >> 12) & 0x000F) as f32 - 8.0) * d;
                let (xi, xi1) = unpack_u16x2(x, x_base + i);
                let (xi16, xi17) = unpack_u16x2(x, x_base + 16 + i);
                acc0 = float::fma_rn_f32(w_i, xi, acc0);
                acc1 = float::fma_rn_f32(w_i1, xi1, acc1);
                acc2 = float::fma_rn_f32(w_i16, xi16, acc2);
                acc3 = float::fma_rn_f32(w_i17, xi17, acc3);
                i += 2;
            }
            (acc0 + acc1) + (acc2 + acc3)
        }
    }

    /// Q4_1 (ggml): 32-elem, f16 d, f16 m, 16 nibble bytes. w = d*q + m.
    pub struct Q4_1;
    impl QuantVariant for Q4_1 {
        const FUSED_IDX: usize = 3;
        const BLOCK_BYTES: usize = 20;
        const BLOCK_ELEMS: usize = 32;
        const FUSED_DUAL: bool = true;
        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d = f16::from_bits(u16::from_le_bytes([unsafe { *packed.get_unchecked(off) }, unsafe { *packed.get_unchecked(off + 1) }])) as f32;
            let m = f16::from_bits(u16::from_le_bytes([unsafe { *packed.get_unchecked(off + 2) }, unsafe { *packed.get_unchecked(off + 3) }])) as f32;
            let mut acc = 0.0f32;
            let mut i = 0usize;
            while i < 16 {
                let q = u16::from_le_bytes([unsafe { *packed.get_unchecked(off + 4 + i) }, unsafe { *packed.get_unchecked(off + 5 + i) }]);
                let w_i = (q & 0x000F) as f32 * d + m;
                let w_i1 = ((q >> 8) & 0x000F) as f32 * d + m;
                let w_i16 = ((q >> 4) & 0x000F) as f32 * d + m;
                let w_i17 = ((q >> 12) & 0x000F) as f32 * d + m;
                let (xi, xi1) = unpack_u16x2(x, x_base + i);
                let (xi16, xi17) = unpack_u16x2(x, x_base + 16 + i);
                acc = float::fma_rn_f32(w_i, xi, acc);
                acc = float::fma_rn_f32(w_i1, xi1, acc);
                acc = float::fma_rn_f32(w_i16, xi16, acc);
                acc = float::fma_rn_f32(w_i17, xi17, acc);
                i += 2;
            }
            acc
        }
    }

    /// Q5_0 (ggml): 32-elem, f16 d, u32 qh, 16 nibble bytes. w = d*(q | qh<<4 - 16).
    pub struct Q5_0;
    impl QuantVariant for Q5_0 {
        const FUSED_IDX: usize = 4;
        const BLOCK_BYTES: usize = 22;
        const BLOCK_ELEMS: usize = 32;
        const FUSED_DUAL: bool = true;
        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d = f16::from_bits(u16::from_le_bytes([unsafe { *packed.get_unchecked(off) }, unsafe { *packed.get_unchecked(off + 1) }])) as f32;
            let qh0 = u16::from_le_bytes([unsafe { *packed.get_unchecked(off + 2) }, unsafe { *packed.get_unchecked(off + 3) }]);
            let qh1 = u16::from_le_bytes([unsafe { *packed.get_unchecked(off + 4) }, unsafe { *packed.get_unchecked(off + 5) }]);
            let qh = (qh0 as u32) | ((qh1 as u32) << 16);
            let mut acc = 0.0f32;
            let mut j = 0usize;
            while j < 16 {
                let q = u16::from_le_bytes([unsafe { *packed.get_unchecked(off + 6 + j) }, unsafe { *packed.get_unchecked(off + 7 + j) }]);
                let xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
                let xh_1 = ((qh >> (j + 1 + 0)) << 4) & 0x10;
                let xi0 = ((q & 0x000F) as u32 | xh_0) as f32 - 16.0;
                let xi1 = (((q >> 8) & 0x000F) as u32 | xh_1) as f32 - 16.0;
                let (x0, x1) = unpack_u16x2(x, x_base + j);
                acc = float::fma_rn_f32(d * xi0, x0, acc);
                acc = float::fma_rn_f32(d * xi1, x1, acc);
                j += 2;
            }
            let mut j = 0usize;
            while j < 16 {
                let q = u16::from_le_bytes([unsafe { *packed.get_unchecked(off + 6 + j) }, unsafe { *packed.get_unchecked(off + 7 + j) }]);
                let xh_0 = (qh >> (j + 12)) & 0x10;
                let xh_1 = (qh >> (j + 13)) & 0x10;
                let xi0 = ((((q >> 4) & 0x000F) as u32) | xh_0) as f32 - 16.0;
                let xi1 = ((((q >> 12) & 0x000F) as u32) | xh_1) as f32 - 16.0;
                let (x0, x1) = unpack_u16x2(x, x_base + 16 + j);
                acc = float::fma_rn_f32(d * xi0, x0, acc);
                acc = float::fma_rn_f32(d * xi1, x1, acc);
                j += 2;
            }
            acc
        }
    }

    /// Q5_1 (ggml): 32-elem, f16 d, f16 m, u32 qh, 16 nibble bytes. w = d*q + m.
    pub struct Q5_1;
    impl QuantVariant for Q5_1 {
        const FUSED_IDX: usize = 5;
        const BLOCK_BYTES: usize = 24;
        const BLOCK_ELEMS: usize = 32;
        const FUSED_DUAL: bool = true;
        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d = f16::from_bits(u16::from_le_bytes([unsafe { *packed.get_unchecked(off) }, unsafe { *packed.get_unchecked(off + 1) }])) as f32;
            let m = f16::from_bits(u16::from_le_bytes([unsafe { *packed.get_unchecked(off + 2) }, unsafe { *packed.get_unchecked(off + 3) }])) as f32;
            let qh0 = u16::from_le_bytes([unsafe { *packed.get_unchecked(off + 4) }, unsafe { *packed.get_unchecked(off + 5) }]);
            let qh1 = u16::from_le_bytes([unsafe { *packed.get_unchecked(off + 6) }, unsafe { *packed.get_unchecked(off + 7) }]);
            let qh = (qh0 as u32) | ((qh1 as u32) << 16);
            let mut acc = 0.0f32;
            let mut j = 0usize;
            while j < 16 {
                let q = u16::from_le_bytes([unsafe { *packed.get_unchecked(off + 8 + j) }, unsafe { *packed.get_unchecked(off + 9 + j) }]);
                let xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
                let xh_1 = ((qh >> (j + 1)) << 4) & 0x10;
                let xi0 = ((q & 0x000F) as u32 | xh_0) as f32 * d + m;
                let xi1 = (((q >> 8) & 0x000F) as u32 | xh_1) as f32 * d + m;
                let (x0, x1) = unpack_u16x2(x, x_base + j);
                acc = float::fma_rn_f32(xi0, x0, acc);
                acc = float::fma_rn_f32(xi1, x1, acc);
                j += 2;
            }
            let mut j = 0usize;
            while j < 16 {
                let q = u16::from_le_bytes([unsafe { *packed.get_unchecked(off + 8 + j) }, unsafe { *packed.get_unchecked(off + 9 + j) }]);
                let xh_0 = (qh >> (j + 12)) & 0x10;
                let xh_1 = (qh >> (j + 13)) & 0x10;
                let xi0 = ((((q >> 4) & 0x000F) as u32) | xh_0) as f32 * d + m;
                let xi1 = ((((q >> 12) & 0x000F) as u32) | xh_1) as f32 * d + m;
                let (x0, x1) = unpack_u16x2(x, x_base + 16 + j);
                acc = float::fma_rn_f32(xi0, x0, acc);
                acc = float::fma_rn_f32(xi1, x1, acc);
                j += 2;
            }
            acc
        }
    }

    /// Q8_0 (ggml): 32-elem, f16 d + 32 int8. w = d * q.
    pub struct Q8_0;
    impl QuantVariant for Q8_0 {
        const FUSED_IDX: usize = 6;
        const BLOCK_BYTES: usize = 34;
        const BLOCK_ELEMS: usize = 32;
        const FUSED_DUAL: bool = true;
        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d = f16::from_bits(u16::from_le_bytes([unsafe { *packed.get_unchecked(off) }, unsafe { *packed.get_unchecked(off + 1) }])) as f32;
            let mut acc = 0.0f32;
            let mut i = 0usize;
            while i < 32 {
                let q = u16::from_le_bytes([unsafe { *packed.get_unchecked(off + 2 + i) }, unsafe { *packed.get_unchecked(off + 3 + i) }]);
                let q0 = (q & 0x00FF) as u8 as i8 as f32;
                let q1 = (q >> 8) as u8 as i8 as f32;
                let (x0, x1) = unpack_u16x2(x, x_base + i);
                acc = float::fma_rn_f32(d * q0, x0, acc);
                acc = float::fma_rn_f32(d * q1, x1, acc);
                i += 2;
            }
            acc
        }
    }

    /// Q8_1 (ggml): 32-elem, f16 d, f16 s (ignored), 32 int8. w = d * q.
    pub struct Q8_1;
    impl QuantVariant for Q8_1 {
        const FUSED_IDX: usize = 7;
        const BLOCK_BYTES: usize = 36;
        const BLOCK_ELEMS: usize = 32;
        const FUSED_DUAL: bool = true;
        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d = f16::from_bits(u16::from_le_bytes([unsafe { *packed.get_unchecked(off) }, unsafe { *packed.get_unchecked(off + 1) }])) as f32;
            let mut acc = 0.0f32;
            let mut i = 0usize;
            while i < 32 {
                let q = u16::from_le_bytes([unsafe { *packed.get_unchecked(off + 4 + i) }, unsafe { *packed.get_unchecked(off + 5 + i) }]);
                let q0 = (q & 0x00FF) as u8 as i8 as f32;
                let q1 = (q >> 8) as u8 as i8 as f32;
                let (x0, x1) = unpack_u16x2(x, x_base + i);
                acc = float::fma_rn_f32(d * q0, x0, acc);
                acc = float::fma_rn_f32(d * q1, x1, acc);
                i += 2;
            }
            acc
        }
    }

    const WARP_SUMS: usize = 32;



    /// Fused dual dot for q1_0.
    #[inline(always)]
    fn fused_dot_q1_0(w1: &[u8], w3: &[u8], off: usize, x: &[u16], x_base: usize) -> (f32, f32) {
        let d1 = f16::from_bits(u16::from_le_bytes([unsafe { *w1.get_unchecked(off) }, unsafe { *w1.get_unchecked(off + 1) }])) as f32;
        let d3 = f16::from_bits(u16::from_le_bytes([unsafe { *w3.get_unchecked(off) }, unsafe { *w3.get_unchecked(off + 1) }])) as f32;
        let mut a1 = 0.0f32;
        let mut a3 = 0.0f32;
        let mut w_idx = 0usize;
        while w_idx < 8 {
            let q1 = u16::from_le_bytes([unsafe { *w1.get_unchecked(off + 2 + w_idx * 2) }, unsafe { *w1.get_unchecked(off + 3 + w_idx * 2) }]);
            let q3 = u16::from_le_bytes([unsafe { *w3.get_unchecked(off + 2 + w_idx * 2) }, unsafe { *w3.get_unchecked(off + 3 + w_idx * 2) }]);
            let lo1 = (q1 & 0x00FF) as u8;
            let hi1 = (q1 >> 8) as u8;
            let lo3 = (q3 & 0x00FF) as u8;
            let hi3 = (q3 >> 8) as u8;
            let mut bit = 0usize;
            while bit < 16 {
                let b1 = (if bit < 8 { lo1 } else { hi1 } >> (bit % 8)) & 1;
                let b3 = (if bit < 8 { lo3 } else { hi3 } >> (bit % 8)) & 1;
                let xj = x_unchecked(x, x_base + w_idx * 16 + bit);
                a1 = float::fma_rn_f32(if b1 != 0 { d1 } else { -d1 }, xj, a1);
                a3 = float::fma_rn_f32(if b3 != 0 { d3 } else { -d3 }, xj, a3);
                bit += 1;
            }
            w_idx += 1;
        }
        (a1, a3)
    }

    /// Fused dual dot for q2_0.
    #[inline(always)]
    fn fused_dot_q2_0(w1: &[u8], w3: &[u8], off: usize, x: &[u16], x_base: usize) -> (f32, f32) {
        let d1 = f16::from_bits(u16::from_le_bytes([unsafe { *w1.get_unchecked(off) }, unsafe { *w1.get_unchecked(off + 1) }])) as f32;
        let d3 = f16::from_bits(u16::from_le_bytes([unsafe { *w3.get_unchecked(off) }, unsafe { *w3.get_unchecked(off + 1) }])) as f32;
        let mut a1 = 0.0f32;
        let mut a3 = 0.0f32;
        let mut i = 0usize;
        while i < 16 {
            let q1 = u16::from_le_bytes([unsafe { *w1.get_unchecked(off + 2 + i) }, unsafe { *w1.get_unchecked(off + 3 + i) }]);
            let q3 = u16::from_le_bytes([unsafe { *w3.get_unchecked(off + 2 + i) }, unsafe { *w3.get_unchecked(off + 3 + i) }]);
            let bytes1 = [(q1 & 0x00FF) as u8, (q1 >> 8) as u8];
            let bytes3 = [(q3 & 0x00FF) as u8, (q3 >> 8) as u8];
            for bi in 0..2 {
                let b1 = bytes1[bi];
                let b3 = bytes3[bi];
                let elem_base = (i + bi) * 4;
                let mut k = 0usize;
                while k < 4 {
                    let v1 = ((b1 >> (k * 2)) & 0x03) as f32 - 1.0;
                    let v3 = ((b3 >> (k * 2)) & 0x03) as f32 - 1.0;
                    let xj = x_unchecked(x, x_base + elem_base + k);
                    a1 = float::fma_rn_f32(v1 * d1, xj, a1);
                    a3 = float::fma_rn_f32(v3 * d3, xj, a3);
                    k += 1;
                }
            }
            i += 2;
        }
        (a1, a3)
    }

    /// Fused dual dot for q4_1.
    #[inline(always)]
    fn fused_dot_q4_1(w1: &[u8], w3: &[u8], off: usize, x: &[u16], x_base: usize) -> (f32, f32) {
        let d1 = f16::from_bits(u16::from_le_bytes([unsafe { *w1.get_unchecked(off) }, unsafe { *w1.get_unchecked(off + 1) }])) as f32;
        let m1 = f16::from_bits(u16::from_le_bytes([unsafe { *w1.get_unchecked(off + 2) }, unsafe { *w1.get_unchecked(off + 3) }])) as f32;
        let d3 = f16::from_bits(u16::from_le_bytes([unsafe { *w3.get_unchecked(off) }, unsafe { *w3.get_unchecked(off + 1) }])) as f32;
        let m3 = f16::from_bits(u16::from_le_bytes([unsafe { *w3.get_unchecked(off + 2) }, unsafe { *w3.get_unchecked(off + 3) }])) as f32;
        let mut a1 = 0.0f32;
        let mut b1 = 0.0f32;
        let mut a3 = 0.0f32;
        let mut b3 = 0.0f32;
        let mut i = 0usize;
        while i < 16 {
            let q1 = u16::from_le_bytes([unsafe { *w1.get_unchecked(off + 4 + i) }, unsafe { *w1.get_unchecked(off + 5 + i) }]);
            let q3 = u16::from_le_bytes([unsafe { *w3.get_unchecked(off + 4 + i) }, unsafe { *w3.get_unchecked(off + 5 + i) }]);
            let w_i = (q1 & 0x000F) as f32 * d1 + m1;
            let w_i1 = ((q1 >> 8) & 0x000F) as f32 * d1 + m1;
            let w_i16 = ((q1 >> 4) & 0x000F) as f32 * d1 + m1;
            let w_i17 = ((q1 >> 12) & 0x000F) as f32 * d1 + m1;
            let v_i = (q3 & 0x000F) as f32 * d3 + m3;
            let v_i1 = ((q3 >> 8) & 0x000F) as f32 * d3 + m3;
            let v_i16 = ((q3 >> 4) & 0x000F) as f32 * d3 + m3;
            let v_i17 = ((q3 >> 12) & 0x000F) as f32 * d3 + m3;
            let (xi, xi1) = unpack_u16x2(x, x_base + i);
            let (xi16, xi17) = unpack_u16x2(x, x_base + 16 + i);
            a1 = float::fma_rn_f32(w_i, xi, a1);
            b1 = float::fma_rn_f32(w_i1, xi1, b1);
            a1 = float::fma_rn_f32(w_i16, xi16, a1);
            b1 = float::fma_rn_f32(w_i17, xi17, b1);
            a3 = float::fma_rn_f32(v_i, xi, a3);
            b3 = float::fma_rn_f32(v_i1, xi1, b3);
            a3 = float::fma_rn_f32(v_i16, xi16, a3);
            b3 = float::fma_rn_f32(v_i17, xi17, b3);
            i += 2;
        }
        ((a1 + b1), (a3 + b3))
    }

    /// Fused dual dot for q5_0.
    #[inline(always)]
    fn fused_dot_q5_0(w1: &[u8], w3: &[u8], off: usize, x: &[u16], x_base: usize) -> (f32, f32) {
        let d1 = f16::from_bits(u16::from_le_bytes([unsafe { *w1.get_unchecked(off) }, unsafe { *w1.get_unchecked(off + 1) }])) as f32;
        let qh1_0 = u16::from_le_bytes([unsafe { *w1.get_unchecked(off + 2) }, unsafe { *w1.get_unchecked(off + 3) }]);
        let qh1_1 = u16::from_le_bytes([unsafe { *w1.get_unchecked(off + 4) }, unsafe { *w1.get_unchecked(off + 5) }]);
        let qh1 = (qh1_0 as u32) | ((qh1_1 as u32) << 16);
        let d3 = f16::from_bits(u16::from_le_bytes([unsafe { *w3.get_unchecked(off) }, unsafe { *w3.get_unchecked(off + 1) }])) as f32;
        let qh3_0 = u16::from_le_bytes([unsafe { *w3.get_unchecked(off + 2) }, unsafe { *w3.get_unchecked(off + 3) }]);
        let qh3_1 = u16::from_le_bytes([unsafe { *w3.get_unchecked(off + 4) }, unsafe { *w3.get_unchecked(off + 5) }]);
        let qh3 = (qh3_0 as u32) | ((qh3_1 as u32) << 16);
        let mut a1 = 0.0f32;
        let mut a3 = 0.0f32;
        let mut j = 0usize;
        while j < 16 {
            let q1 = u16::from_le_bytes([unsafe { *w1.get_unchecked(off + 6 + j) }, unsafe { *w1.get_unchecked(off + 7 + j) }]);
            let q3 = u16::from_le_bytes([unsafe { *w3.get_unchecked(off + 6 + j) }, unsafe { *w3.get_unchecked(off + 7 + j) }]);
            let xh10 = ((qh1 >> (j + 0)) << 4) & 0x10;
            let xh11 = ((qh1 >> (j + 1)) << 4) & 0x10;
            let xh30 = ((qh3 >> (j + 0)) << 4) & 0x10;
            let xh31 = ((qh3 >> (j + 1)) << 4) & 0x10;
            let xi0 = ((q1 & 0x000F) as u32 | xh10) as f32 - 16.0;
            let xi1 = (((q1 >> 8) & 0x000F) as u32 | xh11) as f32 - 16.0;
            let yi0 = ((q3 & 0x000F) as u32 | xh30) as f32 - 16.0;
            let yi1 = (((q3 >> 8) & 0x000F) as u32 | xh31) as f32 - 16.0;
            let x0 = f16::from_bits(x[x_base + j]) as f32;
            let x1 = f16::from_bits(x[x_base + j + 1]) as f32;
            a1 = float::fma_rn_f32(d1 * xi0, x0, a1);
            a1 = float::fma_rn_f32(d1 * xi1, x1, a1);
            a3 = float::fma_rn_f32(d3 * yi0, x0, a3);
            a3 = float::fma_rn_f32(d3 * yi1, x1, a3);
            j += 2;
        }
        let mut j = 0usize;
        while j < 16 {
            let q1 = u16::from_le_bytes([unsafe { *w1.get_unchecked(off + 6 + j) }, unsafe { *w1.get_unchecked(off + 7 + j) }]);
            let q3 = u16::from_le_bytes([unsafe { *w3.get_unchecked(off + 6 + j) }, unsafe { *w3.get_unchecked(off + 7 + j) }]);
            let xh10 = (qh1 >> (j + 12)) & 0x10;
            let xh11 = (qh1 >> (j + 13)) & 0x10;
            let xh30 = (qh3 >> (j + 12)) & 0x10;
            let xh31 = (qh3 >> (j + 13)) & 0x10;
            let xi0 = ((((q1 >> 4) & 0x000F) as u32) | xh10) as f32 - 16.0;
            let xi1 = ((((q1 >> 12) & 0x000F) as u32) | xh11) as f32 - 16.0;
            let yi0 = ((((q3 >> 4) & 0x000F) as u32) | xh30) as f32 - 16.0;
            let yi1 = ((((q3 >> 12) & 0x000F) as u32) | xh31) as f32 - 16.0;
            let x0 = f16::from_bits(x[x_base + 16 + j]) as f32;
            let x1 = f16::from_bits(x[x_base + 16 + j + 1]) as f32;
            a1 = float::fma_rn_f32(d1 * xi0, x0, a1);
            a1 = float::fma_rn_f32(d1 * xi1, x1, a1);
            a3 = float::fma_rn_f32(d3 * yi0, x0, a3);
            a3 = float::fma_rn_f32(d3 * yi1, x1, a3);
            j += 2;
        }
        (a1, a3)
    }

    /// Fused dual dot for q5_1.
    #[inline(always)]
    fn fused_dot_q5_1(w1: &[u8], w3: &[u8], off: usize, x: &[u16], x_base: usize) -> (f32, f32) {
        let d1 = f16::from_bits(u16::from_le_bytes([unsafe { *w1.get_unchecked(off) }, unsafe { *w1.get_unchecked(off + 1) }])) as f32;
        let m1 = f16::from_bits(u16::from_le_bytes([unsafe { *w1.get_unchecked(off + 2) }, unsafe { *w1.get_unchecked(off + 3) }])) as f32;
        let qh1_0 = u16::from_le_bytes([unsafe { *w1.get_unchecked(off + 4) }, unsafe { *w1.get_unchecked(off + 5) }]);
        let qh1_1 = u16::from_le_bytes([unsafe { *w1.get_unchecked(off + 6) }, unsafe { *w1.get_unchecked(off + 7) }]);
        let qh1 = (qh1_0 as u32) | ((qh1_1 as u32) << 16);
        let d3 = f16::from_bits(u16::from_le_bytes([unsafe { *w3.get_unchecked(off) }, unsafe { *w3.get_unchecked(off + 1) }])) as f32;
        let m3 = f16::from_bits(u16::from_le_bytes([unsafe { *w3.get_unchecked(off + 2) }, unsafe { *w3.get_unchecked(off + 3) }])) as f32;
        let qh3_0 = u16::from_le_bytes([unsafe { *w3.get_unchecked(off + 4) }, unsafe { *w3.get_unchecked(off + 5) }]);
        let qh3_1 = u16::from_le_bytes([unsafe { *w3.get_unchecked(off + 6) }, unsafe { *w3.get_unchecked(off + 7) }]);
        let qh3 = (qh3_0 as u32) | ((qh3_1 as u32) << 16);
        let mut a1 = 0.0f32;
        let mut a3 = 0.0f32;
        let mut j = 0usize;
        while j < 16 {
            let q1 = u16::from_le_bytes([unsafe { *w1.get_unchecked(off + 8 + j) }, unsafe { *w1.get_unchecked(off + 9 + j) }]);
            let q3 = u16::from_le_bytes([unsafe { *w3.get_unchecked(off + 8 + j) }, unsafe { *w3.get_unchecked(off + 9 + j) }]);
            let xh10 = ((qh1 >> (j + 0)) << 4) & 0x10;
            let xh11 = ((qh1 >> (j + 1)) << 4) & 0x10;
            let xh30 = ((qh3 >> (j + 0)) << 4) & 0x10;
            let xh31 = ((qh3 >> (j + 1)) << 4) & 0x10;
            let xi0 = ((q1 & 0x000F) as u32 | xh10) as f32 * d1 + m1;
            let xi1 = (((q1 >> 8) & 0x000F) as u32 | xh11) as f32 * d1 + m1;
            let yi0 = ((q3 & 0x000F) as u32 | xh30) as f32 * d3 + m3;
            let yi1 = (((q3 >> 8) & 0x000F) as u32 | xh31) as f32 * d3 + m3;
            let x0 = f16::from_bits(x[x_base + j]) as f32;
            let x1 = f16::from_bits(x[x_base + j + 1]) as f32;
            a1 = float::fma_rn_f32(xi0, x0, a1);
            a1 = float::fma_rn_f32(xi1, x1, a1);
            a3 = float::fma_rn_f32(yi0, x0, a3);
            a3 = float::fma_rn_f32(yi1, x1, a3);
            j += 2;
        }
        let mut j = 0usize;
        while j < 16 {
            let q1 = u16::from_le_bytes([unsafe { *w1.get_unchecked(off + 8 + j) }, unsafe { *w1.get_unchecked(off + 9 + j) }]);
            let q3 = u16::from_le_bytes([unsafe { *w3.get_unchecked(off + 8 + j) }, unsafe { *w3.get_unchecked(off + 9 + j) }]);
            let xh10 = (qh1 >> (j + 12)) & 0x10;
            let xh11 = (qh1 >> (j + 13)) & 0x10;
            let xh30 = (qh3 >> (j + 12)) & 0x10;
            let xh31 = (qh3 >> (j + 13)) & 0x10;
            let xi0 = ((((q1 >> 4) & 0x000F) as u32) | xh10) as f32 * d1 + m1;
            let xi1 = ((((q1 >> 12) & 0x000F) as u32) | xh11) as f32 * d1 + m1;
            let yi0 = ((((q3 >> 4) & 0x000F) as u32) | xh30) as f32 * d3 + m3;
            let yi1 = ((((q3 >> 12) & 0x000F) as u32) | xh31) as f32 * d3 + m3;
            let x0 = f16::from_bits(x[x_base + 16 + j]) as f32;
            let x1 = f16::from_bits(x[x_base + 16 + j + 1]) as f32;
            a1 = float::fma_rn_f32(xi0, x0, a1);
            a1 = float::fma_rn_f32(xi1, x1, a1);
            a3 = float::fma_rn_f32(yi0, x0, a3);
            a3 = float::fma_rn_f32(yi1, x1, a3);
            j += 2;
        }
        (a1, a3)
    }

    /// Fused dual dot for q8_0.
    #[inline(always)]
    fn fused_dot_q8_0(w1: &[u8], w3: &[u8], off: usize, x: &[u16], x_base: usize) -> (f32, f32) {
        let d1 = f16::from_bits(u16::from_le_bytes([unsafe { *w1.get_unchecked(off) }, unsafe { *w1.get_unchecked(off + 1) }])) as f32;
        let d3 = f16::from_bits(u16::from_le_bytes([unsafe { *w3.get_unchecked(off) }, unsafe { *w3.get_unchecked(off + 1) }])) as f32;
        let mut a1 = 0.0f32;
        let mut a3 = 0.0f32;
        let mut i = 0usize;
        while i < 32 {
            let q1 = u16::from_le_bytes([unsafe { *w1.get_unchecked(off + 2 + i) }, unsafe { *w1.get_unchecked(off + 3 + i) }]);
            let q3 = u16::from_le_bytes([unsafe { *w3.get_unchecked(off + 2 + i) }, unsafe { *w3.get_unchecked(off + 3 + i) }]);
            let q10 = (q1 & 0x00FF) as u8 as i8 as f32;
            let q11 = (q1 >> 8) as u8 as i8 as f32;
            let q30 = (q3 & 0x00FF) as u8 as i8 as f32;
            let q31 = (q3 >> 8) as u8 as i8 as f32;
            let (xi, xi1) = unpack_u16x2(x, x_base + i);
            a1 = float::fma_rn_f32(d1 * q10, xi, a1);
            a1 = float::fma_rn_f32(d1 * q11, xi1, a1);
            a3 = float::fma_rn_f32(d3 * q30, xi, a3);
            a3 = float::fma_rn_f32(d3 * q31, xi1, a3);
            i += 2;
        }
        (a1, a3)
    }

    /// Fused dual dot for q8_1.
    #[inline(always)]
    fn fused_dot_q8_1(w1: &[u8], w3: &[u8], off: usize, x: &[u16], x_base: usize) -> (f32, f32) {
        let d1 = f16::from_bits(u16::from_le_bytes([unsafe { *w1.get_unchecked(off) }, unsafe { *w1.get_unchecked(off + 1) }])) as f32;
        let d3 = f16::from_bits(u16::from_le_bytes([unsafe { *w3.get_unchecked(off) }, unsafe { *w3.get_unchecked(off + 1) }])) as f32;
        let mut a1 = 0.0f32;
        let mut a3 = 0.0f32;
        let mut i = 0usize;
        while i < 32 {
            let q1 = u16::from_le_bytes([unsafe { *w1.get_unchecked(off + 4 + i) }, unsafe { *w1.get_unchecked(off + 5 + i) }]);
            let q3 = u16::from_le_bytes([unsafe { *w3.get_unchecked(off + 4 + i) }, unsafe { *w3.get_unchecked(off + 5 + i) }]);
            let q10 = (q1 & 0x00FF) as u8 as i8 as f32;
            let q11 = (q1 >> 8) as u8 as i8 as f32;
            let q30 = (q3 & 0x00FF) as u8 as i8 as f32;
            let q31 = (q3 >> 8) as u8 as i8 as f32;
            let (xi, xi1) = unpack_u16x2(x, x_base + i);
            a1 = float::fma_rn_f32(d1 * q10, xi, a1);
            a1 = float::fma_rn_f32(d1 * q11, xi1, a1);
            a3 = float::fma_rn_f32(d3 * q30, xi, a3);
            a3 = float::fma_rn_f32(d3 * q31, xi1, a3);
            i += 2;
        }
        (a1, a3)
    }


    #[inline(always)]
    fn call_fused_dot<Q: QuantVariant>(w1: &[u8], w3: &[u8], off: usize, x: &[u16], x_base: usize) -> (f32, f32) {
        // Const-generic static dispatch: resolved at compile time per monomorphization.
        match Q::FUSED_IDX {
            0 => fused_dot_q1_0(w1, w3, off, x, x_base),
            1 => fused_dot_q2_0(w1, w3, off, x, x_base),
            2 => fused_dot_q4_0(w1, w3, off, x, x_base),
            3 => fused_dot_q4_1(w1, w3, off, x, x_base),
            4 => fused_dot_q5_0(w1, w3, off, x, x_base),
            5 => fused_dot_q5_1(w1, w3, off, x, x_base),
            6 => fused_dot_q8_0(w1, w3, off, x, x_base),
            7 => fused_dot_q8_1(w1, w3, off, x, x_base),
            _ => unreachable!(),
        }
    }

    /// Fused dual dot: returns (dot(w1row, x), dot(w3row, x)) sharing the x unpack.
    #[inline(always)]
    fn fused_dot_q4_0(w1: &[u8], w3: &[u8], off: usize, x: &[u16], x_base: usize) -> (f32, f32) {
        let d1 = f16::from_bits(u16::from_le_bytes([unsafe { *w1.get_unchecked(off) }, unsafe { *w1.get_unchecked(off + 1) }])) as f32;
        let d3 = f16::from_bits(u16::from_le_bytes([unsafe { *w3.get_unchecked(off) }, unsafe { *w3.get_unchecked(off + 1) }])) as f32;
        let mut a1 = 0.0f32;
        let mut b1 = 0.0f32;
        let mut a3 = 0.0f32;
        let mut b3 = 0.0f32;
        let mut i = 0usize;
        while i < 16 {
            let q1 = u16::from_le_bytes([unsafe { *w1.get_unchecked(off + 2 + i) }, unsafe { *w1.get_unchecked(off + 3 + i) }]);
            let q3 = u16::from_le_bytes([unsafe { *w3.get_unchecked(off + 2 + i) }, unsafe { *w3.get_unchecked(off + 3 + i) }]);
            let w_i = ((q1 & 0x000F) as f32 - 8.0) * d1;
            let w_i1 = (((q1 >> 8) & 0x000F) as f32 - 8.0) * d1;
            let w_i16 = (((q1 >> 4) & 0x000F) as f32 - 8.0) * d1;
            let w_i17 = (((q1 >> 12) & 0x000F) as f32 - 8.0) * d1;
            let v_i = ((q3 & 0x000F) as f32 - 8.0) * d3;
            let v_i1 = (((q3 >> 8) & 0x000F) as f32 - 8.0) * d3;
            let v_i16 = (((q3 >> 4) & 0x000F) as f32 - 8.0) * d3;
            let v_i17 = (((q3 >> 12) & 0x000F) as f32 - 8.0) * d3;
            let (xi, xi1) = unpack_u16x2(x, x_base + i);
            let (xi16, xi17) = unpack_u16x2(x, x_base + 16 + i);
            a1 = float::fma_rn_f32(w_i, xi, a1);
            b1 = float::fma_rn_f32(w_i1, xi1, b1);
            a1 = float::fma_rn_f32(w_i16, xi16, a1);
            b1 = float::fma_rn_f32(w_i17, xi17, b1);
            a3 = float::fma_rn_f32(v_i, xi, a3);
            b3 = float::fma_rn_f32(v_i1, xi1, b3);
            a3 = float::fma_rn_f32(v_i16, xi16, a3);
            b3 = float::fma_rn_f32(v_i17, xi17, b3);
            i += 2;
        }
        ((a1 + b1), (a3 + b3))
    }

    /// Load two consecutive f16 values from `x` as one aligned u32 read.
    /// `idx` must be even; CUDA buffers are 256-byte aligned.
    #[inline(always)]
    fn unpack_u16x2(x: &[u16], idx: usize) -> (f32, f32) {
        let xp = x.as_ptr() as *const u32;
        // SAFETY: idx even, and x's backing allocation covers at least idx+1.
        let v = unsafe { xp.add(idx / 2).read() };
        convert::cvt_f32x2_f16x2(v)
    }

    /// Load one f16 from `x` without bounds checks.
    #[inline(always)]
    fn x_unchecked(x: &[u16], idx: usize) -> f32 {
        // SAFETY: caller guarantees idx < x.len().
        let v = unsafe { x.get_unchecked(idx) };
        f16::from_bits(*v) as f32
    }

    fn warp_tree_sum(mut val: f32) -> f32 {
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 16);
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 8);
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 4);
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 2);
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 1);
        val
    }

    /// Fused dual GEMV (W1, W3) + SwiGLU activation.
    /// One warp per row, 4 warps per CTA; grid (INTER / 4, B, 1).
    /// act[b*n + row] = f16(up(row) * silu(gate(row))).
    pub fn fused_dual_swiglu_impl<Q: QuantVariant + 'static>(
        w1: &[u8],
        w3: &[u8],
        x: &[u16],
        mut act: DisjointSlice<u16>,
        hidden: u32,
        n_blocks: u32,
        n_elems: u32,
    ) {
        const ROWS_PER_CTA: usize = 4;
        let tid = thread::threadIdx_x() as usize;
        let lane = (tid & 31) as usize;
        let warp_id = tid / 32;
        let row = thread::blockIdx_x() as usize * ROWS_PER_CTA + warp_id;
        let b_idx = thread::blockIdx_y() as usize;
        if row >= n_elems as usize {
            return;
        }
        let nb = n_blocks as usize;
        let row_off = row * nb * Q::BLOCK_BYTES;
        let x_base = b_idx * hidden as usize;

        let mut acc1 = 0.0f32;
        let mut acc3 = 0.0f32;
        let mut b = lane;
        if Q::FUSED_DUAL {
            while b < nb {
                let off = row_off + b * Q::BLOCK_BYTES;
                let xb = b * Q::BLOCK_ELEMS;
                let (d1, d3) = call_fused_dot::<Q>(w1, w3, off, x, x_base + xb);
                acc1 += d1;
                acc3 += d3;
                b += 32;
            }
        } else {
            while b < nb {
                let off = row_off + b * Q::BLOCK_BYTES;
                let xb = b * Q::BLOCK_ELEMS;
                acc1 += Q::dot_block(w1, off, x, x_base + xb);
                acc3 += Q::dot_block(w3, off, x, x_base + xb);
                b += 32;
            }
        }

        acc1 = warp_tree_sum(acc1);
        acc3 = warp_tree_sum(acc3);

        if lane == 0 {
            let ex = float::ex2_approx_f32(acc1 * -1.4426950208889634f32);
            let value = acc3 * (acc1 / (1.0 + ex));
            let packed32 = convert::cvt_f16x2_f32(value, 0.0);
            // SAFETY: one writer per (row, b) pair; rows/batch disjoint.
            unsafe { *act.get_unchecked_mut(b_idx * n_elems as usize + row) = packed32 as u16; }
        }
    }

    /// Plain GEMV over packed weights (used for the W2 down projection).
    /// One warp per row, 4 warps per CTA; grid (N / 4, B, 1).
    pub fn gemv_impl<Q: QuantVariant + 'static>(
        w: &[u8],
        x: &[u16],
        mut y: DisjointSlice<u16>,
        hidden: u32,
        n_blocks: u32,
        n_elems: u32,
    ) {
        const ROWS_PER_CTA: usize = 4;
        let tid = thread::threadIdx_x() as usize;
        let lane = (tid & 31) as usize;
        let warp_id = tid / 32;
        let row = thread::blockIdx_x() as usize * ROWS_PER_CTA + warp_id;
        let b_idx = thread::blockIdx_y() as usize;
        if row >= n_elems as usize {
            return;
        }
        let nb = n_blocks as usize;
        let row_off = row * nb * Q::BLOCK_BYTES;
        let x_base = b_idx * hidden as usize;

        let mut acc = 0.0f32;
        let mut b = lane;
        while b < nb {
            let off = row_off + b * Q::BLOCK_BYTES;
            acc += Q::dot_block(w, off, x, x_base + b * Q::BLOCK_ELEMS);
            b += 32;
        }

        acc = warp_tree_sum(acc);
        if lane == 0 {
            let packed32 = convert::cvt_f16x2_f32(acc, 0.0);
            // SAFETY: one writer per (row, b) pair; rows/batch disjoint.
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

        let cfg = LaunchConfig { grid_dim: ((INTER as u32).div_ceil(4), B as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            $module.$fused(
                &$stream, cfg, &w1_dev, &w3_dev, &x1_dev, &mut act_dev,
                HIDDEN as u32, (HIDDEN / $elems) as u32, INTER as u32,
            )
        }
        .expect(concat!($name, " dual launch failed"));

        let cfg = LaunchConfig { grid_dim: ((HIDDEN as u32).div_ceil(4), B as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
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

    if std::env::var("OXIDE_QUANT_BENCH").is_ok() {
        bench();
        return;
    }
    if worst == 0 {
        println!("all {checks} variant(s) pass: packed kernels match host f64 reference");
    } else {
        eprintln!("{worst} variant(s) FAILED");
        std::process::exit(1);
    }
}

// ---- micro-benchmark of the production-shape packed kernel --------------
fn bench() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    const HIDDEN: usize = 2048;
    const INTER: usize = 7680;
    let quantized = oxide_infer_core::quant::quantize_q4_0(&vec![0.01f32; INTER * HIDDEN]);
    let x_host: Vec<u16> = (0..HIDDEN).map(|i| f16::from_f32((i as f32 * 0.01).sin()).to_bits()).collect();

    let w1_dev = DeviceBuffer::from_host(&stream, &quantized).unwrap();
    let w3_dev = DeviceBuffer::from_host(&stream, &quantized).unwrap();
    let x_dev = DeviceBuffer::from_host(&stream, &x_host).unwrap();
    let mut act_dev = DeviceBuffer::<u16>::zeroed(&stream, INTER).unwrap();

    let cfg = LaunchConfig { grid_dim: (INTER as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
    for _ in 0..5 {
        unsafe {
            module.fused_dual_gemv_swiglu_q4_0(&stream, cfg, &w1_dev, &w3_dev, &x_dev, &mut act_dev,
                HIDDEN as u32, (HIDDEN / 32) as u32, INTER as u32)
        }.expect("launch");
    }
    stream.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    let iters = 50;
    for _ in 0..iters {
        unsafe {
            module.fused_dual_gemv_swiglu_q4_0(&stream, cfg, &w1_dev, &w3_dev, &x_dev, &mut act_dev,
                HIDDEN as u32, (HIDDEN / 32) as u32, INTER as u32)
        }.expect("launch");
    }
    stream.synchronize().unwrap();
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let bytes = 3.0 * quantized.len() as f64 + HIDDEN as f64 * 2.0; // w1+w3 (+ act)
    println!("dual_swiglu q4_0 INTER={INTER} K={HIDDEN}: {:.3} ms/launch | {:.1} GB/s effective", dt * 1e3, bytes / dt / 1e9);

    // CPU-only enqueue cost: launches without waiting for the GPU.
    let t2 = std::time::Instant::now();
    for _ in 0..iters {
        unsafe {
            module.fused_dual_gemv_swiglu_q4_0(&stream, cfg, &w1_dev, &w3_dev, &x_dev, &mut act_dev,
                HIDDEN as u32, (HIDDEN / 32) as u32, INTER as u32)
        }.expect("launch");
    }
    let cpu = t2.elapsed().as_secs_f64() / iters as f64;
    stream.synchronize().unwrap();
    println!("dual_swiglu enqueue cost (CPU, no sync): {:.3} ms/launch", cpu * 1e3);
}
