//! Pilot: quantized GEMV kernels in cuda-oxide — one generic kernel source
//! monomorphized per quant variant -- the cuda-oxide analog of llama.cpp's
//! templated MMQ / vLLM's Marlin instantiation matrix.
//!
//! Coverage: all 8 simple ggml types (q1_0, q2_0, q4_0, q4_1, q5_0, q5_1,
//! q8_0, q8_1). K-types (256-element superblocks) are intentionally not
//! included yet: oxide-infer's K dequantizers have not been verified against
//! a ggml reference (output for K-quantized models is currently garbage),
//! so there is nothing faithful to port to the device yet.
//!
//! For each variant the kernel instantiation is launched on the GPU, and the
//! device dot product is checked against an f64 host reference computed from
//! the packed bytes with the same dequantize formula as
//! oxide-infer-core/src/quant.rs (mirrored here, kept in f64).

#![feature(f16)]

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use half::f16;

#[cuda_module]
mod kernels {
    use cuda_device::{DisjointSlice, SharedArray, float, kernel, thread, warp};

    /// One row of packed quantized weights dotted against the shared f16
    /// activation vector x. `out[row] = sum(dot(dequant(block_k), x_k))`.
    ///
    /// `x` carries f16 bits as u16 elements. `n_blocks` is the block count per
    /// row; each block covers `BLOCK_ELEMS` elements.
    pub trait QuantVariant {
        const BLOCK_BYTES: usize;
        const BLOCK_ELEMS: usize;

        /// Dot of one dequantized block against x[x_base .. x_base + BLOCK_ELEMS].
        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32;
    }

    /// q1_0 (ggml): 128-elem blocks = f16 `d` + 16 bitmap bytes (LSB first). w = bit ? d : -d.
    pub struct Q1_0;
    impl QuantVariant for Q1_0 {
        const BLOCK_BYTES: usize = 2 + 16;
        const BLOCK_ELEMS: usize = 128;

        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d_bits = u16::from_le_bytes([packed[off], packed[off + 1]]);
            let d = f16::from_bits(d_bits) as f32;
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

    /// q2_0 (ggml): 64-elem blocks = f16 `d` + 16 2-bit-packed bytes. w = (q - 1) * d.
    pub struct Q2_0;
    impl QuantVariant for Q2_0 {
        const BLOCK_BYTES: usize = 2 + 16;
        const BLOCK_ELEMS: usize = 64;

        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d_bits = u16::from_le_bytes([packed[off], packed[off + 1]]);
            let d = f16::from_bits(d_bits) as f32;
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

    /// Q4_0 (ggml): 32-elem blocks = f16 `d` + 16 nibble bytes. w = d*(q - 8).
    pub struct Q4_0;
    impl QuantVariant for Q4_0 {
        const BLOCK_BYTES: usize = 18;
        const BLOCK_ELEMS: usize = 32;

        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d_bits = u16::from_le_bytes([packed[off], packed[off + 1]]);
            let d = f16::from_bits(d_bits) as f32;
            let mut acc = 0.0f32;
            let mut i = 0usize;
            while i < 16 {
                let byte = packed[off + 2 + i];
                let x0 = f16::from_bits(x[x_base + 2 * i]) as f32;
                let x1 = f16::from_bits(x[x_base + 2 * i + 1]) as f32;
                let w0 = d * ((byte & 0x0F) as f32 - 8.0);
                let w1 = d * (((byte >> 4) & 0x0F) as f32 - 8.0);
                acc = float::fma_rn_f32(w0, x0, acc);
                acc = float::fma_rn_f32(w1, x1, acc);
                i += 1;
            }
            acc
        }
    }

    /// Q4_1 (ggml): 32-elem blocks = f16 `d`, f16 `m`, 16 nibble bytes. w = d*q + m.
    pub struct Q4_1;
    impl QuantVariant for Q4_1 {
        const BLOCK_BYTES: usize = 20;
        const BLOCK_ELEMS: usize = 32;

        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d_bits = u16::from_le_bytes([packed[off], packed[off + 1]]);
            let m_bits = u16::from_le_bytes([packed[off + 2], packed[off + 3]]);
            let d = f16::from_bits(d_bits) as f32;
            let m = f16::from_bits(m_bits) as f32;
            let mut acc = 0.0f32;
            let mut i = 0usize;
            while i < 16 {
                let byte = packed[off + 4 + i];
                let x0 = f16::from_bits(x[x_base + 2 * i]) as f32;
                let x1 = f16::from_bits(x[x_base + 2 * i + 1]) as f32;
                let w0 = d * (byte & 0x0F) as f32 + m;
                let w1 = d * ((byte >> 4) & 0x0F) as f32 + m;
                acc = float::fma_rn_f32(w0, x0, acc);
                acc = float::fma_rn_f32(w1, x1, acc);
                i += 1;
            }
            acc
        }
    }

    /// Q5_0 (ggml): 32-elem blocks = f16 `d`, 4 qh bytes, 16 nibble bytes. w = (q | h<<4) - 16) * d.
    pub struct Q5_0;
    impl QuantVariant for Q5_0 {
        const BLOCK_BYTES: usize = 22;
        const BLOCK_ELEMS: usize = 32;

        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d_bits = u16::from_le_bytes([packed[off], packed[off + 1]]);
            let d = f16::from_bits(d_bits) as f32;
            let qh = u32::from_le_bytes([packed[off + 2], packed[off + 3], packed[off + 4], packed[off + 5]]);
            let mut acc = 0.0f32;
            let mut j = 0usize;
            while j < 16 {
                let xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
                let xh_1 = ((qh >> (j + 12)) & 0x10) << 0;
                let byte = packed[off + 6 + j];
                let x0 = (((byte & 0x0F) as u32 | xh_0) as f32 - 16.0) * d;
                let x1 = (((byte >> 4) as u32 | xh_1) as f32 - 16.0) * d;
                let xi0 = f16::from_bits(x[x_base + 2 * j]) as f32;
                let xi1 = f16::from_bits(x[x_base + 2 * j + 1]) as f32;
                acc = float::fma_rn_f32(x0, xi0, acc);
                acc = float::fma_rn_f32(x1, xi1, acc);
                j += 1;
            }
            acc
        }
    }

    /// Q5_1 (ggml): 32-elem blocks = f16 `d`, f16 `m`, 4 qh bytes, 16 nibble bytes. w = d*q + m.
    pub struct Q5_1;
    impl QuantVariant for Q5_1 {
        const BLOCK_BYTES: usize = 24;
        const BLOCK_ELEMS: usize = 32;

        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d_bits = u16::from_le_bytes([packed[off], packed[off + 1]]);
            let m_bits = u16::from_le_bytes([packed[off + 2], packed[off + 3]]);
            let d = f16::from_bits(d_bits) as f32;
            let m = f16::from_bits(m_bits) as f32;
            let qh = u32::from_le_bytes([packed[off + 4], packed[off + 5], packed[off + 6], packed[off + 7]]);
            let mut acc = 0.0f32;
            let mut j = 0usize;
            while j < 16 {
                let xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
                let xh_1 = ((qh >> (j + 12)) & 0x10) << 0;
                let byte = packed[off + 8 + j];
                let x0 = ((byte & 0x0F) as u32 | xh_0) as f32 * d + m;
                let x1 = (((byte >> 4) as u32) | xh_1) as f32 * d + m;
                let xi0 = f16::from_bits(x[x_base + 2 * j]) as f32;
                let xi1 = f16::from_bits(x[x_base + 2 * j + 1]) as f32;
                acc = float::fma_rn_f32(x0, xi0, acc);
                acc = float::fma_rn_f32(x1, xi1, acc);
                j += 1;
            }
            acc
        }
    }

    /// Q8_0 (ggml): 32-elem blocks = f16 `d` + 32 signed int8. w = d * q.
    pub struct Q8_0;
    impl QuantVariant for Q8_0 {
        const BLOCK_BYTES: usize = 34;
        const BLOCK_ELEMS: usize = 32;

        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d_bits = u16::from_le_bytes([packed[off], packed[off + 1]]);
            let d = f16::from_bits(d_bits) as f32;
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

    /// Q8_1 (ggml): 32-elem blocks = f16 `d`, f16 `s` (ignored for plain dot), 32 signed int8. w = d * q.
    pub struct Q8_1;
    impl QuantVariant for Q8_1 {
        const BLOCK_BYTES: usize = 36;
        const BLOCK_ELEMS: usize = 32;

        fn dot_block(packed: &[u8], off: usize, x: &[u16], x_base: usize) -> f32 {
            let d_bits = u16::from_le_bytes([packed[off], packed[off + 1]]);
            let d = f16::from_bits(d_bits) as f32;
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

    #[kernel]
    pub fn gemv_q<Q: QuantVariant>(
        weights: &[u8],
        x: &[u16],
        mut out: DisjointSlice<f32>,
        n_blocks: u32,
    ) {
        static mut S_WARP_SUMS: SharedArray<f32, 32> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x();
        let ntid = thread::blockDim_x();
        let row_off = row * n_blocks as usize * Q::BLOCK_BYTES;

        let mut acc = 0.0f32;
        let mut b = tid as usize;
        while b < n_blocks as usize {
            let off = row_off + b * Q::BLOCK_BYTES;
            acc += Q::dot_block(weights, off, x, b * Q::BLOCK_ELEMS);
            b += ntid as usize;
        }

        // Warp reduction.
        let mut val = acc;
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 16);
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 8);
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 4);
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 2);
        val += warp::shuffle_down_f32_sync(0xffff_ffff, val, 1);

        let lane = warp::lane_id();
        let warp_id = warp::warp_id() as usize;
        if lane == 0 {
            // SAFETY: 32 slots, warp_id < 32 for blockDim <= 1024.
            unsafe { S_WARP_SUMS[warp_id] = val; }
        }
        thread::sync_threads();

        let num_warps = (ntid as usize).div_ceil(32);
        let mut total = 0.0f32;
        for w in 0..num_warps {
            // SAFETY: w < num_warps <= 32, every warp wrote its slot this launch.
            total += unsafe { S_WARP_SUMS[w] };
        }

        if tid == 0 {
            // SAFETY: one writer per row (grid.x == out.len), rows are disjoint.
            unsafe { *out.get_unchecked_mut(row) = total; }
        }
    }
}

use cuda_device::cuda_module;

const ROWS: usize = 3;
const K: usize = 256; // f16 elements per row; multiple of 128 (q1_0's block size)
const TOL: f64 = 2e-2;

fn to_bits(v: f32) -> u16 {
    f16::from_f32(v).to_bits()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 32) as u8
    }
}

fn scale_bytes(d: f32) -> [u8; 2] {
    f16::from_f32(d).to_bits().to_le_bytes()
}

// ---- Host references, mirroring oxide-infer-core::quant exactly (f64). ----
fn dequant_q1_0(packed: &[u8]) -> Vec<f64> {
    let mut out = Vec::new();
    for b in 0..packed.len() / 18 {
        let off = b * 18;
        let d = f16::from_bits(u16::from_le_bytes([packed[off], packed[off + 1]])).to_f64();
        for j in 0..128 {
            let byte = packed[off + 2 + j / 8];
            out.push(if (byte >> (j % 8)) & 1 != 0 { d } else { -d });
        }
    }
    out
}
fn dequant_q2_0(packed: &[u8]) -> Vec<f64> {
    let mut out = Vec::new();
    for b in 0..packed.len() / 18 {
        let off = b * 18;
        let d = f16::from_bits(u16::from_le_bytes([packed[off], packed[off + 1]])).to_f64();
        for j in 0..64 {
            let byte = packed[off + 2 + j / 4];
            let q = ((byte >> ((j % 4) * 2)) & 0x03) as f64;
            out.push((q - 1.0) * d);
        }
    }
    out
}
fn dequant_q4_0(packed: &[u8]) -> Vec<f64> {
    let mut out = Vec::new();
    for b in 0..packed.len() / 18 {
        let off = b * 18;
        let d = f16::from_bits(u16::from_le_bytes([packed[off], packed[off + 1]])).to_f64();
        for i in 0..16 {
            let byte = packed[off + 2 + i];
            out.push(d * ((byte & 0x0F) as f64 - 8.0));
            out.push(d * (((byte >> 4) & 0x0F) as f64 - 8.0));
        }
    }
    out
}
fn dequant_q4_1(packed: &[u8]) -> Vec<f64> {
    let mut out = Vec::new();
    for b in 0..packed.len() / 20 {
        let off = b * 20;
        let d = f16::from_bits(u16::from_le_bytes([packed[off], packed[off + 1]])).to_f64();
        let m = f16::from_bits(u16::from_le_bytes([packed[off + 2], packed[off + 3]])).to_f64();
        for i in 0..16 {
            let byte = packed[off + 4 + i];
            out.push(d * (byte & 0x0F) as f64 + m);
            out.push(d * ((byte >> 4) & 0x0F) as f64 + m);
        }
    }
    out
}
fn dequant_q5_0(packed: &[u8]) -> Vec<f64> {
    let mut out = Vec::new();
    for b in 0..packed.len() / 22 {
        let off = b * 22;
        let d = f16::from_bits(u16::from_le_bytes([packed[off], packed[off + 1]])).to_f64();
        let qh = u32::from_le_bytes([packed[off + 2], packed[off + 3], packed[off + 4], packed[off + 5]]);
        for j in 0..16 {
            let xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            let xh_1 = ((qh >> (j + 12)) & 0x10) << 0;
            let byte = packed[off + 6 + j];
            let x0 = (((byte & 0x0F) as u32 | xh_0) as f64 - 16.0) * d;
            let x1 = ((((byte >> 4) as u32) | xh_1) as f64 - 16.0) * d;
            out.push(x0);
            out.push(x1);
        }
    }
    out
}
fn dequant_q5_1(packed: &[u8]) -> Vec<f64> {
    let mut out = Vec::new();
    for b in 0..packed.len() / 24 {
        let off = b * 24;
        let d = f16::from_bits(u16::from_le_bytes([packed[off], packed[off + 1]])).to_f64();
        let m = f16::from_bits(u16::from_le_bytes([packed[off + 2], packed[off + 3]])).to_f64();
        let qh = u32::from_le_bytes([packed[off + 4], packed[off + 5], packed[off + 6], packed[off + 7]]);
        for j in 0..16 {
            let xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            let xh_1 = ((qh >> (j + 12)) & 0x10) << 0;
            let byte = packed[off + 8 + j];
            let x0 = ((byte & 0x0F) as u32 | xh_0) as f64 * d + m;
            let x1 = (((byte >> 4) as u32) | xh_1) as f64 * d + m;
            out.push(x0);
            out.push(x1);
        }
    }
    out
}
fn dequant_q8_0(packed: &[u8]) -> Vec<f64> {
    let mut out = Vec::new();
    for b in 0..packed.len() / 34 {
        let off = b * 34;
        let d = f16::from_bits(u16::from_le_bytes([packed[off], packed[off + 1]])).to_f64();
        for j in 0..32 {
            out.push(packed[off + 2 + j] as i8 as f64 * d);
        }
    }
    out
}
fn dequant_q8_1(packed: &[u8]) -> Vec<f64> {
    let mut out = Vec::new();
    for b in 0..packed.len() / 36 {
        let off = b * 36;
        let d = f16::from_bits(u16::from_le_bytes([packed[off], packed[off + 1]])).to_f64();
        for j in 0..32 {
            out.push(packed[off + 4 + j] as i8 as f64 * d);
        }
    }
    out
}

fn dot_row(ws: &[f64], x: &[f32]) -> f64 {
    ws.iter().zip(x).map(|(w, x)| w * (*x as f64)).sum()
}

fn gen_weights(block_bytes: usize, blocks_per_row: usize, header: &dyn Fn(&mut [u8], u64)) -> Vec<u8> {
    let mut v = Vec::with_capacity(block_bytes * blocks_per_row * ROWS);
    for row in 0..ROWS {
        for b in 0..blocks_per_row {
            let mut block = vec![0u8; block_bytes];
            let mut r = Rng(0xABCD_0000_0000 + (row as u64) * 0x1000 + b as u64);
            for slot in block.iter_mut() {
                *slot = r.byte();
            }
            header(&mut block, row as u64 * 1000 + b as u64);
            v.extend_from_slice(&block);
        }
    }
    v
}

fn main() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let x_f32: Vec<f32> = (0..K).map(|i| (i as f32 * 0.11).sin()).collect();
    let x_host: Vec<u16> = x_f32.iter().map(|&v| to_bits(v)).collect();
    let x_dev = DeviceBuffer::from_host(&stream, &x_host).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let mut status = true;

macro_rules! run_case {
        ($name:literal, $Q:ty, $block_bytes:literal, $block_elems:literal, $dequant:path, $d:literal, $m:literal) => {{
            let blocks_per_row = K / $block_elems;
            let weights = gen_weights($block_bytes, blocks_per_row, &|block: &mut [u8], _seed: u64| {
                block[..2].copy_from_slice(&scale_bytes($d));
                if $block_bytes == 20 || $block_bytes == 24 {
                    block[2..4].copy_from_slice(&scale_bytes($m));
                }
            });
            let mut refs = Vec::with_capacity(ROWS);
                for row in 0..ROWS {
                let start = row * blocks_per_row * $block_bytes;
                let row_bytes = &weights[start..start + blocks_per_row * $block_bytes];
                let ws = $dequant(row_bytes);
                assert_eq!(ws.len(), K, "{}: dequantized length mismatch", $name);
                refs.push(dot_row(&ws, &x_f32));
            }
    
            let w_dev = DeviceBuffer::from_host(&stream, &weights).unwrap();
            let mut o_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS).unwrap();
            let cfg = LaunchConfig {
                grid_dim: (ROWS as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                module.gemv_q::<$Q>(&stream, cfg, &w_dev, &x_dev, &mut o_dev, blocks_per_row as u32)
            }
            .expect(concat!($name, " launch failed"));
            let got = o_dev.to_host_vec(&stream).unwrap();
    
            let mut worst = 0.0f64;
            for r in 0..ROWS {
                worst = worst.max((got[r] as f64 - refs[r]).abs());
            }
            let ok = worst <= TOL;
            status &= ok;
            println!(
                "{:>5}: worst |dev - f64| = {:.3e} {}",
                $name,
                worst,
                if ok { "ok" } else { "FAIL" }
            );
        }};
    }

    run_case!("q1_0", kernels::Q1_0, 18, 128, dequant_q1_0, 0.02, 0.0);
    run_case!("q2_0", kernels::Q2_0, 18, 64, dequant_q2_0, 0.015, 0.0);
    run_case!("q4_0", kernels::Q4_0, 18, 32, dequant_q4_0, 0.02, 0.0);
    run_case!("q4_1", kernels::Q4_1, 20, 32, dequant_q4_1, 0.02, 0.005);
    run_case!("q5_0", kernels::Q5_0, 22, 32, dequant_q5_0, 0.02, 0.0);
    run_case!("q5_1", kernels::Q5_1, 24, 32, dequant_q5_1, 0.02, 0.005);
    run_case!("q8_0", kernels::Q8_0, 34, 32, dequant_q8_0, 0.01, 0.0);
    run_case!("q8_1", kernels::Q8_1, 36, 32, dequant_q8_1, 0.01, 0.0);

    if status {
        println!("pilot OK: all 8 simple quant variants match host reference");
    } else {
        eprintln!("pilot FAILED");
        std::process::exit(1);
    }
}
