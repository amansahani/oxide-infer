//! Pilot: quantized GEMV kernels in cuda-oxide, one generic kernel source
//! monomorphized per quant variant -- the cuda-oxide analog of llama.cpp's
//! templated MMQ / vLLM's Marlin instantiation matrix.
//!
//! Verifies for each of Q4_0 and Q8_0:
//!   host -> device -> kernel -> device -> host round-trip,
//!   device dot product vs host f32/f64 references,
//!   and coexistence of both monomorphized PTX entry points in one module.

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

    /// Q4_0 (ggml): 18-byte blocks = f16 `d` + 16 nibble bytes. w = d*(q - 8).
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

    /// Q8_0 (ggml): 34-byte blocks = f16 `d` + 32 signed int8. w = d*q.
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
const K: usize = 256; // f16 elements per row; must be a multiple of 32
const BLOCKS_PER_ROW: usize = K / 32;

fn to_bits(v: f32) -> u16 {
    f16::from_f32(v).to_bits()
}

fn q4_0_block(d: f32, nibbles: &[u8; 16]) -> [u8; 18] {
    let mut block = [0u8; 18];
    block[..2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
    block[2..].copy_from_slice(nibbles);
    block
}

fn q8_0_block(d: f32, qvals: &[i8; 32]) -> [u8; 34] {
    let mut block = [0u8; 34];
    block[..2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
    for (i, q) in qvals.iter().enumerate() {
        block[2 + i] = *q as u8;
    }
    block
}

// Deterministic pseudo-random byte pattern (xorshift).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let x_f32: Vec<f32> = (0..K).map(|i| (i as f32 * 0.11).sin()).collect();
    let x_host: Vec<u16> = x_f32.iter().map(|&v| to_bits(v)).collect();

    // ---- Synthesize quantized weights + references --------------------------
    let mut rng = Rng(0x9E3779B97F4A7C15);

    let mut w_q4 = Vec::with_capacity(ROWS * BLOCKS_PER_ROW * 18);
    let mut ref_q4_f64 = vec![0.0f64; ROWS];
    let mut w_q8 = Vec::with_capacity(ROWS * BLOCKS_PER_ROW * 34);
    let mut ref_q8_f64 = vec![0.0f64; ROWS];
    let mut ref_q8_f32 = vec![0.0f32; ROWS];

    for row in 0..ROWS {
        for b in 0..BLOCKS_PER_ROW {
            let nibbles: [u8; 16] = std::array::from_fn(|_| rng.next() as u8);
            let d4 = 0.02f32;
            let block4 = q4_0_block(d4, &nibbles);
            w_q4.extend_from_slice(&block4);
            for i in 0..16 {
                let byte = nibbles[i];
                let w0 = d4 * ((byte & 0x0F) as f32 - 8.0);
                let w1 = d4 * (((byte >> 4) & 0x0F) as f32 - 8.0);
                let x0 = x_f32[b * 32 + 2 * i] as f64;
                let x1 = x_f32[b * 32 + 2 * i + 1] as f64;
                ref_q4_f64[row] += w0 as f64 * x0 + w1 as f64 * x1;
            }
        }
        for b in 0..BLOCKS_PER_ROW {
            let qvals: [i8; 32] = std::array::from_fn(|_| (rng.next() % 255) as i8);
            let d8 = 0.01f32;
            let block8 = q8_0_block(d8, &qvals);
            w_q8.extend_from_slice(&block8);
            for i in 0..32 {
                let w = d8 * qvals[i] as f32;
                let xi = x_f32[b * 32 + i] as f64;
                ref_q8_f64[row] += w as f64 * xi;
                ref_q8_f32[row] += w * x_f32[b * 32 + i];
            }
        }
    }
    // Recompute f32 reference cleanly (fma-free, sequential).
    let ref_q4_f32: Vec<f32> = (0..ROWS)
        .map(|row| {
            let mut acc = 0.0f32;
            for (j, &xv) in x_f32.iter().enumerate() {
                let byte = w_q4[row * (BLOCKS_PER_ROW * 18) + (j / 32) * 18 + 2 + (j % 32) / 2];
                let d4 = f16::from_bits(u16::from_le_bytes([
                    w_q4[row * (BLOCKS_PER_ROW * 18) + (j / 32) * 18],
                    w_q4[row * (BLOCKS_PER_ROW * 18) + (j / 32) * 18 + 1],
                ]))
                .to_f32();
                let nib = if j % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
                acc += d4 * (nib as f32 - 8.0) * xv;
            }
            acc
        })
        .collect();

    let w4_dev = DeviceBuffer::from_host(&stream, &w_q4).unwrap();
    let w8_dev = DeviceBuffer::from_host(&stream, &w_q8).unwrap();
    let x_dev = DeviceBuffer::from_host(&stream, &x_host).unwrap();
    let mut o4_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS).unwrap();
    let mut o8_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = LaunchConfig {
        grid_dim: (ROWS as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        module.gemv_q::<kernels::Q4_0>(&stream, cfg, &w4_dev, &x_dev, &mut o4_dev, BLOCKS_PER_ROW as u32)
    }
    .expect("Q4_0 launch failed");
    let cfg = LaunchConfig {
        grid_dim: (ROWS as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        module.gemv_q::<kernels::Q8_0>(&stream, cfg, &w8_dev, &x_dev, &mut o8_dev, BLOCKS_PER_ROW as u32)
    }
    .expect("Q8_0 launch failed");

    let o4 = o4_dev.to_host_vec(&stream).unwrap();
    let o8 = o8_dev.to_host_vec(&stream).unwrap();

    let mut worst = 0.0f64;
    for r in 0..ROWS {
        let d4 = (o4[r] as f64 - ref_q4_f64[r]).abs();
        let d8 = (o8[r] as f64 - ref_q8_f64[r]).abs();
        worst = worst.max(d4).max(d8);
        println!(
            "row {r}: q4_0 dev={:+.6} f64={:+.6} f32seq={:+.6} | q8_0 dev={:+.6} f64={:+.6} f32seq={:+.6}",
            o4[r], ref_q4_f64[r], ref_q4_f32[r], o8[r], ref_q8_f64[r], ref_q8_f32[r]
        );
    }
    println!("worst |device - f64| = {worst:.3e}");
    if worst < 1e-2 {
        println!("pilot OK: both monomorphized quant GEMVs match host reference");
    } else {
        eprintln!("pilot FAILED");
        std::process::exit(1);
    }
}
