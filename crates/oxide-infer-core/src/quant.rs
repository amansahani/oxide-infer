//! Quantization block formats and dequantization.
//!
//! Layouts mirror llama.cpp's ggml quantization so checkpoints produced by
//! `quant_weights` (or by llama.cpp's quantizer) dequantize identically.
//! All dequantization happens on the host and produces BF16, which is what
//! the engine loads. This is deliberately the simple, correct path; the
//! performance-focused on-the-fly path is separate work.

use half::{bf16, f16};

/// Quantization type stored in a packed safetensors checkpoint.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuantType {
    Q1_0,
    Q2_0,
    Q2_K,
    Q3_K,
    Q4_0,
    Q4_1,
    Q4_K,
    Q5_0,
    Q5_1,
    Q5_K,
    Q6_K,
    Q8_0,
    Q8_1,
    Q8_K,
}

impl QuantType {
    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "q1_0" => Self::Q1_0,
            "q2_0" => Self::Q2_0,
            "q2_k" | "q2k" => Self::Q2_K,
            "q3_k" | "q3k" => Self::Q3_K,
            "q4_0" => Self::Q4_0,
            "q4_1" => Self::Q4_1,
            "q4_k" | "q4k" | "q4_k_m" | "q4_k_s" => Self::Q4_K,
            "q5_0" => Self::Q5_0,
            "q5_1" => Self::Q5_1,
            "q5_k" | "q5k" | "q5_k_m" | "q5_k_s" => Self::Q5_K,
            "q6_k" | "q6k" | "q6_k" => Self::Q6_K,
            "q8_0" => Self::Q8_0,
            "q8_1" => Self::Q8_1,
            "q8_k" | "q8k" => Self::Q8_K,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Q1_0 => "q1_0",
            Self::Q2_0 => "q2_0",
            Self::Q2_K => "q2_K",
            Self::Q3_K => "q3_K",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q4_K => "q4_K",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q5_K => "q5_K",
            Self::Q6_K => "q6_K",
            Self::Q8_0 => "q8_0",
            Self::Q8_1 => "q8_1",
            Self::Q8_K => "q8_K",
        }
    }

    /// Elements per block (or per super-block for K types).
    pub fn block_elements(self) -> usize {
        match self {
            Self::Q1_0 => 128,
            Self::Q2_0 => 64,
            Self::Q2_K => 256,
            Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q1_0 => 2 + 128 / 8,          // f16 + 16 bytes bits
            Self::Q2_0 => 2 + 64 / 4,           // f16 + 16 bytes
            Self::Q2_K => 4 + 256 / 16 + 256 / 4,
            Self::Q3_K => 2 + 256 / 4 + 256 / 8 + 12, // f16 + qs + hmask + scales
            Self::Q4_0 => 2 + 32 / 2,           // f16 + 16 bytes
            Self::Q4_1 => 4 + 32 / 2,           // d+m (f16x2) + 16 bytes
            Self::Q4_K => 4 + 12 + 256 / 2,     // dm(f16x2) + scales + qs
            Self::Q5_0 => 2 + 4 + 32 / 2,       // f16 + 32 bits high + 16 qs
            Self::Q5_1 => 4 + 4 + 32 / 2,       // d+m + qh + qs
            Self::Q5_K => 4 + 12 + 256 / 2 + 256 / 8,
            Self::Q6_K => 2 + 256 / 16 + 3 * 256 / 4,
            Self::Q8_0 => 2 + 32,               // f16 + 32 int8
            Self::Q8_1 => 4 + 32,               // d+s + 32 int8
            Self::Q8_K => 4 + 256 + 256 / 16 * 2,
        }
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

fn read_f16(packed: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([packed[off], packed[off + 1]]))
}

// ============================================================================
// ggml-compatible K-type primitives (ported from llama.cpp ggml-quants.c).
// These are direct transliterations of the reference implementation so that
// quantized checkpoints are byte-identical to llama.cpp's `llama-quantize`.
// ============================================================================

/// ggml's nearest_int: round-half-to-even (magic-number trick, |fval| <= 2^22).
#[inline]
fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4_194_303.0);
    fval.round_ties_even() as i32
}

const GROUP_MAX_EPS: f32 = 1e-15;

/// Port of ggml's make_qkx2_quants: quantizes x[0..n] to integers in [0, nmax]
/// with an offset min, weighted least squares. Returns the sub-block scale.
/// `the_min` receives the (non-negative) offset that is subtracted from the data.
fn make_qkx2_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l_out: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min { min = x[i]; }
        if x[i] > max { max = x[i]; }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 { min = 0.0; }
    if max == min {
        for i in 0..n { l_out[i] = 0; }
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let l = nearest_int(iscale * (x[i] - min));
        l_out[i] = l.max(0).min(nmax) as u8;
        let diff = scale * l_out[i] as f32 + min - x[i];
        let diff = if use_mad { diff.abs() } else { diff * diff };
        best_error += weights[i] * diff;
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let l = nearest_int(iscale * (x[i] - min));
            let l = l.max(0).min(nmax);
            laux[i] = l as u8;
            let w = weights[i];
            let lf = l as f32;
            sum_l += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let diff = this_scale * laux[i] as f32 + this_min - x[i];
                let diff = if use_mad { diff.abs() } else { diff * diff };
                cur_error += weights[i] * diff;
            }
            if cur_error < best_error {
                for i in 0..n { l_out[i] = laux[i]; }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

/// Port of ggml's make_qx_quants (rmse_type = 1, no weights).
/// Returns the (signed) scale for the sub-block; L output is offset by nmax.
fn make_qx_quants(n: usize, nmax: i32, x: &[f32], l_out: &mut [i8], rmse_type: i32) -> f32 {
    let mut amax = 0.0f32;
    let mut max = 0.0f32;
    for i in 0..n {
        let ax = x[i].abs();
        if ax > amax { amax = ax; max = x[i]; }
    }
    if amax < GROUP_MAX_EPS {
        for i in 0..n { l_out[i] = 0; }
        return 0.0;
    }
    let mut iscale = -nmax as f32 / max;
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let l = nearest_int(iscale * x[i]);
        let l = l.max(-nmax).min(nmax - 1);
        l_out[i] = (l + nmax) as i8;
        let w = if rmse_type == 1 { x[i] * x[i] } else { 1.0 };
        let lf = l as f32;
        sumlx += w * x[i] * lf;
        suml2 += w * lf * lf;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;
    for is in -9..=9 {
        if is == 0 { continue; }
        iscale = -(nmax as f32 + 0.1 * is as f32) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for i in 0..n {
            let l = nearest_int(iscale * x[i]);
            let l = l.max(-nmax).min(nmax - 1);
            let w = if rmse_type == 1 { x[i] * x[i] } else { 1.0 };
            let lf = l as f32;
            sumlx += w * x[i] * lf;
            suml2 += w * lf * lf;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let l = nearest_int(iscale * x[i]);
                l_out[i] = (l.max(-nmax).min(nmax - 1) + nmax) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// Port of ggml's make_q3_quants (do_rmse = true).
/// Returns the (signed) sub-block scale; L output is offset by nmax.
fn make_q3_quants(n: usize, nmax: i32, x: &[f32], l_out: &mut [i8]) -> f32 {
    let mut amax = 0.0f32;
    let mut max = 0.0f32;
    for i in 0..n {
        let ax = x[i].abs();
        if ax > amax { amax = ax; max = x[i]; }
    }
    if amax < GROUP_MAX_EPS {
        for i in 0..n { l_out[i] = 0; }
        return 0.0;
    }
    let iscale = -nmax as f32 / max;
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let l = nearest_int(iscale * x[i]);
        let l = l.max(-nmax).min(nmax - 1);
        l_out[i] = l as i8;
        let w = x[i] * x[i];
        let lf = l as f32;
        sumlx += w * x[i] * lf;
        suml2 += w * lf * lf;
    }
    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = x[i] * x[i];
            let slx = sumlx - w * x[i] * l_out[i] as f32;
            if slx > 0.0 {
                let sl2 = suml2 - w * l_out[i] as f32 * l_out[i] as f32;
                if sl2 > 0.0 {
                    let new_l = nearest_int(x[i] * sl2 / slx);
                    let new_l = new_l.max(-nmax).min(nmax - 1);
                    if new_l != l_out[i] as i32 {
                        let slx = slx + w * x[i] * new_l as f32;
                        let sl2 = sl2 + w * new_l as f32 * new_l as f32;
                        if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                            l_out[i] = new_l as i8;
                            sumlx = slx;
                            suml2 = sl2;
                            n_changed += 1;
                        }
                    }
                }
            }
        }
        if n_changed == 0 { break; }
    }
    for i in 0..n {
        l_out[i] = (l_out[i] as i32 + nmax) as i8;
    }
    if suml2 > 0.0 { sumlx / suml2 } else { 0.0 }
}

fn f16_bytes(v: f32) -> [u8; 2] {
    half::f16::from_f32(v).to_le_bytes()
}

fn write_f16(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&f16_bytes(v));
}

/// Dequantize q1_0 (1 bit per weight, ±d).
pub fn dequantize_q1_0(packed: &[u8]) -> Vec<bf16> {
    let qk = 128;
    let nb = packed.len() / (2 + qk / 8);
    let mut out = Vec::with_capacity(nb * qk);
    for b in 0..nb {
        let off = b * (2 + qk / 8);
        let d = read_f16(packed, off);
        for j in 0..qk {
            let byte = packed[off + 2 + j / 8];
            let bit = (byte >> (j % 8)) & 1;
            out.push(bf16::from_f32(if bit != 0 { d } else { -d }));
        }
    }
    out
}

/// Dequantize q2_0 (2 bits per weight, symbol -1,0,1,2).
pub fn dequantize_q2_0(packed: &[u8]) -> Vec<bf16> {
    let qk = 64;
    let nb = packed.len() / (2 + qk / 4);
    let mut out = Vec::with_capacity(nb * qk);
    for b in 0..nb {
        let off = b * (2 + qk / 4);
        let d = read_f16(packed, off);
        for j in 0..qk {
            let byte = packed[off + 2 + j / 4];
            let q = (byte >> ((j % 4) * 2)) & 0x03;
            out.push(bf16::from_f32((q as f32 - 1.0) * d));
        }
    }
    out
}

/// Layout matches ggml: f16 d, 16 packed nibble bytes, split halves.
pub fn dequantize_q4_0(packed: &[u8]) -> Vec<bf16> {
    let qk = 32;
    let nb = packed.len() / 18;
    let mut out = Vec::with_capacity(nb * qk);
    for b in 0..nb {
        let off = b * 18;
        let d = read_f16(packed, off);
        let qs = &packed[off + 2..off + 18];
        for j in 0..16 {
            let byte = qs[j];
            out.push(bf16::from_f32(d * ((byte & 0x0F) as f32 - 8.0)));
        }
        for j in 0..16 {
            let byte = qs[j];
            out.push(bf16::from_f32(d * (((byte >> 4) & 0x0F) as f32 - 8.0)));
        }
    }
    out
}


/// Layout matches ggml: f16 d, f16 m, 16 packed nibble bytes, split halves.
pub fn dequantize_q4_1(packed: &[u8]) -> Vec<bf16> {
    let qk = 32;
    let nb = packed.len() / 20;
    let mut out = Vec::with_capacity(nb * qk);
    for b in 0..nb {
        let off = b * 20;
        let d = read_f16(packed, off);
        let m = read_f16(packed, off + 2);
        let qs = &packed[off + 4..off + 20];
        for j in 0..16 {
            out.push(bf16::from_f32(d * (qs[j] & 0x0F) as f32 + m));
        }
        for j in 0..16 {
            out.push(bf16::from_f32(d * ((qs[j] >> 4) & 0x0F) as f32 + m));
        }
    }
    out
}


/// Layout matches ggml: f16 d, u32 qh, 16 packed nibble bytes, split halves.
pub fn dequantize_q5_0(packed: &[u8]) -> Vec<bf16> {
    let qk = 32;
    let nb = packed.len() / 22;
    let mut out = Vec::with_capacity(nb * qk);
    for b in 0..nb {
        let off = b * 22;
        let d = read_f16(packed, off);
        let qh = u32::from_le_bytes([packed[off + 2], packed[off + 3], packed[off + 4], packed[off + 5]]);
        let qs = &packed[off + 6..off + 22];
        for j in 0..16 {
            let xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            let x0 = (((qs[j] & 0x0F) as u32 | xh_0) as f32 - 16.0) * d;
            out.push(bf16::from_f32(x0));
        }
        for j in 0..16 {
            let xh_1 = ((qh >> (j + 12)) & 0x10) << 0;
            let x1 = ((((qs[j] >> 4) & 0x0F) as u32 | xh_1) as f32 - 16.0) * d;
            out.push(bf16::from_f32(x1));
        }
    }
    out
}


/// Layout matches ggml: f16 d, f16 m, u32 qh, 16 packed nibble bytes, split halves.
pub fn dequantize_q5_1(packed: &[u8]) -> Vec<bf16> {
    let qk = 32;
    let nb = packed.len() / 24;
    let mut out = Vec::with_capacity(nb * qk);
    for b in 0..nb {
        let off = b * 24;
        let d = read_f16(packed, off);
        let m = read_f16(packed, off + 2);
        let qh = u32::from_le_bytes([packed[off + 4], packed[off + 5], packed[off + 6], packed[off + 7]]);
        let qs = &packed[off + 8..off + 24];
        for j in 0..16 {
            let xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            let x0 = ((qs[j] & 0x0F) as u32 | xh_0) as f32 * d + m;
            out.push(bf16::from_f32(x0));
        }
        for j in 0..16 {
            let xh_1 = ((qh >> (j + 12)) & 0x10) << 0;
            let x1 = (((qs[j] >> 4) & 0x0F) as u32 | xh_1) as f32 * d + m;
            out.push(bf16::from_f32(x1));
        }
    }
    out
}


/// Dequantize q8_0 (8-bit, no min).
pub fn dequantize_q8_0(packed: &[u8]) -> Vec<bf16> {
    let qk = 32;
    let nb = packed.len() / 34;
    let mut out = Vec::with_capacity(nb * qk);
    for b in 0..nb {
        let off = b * 34;
        let d = read_f16(packed, off);
        for j in 0..32 {
            let q = packed[off + 2 + j] as i8 as f32;
            out.push(bf16::from_f32(q * d));
        }
    }
    out
}

/// Dequantize q8_1 (8-bit with per-block min/sum).
pub fn dequantize_q8_1(packed: &[u8]) -> Vec<bf16> {
    let qk = 32;
    let nb = packed.len() / 36;
    let mut out = Vec::with_capacity(nb * qk);
    for b in 0..nb {
        let off = b * 36;
        let d = read_f16(packed, off);
        let s = read_f16(packed, off + 2);
        let _ = s; // d*sum, not needed for plain dequantize
        for j in 0..32 {
            let q = packed[off + 4 + j] as i8 as f32;
            out.push(bf16::from_f32(q * d));
        }
    }
    out
}

fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4),
        )
    }
}

/// Dequantize q4_K (super-block K, 4-bit with per-subblock scale/min).
/// Layout matches ggml: f16 d, f16 dmin, 12 scale bytes, 128 packed nibble bytes.
pub fn dequantize_q4_k(packed: &[u8]) -> Vec<bf16> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 144;
    let nb = packed.len() / BLOCK_BYTES;
    let mut out = Vec::with_capacity(nb * QK_K);
    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let d = read_f16(packed, off);
        let dmin = read_f16(packed, off + 2);
        let scales = &packed[off + 4..off + 16];
        let qs = &packed[off + 16..off + BLOCK_BYTES];
        let mut is = 0usize;
        let mut q_off = 0usize;
        for _ in (0..QK_K).step_by(64) {
            let (sc0, m0) = get_scale_min_k4(is + 0, scales);
            let d1 = d * sc0 as f32;
            let m1 = dmin * m0 as f32;
            let (sc1, m1_) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc1 as f32;
            let m2 = dmin * m1_ as f32;
            for l in 0..32 {
                out.push(bf16::from_f32(d1 * ((qs[q_off + l] & 0x0F) as f32) - m1));
            }
            for l in 0..32 {
                out.push(bf16::from_f32(d2 * ((qs[q_off + l] >> 4) as f32) - m2));
            }
            q_off += 32;
            is += 2;
        }
    }
    out
}

/// Dequantize q2_K (2-bit, per sub-block 4-bit scale/min in one byte).
/// Layout matches ggml: 16 packed scale bytes, 64 quant bytes, f16 d, f16 dmin (i.e. d at the END).
pub fn dequantize_q2_k(packed: &[u8]) -> Vec<bf16> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 84;
    let nb = packed.len() / BLOCK_BYTES;
    let mut out = Vec::with_capacity(nb * QK_K);
    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let scales = &packed[off..off + 16];
        let qs = &packed[off + 16..off + 80];
        let d = read_f16(packed, off + 80);
        let min = read_f16(packed, off + 82);
        let mut is = 0usize;
        let mut q_off = 0usize;
        for _n in (0..QK_K).step_by(128) {
            let mut shift = 0;
            for _j in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0x0F) as f32;
                let ml = min * (sc >> 4) as f32;
                for l in 0..16 {
                    out.push(bf16::from_f32(dl * (((qs[q_off + l] >> shift) & 0x03) as f32) - ml));
                }
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0x0F) as f32;
                let ml = min * (sc >> 4) as f32;
                for l in 0..16 {
                    out.push(bf16::from_f32(dl * (((qs[q_off + 16 + l] >> shift) & 0x03) as f32) - ml));
                }
                shift += 2;
            }
            q_off += 32;
        }
    }
    out
}

/// Dequantize q3_K (3-bit, per sub-block 6-bit signed scale).
/// Layout matches ggml: 32 hmask bytes, 64 qs bytes, 12 scale bytes, f16 d last.
pub fn dequantize_q3_k(packed: &[u8]) -> Vec<bf16> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 110; // 32 hmask + 64 qs + 12 scales + f16 d
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;
    let nb = packed.len() / BLOCK_BYTES;
    let mut out = Vec::with_capacity(nb * QK_K);
    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let d_all = read_f16(packed, off + BLOCK_BYTES - 2);
        let hm = &packed[off..off + 32];
        let qs = &packed[off + 32..off + 96];
        let scales = &packed[off + 96..off + 108];
        // unpack scales (ggml kmask magic)
        let mut aux = [0u32; 4];
        aux[0] = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
        aux[1] = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
        let tmp = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let mut scales_i8 = [0i8; 16];
        for i in 0..4 {
            scales_i8[i * 4] = (aux[i] & 0xFF) as i8;
            scales_i8[i * 4 + 1] = ((aux[i] >> 8) & 0xFF) as i8;
            scales_i8[i * 4 + 2] = ((aux[i] >> 16) & 0xFF) as i8;
            scales_i8[i * 4 + 3] = ((aux[i] >> 24) & 0xFF) as i8;
        }
        let mut is = 0usize;
        let mut q_off = 0usize;
        let mut m = 1u8;
        for _n in (0..QK_K).step_by(128) {
            let mut shift = 0;
            for _j in 0..4 {
                let dl = d_all * (scales_i8[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let q = (qs[q_off + l] >> shift) & 3;
                    let h = if hm[l] & m != 0 { 0 } else { 4 };
                    out.push(bf16::from_f32(dl * (q as i32 - h) as f32));
                }
                let dl = d_all * (scales_i8[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let q = (qs[q_off + 16 + l] >> shift) & 3;
                    let h = if hm[16 + l] & m != 0 { 0 } else { 4 };
                    out.push(bf16::from_f32(dl * (q as i32 - h) as f32));
                }
                shift += 2;
                m <<= 1;
            }
            q_off += 32;
        }
    }
    out
}

/// Dequantize q5_K (5-bit, per sub-block scale/min, high bit in qh).
/// Layout matches ggml: f16 d, f16 dmin, 12 scale bytes, 32 qh bytes, 128 packed nibble bytes.
pub fn dequantize_q5_k(packed: &[u8]) -> Vec<bf16> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 176;
    let nb = packed.len() / BLOCK_BYTES;
    let mut out = Vec::with_capacity(nb * QK_K);
    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let d = read_f16(packed, off);
        let dmin = read_f16(packed, off + 2);
        let scales = &packed[off + 4..off + 16];
        let qh = &packed[off + 16..off + 48];
        let qs = &packed[off + 48..off + BLOCK_BYTES];
        let mut is = 0usize;
        let mut q_off = 0usize;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for _n in (0..QK_K).step_by(64) {
            let (sc0, m0) = get_scale_min_k4(is + 0, scales);
            let d1 = d * sc0 as f32;
            let m1 = dmin * m0 as f32;
            let (sc1, m1_) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc1 as f32;
            let m2 = dmin * m1_ as f32;
            for l in 0..32 {
                let lo = (qs[q_off + l] & 0x0F) as f32;
                let hi = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                out.push(bf16::from_f32(d1 * (lo + hi) - m1));
            }
            for l in 0..32 {
                let lo = (qs[q_off + l] >> 4) as f32;
                let hi = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                out.push(bf16::from_f32(d2 * (lo + hi) - m2));
            }
            q_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    out
}

/// Dequantize q6_K (6-bit, per sub-block signed int8 scale).
/// Layout matches ggml: 128 ql bytes, 64 qh bytes, 16 int8 scales, f16 d last.
pub fn dequantize_q6_k(packed: &[u8]) -> Vec<bf16> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 210; // 128 ql + 64 qh + 16 scales + f16 d
    let nb = packed.len() / BLOCK_BYTES;
    let mut out = Vec::with_capacity(nb * QK_K);
    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let ql_all = &packed[off..off + 128];
        let qh_all = &packed[off + 128..off + 192];
        let scales = &packed[off + 192..off + 208];
        let d = read_f16(packed, off + 208);
        for n in (0..QK_K).step_by(128) {
            let ql = &ql_all[n / 128 * 64..n / 128 * 64 + 64];
            let qh = &qh_all[n / 128 * 32..n / 128 * 32 + 32];
            let sc = &scales[n / 128 * 8..n / 128 * 8 + 8];
            // ggml writes y[l], y[l+32], y[l+64], y[l+96] per group, not interleaved.
            let mut y = [0.0f32; 128];
            for l in 0..32 {
                let is = l / 16;
                let sc0 = sc[is + 0] as i8 as f32;
                let sc2 = sc[is + 2] as i8 as f32;
                let sc4 = sc[is + 4] as i8 as f32;
                let sc6 = sc[is + 6] as i8 as f32;
                let q1 = (((ql[l] & 0x0F) as u32) | (((qh[l] >> 0) & 3) as u32) << 4) as i32 - 32;
                let q2 = (((ql[l + 32] & 0x0F) as u32) | (((qh[l] >> 2) & 3) as u32) << 4) as i32 - 32;
                let q3 = (((ql[l] >> 4) as u32) | (((qh[l] >> 4) & 3) as u32) << 4) as i32 - 32;
                let q4 = (((ql[l + 32] >> 4) as u32) | (((qh[l] >> 6) & 3) as u32) << 4) as i32 - 32;
                y[l] = d * sc0 * q1 as f32;
                y[l + 32] = d * sc2 * q2 as f32;
                y[l + 64] = d * sc4 * q3 as f32;
                y[l + 96] = d * sc6 * q4 as f32;
            }
            for v in y {
                out.push(bf16::from_f32(v));
            }
        }
    }
    out
}

/// Dequantize q8_K (int8, per super-block; d is f32).
/// Layout matches ggml: f32 d, 256 int8 quants, 32 i16 bsums (unused here).
pub fn dequantize_q8_k(packed: &[u8]) -> Vec<bf16> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 292;
    let nb = packed.len() / BLOCK_BYTES;
    let mut out = Vec::with_capacity(nb * QK_K);
    for b in 0..nb {
        let off = b * BLOCK_BYTES;
        let d = f32::from_le_bytes([packed[off], packed[off + 1], packed[off + 2], packed[off + 3]]);
        for l in 0..QK_K {
            out.push(bf16::from_f32(packed[off + 4 + l] as i8 as f32 * d));
        }
    }
    out
}

pub fn dequantize(packed: &[u8], qt: QuantType) -> Vec<bf16> {
    match qt {
        QuantType::Q1_0 => dequantize_q1_0(packed),
        QuantType::Q2_0 => dequantize_q2_0(packed),
        QuantType::Q4_0 => dequantize_q4_0(packed),
        QuantType::Q4_1 => dequantize_q4_1(packed),
        QuantType::Q5_0 => dequantize_q5_0(packed),
        QuantType::Q5_1 => dequantize_q5_1(packed),
        QuantType::Q8_0 => dequantize_q8_0(packed),
        QuantType::Q8_1 => dequantize_q8_1(packed),
        QuantType::Q2_K => dequantize_q2_k(packed),
        QuantType::Q3_K => dequantize_q3_k(packed),
        QuantType::Q4_K => dequantize_q4_k(packed),
        QuantType::Q5_K => dequantize_q5_k(packed),
        QuantType::Q6_K => dequantize_q6_k(packed),
        QuantType::Q8_K => dequantize_q8_k(packed),
    }
}

/// Quantize to q4_0 (ggml reference algorithm, byte-identical to llama.cpp).
pub fn quantize_q4_0(values: &[f32]) -> Vec<u8> {
    let nb = values.len().div_ceil(32);
    let mut out = Vec::with_capacity(nb * 18);
    for b in 0..nb {
        let start = b * 32;
        let end = (start + 32).min(values.len());
        let mut block = [0.0f32; 32];
        block[..end - start].copy_from_slice(&values[start..end]);
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in block.iter() {
            if amax < v.abs() { amax = v.abs(); max = v; }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for j in 0..16 {
            let x0 = block[j] * id;
            let x1 = block[j + 16] * id;
            let xi0 = (((x0 + 8.5).trunc()) as i32).min(15);
            let xi1 = (((x1 + 8.5).trunc()) as i32).min(15);
            out.push((xi0 as u8 & 0x0F) | (((xi1 as u8) & 0x0F) << 4));
        }
    }
    out
}


/// Quantize to q1_0 (1 bit per weight).
pub fn quantize_q1_0(values: &[f32]) -> Vec<u8> {
    const QK: usize = 128;
    let nb = values.len().div_ceil(QK);
    let mut out = Vec::with_capacity(nb * (2 + QK / 8));
    for b in 0..nb {
        let mut sum_abs = 0.0f32;
        let start = b * QK;
        let end = (start + QK).min(values.len());
        for &v in &values[start..end] { sum_abs += v.abs(); }
        let d = sum_abs / QK as f32;
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        let mut qs = [0u8; QK / 8];
        for j in 0..QK {
            let v = if j < end - start { values[start + j] } else { 0.0 };
            if v >= 0.0 { qs[j / 8] |= 1 << (j % 8); }
        }
        out.extend_from_slice(&qs);
    }
    out
}

/// Quantize to q2_0 (2 bits per weight).
pub fn quantize_q2_0(values: &[f32]) -> Vec<u8> {
    const QK: usize = 64;
    let nb = values.len().div_ceil(QK);
    let mut out = Vec::with_capacity(nb * (2 + QK / 4));
    for b in 0..nb {
        let start = b * QK;
        let end = (start + QK).min(values.len());
        let mut amax = 0.0f32;
        for &v in &values[start..end] { amax = amax.max(v.abs()); }
        let d = if amax > 0.0 { amax } else { 1e-8 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        let mut qs = [0u8; QK / 4];
        for j in 0..QK {
            let v = if j < end - start { values[start + j] } else { 0.0 };
            let q = (v / d).round() as i32 + 1;
            let q = q.clamp(0, 3) as u8;
            qs[j / 4] |= q << ((j % 4) * 2);
        }
        out.extend_from_slice(&qs);
    }
    out
}

/// Quantize to q4_1 (ggml reference algorithm, byte-identical to llama.cpp).
pub fn quantize_q4_1(values: &[f32]) -> Vec<u8> {
    let nb = values.len().div_ceil(32);
    let mut out = Vec::with_capacity(nb * 20);
    for b in 0..nb {
        let start = b * 32;
        let end = (start + 32).min(values.len());
        let mut block = [0.0f32; 32];
        block[..end - start].copy_from_slice(&values[start..end]);
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in block.iter() {
            if v < min { min = v; }
            if v > max { max = v; }
        }
        let d = (max - min) / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(min).to_le_bytes());
        for j in 0..16 {
            let x0 = (block[j] - min) * id;
            let x1 = (block[j + 16] - min) * id;
            let xi0 = ((x0 + 0.5).trunc() as i32).min(15);
            let xi1 = ((x1 + 0.5).trunc() as i32).min(15);
            out.push((xi0 as u8 & 0x0F) | (((xi1 as u8) & 0x0F) << 4));
        }
    }
    out
}


/// Quantize to q5_0 (ggml reference algorithm, byte-identical to llama.cpp).
pub fn quantize_q5_0(values: &[f32]) -> Vec<u8> {
    let nb = values.len().div_ceil(32);
    let mut out = Vec::with_capacity(nb * 22);
    for b in 0..nb {
        let start = b * 32;
        let end = (start + 32).min(values.len());
        let mut block = [0.0f32; 32];
        block[..end - start].copy_from_slice(&values[start..end]);
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in block.iter() {
            if amax < v.abs() { amax = v.abs(); max = v; }
        }
        let d = max / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        let mut qh = 0u32;
        let mut qs = [0u8; 16];
        for j in 0..16 {
            let x0 = block[j] * id;
            let x1 = block[j + 16] * id;
            let xi0 = ((x0 + 16.5).trunc() as i32).min(31) as u8;
            let xi1 = ((x1 + 16.5).trunc() as i32).min(31) as u8;
            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= ((xi0 as u32 & 0x10) >> 4) << (j + 0);
            qh |= ((xi1 as u32 & 0x10) >> 4) << (j + 16);
        }
        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&qs);
    }
    out
}


/// Quantize to q5_1 (5-bit + per-block min, split halves).
pub fn quantize_q5_1(values: &[f32]) -> Vec<u8> {
    let nb = values.len().div_ceil(32);
    let mut out = Vec::with_capacity(nb * 24);
    for b in 0..nb {
        let start = b * 32;
        let end = (start + 32).min(values.len());
        let mut block = [0.0f32; 32];
        block[..end - start].copy_from_slice(&values[start..end]);
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in block.iter() {
            if v < min { min = v; }
            if v > max { max = v; }
        }
        let d = (max - min) / 31.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(min).to_le_bytes());
        let mut qh = 0u32;
        let mut qs = [0u8; 16];
        for j in 0..16 {
            let x0 = (block[j] - min) * id;
            let x1 = (block[j + 16] - min) * id;
            let xi0 = ((x0 + 0.5).trunc() as i32).min(31).max(0) as u8;
            let xi1 = ((x1 + 0.5).trunc() as i32).min(31).max(0) as u8;
            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= ((xi0 as u32 & 0x10) >> 4) << (j + 0);
            qh |= ((xi1 as u32 & 0x10) >> 4) << (j + 16);
        }
        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&qs);
    }
    out
}

/// Quantize to q8_0 (8-bit per block).
pub fn quantize_q8_0(values: &[f32]) -> Vec<u8> {
    const QK: usize = 32;
    let nb = values.len().div_ceil(QK);
    let mut out = Vec::with_capacity(nb * 34);
    for b in 0..nb {
        let start = b * QK;
        let end = (start + QK).min(values.len());
        let mut amax = 0.0f32;
        for &v in &values[start..end] { amax = amax.max(v.abs()); }
        let d = if amax > 0.0 { amax / 127.0 } else { 1e-8 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for j in 0..32 {
            let v = if j < end - start { values[start + j] } else { 0.0 };
            let q = (v / d).round() as i8;
            out.push(q as u8);
        }
    }
    out
}

/// Quantize to q8_1 (8-bit per block with min).
pub fn quantize_q8_1(values: &[f32]) -> Vec<u8> {
    const QK: usize = 32;
    let nb = values.len().div_ceil(QK);
    let mut out = Vec::with_capacity(nb * 36);
    for b in 0..nb {
        let start = b * QK;
        let end = (start + QK).min(values.len());
        let mut block = [0.0f32; QK];
        block[..end - start].copy_from_slice(&values[start..end]);
        let mut amax = 0.0f32;
        for &v in block.iter() { amax = amax.max(v.abs()); }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        let mut sum: i32 = 0;
        let mut qs = [0i8; QK];
        for j in 0..QK {
            let q = (block[j] * id).round() as i32;
            qs[j] = q as i8;
            sum += qs[j] as i32;
        }
        out.extend_from_slice(&f16::from_f32(sum as f32 * d).to_le_bytes());
        for &q in qs.iter() { out.push(q as u8); }
    }
    out
}

/// Dispatch to the right quantizer.
pub fn quantize(qt: QuantType, values: &[f32]) -> Vec<u8> {
    match qt {
        QuantType::Q1_0 => quantize_q1_0(values),
        QuantType::Q2_0 => quantize_q2_0(values),
        QuantType::Q2_K => quantize_q2_k(values),
        QuantType::Q3_K => quantize_q3_k(values),
        QuantType::Q4_0 => quantize_q4_0(values),
        QuantType::Q4_K => quantize_q4_k(values),
        QuantType::Q4_1 => quantize_q4_1(values),
        QuantType::Q5_K => quantize_q5_k(values),
        QuantType::Q6_K => quantize_q6_k(values),
        QuantType::Q8_K => quantize_q8_k(values),
        QuantType::Q5_0 => quantize_q5_0(values),
        QuantType::Q8_0 => quantize_q8_0(values),
        QuantType::Q5_1 => quantize_q5_1(values),
        QuantType::Q8_1 => quantize_q8_1(values),
    }
}

/// Quantize to q2_K (ggml reference algorithm, byte-identical to llama.cpp).
/// 16 sub-blocks of 16 elements; per-sub-block 4-bit scale + 4-bit min.
pub fn quantize_q2_k(values: &[f32]) -> Vec<u8> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 84;
    let nb = values.len().div_ceil(QK_K);
    let mut out = Vec::with_capacity(nb * BLOCK_BYTES);
    for b in 0..nb {
        let start = b * QK_K;
        let end = (start + QK_K).min(values.len());
        let mut block = [0.0f32; QK_K];
        block[..end - start].copy_from_slice(&values[start..end]);

        const Q4SCALE: f32 = 15.0;
        let mut l = [0u8; QK_K];
        let mut laux = [0u8; 16];
        let mut weights = [0.0f32; 16];
        let mut mins = [0.0f32; QK_K / 16];
        let mut scales = [0.0f32; QK_K / 16];

        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;
        for j in 0..QK_K / 16 {
            for lx in 0..16 {
                weights[lx] = block[16 * j + lx].abs();
            }
            scales[j] = make_qkx2_quants(
                16, 3, &block[16 * j..16 * j + 16], &weights,
                &mut l[16 * j..16 * j + 16], &mut mins[j], &mut laux,
                -0.5, 0.1, 15, true,
            );
            if scales[j] > max_scale { max_scale = scales[j]; }
            if mins[j] > max_min { max_min = mins[j]; }
        }

        let mut scales_bytes = [0u8; 16];
        let mut dm_bytes = [0u8; 4];
        if max_scale > 0.0 {
            let iscale = Q4SCALE / max_scale;
            for j in 0..16 {
                scales_bytes[j] = nearest_int(iscale * scales[j]) as u8;
            }
            dm_bytes[..2].copy_from_slice(&f16_bytes(max_scale / Q4SCALE));
        } else {
            dm_bytes[..2].copy_from_slice(&f16_bytes(0.0));
        }
        if max_min > 0.0 {
            let iscale = Q4SCALE / max_min;
            for j in 0..16 {
                scales_bytes[j] |= (nearest_int(iscale * mins[j]) as u8) << 4;
            }
            dm_bytes[2..].copy_from_slice(&f16_bytes(max_min / Q4SCALE));
        } else {
            dm_bytes[2..].copy_from_slice(&f16_bytes(0.0));
        }

        let d = f16::from_bits(u16::from_le_bytes([dm_bytes[0], dm_bytes[1]])).to_f32();
        let dmin = f16::from_bits(u16::from_le_bytes([dm_bytes[2], dm_bytes[3]])).to_f32();
        for j in 0..16 {
            let sc = scales_bytes[j];
            let dl = d * (sc & 0x0F) as f32;
            if dl == 0.0 { continue; }
            let dm = dmin * (sc >> 4) as f32;
            for ii in 0..16 {
                let li = nearest_int((block[16 * j + ii] + dm) / dl);
                l[16 * j + ii] = li.max(0).min(3) as u8;
            }
        }

        // ggml block order: scales (16), qs (64), dm (4)
        out.extend_from_slice(&scales_bytes);
        for j in (0..QK_K).step_by(128) {
            for li in 0..32 {
                out.push(
                    l[j + li]
                        | (l[j + li + 32] << 2)
                        | (l[j + li + 64] << 4)
                        | (l[j + li + 96] << 6),
                );
            }
        }
        out.extend_from_slice(&dm_bytes);
    }
    out
}

/// Quantize to q3_K (ggml reference algorithm, byte-identical to llama.cpp).
/// 16 sub-blocks of 16 elements; per sub-block 6-bit signed scale, offset 32.
pub fn quantize_q3_k(values: &[f32]) -> Vec<u8> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 110;
    let nb = values.len().div_ceil(QK_K);
    let mut out = Vec::with_capacity(nb * BLOCK_BYTES);
    for b in 0..nb {
        let start = b * QK_K;
        let end = (start + QK_K).min(values.len());
        let mut block = [0.0f32; QK_K];
        block[..end - start].copy_from_slice(&values[start..end]);

        let mut l = [0i8; QK_K];
        let mut scales = [0.0f32; QK_K / 16];

        let mut max_scale = 0.0f32;
        let mut amax = 0.0f32;
        for j in 0..QK_K / 16 {
            scales[j] = make_q3_quants(16, 4, &block[16 * j..16 * j + 16], &mut l[16 * j..16 * j + 16]);
            let scale = scales[j].abs();
            if scale > amax { amax = scale; max_scale = scales[j]; }
        }

        let mut y_scales = [0u8; 12];
        let d_bytes: [u8; 2] = if max_scale != 0.0 {
            let iscale = -32.0f32 / max_scale;
            for j in 0..QK_K / 16 {
                let mut li = nearest_int(iscale * scales[j]);
                li = li.max(-32).min(31) + 32;
                if j < 8 {
                    y_scales[j] = (li & 0x0F) as u8;
                } else {
                    y_scales[j - 8] |= ((li & 0x0F) as u8) << 4;
                }
                li >>= 4;
                y_scales[j % 4 + 8] |= (li as u8) << (2 * (j / 4) as u8);
            }
            f16_bytes(1.0 / iscale)
        } else {
            f16_bytes(0.0)
        };

        let mut block_out = Vec::with_capacity(BLOCK_BYTES);

        let d = f16::from_bits(u16::from_le_bytes(d_bytes)).to_f32();
        for j in 0..QK_K / 16 {
            let sc_u8 = if j < 8 {
                y_scales[j] & 0x0F
            } else {
                y_scales[j - 8] >> 4
            };
            let sc = ((sc_u8 | (((y_scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) as i32 - 32) as i8;
            let dl = d * sc as f32;
            if dl == 0.0 { continue; }
            for ii in 0..16 {
                let li = nearest_int(block[16 * j + ii] / dl);
                l[16 * j + ii] = li.max(-4).min(3) as i8 + 4;
            }
        }

        let mut hmask = [0u8; 32];
        let mut m = 0usize;
        let mut hm = 1u8;
        for j in 0..QK_K {
            if l[j] > 3 {
                hmask[m] |= hm;
                l[j] -= 4;
            }
            m += 1;
            if m == QK_K / 8 { m = 0; hm <<= 1; }
        }

        let mut qs = [0u8; 64];
        for j in (0..QK_K).step_by(128) {
            for li in 0..32 {
                qs[j / 4 + li] = l[j + li] as u8
                    | ((l[j + li + 32] as u8) << 2)
                    | ((l[j + li + 64] as u8) << 4)
                    | ((l[j + li + 96] as u8) << 6);
            }
        }

        // ggml block order: hmask, qs, scales, d
        block_out.extend_from_slice(&hmask);
        block_out.extend_from_slice(&qs);
        block_out.extend_from_slice(&y_scales);
        block_out.extend_from_slice(&d_bytes);
        out.extend_from_slice(&block_out);
    }
    out
}

/// Quantize to q4_K (ggml reference algorithm, byte-identical to llama.cpp).
/// 8 sub-blocks of 32 elements; per sub-block 6-bit scale and min.
pub fn quantize_q4_k(values: &[f32]) -> Vec<u8> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 144;
    let nb = values.len().div_ceil(QK_K);
    let mut out = Vec::with_capacity(nb * BLOCK_BYTES);
    for b in 0..nb {
        let start = b * QK_K;
        let end = (start + QK_K).min(values.len());
        let mut block = [0.0f32; QK_K];
        block[..end - start].copy_from_slice(&values[start..end]);

        let mut l = [0u8; QK_K];
        let mut lau = [0u8; 32];
        let mut weights = [0.0f32; 32];
        let mut mins = [0.0f32; QK_K / 32];
        let mut scales = [0.0f32; QK_K / 32];

        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;
        for j in 0..QK_K / 32 {
            let sub = &block[j * 32..(j + 1) * 32];
            let mut sum_x2 = 0.0f32;
            for &v in sub { sum_x2 += v * v; }
            let av_x = (sum_x2 / 32.0).sqrt();
            for lx in 0..32 { weights[lx] = av_x + sub[lx].abs(); }
            scales[j] = make_qkx2_quants(
                32, 15, sub, &weights,
                &mut l[j * 32..(j + 1) * 32], &mut mins[j], &mut lau,
                -1.0, 0.1, 20, false,
            );
            if scales[j] > max_scale { max_scale = scales[j]; }
            if mins[j] > max_min { max_min = mins[j]; }
        }

        let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        let mut y_scales = [0u8; 12];
        for j in 0..QK_K / 32 {
            let ls = nearest_int(inv_scale * scales[j]).min(63);
            let lm = nearest_int(inv_min * mins[j]).min(63);
            let ls = ls as u8;
            let lm = lm as u8;
            if j < 4 {
                y_scales[j] = ls;
                y_scales[j + 4] = lm;
            } else {
                y_scales[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
                y_scales[j - 4] |= (ls >> 4) << 6;
                y_scales[j] |= (lm >> 4) << 6;
            }
        }
        write_f16(&mut out, max_scale / 63.0);
        write_f16(&mut out, max_min / 63.0);
        out.extend_from_slice(&y_scales);

        // d/dmin are f16 at offsets 0/2; read them back (ggml does the same).
        let block_head = out.len() - 16;
        let d = read_f16(&out[block_head..], 0);
        let dmin = read_f16(&out[block_head..], 2);
        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, &y_scales);
            let dl = d * sc as f32;
            if dl == 0.0 { continue; }
            let dm = dmin * m as f32;
            for ii in 0..32 {
                let li = nearest_int((block[32 * j + ii] + dm) / dl);
                l[32 * j + ii] = li.max(0).min(15) as u8;
            }
        }

        let mut qs = [0u8; 128];
        for j in (0..QK_K).step_by(64) {
            for li in 0..32 {
                qs[j / 2 + li] = l[j + li] | (l[j + li + 32] << 4);
            }
        }
        out.extend_from_slice(&qs);
    }
    out
}

/// Quantize to q5_K (ggml reference algorithm, byte-identical to llama.cpp).
/// 8 sub-blocks of 32 elements; 6-bit scale/min + one high bit per quant.
pub fn quantize_q5_k(values: &[f32]) -> Vec<u8> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 176;
    let nb = values.len().div_ceil(QK_K);
    let mut out = Vec::with_capacity(nb * BLOCK_BYTES);
    for b in 0..nb {
        let start = b * QK_K;
        let end = (start + QK_K).min(values.len());
        let mut block = [0.0f32; QK_K];
        block[..end - start].copy_from_slice(&values[start..end]);

        let mut l = [0u8; QK_K];
        let mut lau = [0u8; 32];
        let mut weights = [0.0f32; 32];
        let mut mins = [0.0f32; QK_K / 32];
        let mut scales = [0.0f32; QK_K / 32];

        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;
        for j in 0..QK_K / 32 {
            let sub = &block[j * 32..(j + 1) * 32];
            let mut sum_x2 = 0.0f32;
            for &v in sub { sum_x2 += v * v; }
            let av_x = (sum_x2 / 32.0).sqrt();
            for lx in 0..32 { weights[lx] = av_x + sub[lx].abs(); }
            scales[j] = make_qkx2_quants(
                32, 31, sub, &weights,
                &mut l[j * 32..(j + 1) * 32], &mut mins[j], &mut lau,
                -0.5, 0.1, 15, false,
            );
            if scales[j] > max_scale { max_scale = scales[j]; }
            if mins[j] > max_min { max_min = mins[j]; }
        }

        let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        let mut y_scales = [0u8; 12];
        for j in 0..QK_K / 32 {
            let ls = nearest_int(inv_scale * scales[j]).min(63);
            let lm = nearest_int(inv_min * mins[j]).min(63);
            let ls = ls as u8;
            let lm = lm as u8;
            if j < 4 {
                y_scales[j] = ls;
                y_scales[j + 4] = lm;
            } else {
                y_scales[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
                y_scales[j - 4] |= (ls >> 4) << 6;
                y_scales[j] |= (lm >> 4) << 6;
            }
        }
        write_f16(&mut out, max_scale / 63.0);
        write_f16(&mut out, max_min / 63.0);
        out.extend_from_slice(&y_scales);

        let block_head = out.len() - 16;
        let d = read_f16(&out[block_head..], 0);
        let dmin = read_f16(&out[block_head..], 2);
        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, &y_scales);
            let dl = d * sc as f32;
            if dl == 0.0 { continue; }
            let dm = dmin * m as f32;
            for ii in 0..32 {
                let li = nearest_int((block[32 * j + ii] + dm) / dl);
                l[32 * j + ii] = li.max(0).min(31) as u8;
            }
        }

        let mut qh = [0u8; 32];
        let mut qs = [0u8; 128];
        let mut m1 = 1u8;
        let mut m2 = 2u8;
        let mut ql_pos = 0usize;
        for n in (0..QK_K).step_by(64) {
            for j in 0..32 {
                let mut l1 = l[n + j];
                if l1 > 15 { l1 -= 16; qh[j] |= m1; }
                let mut l2 = l[n + j + 32];
                if l2 > 15 { l2 -= 16; qh[j] |= m2; }
                qs[ql_pos + j] = l1 | (l2 << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
            ql_pos += 32;
        }
        out.extend_from_slice(&qh);
        out.extend_from_slice(&qs);
    }
    out
}

/// Quantize to q6_K (ggml reference algorithm, byte-identical to llama.cpp).
/// 16 sub-blocks of 16 elements; per sub-block SIGNED int8 scale.
pub fn quantize_q6_k(values: &[f32]) -> Vec<u8> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 210;
    let nb = values.len().div_ceil(QK_K);
    let mut out = Vec::with_capacity(nb * BLOCK_BYTES);
    for b in 0..nb {
        let start = b * QK_K;
        let end = (start + QK_K).min(values.len());
        let mut block = [0.0f32; QK_K];
        block[..end - start].copy_from_slice(&values[start..end]);

        let mut l = [0i8; QK_K];
        let mut scales = [0.0f32; QK_K / 16];

        let mut max_scale = 0.0f32;
        let mut max_abs_scale = 0.0f32;
        for j in 0..QK_K / 16 {
            scales[j] = make_qx_quants(16, 32, &block[16 * j..16 * j + 16], &mut l[16 * j..16 * j + 16], 1);
            let abs_scale = scales[j].abs();
            if abs_scale > max_abs_scale {
                max_abs_scale = abs_scale;
                max_scale = scales[j];
            }
        }

        if max_abs_scale < GROUP_MAX_EPS {
            out.resize(out.len() + BLOCK_BYTES, 0);
            continue;
        }

        let iscale = -128.0f32 / max_scale;
        let mut y_scales = [0i8; 16];
        for j in 0..QK_K / 16 {
            y_scales[j] = nearest_int(iscale * scales[j]).min(127) as i8;
        }
        let d_bytes = f16_bytes(1.0 / iscale);
        let d = f16::from_bits(u16::from_le_bytes(d_bytes)).to_f32();

        for j in 0..QK_K / 16 {
            let dl = d * y_scales[j] as f32;
            if dl == 0.0 { continue; }
            for ii in 0..16 {
                let li = nearest_int(block[16 * j + ii] / dl);
                l[16 * j + ii] = li.max(-32).min(31) as i8 + 32;
            }
        }

        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for j in (0..QK_K).step_by(128) {
            let ql_out = &mut ql[j / 128 * 64..j / 128 * 64 + 64];
            let qh_out = &mut qh[j / 128 * 32..j / 128 * 32 + 32];
            for li in 0..32 {
                let q1 = l[j + li] as u8;
                let q2 = l[j + li + 32] as u8;
                let q3 = l[j + li + 64] as u8;
                let q4 = l[j + li + 96] as u8;
                ql_out[li] = (q1 & 0x0F) | (q3 << 4);
                ql_out[li + 32] = (q2 & 0x0F) | (q4 << 4);
                qh_out[li] = (q1 >> 4)
                    | ((q2 >> 4) << 2)
                    | ((q3 >> 4) << 4)
                    | ((q4 >> 4) << 6);
            }
        }

        // ggml block order: ql, qh, scales, d
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        for v in y_scales { out.push(v as u8); }
        out.extend_from_slice(&d_bytes);
    }
    out
}

/// Quantize to q8_K (ggml reference algorithm, byte-identical to llama.cpp).
/// f32 d, 256 int8 quants, 16 int16 block sums (bsums, unused by dequantize).
pub fn quantize_q8_k(values: &[f32]) -> Vec<u8> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 292;
    let nb = values.len().div_ceil(QK_K);
    let mut out = Vec::with_capacity(nb * BLOCK_BYTES);
    for b in 0..nb {
        let start = b * QK_K;
        let end = (start + QK_K).min(values.len());
        let mut block = [0.0f32; QK_K];
        block[..end - start].copy_from_slice(&values[start..end]);

        let mut amax = 0.0f32;
        let mut max = 0.0f32; // signed extreme, ggml uses it for the scale
        for &v in &block {
            let a = v.abs();
            if a > amax { amax = a; max = v; }
        }
        if amax == 0.0 {
            out.extend_from_slice(&0.0f32.to_le_bytes());
            out.resize(out.len() + QK_K + QK_K / 16 * 2, 0);
            continue;
        }
        let iscale = -127.0f32 / max;
        let mut q_bytes = [0u8; QK_K];
        for j in 0..QK_K {
            let v = nearest_int(iscale * block[j]);
            q_bytes[j] = v.min(127) as i8 as u8;
        }
        // ggml block order: d (f32), qs (256), bsums (16 x i16)
        out.extend_from_slice(&(1.0f32 / iscale).to_le_bytes());
        out.extend_from_slice(&q_bytes);
        for j in 0..QK_K / 16 {
            let sum: i16 = q_bytes[j * 16..j * 16 + 16].iter().map(|&q| q as i8 as i16).sum();
            out.extend_from_slice(&sum.to_le_bytes());
        }
    }
    out
}
