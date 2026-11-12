//! Differential test against llama.cpp's ggml quantization implementation.
//!
//! llama.cpp's quantize_row_*_ref / dequantize_row_* are the ground truth for
//! block layouts and math. This test reads pre-generated oracle artifacts
//! (produced by /tmp/k_oracle rebuilt from a local llama.cpp checkout) and
//! compares byte-exactly.
//!
//! Set OXIDE_K_ORACLE_DIR to the directory holding:
//!   x.bin                     - 2048 little-endian f32 input weights
//!   <type>_packed.bin         - llama.cpp-packed blocks for that input
//!   <type>_dequant.bin        - llama.cpp-dequantized f32 values
//!
//! q8_1 has no ggml dequantizer (packed bytes compared only). If the variable
//! is unset the test passes silently (oracle not available).

use half::bf16;
use oxide_infer_core::quant::{
    dequantize_q1_0, dequantize_q2_0, dequantize_q2_k, dequantize_q3_k, dequantize_q4_0,
    dequantize_q4_1, dequantize_q4_k, dequantize_q5_0, dequantize_q5_1, dequantize_q5_k,
    dequantize_q6_k, dequantize_q8_0, dequantize_q8_k,
    quantize_q1_0, quantize_q2_0, quantize_q2_k, quantize_q3_k, quantize_q4_0,
    quantize_q4_1, quantize_q4_k, quantize_q5_0, quantize_q5_1, quantize_q5_k,
    quantize_q6_k, quantize_q8_0, quantize_q8_1, quantize_q8_k,
};

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn bf16s_to_f32(v: &[bf16]) -> Vec<f32> {
    v.iter().map(|b| b.to_f32()).collect()
}

struct Case {
    name: &'static str,
    /// None means ggml has no dequantizer for this type (compare packed bytes only).
    dequant: Option<fn(&[u8]) -> Vec<bf16>>,
    quant: fn(&[f32]) -> Vec<u8>,
}

const CASES: &[Case] = &[
    Case { name: "q1_0", quant: quantize_q1_0, dequant: Some(dequantize_q1_0) },
    Case { name: "q2_0", quant: quantize_q2_0, dequant: Some(dequantize_q2_0) },
    Case { name: "q4_0", quant: quantize_q4_0, dequant: Some(dequantize_q4_0) },
    Case { name: "q4_1", quant: quantize_q4_1, dequant: Some(dequantize_q4_1) },
    Case { name: "q5_0", quant: quantize_q5_0, dequant: Some(dequantize_q5_0) },
    Case { name: "q5_1", quant: quantize_q5_1, dequant: Some(dequantize_q5_1) },
    Case { name: "q8_0", quant: quantize_q8_0, dequant: Some(dequantize_q8_0) },
    Case { name: "q8_1", quant: quantize_q8_1, dequant: None },
    Case { name: "q2_k", quant: quantize_q2_k, dequant: Some(dequantize_q2_k) },
    Case { name: "q3_k", quant: quantize_q3_k, dequant: Some(dequantize_q3_k) },
    Case { name: "q4_k", quant: quantize_q4_k, dequant: Some(dequantize_q4_k) },
    Case { name: "q5_k", quant: quantize_q5_k, dequant: Some(dequantize_q5_k) },
    Case { name: "q6_k", quant: quantize_q6_k, dequant: Some(dequantize_q6_k) },
    Case { name: "q8_k", quant: quantize_q8_k, dequant: Some(dequantize_q8_k) },
];

#[test]
fn ggml_oracle() -> anyhow::Result<()> {
    let dir = match std::env::var("OXIDE_K_ORACLE_DIR") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("OXIDE_K_ORACLE_DIR not set; skipping ggml differential test");
            return Ok(());
        }
    };
    let x = f32s(&std::fs::read(format!("{dir}/x.bin"))?);
    assert_eq!(x.len() % 256, 0, "x.bin must hold whole super-blocks");

    let mut failures = 0usize;
    for c in CASES {
        let packed = std::fs::read(format!("{dir}/{}_packed.bin", c.name))?;

        // 1. Rust quantize must reproduce llama.cpp's packed bytes exactly.
        let rust_packed = (c.quant)(&x);
        let bytes_equal = rust_packed.len() == packed.len()
            && rust_packed.iter().zip(&packed).all(|(a, b)| a == b);
        let byte_mismatch = rust_packed
            .iter()
            .zip(&packed)
            .filter(|(a, b)| a != b)
            .count();

        // 2. Rust dequantize of llama.cpp's packed bytes ≈ llama.cpp dequant.
        let mut deq_ok = true;
        let mut max_err = 0.0f32;
        if let Some(dequant) = c.dequant {
            let rust_deq = bf16s_to_f32(&dequant(&packed));
            let expected_f32 = f32s(&std::fs::read(format!("{dir}/{}_dequant.bin", c.name))?);
            max_err = rust_deq
                .iter()
                .zip(&expected_f32)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            deq_ok = max_err <= 0.05; // bf16 output granularity
        }

        println!(
            "{:>5}: rust-quant bytes == llama.cpp: {} ({} mismatches) | rust-dequant max|err|: {:.4} {}",
            c.name,
            bytes_equal,
            byte_mismatch,
            max_err,
            if deq_ok { "ok" } else { "FAIL" }
        );
        if !bytes_equal || !deq_ok {
            failures += 1;
        }
    }
    assert_eq!(failures, 0, "ggml differential FAILED");
    Ok(())
}
