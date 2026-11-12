//! Self-contained roundtrip test (no ggml oracle required).
//!
//! quantize_* then dequantize_* on deterministic pseudo-random data; checks
//! that reconstruction error stays inside quantization-noise expectations.
//! (Byte-exact ggml compatibility is checked separately by tests/k_oracle.rs,
//! which needs a built llama.cpp.)

use oxide_infer_core::quant::{
    dequantize_q1_0, dequantize_q2_0, dequantize_q2_k, dequantize_q3_k, dequantize_q4_0,
    dequantize_q4_1, dequantize_q4_k, dequantize_q5_0, dequantize_q5_1, dequantize_q5_k,
    dequantize_q6_k, dequantize_q8_0, dequantize_q8_1, dequantize_q8_k,
    quantize_q1_0, quantize_q2_0, quantize_q2_k, quantize_q3_k, quantize_q4_0,
    quantize_q4_1, quantize_q4_k, quantize_q5_0, quantize_q5_1, quantize_q5_k,
    quantize_q6_k, quantize_q8_0, quantize_q8_1, quantize_q8_k,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f32_pm(&mut self) -> f32 {
        // deterministic uniform in [-3, 3]
        ((self.next() % 2000) as f32 / 1000.0) - 1.0 * 3.0
    }
}

fn data() -> Vec<f32> {
    let mut r = Rng(0x1234_5678_9abc_def0);
    (0..2048).map(|_| r.f32_pm()).collect()
}

struct Case {
    name: &'static str,
    quant: fn(&[f32]) -> Vec<u8>,
    dequant: fn(&[u8]) -> Vec<half::bf16>,
    tol: f32,
}

const CASES: &[Case] = &[
    Case { name: "q1_0", quant: quantize_q1_0, dequant: dequantize_q1_0, tol: 4.5 },
    Case { name: "q2_0", quant: quantize_q2_0, dequant: dequantize_q2_0, tol: 4.5 },
    Case { name: "q4_0", quant: quantize_q4_0, dequant: dequantize_q4_0, tol: 0.5 },
    Case { name: "q4_1", quant: quantize_q4_1, dequant: dequantize_q4_1, tol: 0.5 },
    Case { name: "q5_0", quant: quantize_q5_0, dequant: dequantize_q5_0, tol: 0.5 },
    Case { name: "q5_1", quant: quantize_q5_1, dequant: dequantize_q5_1, tol: 0.5 },
    Case { name: "q8_0", quant: quantize_q8_0, dequant: dequantize_q8_0, tol: 0.2 },
    Case { name: "q8_1", quant: quantize_q8_1, dequant: dequantize_q8_1, tol: 0.2 },
    Case { name: "q2_k", quant: quantize_q2_k, dequant: dequantize_q2_k, tol: 1.0 },
    Case { name: "q3_k", quant: quantize_q3_k, dequant: dequantize_q3_k, tol: 1.0 },
    Case { name: "q4_k", quant: quantize_q4_k, dequant: dequantize_q4_k, tol: 0.5 },
    Case { name: "q5_k", quant: quantize_q5_k, dequant: dequantize_q5_k, tol: 0.5 },
    Case { name: "q6_k", quant: quantize_q6_k, dequant: dequantize_q6_k, tol: 0.5 },
    Case { name: "q8_k", quant: quantize_q8_k, dequant: dequantize_q8_k, tol: 0.2 },
];

#[test]
fn k_type_roundtrip() {
    let x = data();
    let mut all_ok = true;
    for c in CASES {
        let packed = (c.quant)(&x);
        let deq = (c.dequant)(&packed);
        let deq_f32: Vec<f32> = deq.iter().map(|b| b.to_f32()).collect();
        assert_eq!(deq_f32.len(), x.len(), "{}: dequantized length mismatch", c.name);
        let max_err = deq_f32
            .iter()
            .zip(&x)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let ok = max_err <= c.tol;
        all_ok &= ok;
        println!("{:>5}: max |recon - x| = {:.4} {}", c.name, max_err, if ok { "ok" } else { "FAIL" });
    }
    assert!(all_ok, "roundtrip FAILED");
}
