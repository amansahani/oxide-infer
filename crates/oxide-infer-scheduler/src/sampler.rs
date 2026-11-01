//! Logit sampling implementations (Greedy Argmax, Nucleus Top-P, Temperature).

/// Select token with the highest logit value (Greedy decoding).
pub fn sample_greedy(logits: &[f32]) -> u32 {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;

    for (idx, &val) in logits.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }

    best_idx as u32
}

/// Apply temperature scaling in place.
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature > 0.0 && (temperature - 1.0).abs() > 1e-5 {
        let inv_temp = 1.0 / temperature;
        for val in logits.iter_mut() {
            *val *= inv_temp;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_greedy() {
        let logits = vec![-1.2, 0.5, 4.8, 2.1, 4.79];
        assert_eq!(sample_greedy(&logits), 2);
    }
}
