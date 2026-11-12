//! Quantize f16/bf16 safetensors to ggml quantized formats.
//!
//! Usage: quant_weights <input.safetensors> <output.safetensors> <type> [filter]
//!
//! Supported types (dequantize-at-load supported in the engine):
//!   q1_0, q2_0, q3_K, q4_0, q4_1, q4_K, q5_0, q5_1, q5_K, q6_K, q8_0, q8_1, q8_K
//!
//! The packed format is stored as U8 tensors and the checkpoint metadata records
//! `__quantized__: "<type>"`. The engine's loader reads the marker and
//! dequantizes to BF16 at load time; no GEMM code needs to know the type.

use anyhow::{bail, Context, Result};
use half::f16;
use oxide_infer_core::quant::{quantize, QuantType};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::io::Write;

fn bf16_to_f32_bytes(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|c| f32::from_bits((((c[1] as u32) << 8) | c[0] as u32) << 16))
        .collect()
}

fn f16_to_f32_bytes(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        bail!("usage: quant_weights <input.safetensors> <output.safetensors> <type> [filter]");
    }
    let input = &args[1];
    let output = &args[2];
    let qtype = QuantType::from_str(&args[3])
        .with_context(|| format!("unknown quant type '{}'", args[3]))?;
    let filter = args.get(4).map(|s| s.as_str());

    let buf = fs::read(input)?;
    let tensors = SafeTensors::deserialize(&buf)?;
    let names: Vec<String> = tensors.names().into_iter().map(String::from).collect();
    println!("input: {} tensors, target type: {}", names.len(), qtype.name());

    let mut new_tensors: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)> = HashMap::new();
    let mut qcount = 0usize;
    for name in &names {
        let t = tensors.tensor(name)?;
        let numel: usize = t.shape().iter().product();
        let dtype = t.dtype();
        let raw = t.data().to_vec();
        let keep_filter = filter.map(|f| name.contains(f)).unwrap_or(true);
        if (dtype == Dtype::F16 || dtype == Dtype::BF16) && numel >= qtype.block_elements() && keep_filter {
            let values: Vec<f32> = match dtype {
                Dtype::BF16 => bf16_to_f32_bytes(t.data()),
                Dtype::F16 => f16_to_f32_bytes(t.data()),
                _ => unreachable!(),
            };
            let packed = quantize(qtype, &values);
            println!(
                "  {:60} {:>12} els -> {} bytes",
                name,
                numel,
                packed.len()
            );
            new_tensors.insert(name.clone(), (Dtype::U8, vec![packed.len()], packed));
            qcount += numel;
        } else {
            new_tensors.insert(name.clone(), (dtype, t.shape().to_vec(), raw));
        }
    }
    println!("quantized {} elements", qcount);

    let mut header = serde_json::Map::new();
    let mut offset: usize = 0;
    let mut data_parts: Vec<Vec<u8>> = Vec::new();
    for (name, (dtype, shape, data)) in new_tensors.iter() {
        // No padding: the safetensors format requires tensors to be packed
        // contiguously with no gaps. Alignment for typed casts is handled by
        // the loader copying into an owned Vec before any Pod cast.
        let start = offset;
        let end = start + data.len();
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": format!("{:?}", dtype),
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
        data_parts.push(data.clone());
        offset = end;
    }
    header.insert(
        "__metadata__".to_string(),
        serde_json::json!({ "__quantized__": qtype.name() }),
    );
    let hdr = serde_json::to_vec(&header)?;
    let mut out_file = fs::File::create(output)?;
    out_file.write_all(&(hdr.len() as u64).to_le_bytes())?;
    out_file.write_all(&hdr)?;
    for part in &data_parts {
        out_file.write_all(part)?;
    }
    let out_len = fs::metadata(output)?.len();
    println!("wrote {} ({:.2} GiB)", output, out_len as f64 / 1073741824.0);
    Ok(())
}
