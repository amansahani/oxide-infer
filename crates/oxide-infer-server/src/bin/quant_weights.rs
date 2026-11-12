//! Quantize f16/bf16 safetensors to Q4_0 (ggml block format).
//!
//! Usage: quant_weights <input.safetensors> <output.safetensors> [tensor_filter]
//!
//! Q4_0 block layout (18 bytes per 32 weights, 4.5 bpw):
//!   f16  d          (scale, super-block scale)
//!   u8   qs[16]     (2 x 4-bit weights per byte, signed via w = d*(q-8))
//!
//! Every tensor with dtype F16/BF16 and numel >= 32 is quantized. The output
//! is a safetensors file identical to the input except those tensors become
//! U8 with shape [num_blocks * 18], and the metadata records
//! `__quantized__: ["q4_0"]` so the loader can dequantize.

use anyhow::{bail, Context, Result};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

const BLOCK: usize = 32;
const BYTES_PER_BLOCK: usize = 18;

fn bf16_to_f32_bytes(raw: &[u8]) -> Vec<f32> {
    assert_eq!(raw.len() % 2, 0);
    raw.chunks_exact(2)
        .map(|c| {
            let lo = c[0] as u32;
            let hi = c[1] as u32;
            let bits = ((hi << 8) | lo) << 16;
            f32::from_bits(bits)
        })
        .collect()
}

fn f16_to_f32_bytes(raw: &[u8]) -> Vec<f32> {
    assert_eq!(raw.len() % 2, 0);
    raw.chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect()
}

fn quantize_f32_to_q4_0(values: &[f32]) -> Vec<u8> {
    let nblocks = values.len().div_ceil(BLOCK);
    let mut out = Vec::with_capacity(nblocks * BYTES_PER_BLOCK);
    for b in 0..nblocks {
        let mut block = [0.0f32; BLOCK];
        let start = b * BLOCK;
        let end = (start + BLOCK).min(values.len());
        block[..end - start].copy_from_slice(&values[start..end]);
        let maxabs = block.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let d = if maxabs > 0.0 { maxabs / 7.0 } else { 1e-8 };
        let d_f16 = half::f16::from_f32(d);
        // pack 2 x 4-bit per byte
        let mut qs = [0u8; 16];
        for i in 0..16 {
            let v0 = block[2 * i];
            let v1 = block[2 * i + 1];
            let q0 = (v0 / d).round() as i32 + 8;
            let q1 = (v1 / d).round() as i32 + 8;
            qs[i] = ((q0.clamp(0, 15) as u8) & 0x0F) | (((q1.clamp(0, 15) as u8) & 0x0F) << 4);
        }
        out.extend_from_slice(&d_f16.to_le_bytes());
        out.extend_from_slice(&qs);
    }
    out
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: quant_weights <input.safetensors> <output.safetensors> [filter_substring]");
    }
    let input = &args[1];
    let output = &args[2];
    let filter = args.get(3).map(|s| s.as_str());
    let buf = fs::read(input)?;
    let tensors = SafeTensors::deserialize(&buf)?;
    let names: Vec<String> = tensors.names().into_iter().map(String::from).collect();
    println!("input: {} tensors", names.len());

    // Decide which tensors to quantize.
    let mut quant_names = Vec::new();
    let mut kept: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)> = HashMap::new();
    for name in &names {
        let t = tensors.tensor(name)?;
        let numel: usize = t.shape().iter().product();
        let dtype = t.dtype();
        let raw = t.data().to_vec();
        let keep_filter = filter.map(|f| name.contains(f)).unwrap_or(true);
        if (dtype == Dtype::F16 || dtype == Dtype::BF16) && numel >= BLOCK && keep_filter {
            quant_names.push(name.clone());
        } else {
            kept.insert(name.clone(), (dtype, t.shape().to_vec(), raw));
        }
    }
    println!("quantizing {} tensors", quant_names.len());

    let mut new_tensors: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)> = kept;
    for name in &quant_names {
        let t = tensors.tensor(name)?;
        let numel: usize = t.shape().iter().product();
        let values: Vec<f32> = match t.dtype() {
            Dtype::BF16 => bf16_to_f32_bytes(t.data()),
            Dtype::F16 => f16_to_f32_bytes(t.data()),
            _ => unreachable!(),
        };
        let packed = quantize_f32_to_q4_0(&values);
        println!(
            "  {:60} {:>12} els -> {} blocks ({} bytes)",
            name,
            numel,
            packed.len() / BYTES_PER_BLOCK,
            packed.len()
        );
        new_tensors.insert(name.clone(), (Dtype::U8, vec![packed.len()], packed));
    }

    // Write safetensors manually with the metadata marker.
    let mut header = serde_json::Map::new();
    let mut offset: usize = 0;
    let mut data_parts: Vec<Vec<u8>> = Vec::new();
    for (name, (dtype, shape, data)) in new_tensors.iter() {
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
        serde_json::json!({
            "__quantized__": "q4_0",
            "source": "quant_weights (ggml q4_0 block format)",
        }),
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
