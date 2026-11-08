//! Pure-Rust Vision Tower for LFM3-VL-2B.
//!
//! Implements the 14-layer Vision Transformer:
//! - 3D Patchifier: [N, 1536] -> [N, 1024]
//! - Embedding RMSNorm
//! - Axial 2D RoPE
//! - 8 Spatial Conv2D layers (depthwise 3x3 with merge-block permutation)
//! - 6 Grouped Query Attention layers (16 Q heads, 8 KV heads, head_dim 64)
//! - Fused SwiGLU MLP layers

use anyhow::Result;
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext, SafeTensorsLoader};
use oxide_infer_kernels::gemm::FUSED_GEMV_FP16_FP32ACCUM_PTX;
use oxide_infer_kernels::norm::RMS_NORM_FP16_PTX;
use oxide_infer_kernels::activations::SWIGLU_FP16_PTX;
use oxide_infer_kernels::vision::VISION_OPS_PTX;
use oxide_infer_kernels::PtxModule;

pub enum VisionLayerKind {
    Conv {
        in_proj_w: DeviceBuffer<f16>,
        spatial_conv_w: DeviceBuffer<f16>,
        out_proj_w: DeviceBuffer<f16>,
    },
    Attention {
        q_proj_w: DeviceBuffer<f16>,
        k_proj_w: DeviceBuffer<f16>,
        v_proj_w: DeviceBuffer<f16>,
        out_proj_w: DeviceBuffer<f16>,
        q_layernorm_w: DeviceBuffer<f16>,
        k_layernorm_w: DeviceBuffer<f16>,
    },
}

pub struct VisionLayerWeights {
    pub operator_norm_w: DeviceBuffer<f16>,
    pub ffn_norm_w: DeviceBuffer<f16>,
    pub mlp_w1: DeviceBuffer<f16>,
    pub mlp_w2: DeviceBuffer<f16>,
    pub mlp_w3: DeviceBuffer<f16>,
    pub kind: VisionLayerKind,
}

pub struct VisionTower {
    pub patchifier_w: DeviceBuffer<f16>,
    pub embedding_norm_w: DeviceBuffer<f16>,
    pub layers: Vec<VisionLayerWeights>,

    // PTX Microkernel modules
    gemv_ptx: PtxModule,
    rms_ptx: PtxModule,
    swiglu_ptx: PtxModule,
    ops_ptx: PtxModule,

    // Precomputed scratch buffers for execution
    pub scratch_h1: DeviceBuffer<f16>,
    pub scratch_h2: DeviceBuffer<f16>,
    pub scratch_norm: DeviceBuffer<f16>,
    pub scratch_proj_3c: DeviceBuffer<f16>,
    pub scratch_gate_bx: DeviceBuffer<f16>,
    pub scratch_gate_c: DeviceBuffer<f16>,
    pub scratch_raster_in: DeviceBuffer<f16>,
    pub scratch_raster_out: DeviceBuffer<f16>,
    pub scratch_q: DeviceBuffer<f16>,
    pub scratch_k: DeviceBuffer<f16>,
    pub scratch_v: DeviceBuffer<f16>,
    pub scratch_scores: DeviceBuffer<f32>,
    pub scratch_attn_out: DeviceBuffer<f16>,
    pub scratch_mlp_w1: DeviceBuffer<f16>,
    pub scratch_mlp_w3: DeviceBuffer<f16>,
    pub scratch_mlp_swiglu: DeviceBuffer<f16>,
}

const LAYER_TYPES: [&str; 14] = [
    "conv", "conv", "full_attention", "conv", "full_attention", "conv",
    "full_attention", "conv", "full_attention", "conv", "full_attention",
    "conv", "full_attention", "conv"
];

impl VisionTower {
    pub fn load_from_safetensors(gpu: &GpuContext, loader: &SafeTensorsLoader) -> Result<Self> {
        let stream = gpu.compute_stream();
        let convert_bf16_to_f16 = |name: &str| -> Result<DeviceBuffer<f16>> {
            let buf = loader.load_tensor::<half::bf16>(gpu, name)?;
            let host_bf16 = buf.inner().to_host_vec(stream)?;
            let host_f16: Vec<f16> = host_bf16.into_iter().map(|b| f16::from_f32(b.to_f32())).collect();
            DeviceBuffer::from_slice(gpu, &host_f16)
        };

        let patchifier_w = convert_bf16_to_f16("model.vision_tower.patchifier.weight")?;
        let embedding_norm_w = convert_bf16_to_f16("model.vision_tower.embedding_norm.weight")?;

        let mut layers = Vec::with_capacity(14);
        for l_idx in 0..14 {
            let prefix = format!("model.vision_tower.layers.{}", l_idx);
            let operator_norm_w = convert_bf16_to_f16(&format!("{}.operator_norm.weight", prefix))?;
            let ffn_norm_w = convert_bf16_to_f16(&format!("{}.ffn_norm.weight", prefix))?;
            let mlp_w1 = convert_bf16_to_f16(&format!("{}.feed_forward.w1.weight", prefix))?;
            let mlp_w2 = convert_bf16_to_f16(&format!("{}.feed_forward.w2.weight", prefix))?;
            let mlp_w3 = convert_bf16_to_f16(&format!("{}.feed_forward.w3.weight", prefix))?;

            let kind = if LAYER_TYPES[l_idx] == "conv" {
                let in_proj_w = convert_bf16_to_f16(&format!("{}.conv.in_proj.weight", prefix))?;
                let spatial_conv_w = convert_bf16_to_f16(&format!("{}.conv.spatial_conv.weight", prefix))?;
                let out_proj_w = convert_bf16_to_f16(&format!("{}.conv.out_proj.weight", prefix))?;
                VisionLayerKind::Conv { in_proj_w, spatial_conv_w, out_proj_w }
            } else {
                let q_proj_w = convert_bf16_to_f16(&format!("{}.self_attn.q_proj.weight", prefix))?;
                let k_proj_w = convert_bf16_to_f16(&format!("{}.self_attn.k_proj.weight", prefix))?;
                let v_proj_w = convert_bf16_to_f16(&format!("{}.self_attn.v_proj.weight", prefix))?;
                let out_proj_w = convert_bf16_to_f16(&format!("{}.self_attn.out_proj.weight", prefix))?;
                let q_layernorm_w = convert_bf16_to_f16(&format!("{}.self_attn.q_layernorm.weight", prefix))?;
                let k_layernorm_w = convert_bf16_to_f16(&format!("{}.self_attn.k_layernorm.weight", prefix))?;
                VisionLayerKind::Attention {
                    q_proj_w, k_proj_w, v_proj_w, out_proj_w, q_layernorm_w, k_layernorm_w,
                }
            };

            layers.push(VisionLayerWeights {
                operator_norm_w, ffn_norm_w, mlp_w1, mlp_w2, mlp_w3, kind,
            });
        }

        let gemv_ptx = PtxModule::load_from_ptx(gpu, FUSED_GEMV_FP16_FP32ACCUM_PTX)?;
        let rms_ptx = PtxModule::load_from_ptx(gpu, RMS_NORM_FP16_PTX)?;
        let swiglu_ptx = PtxModule::load_from_ptx(gpu, SWIGLU_FP16_PTX)?;
        let ops_ptx = PtxModule::load_from_ptx(gpu, VISION_OPS_PTX)?;

        let max_n = 256usize;
        let c = 1024usize;
        let scratch_h1 = DeviceBuffer::<f16>::zeroed(gpu, max_n * c)?;
        let scratch_h2 = DeviceBuffer::<f16>::zeroed(gpu, max_n * c)?;
        let scratch_norm = DeviceBuffer::<f16>::zeroed(gpu, max_n * c)?;
        let scratch_proj_3c = DeviceBuffer::<f16>::zeroed(gpu, max_n * 3 * c)?;
        let scratch_gate_bx = DeviceBuffer::<f16>::zeroed(gpu, max_n * c)?;
        let scratch_gate_c = DeviceBuffer::<f16>::zeroed(gpu, max_n * c)?;
        let scratch_raster_in = DeviceBuffer::<f16>::zeroed(gpu, max_n * c)?;
        let scratch_raster_out = DeviceBuffer::<f16>::zeroed(gpu, max_n * c)?;
        let scratch_q = DeviceBuffer::<f16>::zeroed(gpu, max_n * c)?;
        let scratch_k = DeviceBuffer::<f16>::zeroed(gpu, max_n * (c / 2))?;
        let scratch_v = DeviceBuffer::<f16>::zeroed(gpu, max_n * (c / 2))?;
        let scratch_scores = DeviceBuffer::<f32>::zeroed(gpu, 16 * max_n * max_n)?;
        let scratch_attn_out = DeviceBuffer::<f16>::zeroed(gpu, max_n * c)?;
        let scratch_mlp_w1 = DeviceBuffer::<f16>::zeroed(gpu, max_n * 2560)?;
        let scratch_mlp_w3 = DeviceBuffer::<f16>::zeroed(gpu, max_n * 2560)?;
        let scratch_mlp_swiglu = DeviceBuffer::<f16>::zeroed(gpu, max_n * 2560)?;

        Ok(Self {
            patchifier_w,
            embedding_norm_w,
            layers,
            gemv_ptx,
            rms_ptx,
            swiglu_ptx,
            ops_ptx,
            scratch_h1,
            scratch_h2,
            scratch_norm,
            scratch_proj_3c,
            scratch_gate_bx,
            scratch_gate_c,
            scratch_raster_in,
            scratch_raster_out,
            scratch_q,
            scratch_k,
            scratch_v,
            scratch_scores,
            scratch_attn_out,
            scratch_mlp_w1,
            scratch_mlp_w3,
            scratch_mlp_swiglu,
        })
    }
}
