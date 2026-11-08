//! Pure-Rust Vision Tower & MultiModal Projector for LFM3-VL-2B.
//!
//! Implements the 14-layer Vision Transformer + Multi-Modal Projector:
//! - 3D Patchifier: [N, 1536] -> [N, 1024]
//! - Embedding RMSNorm
//! - Axial 2D RoPE
//! - 8 Spatial Conv2D layers (depthwise 3x3 with merge-block permutation)
//! - 6 Grouped Query Attention layers (16 Q heads, 8 KV heads, head_dim 64)
//! - Fused SwiGLU MLP layers
//! - Multi-Modal Projector: LayerNorm(1024) -> Merge [N/4, 4096] -> Linear 1 -> GELU -> Linear 2

use anyhow::Result;
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext, SafeTensorsLoader};
use oxide_infer_kernels::gemm::FUSED_GEMV_FP16_FP32ACCUM_PTX;
use oxide_infer_kernels::norm::RMS_NORM_FP16_PTX;
use oxide_infer_kernels::activations::SWIGLU_FP16_PTX;
use oxide_infer_kernels::vision::{VISION_ATTN_PTX, VISION_BIAS_ADD_PTX, VISION_GELU_PTX, VISION_LAYER_NORM_PTX, VISION_OPS_PTX, VISION_RMSNORM_PTX};
use oxide_infer_kernels::PtxModule;
use cuda_core::launch_kernel_on_stream;

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
    pub cublas_handle: crate::cublas::CublasHandle,
    pub patchifier_w: DeviceBuffer<f16>,
    pub embedding_norm_w: DeviceBuffer<f16>,
    pub layers: Vec<VisionLayerWeights>,

    // Projector weights
    pub proj_ln_w: DeviceBuffer<f16>,
    pub proj_ln_b: DeviceBuffer<f16>,
    pub proj_l1_w: DeviceBuffer<f16>,
    pub proj_l1_b: DeviceBuffer<f16>,
    pub proj_l2_w: DeviceBuffer<f16>,
    pub proj_l2_b: DeviceBuffer<f16>,

    // Precomputed Axial RoPE tables on GPU: [256, 64]
    pub rope_cos: DeviceBuffer<f16>,
    pub rope_sin: DeviceBuffer<f16>,

    // PTX modules
    gemv_ptx: PtxModule,
    rms_ptx: PtxModule,
    swiglu_ptx: PtxModule,
    ops_ptx: PtxModule,
    ln_ptx: PtxModule,
    gelu_ptx: PtxModule,
    bias_ptx: PtxModule,
    attn_ptx: PtxModule,

    // Reusable GPU scratch buffers
    scratch_h1: DeviceBuffer<f16>,
    scratch_h2: DeviceBuffer<f16>,
    scratch_norm: DeviceBuffer<f16>,
    scratch_proj_3c: DeviceBuffer<f16>,
    scratch_gate_bx: DeviceBuffer<f16>,
    scratch_gate_c: DeviceBuffer<f16>,
    scratch_raster_in: DeviceBuffer<f16>,
    scratch_raster_out: DeviceBuffer<f16>,
    scratch_conv_f: DeviceBuffer<f16>,
    scratch_q: DeviceBuffer<f16>,
    scratch_k: DeviceBuffer<f16>,
    scratch_v: DeviceBuffer<f16>,
    scratch_attn_out: DeviceBuffer<f16>,
    scratch_mlp_w1: DeviceBuffer<f16>,
    scratch_mlp_w3: DeviceBuffer<f16>,
    scratch_mlp_swiglu: DeviceBuffer<f16>,
    scratch_proj_norm: DeviceBuffer<f16>,
    scratch_proj_l1: DeviceBuffer<f16>,
    scratch_proj_gelu: DeviceBuffer<f16>,
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

        let mut cublas_handle: crate::cublas::CublasHandle = std::ptr::null_mut();
        unsafe {
            assert_eq!(crate::cublas::cublasCreate_v2(&mut cublas_handle), 0);
            assert_eq!(crate::cublas::cublasSetStream_v2(cublas_handle, stream.cu_stream() as *mut _), 0);
        }
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

        // Projector weights
        let proj_ln_w = convert_bf16_to_f16("model.multi_modal_projector.layer_norm.weight")?;
        let proj_ln_b = convert_bf16_to_f16("model.multi_modal_projector.layer_norm.bias")?;
        let proj_l1_w = convert_bf16_to_f16("model.multi_modal_projector.linear_1.weight")?;
        let proj_l1_b = convert_bf16_to_f16("model.multi_modal_projector.linear_1.bias")?;
        let proj_l2_w = convert_bf16_to_f16("model.multi_modal_projector.linear_2.weight")?;
        let proj_l2_b = convert_bf16_to_f16("model.multi_modal_projector.linear_2.bias")?;

        // Precompute Axial 2D RoPE table for 16x16 grid (256 tokens)
        let mut cos_table = Vec::with_capacity(256 * 64);
        let mut sin_table = Vec::with_capacity(256 * 64);
        let theta = 1000000.0f32;
        let spatial_dim = 32usize;
        let inv_freq: Vec<f32> = (0..spatial_dim)
            .step_by(2)
            .map(|i| 1.0f32 / theta.powf((i as f32) / (spatial_dim as f32)))
            .collect();

        // 16x16 grid, block size m=2
        // token order is block major: (bh, bw, mh, mw)
        for bh in 0..8 {
            for bw in 0..8 {
                for mh in 0..2 {
                    for mw in 0..2 {
                        let h = (bh * 2 + mh) as f32;
                        let w = (bw * 2 + mw) as f32;

                        let mut freq_hw = Vec::with_capacity(32);
                        for &f in &inv_freq {
                            freq_hw.push(h * f);
                        }
                        for &f in &inv_freq {
                            freq_hw.push(w * f);
                        }

                        // full 64 dim: cat([cos_hw, cos_hw])
                        for _ in 0..2 {
                            for &val in &freq_hw {
                                cos_table.push(f16::from_f32(val.cos()));
                                sin_table.push(f16::from_f32(val.sin()));
                            }
                        }
                    }
                }
            }
        }
        let rope_cos = DeviceBuffer::from_slice(gpu, &cos_table)?;
        let rope_sin = DeviceBuffer::from_slice(gpu, &sin_table)?;

        let gemv_ptx = PtxModule::load_from_ptx(gpu, FUSED_GEMV_FP16_FP32ACCUM_PTX)?;
        let rms_ptx = PtxModule::load_from_ptx(gpu, VISION_RMSNORM_PTX)?;
        let swiglu_ptx = PtxModule::load_from_ptx(gpu, SWIGLU_FP16_PTX)?;
        let ops_ptx = PtxModule::load_from_ptx(gpu, VISION_OPS_PTX)?;
        let ln_ptx = PtxModule::load_from_ptx(gpu, VISION_LAYER_NORM_PTX)?;
        let gelu_ptx = PtxModule::load_from_ptx(gpu, VISION_GELU_PTX)?;
        let bias_ptx = PtxModule::load_from_ptx(gpu, VISION_BIAS_ADD_PTX)?;
        let attn_ptx = PtxModule::load_from_ptx(gpu, VISION_ATTN_PTX)?;

        let n = 256usize;
        let c = 1024usize;
        let scratch_h1 = DeviceBuffer::<f16>::zeroed(gpu, n * c)?;
        let scratch_h2 = DeviceBuffer::<f16>::zeroed(gpu, n * c)?;
        let scratch_norm = DeviceBuffer::<f16>::zeroed(gpu, n * c)?;
        let scratch_proj_3c = DeviceBuffer::<f16>::zeroed(gpu, n * 3 * c)?;
        let scratch_gate_bx = DeviceBuffer::<f16>::zeroed(gpu, n * c)?;
        let scratch_gate_c = DeviceBuffer::<f16>::zeroed(gpu, n * c)?;
        let scratch_raster_in = DeviceBuffer::<f16>::zeroed(gpu, n * c)?;
        let scratch_raster_out = DeviceBuffer::<f16>::zeroed(gpu, n * c)?;
        let scratch_conv_f = DeviceBuffer::<f16>::zeroed(gpu, n * c)?;
        let scratch_q = DeviceBuffer::<f16>::zeroed(gpu, n * c)?;
        let scratch_k = DeviceBuffer::<f16>::zeroed(gpu, n * (c / 2))?;
        let scratch_v = DeviceBuffer::<f16>::zeroed(gpu, n * (c / 2))?;
        let scratch_attn_out = DeviceBuffer::<f16>::zeroed(gpu, n * c)?;
        let scratch_mlp_w1 = DeviceBuffer::<f16>::zeroed(gpu, n * 2560)?;
        let scratch_mlp_w3 = DeviceBuffer::<f16>::zeroed(gpu, n * 2560)?;
        let scratch_mlp_swiglu = DeviceBuffer::<f16>::zeroed(gpu, n * 2560)?;

        let scratch_proj_norm = DeviceBuffer::<f16>::zeroed(gpu, n * c)?;
        let scratch_proj_l1 = DeviceBuffer::<f16>::zeroed(gpu, (n / 4) * 4096)?;
        let scratch_proj_gelu = DeviceBuffer::<f16>::zeroed(gpu, (n / 4) * 4096)?;

        Ok(Self {
            cublas_handle,
            patchifier_w,
            embedding_norm_w,
            layers,
            proj_ln_w,
            proj_ln_b,
            proj_l1_w,
            proj_l1_b,
            proj_l2_w,
            proj_l2_b,
            rope_cos,
            rope_sin,
            gemv_ptx,
            rms_ptx,
            swiglu_ptx,
            ops_ptx,
            ln_ptx,
            gelu_ptx,
            bias_ptx,
            attn_ptx,
            scratch_h1,
            scratch_h2,
            scratch_norm,
            scratch_proj_3c,
            scratch_gate_bx,
            scratch_gate_c,
            scratch_raster_in,
            scratch_raster_out,
            scratch_conv_f,
            scratch_q,
            scratch_k,
            scratch_v,
            scratch_attn_out,
            scratch_mlp_w1,
            scratch_mlp_w3,
            scratch_mlp_swiglu,
            scratch_proj_norm,
            scratch_proj_l1,
            scratch_proj_gelu,
        })
    }

    /// Forward pass of the full Vision Tower + Projector on 256 patches (1536 dim).
    /// Produces 64 visual tokens (2048 dim) on GPU.
    pub fn forward(
        &mut self,
        gpu: &GpuContext,
        pixel_patches: &DeviceBuffer<f16>,
        out_projected_tokens: &mut DeviceBuffer<f16>, // [64, 2048]
    ) -> Result<()> {
        let stream = gpu.compute_stream();
        let num_patches = 256usize;
        let c = 1024usize;

        // 1. Patchifier: [256, 1536] x [1536, 1024] -> [256, 1024] (cuBLAS HGEMM)
        let gemv_func = self.gemv_ptx.get_function("fused_gemv_fp16_fp32accum")?;
        unsafe {
            crate::cublas::cublas_matmul_rowmajor(
                self.cublas_handle,
                num_patches as i32,
                c as i32,
                1536,
                pixel_patches.inner().cu_deviceptr() as *const _,
                self.patchifier_w.inner().cu_deviceptr() as *const _,
                self.scratch_h1.inner_mut().cu_deviceptr() as *mut _,
            )?;
        }

        // 2. Embedding RMSNorm
        let rms_func = self.rms_ptx.get_function("vision_rmsnorm_fp16")?;
        for i in 0..num_patches {
            let off = i * c;
            let mut d_out = self.scratch_h1.inner().cu_deviceptr() + (off * 2) as u64;
            let mut d_in = self.scratch_h1.inner().cu_deviceptr() + (off * 2) as u64;
            let mut d_w = self.embedding_norm_w.inner().cu_deviceptr();
            let mut dim_arg = c as u32;
            let mut eps_arg = 1e-5f32;

            let mut args: [*mut std::ffi::c_void; 5] = [
                &mut d_out as *mut _ as *mut std::ffi::c_void,
                &mut d_in as *mut _ as *mut std::ffi::c_void,
                &mut d_w as *mut _ as *mut std::ffi::c_void,
                &mut dim_arg as *mut _ as *mut std::ffi::c_void,
                &mut eps_arg as *mut _ as *mut std::ffi::c_void,
            ];
            unsafe {
                launch_kernel_on_stream(&rms_func, (1, 1, 1), (256, 1, 1), 0, stream, &mut args)?;
            }
        }

        // 3. 14 Vision Transformer Layers
        let unpack_func = self.ops_ptx.get_function("vision_conv_gate_unpack_fp16")?;
        let m2r_func = self.ops_ptx.get_function("vision_merge_to_raster_fp16")?;
        let conv_func = self.ops_ptx.get_function("depthwise_conv2d_3x3_fp16")?;
        let r2m_func = self.ops_ptx.get_function("vision_raster_to_merge_fp16")?;
        let mult_func = self.ops_ptx.get_function("vision_conv_gate_mult_fp16")?;
        let rope_func = self.ops_ptx.get_function("vision_axial_rope_fp16")?;
        let attn_sdpa_func = self.attn_ptx.get_function("vision_gqa_sdpa_fp16")?;
        let add_ptx = PtxModule::load_from_ptx(gpu, crate::RESIDUAL_ADD_PTX)?;
        let add_func = add_ptx.get_function("residual_add_fp16")?;
        let swiglu_func = self.swiglu_ptx.get_function("fused_swiglu_fp16")?;

        for layer in &self.layers {
            // Operator Norm on scratch_h1 -> scratch_norm
            for i in 0..num_patches {
                let off = i * c;
                let mut d_out = self.scratch_norm.inner().cu_deviceptr() + (off * 2) as u64;
                let mut d_in = self.scratch_h1.inner().cu_deviceptr() + (off * 2) as u64;
                let mut d_w = layer.operator_norm_w.inner().cu_deviceptr();
                let mut dim_arg = c as u32;
                let mut eps_arg = 1e-5f32;
                let mut args: [*mut std::ffi::c_void; 5] = [
                    &mut d_out as *mut _ as *mut std::ffi::c_void,
                    &mut d_in as *mut _ as *mut std::ffi::c_void,
                    &mut d_w as *mut _ as *mut std::ffi::c_void,
                    &mut dim_arg as *mut _ as *mut std::ffi::c_void,
                    &mut eps_arg as *mut _ as *mut std::ffi::c_void,
                ];
                unsafe {
                    launch_kernel_on_stream(&rms_func, (1, 1, 1), (256, 1, 1), 0, stream, &mut args)?;
                }
            }

            match &layer.kind {
                VisionLayerKind::Conv { in_proj_w, spatial_conv_w, out_proj_w } => {
                    // in_proj GEMM: [256, 1024] x [1024, 3072] -> [256, 3072] (cuBLAS HGEMM)
                    unsafe {
                        crate::cublas::cublas_matmul_rowmajor(
                            self.cublas_handle,
                            num_patches as i32,
                            (3 * c) as i32,
                            c as i32,
                            self.scratch_norm.inner().cu_deviceptr() as *const _,
                            in_proj_w.inner().cu_deviceptr() as *const _,
                            self.scratch_proj_3c.inner_mut().cu_deviceptr() as *mut _,
                        )?;
                    }

                    // Unpack B*x and C
                    let mut d_bx = self.scratch_gate_bx.inner().cu_deviceptr();
                    let mut d_c = self.scratch_gate_c.inner().cu_deviceptr();
                    let mut d_p3c = self.scratch_proj_3c.inner().cu_deviceptr();
                    let mut n_arg = num_patches as i32;
                    let mut c_arg = c as i32;
                    let mut args_unpack: [*mut std::ffi::c_void; 5] = [
                        &mut d_bx as *mut _ as *mut std::ffi::c_void,
                        &mut d_c as *mut _ as *mut std::ffi::c_void,
                        &mut d_p3c as *mut _ as *mut std::ffi::c_void,
                        &mut n_arg as *mut _ as *mut std::ffi::c_void,
                        &mut c_arg as *mut _ as *mut std::ffi::c_void,
                    ];
                    let mut total_nc = (num_patches * c) as u32;
                    unsafe {
                        launch_kernel_on_stream(&unpack_func, ((total_nc + 255) / 256, 1, 1), (256, 1, 1), 0, stream, &mut args_unpack)?;
                    }

                    // Merge to raster: [256, 1024] -> [1024, 16, 16]
                    let mut d_rin = self.scratch_raster_in.inner().cu_deviceptr();
                    let mut h_arg = 16i32;
                    let mut w_arg = 16i32;
                    let mut m_merge = 2i32;
                    let mut args_m2r: [*mut std::ffi::c_void; 6] = [
                        &mut d_rin as *mut _ as *mut std::ffi::c_void,
                        &mut d_bx as *mut _ as *mut std::ffi::c_void,
                        &mut h_arg as *mut _ as *mut std::ffi::c_void,
                        &mut w_arg as *mut _ as *mut std::ffi::c_void,
                        &mut c_arg as *mut _ as *mut std::ffi::c_void,
                        &mut m_merge as *mut _ as *mut std::ffi::c_void,
                    ];
                    unsafe {
                        launch_kernel_on_stream(&m2r_func, ((total_nc + 255) / 256, 1, 1), (256, 1, 1), 0, stream, &mut args_m2r)?;
                    }

                    // Spatial Conv2D 3x3 depthwise
                    let mut d_rout = self.scratch_raster_out.inner().cu_deviceptr();
                    let mut d_sconv_w = spatial_conv_w.inner().cu_deviceptr();
                    let mut num_frames_i32 = 1i32;
                    let mut args_conv: [*mut std::ffi::c_void; 7] = [
                        &mut d_rout as *mut _ as *mut std::ffi::c_void,
                        &mut d_rin as *mut _ as *mut std::ffi::c_void,
                        &mut d_sconv_w as *mut _ as *mut std::ffi::c_void,
                        &mut num_frames_i32 as *mut _ as *mut std::ffi::c_void,
                        &mut c_arg as *mut _ as *mut std::ffi::c_void,
                        &mut h_arg as *mut _ as *mut std::ffi::c_void,
                        &mut w_arg as *mut _ as *mut std::ffi::c_void,
                    ];
                    unsafe {
                        launch_kernel_on_stream(&conv_func, (1, 1, c as u32), (16, 16, 1), 0, stream, &mut args_conv)?;
                    }

                    // Raster to merge: [1024, 16, 16] -> [256, 1024]
                    let mut d_conv_f = self.scratch_conv_f.inner().cu_deviceptr();
                    let mut args_r2m: [*mut std::ffi::c_void; 6] = [
                        &mut d_conv_f as *mut _ as *mut std::ffi::c_void,
                        &mut d_rout as *mut _ as *mut std::ffi::c_void,
                        &mut h_arg as *mut _ as *mut std::ffi::c_void,
                        &mut w_arg as *mut _ as *mut std::ffi::c_void,
                        &mut c_arg as *mut _ as *mut std::ffi::c_void,
                        &mut m_merge as *mut _ as *mut std::ffi::c_void,
                    ];
                    unsafe {
                        launch_kernel_on_stream(&r2m_func, ((total_nc + 255) / 256, 1, 1), (256, 1, 1), 0, stream, &mut args_r2m)?;
                    }

                    // Gate multiplication: out = C * conv_output
                    let mut d_mult_out = self.scratch_norm.inner().cu_deviceptr();
                    let mut args_mult: [*mut std::ffi::c_void; 4] = [
                        &mut d_mult_out as *mut _ as *mut std::ffi::c_void,
                        &mut d_c as *mut _ as *mut std::ffi::c_void,
                        &mut d_conv_f as *mut _ as *mut std::ffi::c_void,
                        &mut total_nc as *mut _ as *mut std::ffi::c_void,
                    ];
                    unsafe {
                        launch_kernel_on_stream(&mult_func, ((total_nc + 255) / 256, 1, 1), (256, 1, 1), 0, stream, &mut args_mult)?;
                    }

                    // out_proj GEMM: [256, 1024] x [1024, 1024] -> [256, 1024] (cuBLAS HGEMM)
                    unsafe {
                        crate::cublas::cublas_matmul_rowmajor(
                            self.cublas_handle,
                            num_patches as i32,
                            c as i32,
                            c as i32,
                            self.scratch_norm.inner().cu_deviceptr() as *const _,
                            out_proj_w.inner().cu_deviceptr() as *const _,
                            self.scratch_attn_out.inner_mut().cu_deviceptr() as *mut _,
                        )?;
                    }
                }
                VisionLayerKind::Attention { q_proj_w, k_proj_w, v_proj_w, out_proj_w, q_layernorm_w, k_layernorm_w } => {
                    // Q, K, V GEMMs (cuBLAS HGEMM)
                    unsafe {
                        crate::cublas::cublas_matmul_rowmajor(
                            self.cublas_handle,
                            num_patches as i32,
                            c as i32,
                            c as i32,
                            self.scratch_norm.inner().cu_deviceptr() as *const _,
                            q_proj_w.inner().cu_deviceptr() as *const _,
                            self.scratch_q.inner_mut().cu_deviceptr() as *mut _,
                        )?;
                        crate::cublas::cublas_matmul_rowmajor(
                            self.cublas_handle,
                            num_patches as i32,
                            (c / 2) as i32,
                            c as i32,
                            self.scratch_norm.inner().cu_deviceptr() as *const _,
                            k_proj_w.inner().cu_deviceptr() as *const _,
                            self.scratch_k.inner_mut().cu_deviceptr() as *mut _,
                        )?;
                        crate::cublas::cublas_matmul_rowmajor(
                            self.cublas_handle,
                            num_patches as i32,
                            (c / 2) as i32,
                            c as i32,
                            self.scratch_norm.inner().cu_deviceptr() as *const _,
                            v_proj_w.inner().cu_deviceptr() as *const _,
                            self.scratch_v.inner_mut().cu_deviceptr() as *mut _,
                        )?;
                    }

                    // Vectorized RMSNorm on Q (16 heads x 64) and K (8 heads x 64)
                    let qk_norm_func = self.rms_ptx.get_function("vision_qk_head_rmsnorm_fp16")?;
                    let mut d_q = self.scratch_q.inner_mut().cu_deviceptr();
                    let mut d_qw = q_layernorm_w.inner().cu_deviceptr();
                    let mut total_q_heads = (num_patches * 16) as i32;
                    let mut eps_q = 1e-5f32;
                    let mut args_qn: [*mut std::ffi::c_void; 4] = [
                        &mut d_q as *mut _ as *mut std::ffi::c_void,
                        &mut d_qw as *mut _ as *mut std::ffi::c_void,
                        &mut total_q_heads as *mut _ as *mut std::ffi::c_void,
                        &mut eps_q as *mut _ as *mut std::ffi::c_void,
                    ];
                    unsafe { launch_kernel_on_stream(&qk_norm_func, (total_q_heads as u32, 1, 1), (64, 1, 1), 0, stream, &mut args_qn)?; }

                    let mut d_k = self.scratch_k.inner_mut().cu_deviceptr();
                    let mut d_kw = k_layernorm_w.inner().cu_deviceptr();
                    let mut total_k_heads = (num_patches * 8) as i32;
                    let mut args_kn: [*mut std::ffi::c_void; 4] = [
                        &mut d_k as *mut _ as *mut std::ffi::c_void,
                        &mut d_kw as *mut _ as *mut std::ffi::c_void,
                        &mut total_k_heads as *mut _ as *mut std::ffi::c_void,
                        &mut eps_q as *mut _ as *mut std::ffi::c_void,
                    ];
                    unsafe { launch_kernel_on_stream(&qk_norm_func, (total_k_heads as u32, 1, 1), (64, 1, 1), 0, stream, &mut args_kn)?; }

                    // Axial 2D RoPE
                    let mut d_q = self.scratch_q.inner().cu_deviceptr();
                    let mut d_k = self.scratch_k.inner().cu_deviceptr();
                    let mut d_cos = self.rope_cos.inner().cu_deviceptr();
                    let mut d_sin = self.rope_sin.inner().cu_deviceptr();
                    let mut n_i32 = num_patches as i32;
                    let mut q_heads_i32 = 16i32;
                    let mut kv_heads_i32 = 8i32;
                    let mut head_dim_i32 = 64i32;

                    let mut args_rope: [*mut std::ffi::c_void; 8] = [
                        &mut d_q as *mut _ as *mut std::ffi::c_void,
                        &mut d_k as *mut _ as *mut std::ffi::c_void,
                        &mut d_cos as *mut _ as *mut std::ffi::c_void,
                        &mut d_sin as *mut _ as *mut std::ffi::c_void,
                        &mut n_i32 as *mut _ as *mut std::ffi::c_void,
                        &mut q_heads_i32 as *mut _ as *mut std::ffi::c_void,
                        &mut kv_heads_i32 as *mut _ as *mut std::ffi::c_void,
                        &mut head_dim_i32 as *mut _ as *mut std::ffi::c_void,
                    ];
                    let total_q_elem = (num_patches * 16 * 64) as u32;
                    unsafe {
                        launch_kernel_on_stream(&rope_func, ((total_q_elem + 255) / 256, 1, 1), (256, 1, 1), 0, stream, &mut args_rope)?;
                    }

                    // Attention SDPA: on-device native Ampere kernel (zero host copy, fused softmax + V-reduction)
                    let mut d_attn_out = self.scratch_norm.inner().cu_deviceptr();
                    let mut d_q_in = self.scratch_q.inner().cu_deviceptr();
                    let mut d_k_in = self.scratch_k.inner().cu_deviceptr();
                    let mut d_v_in = self.scratch_v.inner().cu_deviceptr();
                    let mut num_p_i32 = num_patches as i32;
                    let mut scale_f32 = 1.0f32 / (64.0f32).sqrt();

                    let mut args_sdpa: [*mut std::ffi::c_void; 6] = [
                        &mut d_attn_out as *mut _ as *mut std::ffi::c_void,
                        &mut d_q_in as *mut _ as *mut std::ffi::c_void,
                        &mut d_k_in as *mut _ as *mut std::ffi::c_void,
                        &mut d_v_in as *mut _ as *mut std::ffi::c_void,
                        &mut num_p_i32 as *mut _ as *mut std::ffi::c_void,
                        &mut scale_f32 as *mut _ as *mut std::ffi::c_void,
                    ];
                    unsafe {
                        launch_kernel_on_stream(
                            &attn_sdpa_func,
                            (16, num_patches as u32, 1),
                            (64, 1, 1),
                            0,
                            stream,
                            &mut args_sdpa,
                        )?;
                    }

                    // out_proj GEMM: [256, 1024] x [1024, 1024] -> [256, 1024] (cuBLAS HGEMM)
                    unsafe {
                        crate::cublas::cublas_matmul_rowmajor(
                            self.cublas_handle,
                            num_patches as i32,
                            c as i32,
                            c as i32,
                            self.scratch_norm.inner().cu_deviceptr() as *const _,
                            out_proj_w.inner().cu_deviceptr() as *const _,
                            self.scratch_attn_out.inner_mut().cu_deviceptr() as *mut _,
                        )?;
                    }                }
            }

            // Residual Add 1: on-device scratch_h1 = scratch_h1 + scratch_attn_out
            let mut d_h1 = self.scratch_h1.inner_mut().cu_deviceptr();
            let mut d_op = self.scratch_attn_out.inner().cu_deviceptr();
            let mut n_res = (num_patches * c) as u32;
            let mut args_add1: [*mut std::ffi::c_void; 3] = [
                &mut d_h1 as *mut _ as *mut std::ffi::c_void,
                &mut d_op as *mut _ as *mut std::ffi::c_void,
                &mut n_res as *mut _ as *mut std::ffi::c_void,
            ];
            unsafe {
                launch_kernel_on_stream(&add_func, ((n_res + 255) / 256, 1, 1), (256, 1, 1), 0, stream, &mut args_add1)?;
            }

            // FFN Norm on scratch_h1 -> scratch_norm
            for i in 0..num_patches {
                let off = i * c;
                let mut d_out = self.scratch_norm.inner().cu_deviceptr() + (off * 2) as u64;
                let mut d_in = self.scratch_h1.inner().cu_deviceptr() + (off * 2) as u64;
                let mut d_w = layer.ffn_norm_w.inner().cu_deviceptr();
                let mut dim_arg = c as u32;
                let mut eps_arg = 1e-5f32;
                let mut args: [*mut std::ffi::c_void; 5] = [
                    &mut d_out as *mut _ as *mut std::ffi::c_void,
                    &mut d_in as *mut _ as *mut std::ffi::c_void,
                    &mut d_w as *mut _ as *mut std::ffi::c_void,
                    &mut dim_arg as *mut _ as *mut std::ffi::c_void,
                    &mut eps_arg as *mut _ as *mut std::ffi::c_void,
                ];
                unsafe {
                    launch_kernel_on_stream(&rms_func, (1, 1, 1), (256, 1, 1), 0, stream, &mut args)?;
                }
            }

            // MLP: w1, w3, swiglu, w2 (cuBLAS HGEMMs)
            let inter_dim = 2560usize;
            unsafe {
                crate::cublas::cublas_matmul_rowmajor(
                    self.cublas_handle,
                    num_patches as i32,
                    inter_dim as i32,
                    c as i32,
                    self.scratch_norm.inner().cu_deviceptr() as *const _,
                    layer.mlp_w1.inner().cu_deviceptr() as *const _,
                    self.scratch_mlp_w1.inner_mut().cu_deviceptr() as *mut _,
                )?;
                crate::cublas::cublas_matmul_rowmajor(
                    self.cublas_handle,
                    num_patches as i32,
                    inter_dim as i32,
                    c as i32,
                    self.scratch_norm.inner().cu_deviceptr() as *const _,
                    layer.mlp_w3.inner().cu_deviceptr() as *const _,
                    self.scratch_mlp_w3.inner_mut().cu_deviceptr() as *mut _,
                )?;
            }

            // SwiGLU: out = silu(w1) * w3
            let mut total_mlp = (num_patches * inter_dim) as u32;
            let mut d_swiglu_out = self.scratch_mlp_swiglu.inner().cu_deviceptr();
            let mut d_gate = self.scratch_mlp_w1.inner().cu_deviceptr();
            let mut d_up = self.scratch_mlp_w3.inner().cu_deviceptr();
            let mut args_swiglu: [*mut std::ffi::c_void; 4] = [
                &mut d_swiglu_out as *mut _ as *mut std::ffi::c_void,
                &mut d_gate as *mut _ as *mut std::ffi::c_void,
                &mut d_up as *mut _ as *mut std::ffi::c_void,
                &mut total_mlp as *mut _ as *mut std::ffi::c_void,
            ];
            unsafe {
                launch_kernel_on_stream(&swiglu_func, ((total_mlp + 255) / 256, 1, 1), (256, 1, 1), 0, stream, &mut args_swiglu)?;
            }

            // w2 GEMM: [256, 2560] x [2560, 1024] -> [256, 1024] (cuBLAS HGEMM)
            unsafe {
                crate::cublas::cublas_matmul_rowmajor(
                    self.cublas_handle,
                    num_patches as i32,
                    c as i32,
                    inter_dim as i32,
                    self.scratch_mlp_swiglu.inner().cu_deviceptr() as *const _,
                    layer.mlp_w2.inner().cu_deviceptr() as *const _,
                    self.scratch_h2.inner_mut().cu_deviceptr() as *mut _,
                )?;
            }

            // Residual Add 2: on-device scratch_h1 = scratch_h1 + scratch_h2
            let mut d_h1_2 = self.scratch_h1.inner_mut().cu_deviceptr();
            let mut d_h2 = self.scratch_h2.inner().cu_deviceptr();
            let mut args_add2: [*mut std::ffi::c_void; 3] = [
                &mut d_h1_2 as *mut _ as *mut std::ffi::c_void,
                &mut d_h2 as *mut _ as *mut std::ffi::c_void,
                &mut n_res as *mut _ as *mut std::ffi::c_void,
            ];
            unsafe {
                launch_kernel_on_stream(&add_func, ((n_res + 255) / 256, 1, 1), (256, 1, 1), 0, stream, &mut args_add2)?;
            }
        }

        // 4. Multi-Modal Projector
        // LayerNorm(1024) on scratch_h1 -> scratch_proj_norm
        let ln_func = self.ln_ptx.get_function("layer_norm_fp16")?;
        let mut d_normed = self.scratch_proj_norm.inner().cu_deviceptr();
        let mut d_h1 = self.scratch_h1.inner().cu_deviceptr();
        let mut d_ln_w = self.proj_ln_w.inner().cu_deviceptr();
        let mut d_ln_b = self.proj_ln_b.inner().cu_deviceptr();
        let mut rows: i32 = num_patches as i32;
        let mut cols: i32 = c as i32;
        let mut eps: f32 = 1e-5;

        let mut args_ln: [*mut std::ffi::c_void; 7] = [
            &mut d_normed as *mut _ as *mut std::ffi::c_void,
            &mut d_h1 as *mut _ as *mut std::ffi::c_void,
            &mut d_ln_w as *mut _ as *mut std::ffi::c_void,
            &mut d_ln_b as *mut _ as *mut std::ffi::c_void,
            &mut rows as *mut _ as *mut std::ffi::c_void,
            &mut cols as *mut _ as *mut std::ffi::c_void,
            &mut eps as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&ln_func, (rows as u32, 1, 1), (256, 1, 1), 0, stream, &mut args_ln)?;
        }

        // Linear 1 GEMM: [64, 4096] x [4096, 4096] -> [64, 4096] (cuBLAS HGEMM)
        let num_merged = 64usize;
        let m1 = 4096u32;
        let k1 = 4096u32;
        unsafe {
            crate::cublas::cublas_matmul_rowmajor(
                self.cublas_handle,
                num_merged as i32,
                m1 as i32,
                k1 as i32,
                self.scratch_proj_norm.inner().cu_deviceptr() as *const _,
                self.proj_l1_w.inner().cu_deviceptr() as *const _,
                self.scratch_proj_l1.inner_mut().cu_deviceptr() as *mut _,
            )?;
        }

        // Bias 1 add
        let bias_func = self.bias_ptx.get_function("bias_add_fp16")?;
        let mut d_l1 = self.scratch_proj_l1.inner().cu_deviceptr();
        let mut d_b1 = self.proj_l1_b.inner().cu_deviceptr();
        let mut num_rows_i32 = 64i32;
        let mut cols_i32 = 4096i32;
        let mut args_b1: [*mut std::ffi::c_void; 4] = [
            &mut d_l1 as *mut _ as *mut std::ffi::c_void,
            &mut d_b1 as *mut _ as *mut std::ffi::c_void,
            &mut num_rows_i32 as *mut _ as *mut std::ffi::c_void,
            &mut cols_i32 as *mut _ as *mut std::ffi::c_void,
        ];
        let total_elems1 = (num_rows_i32 * cols_i32) as u32;
        unsafe {
            launch_kernel_on_stream(&bias_func, ((total_elems1 + 255) / 256, 1, 1), (256, 1, 1), 0, stream, &mut args_b1)?;
        }

        // GELU
        let gelu_func = self.gelu_ptx.get_function("gelu_fp16")?;
        let mut d_gelu = self.scratch_proj_gelu.inner().cu_deviceptr();
        let mut total_u32 = total_elems1;
        let mut args_gelu: [*mut std::ffi::c_void; 3] = [
            &mut d_gelu as *mut _ as *mut std::ffi::c_void,
            &mut d_l1 as *mut _ as *mut std::ffi::c_void,
            &mut total_u32 as *mut _ as *mut std::ffi::c_void,
        ];
        unsafe {
            launch_kernel_on_stream(&gelu_func, ((total_elems1 + 255) / 256, 1, 1), (256, 1, 1), 0, stream, &mut args_gelu)?;
        }

        // Linear 2 GEMM: [64, 4096] x [4096, 2048] -> [64, 2048] (cuBLAS HGEMM)
        let m2 = 2048u32;
        let k2 = 4096u32;
        unsafe {
            crate::cublas::cublas_matmul_rowmajor(
                self.cublas_handle,
                num_merged as i32,
                m2 as i32,
                k2 as i32,
                self.scratch_proj_gelu.inner().cu_deviceptr() as *const _,
                self.proj_l2_w.inner().cu_deviceptr() as *const _,
                out_projected_tokens.inner_mut().cu_deviceptr() as *mut _,
            )?;
        }

        // Bias 2 add
        let mut d_out_ptr = out_projected_tokens.inner().cu_deviceptr();
        let mut d_b2 = self.proj_l2_b.inner().cu_deviceptr();
        let mut cols_i32_2 = 2048i32;
        let mut args_b2: [*mut std::ffi::c_void; 4] = [
            &mut d_out_ptr as *mut _ as *mut std::ffi::c_void,
            &mut d_b2 as *mut _ as *mut std::ffi::c_void,
            &mut num_rows_i32 as *mut _ as *mut std::ffi::c_void,
            &mut cols_i32_2 as *mut _ as *mut std::ffi::c_void,
        ];
        let total_elems2 = (num_rows_i32 * cols_i32_2) as u32;
        unsafe {
            launch_kernel_on_stream(&bias_func, ((total_elems2 + 255) / 256, 1, 1), (256, 1, 1), 0, stream, &mut args_b2)?;
        }

        stream.synchronize()?;
        Ok(())
    }
}

impl Drop for VisionTower {
    fn drop(&mut self) {
        if !self.cublas_handle.is_null() {
            unsafe {
                let _ = crate::cublas::cublasDestroy_v2(self.cublas_handle);
            }
        }
    }
}

unsafe impl Send for VisionTower {}
