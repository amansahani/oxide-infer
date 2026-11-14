pub mod ptx;
pub mod gemm;
pub mod gemv;
pub mod norm;
pub mod attention;
pub mod activations;
pub mod conv;
pub mod batched_attn;
pub mod vision;
pub mod quant_gemv;
pub mod oxide_kernels;

pub use ptx::PtxModule;
pub use quant_gemv::{QuantGemv, QUANT_GEMV_PTX, supported as quant_gemv_supported};
pub use oxide_kernels::{OxideKernels, detect_sm, select_ptx};
pub use gemv::GEMV_FP16_PTX;
pub use attention::PagedAttentionExecutor;
pub use activations::{ROPE_FP16_PTX, SWIGLU_FP16_PTX};
pub use conv::CAUSAL_CONV1D_FP16_PTX;
pub use vision::{
    VISION_ATTN_PTX, VISION_BIAS_ADD_PTX, VISION_EMBED_SPLICE_PTX, VISION_GELU_PTX, VISION_LAYER_NORM_PTX,
    VISION_OPS_PTX, VISION_RMSNORM_PTX, VISION_SPATIAL_MERGE_PTX,
};
