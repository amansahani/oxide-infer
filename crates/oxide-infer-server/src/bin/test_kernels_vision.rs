use anyhow::Result;
use oxide_infer_core::GpuContext;
use oxide_infer_kernels::vision::{VISION_BIAS_ADD_PTX, VISION_GELU_PTX, VISION_LAYER_NORM_PTX, VISION_OPS_PTX};
use oxide_infer_kernels::PtxModule;

fn main() -> Result<()> {
    println!("Testing PTX loading of vision microkernels on GPU...");
    let gpu = GpuContext::init(0)?;

    let ln_mod = PtxModule::load_from_ptx(&gpu, VISION_LAYER_NORM_PTX)?;
    let _ln_func = ln_mod.get_function("layer_norm_fp16")?;
    println!("✓ layer_norm_fp16 loaded");

    let gelu_mod = PtxModule::load_from_ptx(&gpu, VISION_GELU_PTX)?;
    let _gelu_func = gelu_mod.get_function("gelu_fp16")?;
    println!("✓ gelu_fp16 loaded");

    let bias_mod = PtxModule::load_from_ptx(&gpu, VISION_BIAS_ADD_PTX)?;
    let _bias_func = bias_mod.get_function("bias_add_fp16")?;
    println!("✓ bias_add_fp16 loaded");

    let ops_mod = PtxModule::load_from_ptx(&gpu, VISION_OPS_PTX)?;
    let _conv_func = ops_mod.get_function("depthwise_conv2d_3x3_fp16")?;
    let _unpack_func = ops_mod.get_function("vision_conv_gate_unpack_fp16")?;
    let _mult_func = ops_mod.get_function("vision_conv_gate_mult_fp16")?;
    let _m2r_func = ops_mod.get_function("vision_merge_to_raster_fp16")?;
    let _r2m_func = ops_mod.get_function("vision_raster_to_merge_fp16")?;
    let _rope_func = ops_mod.get_function("vision_axial_rope_fp16")?;
    println!("✓ all vision_ops functions loaded");

    println!("All vision PTX microkernels compiled and verified successfully on device!");
    Ok(())
}
