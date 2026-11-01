use oxide_infer_core::{GpuContext, SafeTensorsLoader, DeviceBuffer};
use oxide_infer_models::{EmbeddingTable, CausalConvLayer, FfnLayer};
use oxide_infer_kernels::PtxModule;
use oxide_infer_kernels::norm::RMS_NORM_FP16_PTX;
use cuda_core::launch_kernel_on_stream;

const RESIDUAL_ADD_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry residual_add_fp16(
    .param .u64 a,
    .param .u64 b,
    .param .u32 num_elements
) {
    .reg .pred %p0;
    .reg .b32 %idx, %n, %b_idx, %stride;
    .reg .b64 %a_ptr, %b_ptr, %offset;
    .reg .b16 %ha, %hb;
    .reg .f32 %fa, %fb, %fres;

    ld.param.u32 %n, [num_elements];
    mov.u32 %idx, %tid.x;
    mov.u32 %b_idx, %ctaid.x;
    mov.u32 %stride, %ntid.x;
    mad.lo.u32 %idx, %b_idx, %stride, %idx;

    setp.ge.u32 %p0, %idx, %n;
    @%p0 bra EXIT;

    ld.param.u64 %a_ptr, [a];
    ld.param.u64 %b_ptr, [b];

    cvt.u64.u32 %offset, %idx;
    shl.b64 %offset, %offset, 1;

    add.u64 %b_ptr, %b_ptr, %offset;
    ld.global.u16 %hb, [%b_ptr];

    add.u64 %a_ptr, %a_ptr, %offset;
    ld.global.u16 %ha, [%a_ptr];

    cvt.f32.f16 %fa, %ha;
    cvt.f32.f16 %fb, %hb;
    add.f32 %fres, %fa, %fb;
    cvt.rn.f16.f32 %ha, %fres;

    st.global.u16 [%a_ptr], %ha;

EXIT:
    ret;
}
"#;

fn main() -> anyhow::Result<()> {
    let gpu = GpuContext::init(0)?;
    let stream = gpu.compute_stream();
    let loader = SafeTensorsLoader::open("/mnt/hdd/models/LFM3-VL-2B/model.safetensors")?;

    let embed_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.embed_tokens.weight")?;
    let host_bf16 = embed_bf16.inner().to_host_vec(stream)?;
    let host_f16: Vec<half::f16> = host_bf16.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
    let embed_table = EmbeddingTable::new(&gpu, &host_f16, 128000, 2048)?;

    let mut d_hidden = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let d_tok0 = DeviceBuffer::from_slice(&gpu, &[124894u32])?;
    embed_table.lookup(&gpu, &d_tok0, 1, &mut d_hidden)?;

    let norm_ptx = PtxModule::load_from_ptx(&gpu, RMS_NORM_FP16_PTX)?;
    let add_ptx = PtxModule::load_from_ptx(&gpu, RESIDUAL_ADD_PTX)?;
    let rms_func = norm_ptx.get_function("rms_norm_fp16")?;
    let add_func = add_ptx.get_function("residual_add_fp16")?;

    let op_norm_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.layers.0.operator_norm.weight")?;
    let post_op_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.layers.0.post_operator_norm.weight")?;
    let ffn_norm_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.layers.0.ffn_norm.weight")?;
    let post_ffn_bf16 = loader.load_tensor::<half::bf16>(&gpu, "model.language_model.layers.0.post_ffn_norm.weight")?;

    let to_f16 = |buf: DeviceBuffer<half::bf16>| -> anyhow::Result<DeviceBuffer<half::f16>> {
        let vec: Vec<half::f16> = buf.inner().to_host_vec(stream)?.into_iter().map(|b| half::f16::from_f32(b.to_f32())).collect();
        DeviceBuffer::from_slice(&gpu, &vec)
    };

    let op_norm = to_f16(op_norm_bf16)?;
    let post_op_norm = to_f16(post_op_bf16)?;
    let ffn_norm = to_f16(ffn_norm_bf16)?;
    let post_ffn_norm = to_f16(post_ffn_bf16)?;

    let mut conv = CausalConvLayer::load_from_safetensors(&gpu, &loader, 0, 2048, 7)?;
    let mut ffn = FfnLayer::load_from_safetensors(&gpu, &loader, 0, 2048, 7680)?;

    let mut scratch_norm = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_layer = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;
    let mut scratch_ffn = DeviceBuffer::<half::f16>::zeroed(&gpu, 2048)?;

    let mut eps: f32 = 1e-5;
    let mut hsize: u32 = 2048;

    // 1. operator_norm
    let mut d_normed = scratch_norm.inner_mut().cu_deviceptr();
    let mut d_hid = d_hidden.inner().cu_deviceptr();
    let mut d_w = op_norm.inner().cu_deviceptr();
    let mut args: [*mut std::ffi::c_void; 5] = [
        &mut d_normed as *mut _ as *mut std::ffi::c_void,
        &mut d_hid as *mut _ as *mut std::ffi::c_void,
        &mut d_w as *mut _ as *mut std::ffi::c_void,
        &mut eps as *mut _ as *mut std::ffi::c_void,
        &mut hsize as *mut _ as *mut std::ffi::c_void,
    ];
    unsafe { launch_kernel_on_stream(&rms_func, (1, 1, 1), (128, 1, 1), 4, stream, &mut args)?; }

    // 2. conv
    conv.forward(&gpu, &scratch_norm, &mut scratch_layer)?;

    // 3. post_operator_norm
    let mut d_layer = scratch_layer.inner_mut().cu_deviceptr();
    let mut d_post_op = post_op_norm.inner().cu_deviceptr();
    let mut args_post_op: [*mut std::ffi::c_void; 5] = [
        &mut d_layer as *mut _ as *mut std::ffi::c_void,
        &mut d_layer as *mut _ as *mut std::ffi::c_void,
        &mut d_post_op as *mut _ as *mut std::ffi::c_void,
        &mut eps as *mut _ as *mut std::ffi::c_void,
        &mut hsize as *mut _ as *mut std::ffi::c_void,
    ];
    unsafe { launch_kernel_on_stream(&rms_func, (1, 1, 1), (128, 1, 1), 4, stream, &mut args_post_op)?; }

    println!("Rust post_op first 5: {:?}", &scratch_layer.inner().to_host_vec(stream)?[..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());

    // 4. residual 1
    let mut d_hid_mut = d_hidden.inner_mut().cu_deviceptr();
    let mut args_add1: [*mut std::ffi::c_void; 3] = [
        &mut d_hid_mut as *mut _ as *mut std::ffi::c_void,
        &mut d_layer as *mut _ as *mut std::ffi::c_void,
        &mut hsize as *mut _ as *mut std::ffi::c_void,
    ];
    unsafe { launch_kernel_on_stream(&add_func, (16, 1, 1), (128, 1, 1), 0, stream, &mut args_add1)?; }

    println!("Rust res1 first 5: {:?}", &d_hidden.inner().to_host_vec(stream)?[..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());

    // 5. ffn_norm
    let mut d_ffn_w = ffn_norm.inner().cu_deviceptr();
    let mut args_ffn_norm: [*mut std::ffi::c_void; 5] = [
        &mut d_normed as *mut _ as *mut std::ffi::c_void,
        &mut d_hid_mut as *mut _ as *mut std::ffi::c_void,
        &mut d_ffn_w as *mut _ as *mut std::ffi::c_void,
        &mut eps as *mut _ as *mut std::ffi::c_void,
        &mut hsize as *mut _ as *mut std::ffi::c_void,
    ];
    unsafe { launch_kernel_on_stream(&rms_func, (1, 1, 1), (128, 1, 1), 4, stream, &mut args_ffn_norm)?; }

    println!("Rust ffn_norm first 5: {:?}", &scratch_norm.inner().to_host_vec(stream)?[..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());

    // 6. ffn
    ffn.forward(&gpu, &scratch_norm, &mut scratch_ffn)?;

    println!("Rust ffn_o (raw) first 5: {:?}", &scratch_ffn.inner().to_host_vec(stream)?[..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());

    // 7. post_ffn_norm
    let mut d_ffn_out = scratch_ffn.inner_mut().cu_deviceptr();
    let mut d_post_ffn = post_ffn_norm.inner().cu_deviceptr();
    let mut args_post_ffn: [*mut std::ffi::c_void; 5] = [
        &mut d_ffn_out as *mut _ as *mut std::ffi::c_void,
        &mut d_ffn_out as *mut _ as *mut std::ffi::c_void,
        &mut d_post_ffn as *mut _ as *mut std::ffi::c_void,
        &mut eps as *mut _ as *mut std::ffi::c_void,
        &mut hsize as *mut _ as *mut std::ffi::c_void,
    ];
    unsafe { launch_kernel_on_stream(&rms_func, (1, 1, 1), (128, 1, 1), 4, stream, &mut args_post_ffn)?; }

    // 8. residual 2
    let mut args_add2: [*mut std::ffi::c_void; 3] = [
        &mut d_hid_mut as *mut _ as *mut std::ffi::c_void,
        &mut d_ffn_out as *mut _ as *mut std::ffi::c_void,
        &mut hsize as *mut _ as *mut std::ffi::c_void,
    ];
    unsafe { launch_kernel_on_stream(&add_func, (16, 1, 1), (128, 1, 1), 0, stream, &mut args_add2)?; }

    stream.synchronize()?;
    let h_res = d_hidden.inner().to_host_vec(stream)?;
    println!("Step-by-step layer 0 out: first 5: {:?}", &h_res[..5].iter().map(|x| x.to_f32()).collect::<Vec<_>>());

    Ok(())
}
