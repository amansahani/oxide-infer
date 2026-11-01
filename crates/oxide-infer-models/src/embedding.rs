use anyhow::Result;
use oxide_infer_core::{DeviceBuffer, GpuContext};
use oxide_infer_kernels::PtxModule;
use cuda_core::launch_kernel_on_stream;

const EMBED_LOOKUP_PTX: &str = r#"
.version 7.5
.target sm_80
.address_size 64

.visible .entry embed_tokens_lookup_fp16(
    .param .u64 out_embeddings,
    .param .u64 token_ids,
    .param .u64 embedding_table,
    .param .u32 num_tokens,
    .param .u32 hidden_size
) {
    .reg .pred %p0, %p1;
    .reg .b32 %tok_idx, %dim_idx, %ntok, %hdim, %tok_id, %stride;
    .reg .b64 %tok_ptr, %embed_ptr, %out_ptr, %tok_offset, %row_offset, %elem_offset;
    .reg .b16 %val;

    ld.param.u32 %ntok, [num_tokens];
    ld.param.u32 %hdim, [hidden_size];

    mov.u32 %tok_idx, %ctaid.x;
    mov.u32 %dim_idx, %tid.x;

    setp.ge.u32 %p0, %tok_idx, %ntok;
    @%p0 bra EXIT;

    ld.param.u64 %tok_ptr, [token_ids];
    cvt.u64.u32 %tok_offset, %tok_idx;
    shl.b64 %tok_offset, %tok_offset, 2;
    add.u64 %tok_ptr, %tok_ptr, %tok_offset;
    ld.global.u32 %tok_id, [%tok_ptr];

    ld.param.u64 %embed_ptr, [embedding_table];
    ld.param.u64 %out_ptr, [out_embeddings];

    mul.wide.u32 %row_offset, %tok_id, %hdim;

DIM_LOOP:
    setp.ge.u32 %p1, %dim_idx, %hdim;
    @%p1 bra EXIT;

    cvt.u64.u32 %elem_offset, %dim_idx;
    add.u64 %elem_offset, %row_offset, %elem_offset;
    shl.b64 %elem_offset, %elem_offset, 1;

    add.u64 %tok_offset, %embed_ptr, %elem_offset;
    ld.global.u16 %val, [%tok_offset];

    mul.wide.u32 %tok_offset, %tok_idx, %hdim;
    cvt.u64.u32 %elem_offset, %dim_idx;
    add.u64 %tok_offset, %tok_offset, %elem_offset;
    shl.b64 %tok_offset, %tok_offset, 1;
    add.u64 %tok_offset, %out_ptr, %tok_offset;

    st.global.u16 [%tok_offset], %val;

    mov.u32 %stride, %ntid.x;
    add.u32 %dim_idx, %dim_idx, %stride;
    bra DIM_LOOP;

EXIT:
    ret;
}
"#;

pub struct EmbeddingTable {
    pub table: DeviceBuffer<half::f16>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    ptx: PtxModule,
}

impl EmbeddingTable {
    pub fn new(
        gpu: &GpuContext,
        table_weights: &[half::f16],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<Self> {
        let table = DeviceBuffer::from_slice(gpu, table_weights)?;
        let ptx = PtxModule::load_from_ptx(gpu, EMBED_LOOKUP_PTX)?;
        Ok(Self {
            table,
            vocab_size,
            hidden_size,
            ptx,
        })
    }

    /// Lookup tokens: token_ids [N] -> out_embeddings [N, hidden_size]
    pub fn lookup(
        &self,
        gpu: &GpuContext,
        token_ids: &DeviceBuffer<u32>,
        num_tokens: usize,
        out_embeddings: &mut DeviceBuffer<half::f16>,
    ) -> Result<()> {
        let func = self.ptx.get_function("embed_tokens_lookup_fp16")?;
        let stream = gpu.compute_stream();

        let mut d_out_ptr = out_embeddings.inner_mut().cu_deviceptr();
        let mut d_tok_ptr = token_ids.inner().cu_deviceptr();
        let mut d_tbl_ptr = self.table.inner().cu_deviceptr();
        let mut num_tokens_u32 = num_tokens as u32;
        let mut hidden_size_u32 = self.hidden_size as u32;

        let mut args: [*mut std::ffi::c_void; 5] = [
            &mut d_out_ptr as *mut _ as *mut std::ffi::c_void,
            &mut d_tok_ptr as *mut _ as *mut std::ffi::c_void,
            &mut d_tbl_ptr as *mut _ as *mut std::ffi::c_void,
            &mut num_tokens_u32 as *mut _ as *mut std::ffi::c_void,
            &mut hidden_size_u32 as *mut _ as *mut std::ffi::c_void,
        ];

        let block_dim = 256.min(self.hidden_size as u32);
        let grid_dim = num_tokens as u32;

        unsafe {
            launch_kernel_on_stream(
                &func,
                (grid_dim, 1, 1),
                (block_dim, 1, 1),
                0,
                stream,
                &mut args,
            )?;
        }
        Ok(())
    }
}
