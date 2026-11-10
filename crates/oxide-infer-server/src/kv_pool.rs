//! Pre-allocated KV / Conv1D state pool.
//!
//! Requests no longer allocate per-layer KV caches at admission. One
//! layer-major KV slab and one request-major conv-state slab are allocated at
//! startup; each request leases a contiguous token range plus a conv slot.
//!
//! KV layout: `2 * n_attn` planes, each `capacity_tokens * KV_DIM` f16. For
//! attention layer `l` with attention index `a`, the K plane is `2*a` and the
//! V plane is `2*a + 1`. A lease's token range `[off, off+len)` is used in
//! every plane, so all existing kernels keep their contiguous addressing.
//!
//! Conv layout: `max_batch * n_conv` planes in request-slot-major order, each
//! `HIDDEN * CONV_STATE_ELEMS` f16 (the 6 live channels of the 7-tap window).
//! Conv state is zeroed on lease because positions 0..5 read it as carry.

use std::ffi::c_void;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use half::f16;
use oxide_infer_core::{DeviceBuffer, GpuContext};

use crate::batch_engine::{HIDDEN, KV_DIM};

/// Live conv-state elements per channel: `conv_L_cache - 1 = 6`.
pub const CONV_STATE_ELEMS: usize = 6;

extern "C" {
    fn cuMemsetD8Async(dst: u64, value: u8, bytes: usize, stream: *mut c_void) -> i32;
}

/// Immutable addressing information shared by every leased context.
pub struct KvLayout {
    kv_base: u64,
    conv_base: u64,
    kv_plane_elems: usize,
    conv_plane_elems: usize,
    n_conv: usize,
    attn_plane: Vec<Option<usize>>,
    conv_plane: Vec<Option<usize>>,
    capacity_tokens: usize,
}

impl KvLayout {
    #[inline]
    pub fn k_cache_ptr(&self, layer: usize, tok_off: usize) -> u64 {
        let p = self.attn_plane[layer].expect("k cache pointer on conv layer");
        self.kv_base + ((p * 2 * self.kv_plane_elems + tok_off * KV_DIM) * 2) as u64
    }

    #[inline]
    pub fn v_cache_ptr(&self, layer: usize, tok_off: usize) -> u64 {
        let p = self.attn_plane[layer].expect("v cache pointer on conv layer");
        self.kv_base + (((p * 2 + 1) * self.kv_plane_elems + tok_off * KV_DIM) * 2) as u64
    }

    #[inline]
    pub fn conv_state_ptr(&self, layer: usize, slot: usize) -> u64 {
        let p = self.conv_plane[layer].expect("conv state pointer on attention layer");
        self.conv_base + (((slot * self.n_conv + p) * self.conv_plane_elems) * 2) as u64
    }

    pub fn capacity_tokens(&self) -> usize {
        self.capacity_tokens
    }
}

/// One request's lease on the pool.
#[derive(Clone, Copy, Debug)]
pub struct Lease {
    pub tok_off: usize,
    pub tokens: usize,
    pub slot: usize,
}

/// Startup-allocated KV and conv-state slabs plus a first-fit token allocator.
pub struct KvPool {
    layout: Arc<KvLayout>,
    /// Free token ranges `(start, len)`, sorted by `start` and coalesced.
    free_ranges: Vec<(usize, usize)>,
    free_slots: Vec<usize>,
    n_conv: usize,
    conv_plane_elems: usize,
    _kv: DeviceBuffer<f16>,
    _conv: DeviceBuffer<f16>,
}

impl KvPool {
    /// Allocate the slabs. `capacity_tokens` is a request, not a promise: if
    /// the KV slab does not fit it is halved until allocation succeeds.
    pub fn new(
        gpu: &GpuContext,
        is_conv: &[bool],
        max_batch: usize,
        capacity_tokens: usize,
    ) -> Result<Self> {
        let n_attn = is_conv.iter().filter(|c| !**c).count();
        let n_conv = is_conv.iter().filter(|c| **c).count();
        if n_attn == 0 || n_conv == 0 || max_batch == 0 || capacity_tokens == 0 {
            return Err(anyhow::anyhow!("invalid KV pool geometry"));
        }

        let mut cap = capacity_tokens;
        let kv = loop {
            match DeviceBuffer::<f16>::zeroed(gpu, n_attn * 2 * cap * KV_DIM) {
                Ok(buf) => break buf,
                Err(e) => {
                    if cap > 4096 {
                        let next = cap / 2;
                        eprintln!(
                            "[pool] KV slab allocation for {cap} tokens failed ({e}); retrying with {next}"
                        );
                        cap = next;
                    } else {
                        return Err(e);
                    }
                }
            }
        };

        let conv_plane_elems = HIDDEN * CONV_STATE_ELEMS;
        let conv = DeviceBuffer::<f16>::zeroed(gpu, max_batch * n_conv * conv_plane_elems)
            .map_err(|e| anyhow!("conv-state slab allocation failed: {e}"))?;

        let mut attn_plane = vec![None; is_conv.len()];
        let mut conv_plane = vec![None; is_conv.len()];
        let (mut a, mut c) = (0usize, 0usize);
        for (layer, conv_layer) in is_conv.iter().enumerate() {
            if *conv_layer {
                conv_plane[layer] = Some(c);
                c += 1;
            } else {
                attn_plane[layer] = Some(a);
                a += 1;
            }
        }

        let layout = Arc::new(KvLayout {
            kv_base: kv.inner().cu_deviceptr(),
            conv_base: conv.inner().cu_deviceptr(),
            kv_plane_elems: cap * KV_DIM,
            conv_plane_elems,
            n_conv,
            attn_plane,
            conv_plane,
            capacity_tokens: cap,
        });

        Ok(Self {
            layout,
            free_ranges: vec![(0, cap)],
            free_slots: (0..max_batch).rev().collect(),
            n_conv,
            conv_plane_elems,
            _kv: kv,
            _conv: conv,
        })
    }

    pub fn layout(&self) -> Arc<KvLayout> {
        self.layout.clone()
    }

    pub fn capacity_tokens(&self) -> usize {
        self.layout.capacity_tokens
    }

    pub fn free_tokens(&self) -> usize {
        self.free_ranges.iter().map(|&(_, len)| len).sum()
    }

    /// Lease `tokens` contiguous KV positions and a conv slot. The conv state
    /// of the returned slot is zeroed on the compute stream, stream-ordered
    /// with any subsequent prefill work.
    pub fn acquire(&mut self, gpu: &GpuContext, tokens: usize) -> Option<Lease> {
        if tokens == 0 || tokens > self.layout.capacity_tokens {
            return None;
        }
        let slot = self.free_slots.pop()?;
        let Some(idx) = self
            .free_ranges
            .iter()
            .position(|&(_, len)| len >= tokens)
        else {
            self.free_slots.push(slot);
            return None;
        };

        let (start, len) = self.free_ranges[idx];
        if len == tokens {
            self.free_ranges.remove(idx);
        } else {
            self.free_ranges[idx] = (start + tokens, len - tokens);
        }

        let plane_bytes = self.conv_plane_elems * 2;
        let dst = self.layout.conv_base + (slot * self.n_conv * plane_bytes) as u64;
        let rc = unsafe {
            cuMemsetD8Async(
                dst,
                0,
                self.n_conv * plane_bytes,
                gpu.compute_stream().cu_stream() as *mut c_void,
            )
        };
        if rc != 0 {
            eprintln!("[pool] conv-state zeroing failed (cuda error {rc})");
            self.free_slots.push(slot);
            self.release_range(start, tokens);
            return None;
        }

        Some(Lease {
            tok_off: start,
            tokens,
            slot,
        })
    }

    pub fn release(&mut self, lease: Lease) {
        self.free_slots.push(lease.slot);
        self.release_range(lease.tok_off, lease.tokens);
    }

    fn release_range(&mut self, start: usize, len: usize) {
        let pos = self.free_ranges.partition_point(|&(s, _)| s < start);
        if pos > 0 {
            let (prev_start, prev_len) = self.free_ranges[pos - 1];
            if prev_start + prev_len == start {
                self.free_ranges[pos - 1] = (prev_start, prev_len + len);
                if pos < self.free_ranges.len() {
                    let (next_start, next_len) = self.free_ranges[pos];
                    let (base, base_len) = self.free_ranges[pos - 1];
                    if base + base_len == next_start {
                        self.free_ranges[pos - 1] = (base, base_len + next_len);
                        self.free_ranges.remove(pos);
                    }
                }
                return;
            }
        }
        if pos < self.free_ranges.len() {
            let (next_start, next_len) = self.free_ranges[pos];
            if start + len == next_start {
                self.free_ranges[pos] = (start, len + next_len);
                return;
            }
        }
        self.free_ranges.insert(pos, (start, len));
    }
}
