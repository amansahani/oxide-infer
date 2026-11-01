use anyhow::{Context as AnyhowContext, Result};
use cuda_core::{DeviceBuffer as CudaDeviceBuffer, DeviceCopy, PinnedHostBuffer as CudaPinnedHostBuffer};
use crate::context::GpuContext;

/// Device memory buffer allocated on the active GPU.
pub struct DeviceBuffer<T: DeviceCopy> {
    inner: CudaDeviceBuffer<T>,
}

impl<T: DeviceCopy> DeviceBuffer<T> {
    /// Allocate and zero-initialize a device buffer of `len` elements using the compute stream.
    pub fn zeroed(gpu: &GpuContext, len: usize) -> Result<Self> {
        let inner = CudaDeviceBuffer::<T>::zeroed(gpu.compute_stream(), len)
            .context("Failed to allocate zeroed DeviceBuffer")?;
        Ok(Self { inner })
    }

    /// Allocate a device buffer and copy data from host slice.
    pub fn from_slice(gpu: &GpuContext, data: &[T]) -> Result<Self> {
        let mut buf = Self::zeroed(gpu, data.len())?;
        buf.inner
            .copy_from_host(gpu.compute_stream(), data)
            .context("Failed to copy slice into DeviceBuffer")?;
        Ok(Self { inner: buf.inner })
    }

    /// Update an existing device buffer with host slice asynchronously (zero allocation overhead).
    pub fn copy_from_slice(&mut self, gpu: &GpuContext, data: &[T]) -> Result<()> {
        self.inner
            .copy_from_host(gpu.compute_stream(), data)
            .context("Failed to copy slice into DeviceBuffer asynchronously")?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn inner(&self) -> &CudaDeviceBuffer<T> {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut CudaDeviceBuffer<T> {
        &mut self.inner
    }
}

/// Host-pinned memory buffer for zero-overhead asynchronous DMA transfers.
pub struct HostPinnedBuffer<T: DeviceCopy> {
    inner: CudaPinnedHostBuffer<T>,
}

impl<T: DeviceCopy> HostPinnedBuffer<T> {
    pub fn zeroed(gpu: &GpuContext, len: usize) -> Result<Self> {
        let inner = CudaPinnedHostBuffer::<T>::zeroed(gpu.context(), len)
            .context("Failed to allocate PinnedHostBuffer")?;
        Ok(Self { inner })
    }

    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }

    pub fn inner(&self) -> &CudaPinnedHostBuffer<T> {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut CudaPinnedHostBuffer<T> {
        &mut self.inner
    }
}
