//! Staging-buffer-based upload utilities.

use std::sync::Arc;

use ash::vk;

use crate::{
    Buffer, BufferDesc, BufferState, ImmediateCommandEncoder, RenderDevice,
    BufferBarrier, PipelineStage, PipelineStages,
};

#[derive(Clone)]
struct PendingBufferCopy {
    dst: Arc<Buffer>,
    dst_offset: vk::DeviceSize,
    src_offset: vk::DeviceSize,
    size: vk::DeviceSize,
    final_state: BufferState,
    readonly: bool,
}

/// A simple upload pool backed by a single reusable staging buffer.
///
/// - Supports queueing multiple buffer uploads and flushing them in one submit.
/// - Current behavior is **blocking**: `flush()` waits on a fence.
pub struct UploadPool {
    staging: Buffer,
    staging_size: vk::DeviceSize,
    write_head: vk::DeviceSize,
    pending: Vec<PendingBufferCopy>,
}

impl UploadPool {
    pub fn new(device: &RenderDevice, staging_size: vk::DeviceSize) -> Result<Self, vk::Result> {
        let staging = Buffer::new(device, &BufferDesc::staging(staging_size))?;
        Ok(Self {
            staging,
            staging_size,
            write_head: 0,
            pending: Vec::new(),
        })
    }

    pub fn staging_size(&self) -> vk::DeviceSize { self.staging_size }

    /// Enqueue an upload into `dst` at `dst_offset`.
    ///
    /// If the staging buffer doesn't have enough remaining space, the upload is rejected;
    /// call `flush()` first and retry.
    pub fn enqueue_copy(
        &mut self,
        dst: Arc<Buffer>,
        dst_offset: vk::DeviceSize,
        data: &[u8],
        final_state: BufferState,
    ) -> Result<(), vk::Result> {
        self.enqueue_buffer_ex(dst, dst_offset, data, final_state, true)
    }

    pub fn enqueue_buffer_ex(
        &mut self,
        dst: Arc<Buffer>,
        dst_offset: vk::DeviceSize,
        data: &[u8],
        final_state: BufferState,
        readonly: bool,
    ) -> Result<(), vk::Result> {
        let size = data.len() as vk::DeviceSize;
        if size == 0 {
            return Ok(());
        }
        if size > self.staging_size {
            return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        }
        if self.write_head + size > self.staging_size {
            return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        }

        let src_offset = self.write_head;
        self.staging.write_at(src_offset, data)?;
        self.write_head += size;

        self.pending.push(PendingBufferCopy {
            dst,
            dst_offset,
            src_offset,
            size,
            final_state,
            readonly,
        });

        Ok(())
    }

    pub fn is_empty(&self) -> bool { self.pending.is_empty() }

    /// Flush all pending uploads using an immediate submit, blocking until completion.
    pub fn flush(&mut self, immediate: &ImmediateCommandEncoder, device: &RenderDevice) -> Result<(), vk::Result> {
        if self.pending.is_empty() {
            self.write_head = 0;
            return Ok(());
        }

        let staging_handle = self.staging.handle();
        let staging_size = self.staging.size() as usize;
        let q = device.graphics_queue();

        let pending = std::mem::take(&mut self.pending);

        let result = immediate.submit_and_wait(|encoder| {
            let mut pre: Vec<BufferBarrier> = Vec::with_capacity(1 + pending.len());
            // Staging: HOST_WRITE -> TRANSFER_READ (as TRANSFER_SRC)
            pre.push(
                BufferBarrier::new(
                    &self.staging,
                    BufferState::HostWrite,
                    BufferState::TransferSrc,
                    PipelineStage::Host.into(),
                    PipelineStage::Transfer.into(),
                    q,
                    q,
                    true,
                )
                .with_range(0, staging_size),
            );
            // Dst buffers: Undefined -> TransferDst
            for p in pending.iter() {
                pre.push(BufferBarrier::new(
                    &p.dst,
                    BufferState::Undefined,
                    BufferState::TransferDst,
                    PipelineStages::empty(),
                    PipelineStage::Transfer.into(),
                    q,
                    q,
                    p.readonly,
                ));
            }
            encoder.barrier_buffers(&pre);

            // Copies
            for p in pending.iter() {
                let region = vk::BufferCopy::default()
                    .src_offset(p.src_offset)
                    .dst_offset(p.dst_offset)
                    .size(p.size);
                encoder.copy_buffer(staging_handle, p.dst.handle(), std::slice::from_ref(&region));
            }

            // Post-copy barriers: TRANSFER_DST -> final_state
            let mut post: Vec<BufferBarrier> = Vec::with_capacity(pending.len());
            for p in pending.iter() {
                let dst_stage = match p.final_state {
                    BufferState::Vertex => PipelineStage::VertexAttributeInput.into(),
                    BufferState::Index => PipelineStage::IndexInput.into(),
                    BufferState::TransferSrc | BufferState::TransferDst => PipelineStage::Transfer.into(),
                    BufferState::HostWrite => PipelineStage::Host.into(),
                    BufferState::Uniform | BufferState::Storage | BufferState::Undefined => PipelineStage::AllCommands.into(),
                };
                post.push(BufferBarrier::new(
                    &p.dst,
                    BufferState::TransferDst,
                    p.final_state,
                    PipelineStage::Transfer.into(),
                    dst_stage,
                    q,
                    q,
                    p.readonly,
                ));
            }
            encoder.barrier_buffers(&post);
        });

        if result.is_err() {
            // restore pending on failure (best-effort)
            self.pending = pending;
        }

        result?;
        self.write_head = 0;
        Ok(())
    }

    /// Convenience: enqueue then flush (blocking).
    pub fn upload_buffer(
        &mut self,
        immediate: &ImmediateCommandEncoder,
        device: &RenderDevice,
        dst: Arc<Buffer>,
        dst_offset: vk::DeviceSize,
        data: &[u8],
        final_state: BufferState,
    ) -> Result<(), vk::Result> {
        if self.enqueue_copy(dst.clone(), dst_offset, data, final_state).is_err() {
            self.flush(immediate, device)?;
            self.enqueue_copy(dst, dst_offset, data, final_state)?;
        }
        self.flush(immediate, device)
    }
}


