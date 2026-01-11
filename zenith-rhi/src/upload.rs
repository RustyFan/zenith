//! Staging-buffer-based upload utilities.

use ash::vk;

use crate::{
    Buffer, BufferDesc, BufferState, ImmediateCommandEncoder, RenderDevice,
    BufferBarrier, PipelineStage, PipelineStages,
};
use crate::buffer::BufferRange;

struct PendingBufferCopy<'a> {
    dst: BufferRange<'a>,
    src_offset: vk::DeviceSize,
    size: vk::DeviceSize,
    final_state: BufferState,
}

/// A simple upload pool backed by a single reusable staging buffer.
///
/// - Supports queueing multiple buffer uploads and flushing them in one submit.
/// - Current behavior is **blocking**: `flush()` waits on a fence.
pub struct UploadPool<'a> {
    staging: Buffer,
    staging_size: vk::DeviceSize,
    write_head: vk::DeviceSize,
    pending: Vec<PendingBufferCopy<'a>>,
}

impl<'a> UploadPool<'a> {
    pub fn new(device: &RenderDevice, staging_size: vk::DeviceSize) -> Result<Self, vk::Result> {
        let staging = Buffer::new(device, &BufferDesc::staging("upload_pool_staging", staging_size))?;
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
        dst: BufferRange<'a>,
        data: &[u8],
        final_state: BufferState,
    ) -> Result<(), vk::Result> {
        self.enqueue_buffer_ex(dst, data, final_state)
    }

    pub fn enqueue_buffer_ex(
        &mut self,
        dst: BufferRange<'a>,
        data: &[u8],
        final_state: BufferState,
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
        // Write into staging (exact range).
        self.staging
            .as_range((src_offset as u64)..((src_offset + size) as u64))?
            .write(data)?;
        self.write_head += size;

        self.pending.push(PendingBufferCopy {
            dst,
            src_offset,
            size,
            final_state,
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
                    self.staging.as_range(..).unwrap(),
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
                    p.dst.buffer().as_range(..).unwrap(),
                    BufferState::Undefined,
                    BufferState::TransferDst,
                    PipelineStages::empty(),
                    PipelineStage::Transfer.into(),
                    q,
                    q,
                    false,
                ).with_range(p.dst.offset() as usize, p.size as usize));
            }
            encoder.buffer_barriers(&pre);

            // Copies
            for p in pending.iter() {
                let region = vk::BufferCopy::default()
                    .src_offset(p.src_offset)
                    .dst_offset(p.dst.offset() as vk::DeviceSize)
                    .size(p.size);
                encoder.copy_buffer(staging_handle, p.dst.buffer().handle(), std::slice::from_ref(&region));
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
                    p.dst.buffer().as_range(..).unwrap(),
                    BufferState::TransferDst,
                    p.final_state,
                    PipelineStage::Transfer.into(),
                    dst_stage,
                    q,
                    q,
                    true,
                ).with_range(p.dst.offset() as usize, p.size as usize));
            }
            encoder.buffer_barriers(&post);
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
        dst: BufferRange<'a>,
        data: &[u8],
        final_state: BufferState,
    ) -> Result<(), vk::Result> {
        if self.enqueue_copy(dst, data, final_state).is_err() {
            self.flush(immediate, device)?;
            self.enqueue_copy(dst, data, final_state)?;
        }
        self.flush(immediate, device)
    }
}


