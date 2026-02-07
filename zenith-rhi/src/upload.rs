//! Staging-buffer-based upload utilities.

use ash::vk;

use crate::{
    Buffer, BufferDesc, BufferState, ImmediateCommandEncoder, RenderDevice,
    BufferBarrier, PipelineStage, PipelineStages,
    TextureBarrier, TextureLayout, TextureState,
};
use crate::buffer::BufferRange;
use crate::texture::TextureRange;

struct PendingBufferCopy<'a> {
    dst: BufferRange<'a>,
    src_offset: vk::DeviceSize,
    size: vk::DeviceSize,
    final_state: BufferState,
}

struct PendingTextureCopy<'a> {
    dst: TextureRange<'a>,
    src_offset: vk::DeviceSize,
    final_state: TextureState,
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
    pending_textures: Vec<PendingTextureCopy<'a>>,
}

impl<'a> UploadPool<'a> {
    pub fn new(device: &RenderDevice, staging_size: vk::DeviceSize) -> Result<Self, vk::Result> {
        let staging = Buffer::new(device, &BufferDesc::staging("upload_pool.staging", staging_size))?;
        Ok(Self {
            staging,
            staging_size,
            write_head: 0,
            pending: Vec::new(),
            pending_textures: Vec::new(),
        })
    }

    #[inline]
    pub fn staging_size(&self) -> vk::DeviceSize { self.staging_size }

    pub fn enqueue_copy_buffer(
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

    pub fn enqueue_upload_texture(
        &mut self,
        dst: TextureRange<'a>,
        data: &[u8],
        final_state: TextureState,
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
        self.staging
            .as_range((src_offset as u64)..((src_offset + size) as u64))?
            .write(data)?;
        self.write_head += size;

        self.pending_textures.push(PendingTextureCopy {
            dst,
            src_offset,
            final_state,
        });

        Ok(())
    }

    #[inline]
    pub fn is_empty(&self) -> bool { self.pending.is_empty() && self.pending_textures.is_empty() }

    /// Flush all pending uploads using an immediate submit, blocking until completion.
    pub fn flush(&mut self, immediate: &ImmediateCommandEncoder, device: &RenderDevice) -> Result<(), vk::Result> {
        if self.pending.is_empty() && self.pending_textures.is_empty() {
            self.write_head = 0;
            return Ok(());
        }

        let staging_handle = self.staging.handle();
        let staging_size = self.staging.size() as usize;
        let queue = device.graphics_queue();

        let pending = std::mem::take(&mut self.pending);
        let pending_textures = std::mem::take(&mut self.pending_textures);

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
                    queue,
                    queue,
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
                    queue,
                    queue,
                ).with_range(p.dst.offset() as usize, p.size as usize));
            }
            encoder.buffer_barriers(&pre);

            // Dst textures: Undefined -> TransferDst
            if !pending_textures.is_empty() {
                let mut pre_img: Vec<TextureBarrier> = Vec::with_capacity(pending_textures.len());
                for p in pending_textures.iter() {
                    pre_img.push(TextureBarrier::new(
                        p.dst.texture()
                            .as_range(TextureLayout::Undefined, .., ..)
                            .unwrap(),
                        TextureState::Undefined,
                        TextureState::TransferDst,
                        PipelineStages::empty(),
                        PipelineStage::Transfer.into(),
                        queue,
                        queue,
                        true,
                    ));
                }
                encoder.texture_barriers(&pre_img);
            }

            // Copies
            for p in pending.iter() {
                let region = vk::BufferCopy::default()
                    .src_offset(p.src_offset)
                    .dst_offset(p.dst.offset() as vk::DeviceSize)
                    .size(p.size);
                encoder.copy_buffer(staging_handle, p.dst.buffer().handle(), std::slice::from_ref(&region));
            }

            for p in pending_textures.iter() {
                let aspect = p.dst.texture().aspect();
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(p.src_offset)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(aspect)
                            .mip_level(p.dst.subresource().base_mip)
                            .base_array_layer(p.dst.subresource().base_layer)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(p.dst.texture().extent());
                
                encoder.copy_buffer_to_image(
                    staging_handle,
                    p.dst.texture().handle(),
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    std::slice::from_ref(&region),
                );
            }

            // Post-copy barriers: TRANSFER_DST -> final_state
            let mut post: Vec<BufferBarrier> = Vec::with_capacity(pending.len());
            for p in pending.iter() {
                let dst_stage = match p.final_state {
                    BufferState::Vertex => PipelineStage::VertexAttributeInput.into(),
                    BufferState::Index => PipelineStage::IndexInput.into(),
                    BufferState::TransferSrc | BufferState::TransferDst => PipelineStage::Transfer.into(),
                    BufferState::HostWrite => PipelineStage::Host.into(),
                    BufferState::Uniform | BufferState::StorageRead | BufferState::StorageWrite | BufferState::Undefined => PipelineStage::AllCommands.into(),
                };
                post.push(BufferBarrier::new(
                    p.dst.buffer().as_range(..).unwrap(),
                    BufferState::TransferDst,
                    p.final_state,
                    PipelineStage::Transfer.into(),
                    dst_stage,
                    queue,
                    queue,
                ).with_range(p.dst.offset() as usize, p.size as usize));
            }
            encoder.buffer_barriers(&post);

            if !pending_textures.is_empty() {
                let mut post_img: Vec<TextureBarrier> = Vec::with_capacity(pending_textures.len());
                for p in pending_textures.iter() {
                    let dst_stage = texture_state_to_dst_stage(p.final_state);
                    post_img.push(TextureBarrier::new(
                        p.dst.texture()
                            .as_range(TextureLayout::TransferDst, .., ..)
                            .unwrap(),
                        TextureState::TransferDst,
                        p.final_state,
                        PipelineStage::Transfer.into(),
                        dst_stage,
                        queue,
                        queue,
                        false,
                    ));
                }
                encoder.texture_barriers(&post_img);
            }
        });

        if result.is_err() {
            // restore pending on failure (best-effort)
            self.pending = pending;
            self.pending_textures = pending_textures;
        }

        result?;
        self.write_head = 0;
        Ok(())
    }

    /// Convenience: enqueue then flush (blocking).
    pub fn upload_buffer(
        &mut self,
        device: &RenderDevice,
        immediate: &ImmediateCommandEncoder,
        dst: BufferRange<'a>,
        data: &[u8],
        final_state: BufferState,
    ) -> Result<(), vk::Result> {
        if self.enqueue_copy_buffer(dst, data, final_state).is_err() {
            self.flush(immediate, device)?;
            self.enqueue_copy_buffer(dst, data, final_state)?;
        }
        self.flush(immediate, device)
    }

    /// Convenience: enqueue then flush (blocking).
    pub fn upload_texture(
        &mut self,
        device: &RenderDevice,
        immediate: &ImmediateCommandEncoder,
        dst: TextureRange<'a>,
        data: &[u8],
        final_state: TextureState,
    ) -> Result<(), vk::Result> {
        if self.enqueue_upload_texture(dst, data, final_state).is_err() {
            self.flush(immediate, device)?;
            self.enqueue_upload_texture(dst, data, final_state)?;
        }
        self.flush(immediate, device)
    }
}

fn texture_state_to_dst_stage(state: TextureState) -> PipelineStages {
    match state {
        TextureState::Sampled => PipelineStage::FragmentShader.into(),
        TextureState::Color => PipelineStage::ColorAttachmentOutput.into(),
        TextureState::DepthStencil => PipelineStages::from(PipelineStage::EarlyFragmentTests) | PipelineStages::from(PipelineStage::LateFragmentTests),
        TextureState::TransferSrc | TextureState::TransferDst => PipelineStage::Transfer.into(),
        _ => PipelineStage::AllCommands.into(),
    }
}
