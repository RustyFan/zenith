//! Staging-buffer-based upload utilities.

use ash::vk;
use std::fmt;
use std::sync::Arc;
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

struct UploadBatch<'a> {
    staging: Buffer,
    pending: Vec<PendingBufferCopy<'a>>,
    pending_textures: Vec<PendingTextureCopy<'a>>,
}

#[derive(Debug)]
pub enum UploadError {
    OutOfDeviceMemory,
    ValidationFailed,
    BackendFailure,
}

impl fmt::Display for UploadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UploadError::OutOfDeviceMemory => write!(f, "upload pool ran out of device memory"),
            UploadError::ValidationFailed => write!(f, "upload pool data failed validation"),
            UploadError::BackendFailure => write!(f, "upload pool backend failure"),
        }
    }
}

impl std::error::Error for UploadError {}
/// A simple upload pool backed by a single reusable staging buffer.
///
/// - Supports queueing multiple buffer uploads and flushing them in one submit.
/// - Current behavior is **blocking**: `flush()` waits on a fence.
pub struct UploadPool<'a> {
    default_size: vk::DeviceSize,
    write_head: vk::DeviceSize,
    current: Option<UploadBatch<'a>>,
    batches: Vec<UploadBatch<'a>>,
}

impl<'a> UploadPool<'a> {
    pub const POOL_SIZE_IN_BYTES: u64 = 1024 * 16;

    pub fn new() -> Result<Self, UploadError> {
        Self::new_sized(Self::POOL_SIZE_IN_BYTES)
    }

    pub fn new_sized(pool_size: u64) -> Result<Self, UploadError> {
        Ok(Self {
            default_size: pool_size,
            write_head: 0,
            current: None,
            batches: Vec::new(),
        })
    }

    fn start_new_batch(&mut self, device: &Arc<RenderDevice>, required_size: vk::DeviceSize) -> Result<(), UploadError> {
        if let Some(batch) = self.current.take() {
            if !batch.pending.is_empty() || !batch.pending_textures.is_empty() {
                self.batches.push(batch);
            }
        }

        let new_size = self.default_size.max(required_size);
        let new_staging = Buffer::new(device, &BufferDesc::staging("upload_pool.staging", new_size))
            .map_err(Self::map_vk_error)?;
        self.current = Some(UploadBatch {
            staging: new_staging,
            pending: Vec::new(),
            pending_textures: Vec::new(),
        });
        self.write_head = 0;
        Ok(())
    }

    #[inline]
    pub fn staging_size(&self) -> vk::DeviceSize { self.default_size }

    fn current_size(&self) -> vk::DeviceSize {
        self.current
            .as_ref()
            .map(|batch| batch.staging.size())
            .unwrap_or(self.default_size)
    }

    pub fn enqueue_buffer_copy(
        &mut self,
        device: &Arc<RenderDevice>,
        dst: BufferRange<'a>,
        data: &[u8],
        final_state: BufferState,
    ) -> Result<(), UploadError> {
        let size = data.len() as vk::DeviceSize;
        if size == 0 {
            return Ok(());
        }
        let current_size = self.current_size();
        if self.current.is_none() || size > current_size || self.write_head + size > current_size {
            self.start_new_batch(device, size)?;
        }
        let current_size = self.current_size();
        if size > current_size {
            return Err(UploadError::OutOfDeviceMemory);
        }

        let src_offset = self.write_head;
        // Write into staging (exact range).
        let batch = self.current.as_mut().expect("UploadPool current batch missing");
        batch.staging
            .as_range((src_offset as u64)..((src_offset + size) as u64))
            .write(data)
            .map_err(Self::map_vk_error)?;
        self.write_head += size;

        batch.pending.push(PendingBufferCopy {
            dst,
            src_offset,
            size,
            final_state,
        });

        Ok(())
    }

    pub fn enqueue_texture_upload(
        &mut self,
        device: &Arc<RenderDevice>,
        dst: TextureRange<'a>,
        data: &[u8],
        final_state: TextureState,
    ) -> Result<(), UploadError> {
        let (_, _, block_size) = Self::texture_format_block_info(dst.texture().format());
        let aligned_head = Self::align_up(self.write_head, block_size);
        if aligned_head != self.write_head {
            self.write_head = aligned_head;
        }

        let size = data.len() as vk::DeviceSize;
        if size == 0 {
            return Ok(());
        }
        if block_size > 1 && size % block_size != 0 {
            return Err(UploadError::ValidationFailed);
        }
        let expected_size = Self::texture_upload_size_bytes(&dst);
        if size != expected_size {
            return Err(UploadError::ValidationFailed);
        }

        let current_size = self.current_size();
        if self.current.is_none() || size > current_size || self.write_head + size > current_size {
            self.start_new_batch(device, size)?;
            self.write_head = Self::align_up(self.write_head, block_size);
        }

        let current_size = self.current_size();
        if size > current_size {
            return Err(UploadError::OutOfDeviceMemory);
        }

        let src_offset = self.write_head;
        let batch = self.current.as_mut().expect("UploadPool current batch missing");
        batch.staging
            .as_range((src_offset as u64)..((src_offset + size) as u64))
            .write(data)
            .map_err(Self::map_vk_error)?;
        self.write_head += size;

        batch.pending_textures.push(PendingTextureCopy {
            dst,
            src_offset,
            final_state,
        });

        Ok(())
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        let current_empty = self.current.as_ref()
            .map(|batch| batch.pending.is_empty() && batch.pending_textures.is_empty())
            .unwrap_or(true);
        current_empty && self.batches.is_empty()
    }

    /// Flush all pending uploads using an immediate submit, blocking until completion.
    pub fn flush(&mut self, immediate: &ImmediateCommandEncoder, device: &RenderDevice) -> Result<(), UploadError> {
        if self.is_empty() {
            self.write_head = 0;
            return Ok(());
        }
        
        let queue = device.graphics_queue();
        
        let submit_batch = |staging: &Buffer, pending_buffers: Vec<PendingBufferCopy<'a>>, pending_textures: Vec<PendingTextureCopy<'a>>| -> Result<(), UploadError> {
            let staging_size = staging.size() as usize;
            let result = immediate.submit_and_wait(|encoder| {
                let mut pre: Vec<BufferBarrier> = Vec::with_capacity(1 + pending_buffers.len());
                // Staging: HOST_WRITE -> TRANSFER_READ (as TRANSFER_SRC)
                pre.push(
                    BufferBarrier::new(
                        staging.as_range(..),
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
                for p in pending_buffers.iter() {
                    pre.push(BufferBarrier::new(
                        p.dst.buffer().as_range(..),
                        BufferState::Undefined,
                        BufferState::TransferDst,
                        PipelineStages::empty(),
                        PipelineStage::Transfer.into(),
                        queue,
                        queue,
                    ).with_range(p.dst.offset() as usize, p.size as usize));
                }
                encoder.pipeline_barriers(&[], &[], &pre);

                // Dst textures: Undefined -> TransferDst
                if !pending_textures.is_empty() {
                    let mut pre_img: Vec<TextureBarrier> = Vec::with_capacity(pending_textures.len());
                    for p in pending_textures.iter() {
                        pre_img.push(TextureBarrier::new(
                            p.dst.texture().as_range(TextureLayout::Undefined, .., ..),
                            TextureState::Undefined,
                            TextureState::TransferDst,
                            PipelineStages::empty(),
                            PipelineStage::Transfer.into(),
                            queue,
                            queue,
                            true,
                        ));
                    }
                    encoder.pipeline_barriers(&[], &pre_img, &[]);
                }

                // Copies
                for p in pending_buffers.iter() {
                    let region = vk::BufferCopy::default()
                        .src_offset(p.src_offset)
                        .dst_offset(p.dst.offset() as vk::DeviceSize)
                        .size(p.size);
                    encoder.copy_buffer(staging, p.dst.buffer(), std::slice::from_ref(&region));
                }

                for p in pending_textures.iter() {
                    let aspect = p.dst.texture().aspect();
                    let subresource = p.dst.subresource();
                    let format = p.dst.texture().format();
                    let base_extent = p.dst.texture().extent();

                    let mut regions: Vec<vk::BufferImageCopy> = Vec::with_capacity(subresource.num_mips as usize);
                    for mip_idx in 0..subresource.num_mips {
                        let mip_level = subresource.base_mip + mip_idx;
                        let mip_extent = Self::texture_mip_extent(base_extent, mip_level);
                        let mip_offset = p.src_offset + Self::texture_mip_block_offset_bytes(
                            base_extent,
                            format,
                            subresource.base_mip,
                            subresource.num_layers,
                            mip_idx,
                        );

                        regions.push(
                            vk::BufferImageCopy::default()
                                .buffer_offset(mip_offset)
                                .buffer_row_length(0)
                                .buffer_image_height(0)
                                .image_subresource(
                                    vk::ImageSubresourceLayers::default()
                                        .aspect_mask(aspect)
                                        .mip_level(mip_level)
                                        .base_array_layer(subresource.base_layer)
                                        .layer_count(subresource.num_layers),
                                )
                                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                                .image_extent(mip_extent),
                        );
                    }

                    encoder.copy_buffer_to_image(
                        staging,
                        p.dst.texture(),
                        TextureLayout::TransferDst,
                        &regions,
                    );
                }

                // Post-copy barriers: TRANSFER_DST -> final_state
                let mut post: Vec<BufferBarrier> = Vec::with_capacity(pending_buffers.len());
                for p in pending_buffers.iter() {
                    let dst_stage = match p.final_state {
                        BufferState::Vertex => PipelineStage::VertexAttributeInput.into(),
                        BufferState::Index => PipelineStage::IndexInput.into(),
                        BufferState::TransferSrc | BufferState::TransferDst => PipelineStage::Transfer.into(),
                        BufferState::HostWrite => PipelineStage::Host.into(),
                        BufferState::Uniform | BufferState::StorageRead | BufferState::StorageWrite | BufferState::Undefined => PipelineStage::AllCommands.into(),
                    };
                    post.push(BufferBarrier::new(
                        p.dst.buffer().as_range(..),
                        BufferState::TransferDst,
                        p.final_state,
                        PipelineStage::Transfer.into(),
                        dst_stage,
                        queue,
                        queue,
                    ).with_range(p.dst.offset() as usize, p.size as usize));
                }
                encoder.pipeline_barriers(&[], &[], &post);

                if !pending_textures.is_empty() {
                    let mut post_img: Vec<TextureBarrier> = Vec::with_capacity(pending_textures.len());
                    for p in pending_textures.iter() {
                        let dst_stage = texture_state_to_dst_stage(p.final_state);
                        post_img.push(TextureBarrier::new(
                            p.dst.texture().as_range(TextureLayout::TransferDst, .., ..),
                            TextureState::TransferDst,
                            p.final_state,
                            PipelineStage::Transfer.into(),
                            dst_stage,
                            queue,
                            queue,
                            false,
                        ));
                    }
                    encoder.pipeline_barriers(&[], &post_img, &[]);
                }
            });
            result.map_err(Self::map_vk_error)?;
            Ok(())
        };

        let mut batches = std::mem::take(&mut self.batches);
        if let Some(current) = self.current.take() {
            if !current.pending.is_empty() || !current.pending_textures.is_empty() {
                batches.push(current);
            }
        }

        for batch in batches.into_iter() {
            submit_batch(&batch.staging, batch.pending, batch.pending_textures)?;
        }

        self.write_head = 0;
        Ok(())
    }

    fn map_vk_error(result: vk::Result) -> UploadError {
        match result {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => UploadError::OutOfDeviceMemory,
            vk::Result::ERROR_VALIDATION_FAILED_EXT => UploadError::ValidationFailed,
            _ => UploadError::BackendFailure,
        }
    }
    fn align_up(value: vk::DeviceSize, alignment: vk::DeviceSize) -> vk::DeviceSize {
        if alignment <= 1 {
            return value;
        }
        (value + alignment - 1) / alignment * alignment
    }

    fn texture_upload_size_bytes(dst: &TextureRange<'_>) -> vk::DeviceSize {
        let sub = dst.subresource();
        Self::texture_layer_size_bytes(
            dst.texture().extent(),
            dst.texture().format(),
            sub.base_mip,
            sub.num_mips,
        ) * sub.num_layers as vk::DeviceSize
    }

    fn texture_mip_extent(base_extent: vk::Extent3D, mip_level: u32) -> vk::Extent3D {
        let shift = mip_level.min(31);
        vk::Extent3D {
            width: (base_extent.width >> shift).max(1),
            height: (base_extent.height >> shift).max(1),
            depth: (base_extent.depth >> shift).max(1),
        }
    }

    fn texture_mip_size_bytes(extent: vk::Extent3D, format: vk::Format) -> vk::DeviceSize {
        let (block_w, block_h, bytes_per_block) = Self::texture_format_block_info(format);
        let blocks_x = (extent.width + block_w - 1) / block_w;
        let blocks_y = (extent.height + block_h - 1) / block_h;
        (blocks_x as vk::DeviceSize)
            * (blocks_y as vk::DeviceSize)
            * (extent.depth as vk::DeviceSize)
            * bytes_per_block
    }

    fn texture_format_block_info(format: vk::Format) -> (u32, u32, vk::DeviceSize) {
        match format {
            vk::Format::BC5_UNORM_BLOCK
            | vk::Format::BC6H_UFLOAT_BLOCK
            | vk::Format::BC6H_SFLOAT_BLOCK
            | vk::Format::BC7_UNORM_BLOCK
            | vk::Format::BC7_SRGB_BLOCK => (4u32, 4u32, 16),
            _ => (1u32, 1u32, Self::bytes_per_pixel(format) as vk::DeviceSize),
        }
    }

    fn texture_layer_size_bytes(
        base_extent: vk::Extent3D,
        format: vk::Format,
        base_mip: u32,
        num_mips: u32,
    ) -> vk::DeviceSize {
        let mut total = 0;
        for mip in 0..num_mips {
            let mip_extent = Self::texture_mip_extent(base_extent, base_mip + mip);
            total += Self::texture_mip_size_bytes(mip_extent, format);
        }
        total
    }

    fn texture_mip_block_offset_bytes(
        base_extent: vk::Extent3D,
        format: vk::Format,
        base_mip: u32,
        num_layers: u32,
        mip_idx: u32,
    ) -> vk::DeviceSize {
        let mut offset = 0;
        for m in 0..mip_idx {
            let mip_extent = Self::texture_mip_extent(base_extent, base_mip + m);
            offset += Self::texture_mip_size_bytes(mip_extent, format) * num_layers as vk::DeviceSize;
        }
        offset
    }

    fn bytes_per_pixel(format: vk::Format) -> u32 {
        use vk::Format;
        match format {
            Format::R8_UNORM | Format::R8_SNORM => 1,
            Format::R8G8_UNORM | Format::R8G8_SNORM | Format::R16_UNORM | Format::R16_SFLOAT => 2,
            Format::R8G8B8A8_UNORM
            | Format::R8G8B8A8_SRGB
            | Format::R32_SFLOAT
            | Format::R16G16_UNORM
            | Format::R16G16_SFLOAT => 4,
            Format::R16G16B16A16_UNORM
            | Format::R16G16B16A16_SFLOAT
            | Format::R32G32_SFLOAT => 8,
            Format::R32G32B32A32_SFLOAT => 16,
            _ => 4, // fallback
        }
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
