use ash::vk;
use crate::{Buffer, Texture};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferState {
    Undefined,
    TransferSrc,
    TransferDst,
    Uniform,
    Storage,
    Vertex,
    Index,
}

impl BufferState {
    pub fn into_pipeline_stage(self, shader_used_stage: vk::PipelineStageFlags2) -> vk::PipelineStageFlags2 {
        match self {
            BufferState::Undefined => vk::PipelineStageFlags2::NONE,
            BufferState::TransferSrc |
            BufferState::TransferDst => vk::PipelineStageFlags2::TRANSFER,
            BufferState::Uniform => shader_used_stage,
            BufferState::Storage => shader_used_stage,
            BufferState::Vertex => vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
            BufferState::Index => vk::PipelineStageFlags2::INDEX_INPUT,
        }
    }

    pub fn into_access_flag(self, is_readonly: bool) -> vk::AccessFlags2 {
        match self {
            BufferState::Undefined => vk::AccessFlags2::NONE,
            BufferState::TransferSrc => vk::AccessFlags2::TRANSFER_READ,
            BufferState::TransferDst => vk::AccessFlags2::TRANSFER_WRITE,
            BufferState::Uniform => vk::AccessFlags2::UNIFORM_READ,
            // TODO: Is it better to identify the write-only case?
            BufferState::Storage => if is_readonly { vk::AccessFlags2::SHADER_STORAGE_READ } else { vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE },
            BufferState::Vertex => vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
            BufferState::Index => vk::AccessFlags2::INDEX_READ,
        }
    }
}

pub fn buffer_barrier<'a>(
    buffer: &Buffer,
    src_shader_used_stage: vk::PipelineStageFlags2,
    src_state: BufferState,
    dst_shader_used_stage: vk::PipelineStageFlags2,
    dst_state: BufferState,
    src_queue_family: u32,
    dst_queue_family: u32,
    is_readonly: bool,
) -> vk::BufferMemoryBarrier2<'a> {
    let src_stage = src_state.into_pipeline_stage(src_shader_used_stage);
    let src_access = src_state.into_access_flag(is_readonly);
    let dst_stage = dst_state.into_pipeline_stage(dst_shader_used_stage);
    let dst_access = dst_state.into_access_flag(is_readonly);

    // TODO: The src and dst pipeline stage comes from shader reflection and previous usage.
    vk::BufferMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .src_queue_family_index(src_queue_family)
        .dst_queue_family_index(dst_queue_family)
        .buffer(buffer.handle())
        // TODO: support subresource
        .size(buffer.size())
        .offset(0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureState {
    Undefined,
    TransferSrc,
    TransferDst,
    Sampled,
    Storage,
    General,
    Color,
    DepthStencil,
    Present,
}

impl TextureState {
    pub fn into_pipeline_stage(self, shader_used_stage: vk::PipelineStageFlags2) -> vk::PipelineStageFlags2 {
        match self {
            TextureState::Undefined => vk::PipelineStageFlags2::NONE,
            TextureState::TransferSrc |
            TextureState::TransferDst => vk::PipelineStageFlags2::TRANSFER,
            TextureState::Sampled => shader_used_stage,
            TextureState::Storage => shader_used_stage,
            TextureState::General => shader_used_stage,
            TextureState::Color => vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            TextureState::DepthStencil => vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            TextureState::Present => vk::PipelineStageFlags2::NONE,
        }
    }

    pub fn into_access_flag(self, is_readonly: bool) -> vk::AccessFlags2 {
        match self {
            TextureState::Undefined => vk::AccessFlags2::NONE,
            TextureState::TransferSrc => vk::AccessFlags2::TRANSFER_READ,
            TextureState::TransferDst => vk::AccessFlags2::TRANSFER_WRITE,
            TextureState::Sampled => vk::AccessFlags2::SHADER_SAMPLED_READ,
            // TODO: Is it better to identify the write-only case?
            TextureState::Storage => if is_readonly { vk::AccessFlags2::SHADER_STORAGE_READ } else { vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE },
            TextureState::General => if is_readonly { vk::AccessFlags2::MEMORY_READ } else { vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE },
            TextureState::Color => if is_readonly { vk::AccessFlags2::COLOR_ATTACHMENT_READ } else { vk::AccessFlags2::COLOR_ATTACHMENT_WRITE }
            TextureState::DepthStencil => if is_readonly { vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ } else { vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE }
            TextureState::Present => vk::AccessFlags2::NONE
        }
    }

    pub fn into_image_layout(self) -> vk::ImageLayout {
        match self {
            TextureState::Undefined => vk::ImageLayout::UNDEFINED,
            TextureState::TransferSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            TextureState::TransferDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            TextureState::Sampled => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            TextureState::Storage => vk::ImageLayout::GENERAL,
            TextureState::General => vk::ImageLayout::GENERAL,
            TextureState::Color => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            TextureState::DepthStencil => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            TextureState::Present => vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }
}

pub fn texture_barrier<'a>(
    texture: &Texture,
    src_shader_used_stage: vk::PipelineStageFlags2,
    src_state: TextureState,
    dst_shader_used_stage: vk::PipelineStageFlags2,
    dst_state: TextureState,
    src_queue_family: u32,
    dst_queue_family: u32,
    is_readonly: bool,
    is_discard: bool
) -> vk::ImageMemoryBarrier2<'a> {
    let src_stage = src_state.into_pipeline_stage(src_shader_used_stage);
    let src_access = src_state.into_access_flag(is_readonly);
    let dst_stage = dst_state.into_pipeline_stage(dst_shader_used_stage);
    let dst_access = dst_state.into_access_flag(is_readonly);
    let mut old_layout = src_state.into_image_layout();
    let new_layout = dst_state.into_image_layout();

    if is_discard {
        old_layout = vk::ImageLayout::UNDEFINED;
    }

    // TODO: The src and dst pipeline stage comes from shader reflection and previous usage.
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .src_queue_family_index(src_queue_family)
        .dst_queue_family_index(dst_queue_family)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(texture.handle())
        // TODO: support subresource
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: texture.aspect(),
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        })
}