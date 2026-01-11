use ash::vk;
use crate::queue::Queue;
use enumflags2::BitFlags;
use crate::buffer::BufferRange;
use crate::texture::TextureRange;

/// A global memory barrier (sync2) that does not target a specific buffer/image.
///
/// This is useful as a conservative "make all writes visible" fence within a command buffer.
pub fn global_memory_barrier<'a>(
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) -> vk::MemoryBarrier2<'a> {
    vk::MemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
}

/// Flush all memory writes so they are visible to subsequent GPU operations.
pub fn flush_all_memory_writes<'a>() -> vk::MemoryBarrier2<'a> {
    global_memory_barrier(
        vk::PipelineStageFlags2::ALL_COMMANDS,
        vk::AccessFlags2::MEMORY_WRITE,
        vk::PipelineStageFlags2::ALL_COMMANDS,
        vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
    )
}

#[enumflags2::bitflags]
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineStage {
    Host = 1 << 0,
    Transfer = 1 << 1,
    VertexAttributeInput = 1 << 2,
    IndexInput = 1 << 3,
    VertexShader = 1 << 4,
    FragmentShader = 1 << 5,
    ComputeShader = 1 << 6,
    GeometryShader = 1 << 7,
    TessellationControlShader = 1 << 8,
    TessellationEvaluationShader = 1 << 9,
    ColorAttachmentOutput = 1 << 10,
    EarlyFragmentTests = 1 << 11,
    LateFragmentTests = 1 << 12,
    BottomOfPipe = 1 << 13,
    AllCommands = 1 << 14,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PipelineStages(BitFlags<PipelineStage>);

impl PipelineStage {
    pub fn to_vk(self) -> vk::PipelineStageFlags2 {
        match self {
            PipelineStage::Host => vk::PipelineStageFlags2::HOST,
            PipelineStage::Transfer => vk::PipelineStageFlags2::TRANSFER,
            PipelineStage::VertexAttributeInput => vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
            PipelineStage::IndexInput => vk::PipelineStageFlags2::INDEX_INPUT,
            PipelineStage::VertexShader => vk::PipelineStageFlags2::VERTEX_SHADER,
            PipelineStage::FragmentShader => vk::PipelineStageFlags2::FRAGMENT_SHADER,
            PipelineStage::ComputeShader => vk::PipelineStageFlags2::COMPUTE_SHADER,
            PipelineStage::GeometryShader => vk::PipelineStageFlags2::GEOMETRY_SHADER,
            PipelineStage::TessellationControlShader => vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
            PipelineStage::TessellationEvaluationShader => vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
            PipelineStage::ColorAttachmentOutput => vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            PipelineStage::EarlyFragmentTests => vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
            PipelineStage::LateFragmentTests => vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            PipelineStage::BottomOfPipe => vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            PipelineStage::AllCommands => vk::PipelineStageFlags2::ALL_COMMANDS,
        }
    }
}

impl PipelineStages {
    pub fn empty() -> Self {
        Self(BitFlags::empty())
    }

    pub fn insert(&mut self, stage: PipelineStage) {
        self.0.insert(stage);
    }

    pub fn to_vk(self) -> vk::PipelineStageFlags2 {
        self.0.iter()
            .fold(vk::PipelineStageFlags2::empty(), |acc, s| acc | s.to_vk())
    }

    pub fn from_vk(flags: vk::PipelineStageFlags2) -> Self {
        if flags == vk::PipelineStageFlags2::NONE {
            return PipelineStages::empty();
        }
        // Keep this list in sync with PipelineStage::to_vk()
        const ALL: [PipelineStage; 15] = [
            PipelineStage::Host,
            PipelineStage::Transfer,
            PipelineStage::VertexAttributeInput,
            PipelineStage::IndexInput,
            PipelineStage::VertexShader,
            PipelineStage::FragmentShader,
            PipelineStage::ComputeShader,
            PipelineStage::GeometryShader,
            PipelineStage::TessellationControlShader,
            PipelineStage::TessellationEvaluationShader,
            PipelineStage::ColorAttachmentOutput,
            PipelineStage::EarlyFragmentTests,
            PipelineStage::LateFragmentTests,
            PipelineStage::BottomOfPipe,
            PipelineStage::AllCommands,
        ];
        let mut out = PipelineStages::empty();
        for s in ALL {
            if flags.contains(s.to_vk()) {
                out.insert(s);
            }
        }
        out
    }
}

impl From<PipelineStage> for PipelineStages {
    fn from(value: PipelineStage) -> Self {
        Self(BitFlags::from_flag(value))
    }
}

impl core::ops::BitOr for PipelineStages {
    type Output = PipelineStages;
    fn bitor(self, rhs: PipelineStages) -> Self::Output {
        PipelineStages(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for PipelineStages {
    fn bitor_assign(&mut self, rhs: PipelineStages) {
        self.0 |= rhs.0;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureLayout {
    Undefined,
    General,
    TransferSrc,
    TransferDst,
    ShaderReadOnly,
    Color,
    DepthStencil,
    Present,
}

impl TextureLayout {
    pub fn to_vk(self) -> vk::ImageLayout {
        match self {
            TextureLayout::Undefined => vk::ImageLayout::UNDEFINED,
            TextureLayout::General => vk::ImageLayout::GENERAL,
            TextureLayout::TransferSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            TextureLayout::TransferDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            TextureLayout::ShaderReadOnly => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            TextureLayout::Color => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            TextureLayout::DepthStencil => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            TextureLayout::Present => vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferState {
    Undefined,
    HostWrite,
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
            BufferState::HostWrite => vk::PipelineStageFlags2::HOST,
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
            BufferState::HostWrite => vk::AccessFlags2::HOST_WRITE,
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

pub struct BufferBarrier<'a> {
    pub buffer: BufferRange<'a>,
    pub src_state: BufferState,
    pub dst_state: BufferState,
    pub src_stage: PipelineStages,
    pub dst_stage: PipelineStages,
    pub src_queue: Queue,
    pub dst_queue: Queue,
    pub readonly: bool,
    pub offset: usize,
    pub size: usize,
}

impl<'a> BufferBarrier<'a> {
    pub fn new(
        buffer: BufferRange<'a>,
        src_state: BufferState,
        dst_state: BufferState,
        src_stage: PipelineStages,
        dst_stage: PipelineStages,
        src_queue: Queue,
        dst_queue: Queue,
        readonly: bool,
    ) -> Self {
        Self {
            buffer,
            src_state,
            dst_state,
            src_stage,
            dst_stage,
            src_queue,
            dst_queue,
            readonly,
            offset: 0,
            size: buffer.buffer().size() as usize,
        }
    }

    pub fn with_range(mut self, offset: usize, size: usize) -> Self {
        self.offset = offset;
        self.size = size;
        self
    }

    pub fn to_vk(&self) -> vk::BufferMemoryBarrier2<'a> {
        // Ensure stage mask matches access mask expectations derived from BufferState.
        // This mirrors the old behavior of `buffer_barrier()` which derives stage from state.
        let src_stage_vk = self.src_state.into_pipeline_stage(self.src_stage.to_vk());
        let dst_stage_vk = self.dst_state.into_pipeline_stage(self.dst_stage.to_vk());
        vk::BufferMemoryBarrier2::default()
            .src_stage_mask(src_stage_vk)
            .src_access_mask(self.src_state.into_access_flag(self.readonly))
            .dst_stage_mask(dst_stage_vk)
            .dst_access_mask(self.dst_state.into_access_flag(self.readonly))
            .src_queue_family_index(self.src_queue.family_index())
            .dst_queue_family_index(self.dst_queue.family_index())
            .buffer(self.buffer.buffer().handle())
            .offset(self.offset as vk::DeviceSize)
            .size(self.size as vk::DeviceSize)
    }
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

impl From<TextureState> for TextureLayout {
    fn from(value: TextureState) -> Self {
        match value {
            TextureState::Undefined => TextureLayout::Undefined,
            TextureState::TransferSrc => TextureLayout::TransferSrc,
            TextureState::TransferDst => TextureLayout::TransferDst,
            TextureState::Sampled => TextureLayout::ShaderReadOnly,
            TextureState::Storage => TextureLayout::General,
            TextureState::General => TextureLayout::General,
            TextureState::Color => TextureLayout::Color,
            TextureState::DepthStencil => TextureLayout::DepthStencil,
            TextureState::Present => TextureLayout::Present,
        }
    }
}

pub struct TextureBarrier<'a> {
    pub texture: TextureRange<'a>,
    pub src_state: TextureState,
    pub dst_state: TextureState,
    pub src_stage: PipelineStages,
    pub dst_stage: PipelineStages,
    pub src_queue: Queue,
    pub dst_queue: Queue,
    pub readonly: bool,
    pub discard: bool,
    pub old_layout: TextureLayout,
    pub new_layout: TextureLayout,
}

impl<'a> TextureBarrier<'a> {
    pub fn new(
        texture: TextureRange<'a>,
        src_state: TextureState,
        dst_state: TextureState,
        src_stage: PipelineStages,
        dst_stage: PipelineStages,
        src_queue: Queue,
        dst_queue: Queue,
        readonly: bool,
        discard: bool,
    ) -> Self {
        let old_layout = TextureLayout::from(src_state);
        let new_layout = TextureLayout::from(dst_state);
        Self {
            texture,
            src_state,
            dst_state,
            src_stage,
            dst_stage,
            src_queue,
            dst_queue,
            readonly,
            discard,
            old_layout,
            new_layout,
        }
    }

    pub fn with_layouts(mut self, old_layout: TextureLayout, new_layout: TextureLayout) -> Self {
        self.old_layout = old_layout;
        self.new_layout = new_layout;
        self
    }

    pub fn to_vk(&self) -> vk::ImageMemoryBarrier2<'a> {
        let mut old_layout = self.old_layout.to_vk();
        if self.discard {
            old_layout = vk::ImageLayout::UNDEFINED;
        }
        // Ensure stage mask matches access mask expectations derived from TextureState.
        // This mirrors the old behavior of `texture_barrier()` which derives stage from state.
        let src_stage_vk = self.src_state.into_pipeline_stage(self.src_stage.to_vk());
        let dst_stage_vk = self.dst_state.into_pipeline_stage(self.dst_stage.to_vk());

        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage_vk)
            .src_access_mask(self.src_state.into_access_flag(self.readonly))
            .dst_stage_mask(dst_stage_vk)
            .dst_access_mask(self.dst_state.into_access_flag(self.readonly))
            .src_queue_family_index(self.src_queue.family_index())
            .dst_queue_family_index(self.dst_queue.family_index())
            .old_layout(old_layout)
            .new_layout(self.new_layout.to_vk())
            .image(self.texture.texture().handle())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: self.texture.texture().aspect(),
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MemoryBarrier {
    pub src_stage: PipelineStages,
    pub src_access: vk::AccessFlags2,
    pub dst_stage: PipelineStages,
    pub dst_access: vk::AccessFlags2,
}

impl MemoryBarrier {
    pub fn new(
        src_stage: PipelineStages,
        src_access: vk::AccessFlags2,
        dst_stage: PipelineStages,
        dst_access: vk::AccessFlags2,
    ) -> Self {
        Self { src_stage, src_access, dst_stage, dst_access }
    }

    pub fn flush_all_writes() -> Self {
        Self::new(
            PipelineStage::AllCommands.into(),
            vk::AccessFlags2::MEMORY_WRITE,
            PipelineStage::AllCommands.into(),
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        )
    }

    pub fn to_vk(&self) -> vk::MemoryBarrier2<'static> {
        vk::MemoryBarrier2::default()
            .src_stage_mask(self.src_stage.to_vk())
            .src_access_mask(self.src_access)
            .dst_stage_mask(self.dst_stage.to_vk())
            .dst_access_mask(self.dst_access)
    }
}