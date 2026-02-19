//! Zenith RHI (Render Hardware Interface) - Pure Vulkan backend.
//!
//! This crate provides low-level Vulkan abstractions for the Zenith engine.

pub mod buffer;
pub mod command;
pub mod core;
pub mod descriptor;
pub mod device;
pub mod pipeline;
pub mod pipeline_registry;
pub mod resource_cache;
pub mod sampler;
pub mod shader;
mod slang_compiler;
pub mod swapchain;
pub mod texture;
pub mod upload;
pub mod queue;
mod defer_release;
mod barrier;
mod synchronization;
mod utility;
pub mod bindless;

pub(crate) use paste::paste;

pub(crate) const NUM_BACK_BUFFERS: u32 = 3;

pub use memoffset;
pub use zenith_rhi_derive::VertexLayout;

pub use ash::{vk, Device};
pub use buffer::{Buffer, BufferDesc};
pub use command::{CommandPool, CommandEncoder, ImmediateCommandEncoder};
pub use core::RhiCore;
pub use queue::Queue;
pub use descriptor::{
    DescriptorPool, DescriptorSetLayout, LayoutBinding,
    DescriptorBindingError, DescriptorSetBinder,
};
pub use device::RenderDevice;
pub use pipeline::{
    ColorAttachmentDesc, ColorAttachmentDescBuilder, ColorAttachmentDescBuilderError,
    ComputePipeline, ComputePipelineDesc,
    DepthStencilState, DepthStencilStateBuilder, DepthStencilStateBuilderError,
    DepthStencilAttachmentDesc, DepthStencilAttachmentDescBuilder, DepthStencilAttachmentDescBuilderError,
    GraphicPipeline, GraphicPipelineDesc, GraphicPipelineState, GraphicPipelineStateBuilder,
    GraphicShaderInput, GraphicShaderInputBuilder, GraphicShaderInputBuildError,
    GraphicPipelineAttachments,
    InputAssemblyState, RasterizationState, MultisampleState,
    VertexAttribute, VertexBinding, VertexLayout,
};
pub use pipeline_registry::{PipelineRegistry, PipelineCacheStats, PipelineEntry, PipelineHandle};
pub use resource_cache::TransientResourceCache;
pub use sampler::{Sampler, SamplerDesc};
pub use shader::{
    Shader, ShaderBinding, ShaderError, ShaderReflection, ShaderStage,
};
pub use swapchain::{FrameSync, SwapchainConfig, Swapchain};
pub use texture::{Texture, TextureDesc};
pub use barrier::{
    BufferState, TextureState,
    global_memory_barrier, flush_all_memory_writes,
    PipelineStage, PipelineStages, TextureLayout,
    BufferBarrier, TextureBarrier, MemoryBarrier,
};
pub use synchronization::{Semaphore, Fence};
pub use upload::UploadPool;

pub use defer_release::{DeferRelease, LastFreedStats};

pub use bindless::{
    BindlessCaps, BindlessPool, BindlessHandle, ResourceType,
};