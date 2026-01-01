//! Zenith RHI (Render Hardware Interface) - Pure Vulkan backend.
//!
//! This crate provides low-level Vulkan abstractions for the Zenith engine.

pub mod buffer;
pub mod command;
pub mod core;
pub mod descriptor;
pub mod device;
pub mod pipeline;
pub mod pipeline_cache;
pub mod sampler;
pub mod shader;
pub mod swapchain;
pub mod texture;
mod defer_release;
mod barrier;
mod synchronization;

pub use ash::{vk, Device};
pub use buffer::{find_memory_type, Buffer, BufferDesc};
pub use command::{CommandPool, CommandRecorder};
pub use core::RhiCore;
pub use descriptor::{
    BindingError, DescriptorPool, DescriptorSetLayout, LayoutBinding, ResourceBinder,
    ShaderBindingError, ShaderResourceBinder, create_layouts_from_reflection,
};
pub use device::RenderDevice;
pub use pipeline::{GraphicPipeline, GraphicPipelineInfo};
pub use pipeline_cache::{
    ColorBlendAttachment, GraphicPipelineKey, PipelineCache, StencilOpKey, VertexAttribute,
    VertexBinding,
};
pub use sampler::{Sampler, SamplerConfig};
pub use shader::{
    compile_hlsl, reflect_spirv, Shader, ShaderBinding, ShaderError, ShaderReflection, ShaderStage,
};
pub use swapchain::{FrameSync, SwapchainConfig, Swapchain};
pub use texture::{Texture, TextureDesc};
pub use barrier::{BufferState, buffer_barrier, TextureState, texture_barrier};
pub use synchronization::Semaphore;