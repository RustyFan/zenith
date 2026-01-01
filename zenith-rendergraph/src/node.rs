use std::sync::Arc;
use derive_builder::Builder;
use zenith_rhi::{vk, Shader, ShaderReflection, DescriptorSetLayout};
use crate::graph::{GraphicNodeExecutionContext, LambdaNodeExecutionContext};
use crate::builder::ResourceAccessStorage;
use crate::interface::Texture;
use crate::resource::{RenderGraphResourceAccess, Rt};

#[derive(Clone, Debug, Builder)]
#[builder(setter(into))]
pub struct ColorInfo {
    #[builder(default)]
    pub blend_enable: bool,
    #[builder(default = "vk::BlendFactor::ONE")]
    pub src_color_blend: vk::BlendFactor,
    #[builder(default = "vk::BlendFactor::ZERO")]
    pub dst_color_blend: vk::BlendFactor,
    #[builder(default = "vk::BlendOp::ADD")]
    pub color_blend_op: vk::BlendOp,
    #[builder(default = "vk::BlendFactor::ONE")]
    pub src_alpha_blend: vk::BlendFactor,
    #[builder(default = "vk::BlendFactor::ZERO")]
    pub dst_alpha_blend: vk::BlendFactor,
    #[builder(default = "vk::BlendOp::ADD")]
    pub alpha_blend_op: vk::BlendOp,
    #[builder(default = "vk::ColorComponentFlags::RGBA")]
    pub write_mask: vk::ColorComponentFlags,
    #[builder(default = "vk::AttachmentLoadOp::CLEAR")]
    pub load_op: vk::AttachmentLoadOp,
    #[builder(default = "vk::AttachmentStoreOp::STORE")]
    pub store_op: vk::AttachmentStoreOp,
    #[builder(default)]
    pub clear_value: [f32; 4],
}

impl Default for ColorInfo {
    fn default() -> Self {
        Self {
            blend_enable: false,
            src_color_blend: vk::BlendFactor::ONE,
            dst_color_blend: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend: vk::BlendFactor::ONE,
            dst_alpha_blend: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            write_mask: vk::ColorComponentFlags::RGBA,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

#[derive(Clone, Debug, Builder)]
#[builder(setter(into))]
pub struct DepthStencilInfo {
    #[builder(default)]
    pub depth_test_enable: bool,
    #[builder(default)]
    pub depth_write_enable: bool,
    #[builder(default = "vk::CompareOp::LESS")]
    pub depth_compare_op: vk::CompareOp,
    #[builder(default = "vk::AttachmentLoadOp::CLEAR")]
    pub depth_load_op: vk::AttachmentLoadOp,
    #[builder(default = "vk::AttachmentStoreOp::STORE")]
    pub depth_store_op: vk::AttachmentStoreOp,
    #[builder(default = "1.0")]
    pub depth_clear_value: f32,
    #[builder(default)]
    pub stencil_test_enable: bool,
    #[builder(default)]
    pub stencil_front: vk::StencilOpState,
    #[builder(default)]
    pub stencil_back: vk::StencilOpState,
    #[builder(default = "vk::AttachmentLoadOp::DONT_CARE")]
    pub stencil_load_op: vk::AttachmentLoadOp,
    #[builder(default = "vk::AttachmentStoreOp::DONT_CARE")]
    pub stencil_store_op: vk::AttachmentStoreOp,
    #[builder(default)]
    pub stencil_clear_value: u32,
}

impl Default for DepthStencilInfo {
    fn default() -> Self {
        Self {
            depth_test_enable: true,
            depth_write_enable: true,
            depth_compare_op: vk::CompareOp::LESS,
            depth_load_op: vk::AttachmentLoadOp::CLEAR,
            depth_store_op: vk::AttachmentStoreOp::STORE,
            depth_clear_value: 1.0,
            stencil_test_enable: false,
            stencil_front: vk::StencilOpState::default(),
            stencil_back: vk::StencilOpState::default(),
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_clear_value: 0,
        }
    }
}

/// Vertex binding description for pipeline creation.
#[derive(Clone, Debug)]
pub struct VertexBindingDesc {
    pub binding: u32,
    pub stride: u32,
    pub input_rate: vk::VertexInputRate,
}

/// Vertex attribute description for pipeline creation.
#[derive(Clone, Debug)]
pub struct VertexAttributeDesc {
    pub location: u32,
    pub binding: u32,
    pub format: vk::Format,
    pub offset: u32,
}

#[derive(Default)]
pub struct GraphicPipelineDescriptor {
    pub(crate) vertex_shader: Option<Arc<Shader>>,
    pub(crate) fragment_shader: Option<Arc<Shader>>,
    pub(crate) color_attachments: Vec<(RenderGraphResourceAccess<Texture, Rt>, ColorInfo)>,
    pub(crate) depth_stencil_attachment: Option<(RenderGraphResourceAccess<Texture, Rt>, DepthStencilInfo)>,
    pub(crate) vertex_bindings: Vec<VertexBindingDesc>,
    pub(crate) vertex_attributes: Vec<VertexAttributeDesc>,
    pub(crate) merged_reflection: Option<ShaderReflection>,
    pub(crate) descriptor_set_layouts: Vec<Arc<DescriptorSetLayout>>,
}

impl GraphicPipelineDescriptor {
    pub fn valid(&self) -> bool {
        self.vertex_shader.is_some() && !self.color_attachments.is_empty()
    }
}

#[derive(Default, Debug)]
pub struct ComputePipelineDescriptor {
}

impl ComputePipelineDescriptor {
    #[allow(dead_code)]
    pub fn name(&self) -> &str {
        "Unknown"
    }

    pub fn valid(&self) -> bool {
        false
    }
}

pub(crate) enum NodePipelineState {
    Graphic {
        pipeline_desc: GraphicPipelineDescriptor,
        job_functor: Option<Box<dyn FnOnce(&mut GraphicNodeExecutionContext, vk::CommandBuffer)>>,
    },
    #[allow(dead_code)]
    Compute {
        pipeline_desc: ComputePipelineDescriptor,
        job_functor: Option<Box<dyn FnOnce(&mut GraphicNodeExecutionContext, vk::CommandBuffer)>>,
    },
    Lambda {
        job_functor: Option<Box<dyn FnOnce(&mut LambdaNodeExecutionContext, vk::CommandBuffer)>>,
    }
}

impl NodePipelineState {
    pub(crate) fn valid(&self) -> bool {
        match self {
            NodePipelineState::Graphic { pipeline_desc, job_functor } => {
                pipeline_desc.valid() && job_functor.is_some()
            }
            NodePipelineState::Compute { pipeline_desc, job_functor } => {
                pipeline_desc.valid() && job_functor.is_some()
            }
            NodePipelineState::Lambda { job_functor } => {
                job_functor.is_some()
            }
        }
    }
}

pub struct RenderGraphNode {
    // TODO: debug only
    #[allow(dead_code)]
    pub(crate) name: String,
    pub(crate) inputs: Vec<ResourceAccessStorage>,
    pub(crate) outputs: Vec<ResourceAccessStorage>,

    pub(crate) pipeline_state: NodePipelineState,
}

impl RenderGraphNode {
    pub fn name(&self) -> &str {
        &self.name
    }
}