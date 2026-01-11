use zenith_rhi::{ColorAttachmentDesc, DepthStencilDesc, GraphicPipelineDesc};
use crate::resource::GraphResourceId;
use crate::graph::{GraphicNodeExecutionContext, LambdaNodeExecutionContext};
use crate::builder::ResourceAccessStorage;

#[derive(Default, Debug)]
pub struct ComputePipelineDescriptor {
}

impl ComputePipelineDescriptor {
    #[allow(dead_code)]
    pub fn name(&self) -> &str {
        "Unknown"
    }
}

pub(crate) enum NodePipelineState {
    Graphic {
        pipeline_desc: Option<GraphicPipelineDesc>,
        color_attachments: Vec<(GraphResourceId, ColorAttachmentDesc)>,
        depth_attachment: Option<(GraphResourceId, DepthStencilDesc)>,
        job_functor: Option<Box<dyn FnOnce(&mut GraphicNodeExecutionContext) -> anyhow::Result<()>>>,
    },
    #[allow(dead_code)]
    Compute {
        pipeline_desc: ComputePipelineDescriptor,
        job_functor: Option<Box<dyn FnOnce(&mut GraphicNodeExecutionContext) -> anyhow::Result<()>>>,
    },
    Lambda {
        job_functor: Option<Box<dyn FnOnce(&mut LambdaNodeExecutionContext) -> anyhow::Result<()>>>,
    }
}

impl NodePipelineState {
    pub(crate) fn valid(&self) -> bool {
        match self {
            NodePipelineState::Graphic { pipeline_desc, job_functor, .. } => pipeline_desc.is_some() && job_functor.is_some(),
            NodePipelineState::Compute { .. } => {
                false
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