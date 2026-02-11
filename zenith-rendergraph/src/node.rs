use zenith_rhi::{GraphicPipelineDesc};
use crate::graph::{GraphicNodeExecutionContext, LambdaNodeExecutionContext};
use crate::builder::ResourceAccessStorage;

/// Handle to a registered pipeline within a node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphicPipelineHandle(pub(crate) u32);

pub(crate) enum NodePipelineState {
    Graphic {
        pipeline_descs: Vec<GraphicPipelineDesc>,
        job_functor: Option<Box<dyn FnOnce(&mut GraphicNodeExecutionContext) -> anyhow::Result<()>>>,
    },
    #[allow(dead_code)]
    Compute {
        job_functor: Option<Box<dyn FnOnce(&mut GraphicNodeExecutionContext) -> anyhow::Result<()>>>,
    },
    Lambda {
        job_functor: Option<Box<dyn FnOnce(&mut LambdaNodeExecutionContext) -> anyhow::Result<()>>>,
    }
}

impl NodePipelineState {
    pub(crate) fn valid(&self) -> bool {
        match self {
            NodePipelineState::Graphic { pipeline_descs, job_functor, .. } => {
                !pipeline_descs.is_empty() && job_functor.is_some()
            }
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