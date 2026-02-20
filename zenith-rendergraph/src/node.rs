use zenith_rhi::PipelineHandle;
use crate::graph::{ComputeNodeExecutionContext, GraphicNodeExecutionContext, LambdaNodeExecutionContext};
use crate::builder::ResourceAccessStorage;

/// Graph-scoped handle to a registered pipeline (valid only when used inside a render graph node).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphPipelineHandle(pub(crate) u32);

pub(crate) enum NodeState {
    Graphic {
        pipeline_handles: Vec<PipelineHandle>,
        job_functor: Option<Box<dyn FnOnce(&mut GraphicNodeExecutionContext) -> anyhow::Result<()>>>,
    },
    Compute {
        pipeline_handles: Vec<PipelineHandle>,
        job_functor: Option<Box<dyn FnOnce(&mut ComputeNodeExecutionContext) -> anyhow::Result<()>>>,
    },
    Lambda {
        job_functor: Option<Box<dyn FnOnce(&mut LambdaNodeExecutionContext) -> anyhow::Result<()>>>,
    }
}

impl NodeState {
    pub(crate) fn valid(&self) -> bool {
        match self {
            NodeState::Graphic { pipeline_handles, job_functor, .. } => {
                !pipeline_handles.is_empty() && job_functor.is_some()
            }
            NodeState::Compute { pipeline_handles, job_functor } => {
                !pipeline_handles.is_empty() && job_functor.is_some()
            }
            NodeState::Lambda { job_functor } => {
                job_functor.is_some()
            }
        }
    }

    /// Returns the pipeline handles for this node (empty for Lambda).
    pub(crate) fn pipeline_handles(&self) -> &[PipelineHandle] {
        match self {
            NodeState::Graphic { pipeline_handles, .. } => pipeline_handles.as_slice(),
            NodeState::Compute { pipeline_handles, .. } => pipeline_handles.as_slice(),
            NodeState::Lambda { .. } => &[],
        }
    }
}

pub struct RenderGraphNode {
    pub(crate) name: String,
    pub(crate) inputs: Vec<ResourceAccessStorage>,
    pub(crate) outputs: Vec<ResourceAccessStorage>,
    pub(crate) pipeline_state: NodeState,
}

impl RenderGraphNode {
    pub fn name(&self) -> &str {
        &self.name
    }
}