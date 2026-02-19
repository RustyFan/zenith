mod builder;
mod node;
mod graph;
mod interface;
mod resource;

pub use resource::{
    GraphBindable, RenderGraphResource, RenderGraphResourceAccess,
};
pub use builder::{RenderGraphBuilder, GraphicNodeBuilder};
pub use zenith_rhi::{
    ColorAttachmentDesc, ColorAttachmentDescBuilder, ColorAttachmentDescBuilderError,
    ComputePipelineDesc, GraphicPipelineDesc, GraphicShaderInput, GraphicPipelineState,
    GraphicShaderInputBuilder, GraphicShaderInputBuildError,
    GraphicPipelineStateBuilder,
    VertexLayout,
};
pub use node::{
    RenderGraphNode,
    GraphPipelineHandle,
};
pub use zenith_rhi::PipelineHandle;
pub use graph::{
    RenderGraph, CompiledRenderGraph, RetiredRenderGraph,
    GraphicNodeExecutionContext, LambdaNodeExecutionContext, ComputeNodeExecutionContext,
    ColorAttachment, DepthStencilAttachment,
};
