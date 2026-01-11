mod builder;
mod node;
mod graph;
mod interface;
mod resource;

pub use resource::{
    RenderGraphResource, RenderGraphResourceAccess
};
pub use builder::{RenderGraphBuilder, GraphicNodeBuilder};
pub use zenith_rhi::{
    ColorAttachmentDesc, ColorAttachmentDescBuilder, ColorAttachmentDescBuilderError,
    DepthStencilDesc, DepthStencilDescBuilder, DepthStencilDescBuilderError,
    GraphicPipelineDesc, GraphicShaderInput, GraphicPipelineState,
    GraphicShaderInputBuilder, GraphicShaderInputBuildError,
    GraphicPipelineStateBuilder,
    VertexLayout,
};
pub use node::{
    RenderGraphNode,
};
pub use graph::{
    RenderGraph, CompiledRenderGraph, RetiredRenderGraph,
    GraphicNodeExecutionContext, LambdaNodeExecutionContext,
};
