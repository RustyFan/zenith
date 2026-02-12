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
    GraphicPipelineDesc, GraphicShaderInput, GraphicPipelineState,
    GraphicShaderInputBuilder, GraphicShaderInputBuildError,
    GraphicPipelineStateBuilder,
    VertexLayout,
};
pub use node::{
    RenderGraphNode,
    GraphicPipelineHandle,
};
pub use graph::{
    RenderGraph, CompiledRenderGraph, RetiredRenderGraph,
    GraphicNodeExecutionContext, LambdaNodeExecutionContext,
    ColorAttachment, DepthStencilAttachment,
};
