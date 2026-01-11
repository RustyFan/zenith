use crate::graph::{GraphicNodeExecutionContext, LambdaNodeExecutionContext, RenderGraph};
use crate::interface::{ResourceDescriptor, ResourceState};
use crate::node::{NodePipelineState, RenderGraphNode};
use crate::resource::{
    ExportResourceStorage, ExportedRenderGraphResource, GraphImportExportResource,
    GraphResource, GraphResourceDescriptor, GraphResourceId,
    GraphResourceView, InitialResourceStorage,
    RenderGraphResource, RenderGraphResourceAccess, Rt, Srv, Uav};
use log::warn;
use std::marker::PhantomData;
use std::sync::Arc;
use zenith_rhi::{vk, ColorAttachmentDesc, DepthStencilDesc, GraphicPipelineDesc, GraphicPipelineState, GraphicShaderInput, GraphicPipelineAttachments};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResourceAccessStorage {
    pub(crate) id: GraphResourceId,
    pub(crate) access: ResourceState,
    pub(crate) stage_hint: Option<vk::PipelineStageFlags2>,
}

#[derive(Default)]
pub struct RenderGraphBuilder {
    nodes: Vec<RenderGraphNode>,
    pub(crate) initial_resources: Vec<InitialResourceStorage>,
    #[allow(dead_code)]
    pub(crate) export_resources: Vec<ExportResourceStorage>,
}

impl RenderGraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn create<D: GraphResourceDescriptor>(
        &mut self,
        desc: D,
    ) -> RenderGraphResource<D::Resource> {
        let id = self.initial_resources.len() as u32;
        let desc: ResourceDescriptor = desc.into();

        match desc {
            ResourceDescriptor::Buffer(desc) => {
                self.initial_resources.push(desc.into());
            }
            ResourceDescriptor::Texture(desc) => {
                self.initial_resources.push(desc.into());
            }
        }

        RenderGraphResource {
            id,
            _marker: PhantomData,
        }
    }

    #[must_use]
    pub fn import<R: GraphImportExportResource>(
        &mut self,
        import_resource: Arc<R>,
        access: <R as GraphResource>::State,
    ) -> RenderGraphResource<R> {
        GraphImportExportResource::import(import_resource, self, access)
    }

    #[must_use]
    pub fn export<R: GraphImportExportResource>(
        &mut self,
        resource: RenderGraphResource<R>,
        access: <R as GraphResource>::State,
    ) -> ExportedRenderGraphResource<R> {
        GraphImportExportResource::export(resource, self, access)
    }

    #[must_use]
    pub fn add_graphic_node(&mut self, name: &str) -> GraphicNodeBuilder<'_, '_> {
        let index = self.nodes.len();

        self.nodes.push(RenderGraphNode {
            name: name.to_string(),
            inputs: vec![],
            outputs: vec![],
            pipeline_state: NodePipelineState::Graphic {
                pipeline_desc: None,
                color_attachments: vec![],
                depth_attachment: None,
                job_functor: None,
            },
        });

        GraphicNodeBuilder {
            common: CommonNodeBuilder {
                node: &mut self.nodes[index],
                resources: &self.initial_resources,
            },
        }
    }

    #[must_use]
    pub fn add_lambda_node(&mut self, name: &str) -> LambdaNodeBuilder<'_, '_> {
        let index = self.nodes.len();

        self.nodes.push(RenderGraphNode {
            name: name.to_string(),
            inputs: vec![],
            outputs: vec![],
            pipeline_state: NodePipelineState::Lambda {
                job_functor: None,
            },
        });

        LambdaNodeBuilder {
            common: CommonNodeBuilder {
                node: &mut self.nodes[index],
                resources: &self.initial_resources,
            }
        }
    }

    // #[must_use]
    // pub fn add_compute_node(&mut self, name: &str) -> GraphComputeNodeBuilder {
    //     let index = self.nodes.len();
    //     self.nodes.push(RenderGraphNode {
    //         node_name: name.to_string(),
    //         ..Default::default()
    //     });
    //
    //     GraphComputeNodeBuilder {
    //         node: &mut self.nodes[index]
    //     }
    // }

    #[profiling::function]
    pub fn build(self) -> RenderGraph {
        RenderGraph {
            nodes: self.nodes,
            initial_resources: self.initial_resources,
        }
    }
}

pub struct CommonNodeBuilder<'node, 'res> {
    node: &'node mut RenderGraphNode,
    resources: &'res Vec<InitialResourceStorage>,
}

impl CommonNodeBuilder<'_, '_> {
    #[must_use]
    fn read<R: GraphResource, V: GraphResourceView>(
        &mut self,
        resource: &RenderGraphResource<R>,
        access: impl Into<ResourceState>,
    ) -> RenderGraphResourceAccess<R, V> {
        let access = RenderGraphResourceAccess {
            id: resource.id,
            access: access.into(),
            _marker: PhantomData,
        };

        if let None = self.node.inputs.iter().find(|h| h.id == resource.id) {
            self.node.inputs.push(access.as_untyped());
        } else {
            let name = self.resources
                .get(resource.id as usize)
                .expect("Graph resource id out of bound!")
                .name();

            warn!("Try to read resource[{name}] multiple time!")
        }

        access
    }

    #[must_use]
    fn read_hint<R: GraphResource, V: GraphResourceView>(
        &mut self,
        resource: &RenderGraphResource<R>,
        access: impl Into<ResourceState>,
        stage_hint: vk::PipelineStageFlags2,
    ) -> RenderGraphResourceAccess<R, V> {
        let access = RenderGraphResourceAccess {
            id: resource.id,
            access: access.into(),
            _marker: PhantomData,
        };

        if let None = self.node.inputs.iter().find(|h| h.id == resource.id) {
            self.node.inputs.push(access.as_untyped_with_hint(stage_hint));
        } else {
            let name = self.resources
                .get(resource.id as usize)
                .expect("Graph resource id out of bound!")
                .name();

            warn!("Try to read resource[{name}] multiple time!")
        }

        access
    }

    #[must_use]
    fn write<R: GraphResource, V: GraphResourceView>(
        &mut self,
        resource: &mut RenderGraphResource<R>,
        access: impl Into<ResourceState>,
    ) -> RenderGraphResourceAccess<R, V>  {
        let access = RenderGraphResourceAccess {
            id: resource.id,
            access: access.into(),
            _marker: PhantomData,
        };

        if let None = self.node.outputs.iter().find(|h| h.id == resource.id) {
            self.node.outputs.push(access.as_untyped());
        } else {
            let name = self.resources
                .get(resource.id as usize)
                .expect("Graph resource id out of bound!")
                .name();

            warn!("Try to write to resource[{name}] multiple time!")
        }

        access
    }

    #[must_use]
    fn write_hint<R: GraphResource, V: GraphResourceView>(
        &mut self,
        resource: &mut RenderGraphResource<R>,
        access: impl Into<ResourceState>,
        stage_hint: vk::PipelineStageFlags2,
    ) -> RenderGraphResourceAccess<R, V>  {
        let access = RenderGraphResourceAccess {
            id: resource.id,
            access: access.into(),
            _marker: PhantomData,
        };

        if let None = self.node.outputs.iter().find(|h| h.id == resource.id) {
            self.node.outputs.push(access.as_untyped_with_hint(stage_hint));
        } else {
            let name = self.resources
                .get(resource.id as usize)
                .expect("Graph resource id out of bound!")
                .name();

            warn!("Try to write to resource[{name}] multiple time!")
        }

        access
    }
}

macro_rules! inject_common_node_builder_methods {
    ($read_view:ty, $write_view:ty) => {
        #[must_use]
        #[inline]
        pub fn read<R: GraphResource>(
            &mut self,
            resource: &RenderGraphResource<R>,
            access: <R as GraphResource>::State,
        ) -> RenderGraphResourceAccess<R, $read_view> {
            self.common.read(resource, access)
        }

        #[must_use]
        #[inline]
        pub fn read_hint<R: GraphResource>(
            &mut self,
            resource: &RenderGraphResource<R>,
            access: <R as GraphResource>::State,
            stage_hint: vk::PipelineStageFlags2,
        ) -> RenderGraphResourceAccess<R, $read_view> {
            self.common.read_hint(resource, access, stage_hint)
        }

        #[must_use]
        #[inline]
        pub fn write<R: GraphResource>(
            &mut self,
            resource: &mut RenderGraphResource<R>,
            access: <R as GraphResource>::State,
        ) -> RenderGraphResourceAccess<R, $write_view>  {
            self.common.write(resource, access)
        }

        #[must_use]
        #[inline]
        pub fn write_hint<R: GraphResource>(
            &mut self,
            resource: &mut RenderGraphResource<R>,
            access: <R as GraphResource>::State,
            stage_hint: vk::PipelineStageFlags2,
        ) -> RenderGraphResourceAccess<R, $write_view>  {
            self.common.write_hint(resource, access, stage_hint)
        }
    };
}

pub struct GraphicNodeBuilder<'node, 'res> {
    common: CommonNodeBuilder<'node, 'res>,
}

impl<'node, 'res> GraphicNodeBuilder<'node, 'res> {
    inject_common_node_builder_methods!(Srv, Rt);

    #[inline]
    pub fn execute<F>(&mut self, node_job: F)
    where
        F: FnOnce(&mut GraphicNodeExecutionContext) -> anyhow::Result<()> + 'static
    {
        if let NodePipelineState::Graphic { job_functor, .. } = &mut self.common.node.pipeline_state {
            job_functor.replace(Box::new(node_job));
        } else {
            unreachable!("Use other node execution context in graphic node: {}", self.common.node.name());
        }
    }

    pub fn pipeline(&mut self, shader: GraphicShaderInput, state: GraphicPipelineState) -> AttachmentBinder<'_, 'res> {
        // Clear any previous attachment bindings / desc.
        if let NodePipelineState::Graphic { pipeline_desc, color_attachments, depth_attachment, .. } =
            &mut self.common.node.pipeline_state
        {
            *pipeline_desc = None;
            color_attachments.clear();
            *depth_attachment = None;
        } else {
            unreachable!();
        }

        AttachmentBinder {
            node: self.common.node,
            resources: self.common.resources,
            shader: Some(shader),
            state: Some(state),
            finished: false,
        }
    }
}

pub struct AttachmentBinder<'node, 'res> {
    node: &'node mut RenderGraphNode,
    resources: &'res Vec<InitialResourceStorage>,
    shader: Option<GraphicShaderInput>,
    state: Option<GraphicPipelineState>,
    finished: bool,
}

impl<'node, 'res> AttachmentBinder<'node, 'res> {
    pub fn push_color(
        &mut self,
        rt: RenderGraphResourceAccess<crate::interface::Texture, Rt>,
        desc: ColorAttachmentDesc,
    ) -> &mut Self {
        if let NodePipelineState::Graphic { color_attachments, .. } = &mut self.node.pipeline_state {
            color_attachments.push((rt.id, desc));
        } else {
            unreachable!();
        }
        self
    }

    pub fn depth(
        &mut self,
        rt: RenderGraphResourceAccess<crate::interface::Texture, Rt>,
        desc: DepthStencilDesc,
    ) -> &mut Self {
        if let NodePipelineState::Graphic { depth_attachment, .. } = &mut self.node.pipeline_state {
            *depth_attachment = Some((rt.id, desc));
        } else {
            unreachable!();
        }
        self
    }

    pub fn finish(mut self) -> GraphicPipelineDesc {
        let desc = self.finalize();
        self.finished = true;
        desc
    }

    fn finalize(&mut self) -> GraphicPipelineDesc {
        let shader = self.shader.take().expect("AttachmentBinder finalized twice");
        let mut state = self.state.take().expect("AttachmentBinder finalized twice");

        let (color_ids, color_descs, depth_id, depth_desc) = match &mut self.node.pipeline_state {
            NodePipelineState::Graphic { color_attachments, depth_attachment, .. } => {
                let (ids, descs): (Vec<_>, Vec<_>) = color_attachments.iter().cloned().unzip();
                let (depth_id, depth_desc) = depth_attachment.clone().map(|(id, d)| (Some(id), Some(d))).unwrap_or((None, None));
                (ids, descs, depth_id, depth_desc)
            }
            _ => unreachable!(),
        };

        // Attachments formats (dynamic rendering order).
        let mut attachments = GraphicPipelineAttachments::default();
        attachments.color_formats = color_ids
            .iter()
            .map(|id| texture_format(self.resources, *id))
            .collect();
        attachments.depth_format = depth_id.map(|id| texture_format(self.resources, id));
        attachments.stencil_format = None;

        // Populate state attachment descs used for blend state + begin_rendering.
        state.color_blend.attachments = color_descs;
        if let Some(ds) = depth_desc {
            state.depth_stencil = Some(ds);
        }

        let pipeline_desc = GraphicPipelineDesc::new(shader, state, attachments);

        if let NodePipelineState::Graphic { pipeline_desc: slot, .. } = &mut self.node.pipeline_state {
            *slot = Some(pipeline_desc.clone());
        }

        pipeline_desc
    }
}

impl Drop for AttachmentBinder<'_, '_> {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.finalize();
        }
    }
}

fn texture_format(resources: &Vec<InitialResourceStorage>, id: GraphResourceId) -> vk::Format {
    let storage = resources.get(id as usize).expect("Graph resource id out of bound!");
    match storage {
        InitialResourceStorage::ManagedTexture(desc) => desc.format,
        InitialResourceStorage::ImportedTexture(tex, _) => tex.format(),
        _ => panic!("AttachmentBinder expects a texture resource id."),
    }
}

pub struct LambdaNodeBuilder<'node, 'res> {
    common: CommonNodeBuilder<'node, 'res>,
}

impl<'node, 'res> LambdaNodeBuilder<'node, 'res> {
    inject_common_node_builder_methods!(Srv, Uav);

    #[inline]
    pub fn execute<F>(&mut self, node_job: F)
    where
        F: FnOnce(&mut LambdaNodeExecutionContext) -> anyhow::Result<()> + 'static
    {
        if let NodePipelineState::Lambda { job_functor } = &mut self.common.node.pipeline_state {
            job_functor.replace(Box::new(node_job));
        } else {
            unreachable!("Use other node execution context in lambda node: {}", self.common.node.name());
        }
    }
}

