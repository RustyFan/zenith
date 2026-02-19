use crate::graph::{ComputeNodeExecutionContext, GraphicNodeExecutionContext, LambdaNodeExecutionContext, RenderGraph};
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
use zenith_rhi::{vk, ComputePipelineDesc, GraphicPipelineDesc, PipelineRegistry, RenderDevice};
use crate::GraphPipelineHandle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResourceAccessStorage {
    pub(crate) id: GraphResourceId,
    pub(crate) access: ResourceState,
    pub(crate) stage_hint: Option<vk::PipelineStageFlags2>,
}

pub struct RenderGraphBuilder<'a> {
    nodes: Vec<RenderGraphNode>,
    pub(crate) initial_resources: Vec<InitialResourceStorage>,
    #[allow(dead_code)]
    pub(crate) export_resources: Vec<ExportResourceStorage>,
    pipeline_cache: &'a mut PipelineRegistry,
    device: &'a RenderDevice,
}

impl<'a> RenderGraphBuilder<'a> {
    pub fn new(pipeline_cache: &'a mut PipelineRegistry, device: &'a RenderDevice) -> Self {
        Self {
            nodes: Vec::new(),
            initial_resources: Vec::new(),
            export_resources: Vec::new(),
            pipeline_cache,
            device,
        }
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
    pub fn import_simplified<R: GraphImportExportResource>(
        &mut self,
        import_resource: Arc<R>,
        access: <R as GraphResource>::State,
    ) -> RenderGraphResource<R> {
        GraphImportExportResource::import(import_resource, self, access)
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
    pub fn add_graphic_node(&mut self, name: &str) -> GraphicNodeBuilder<'_> {
        let index = self.nodes.len();

        self.nodes.push(RenderGraphNode {
            name: name.to_string(),
            inputs: vec![],
            outputs: vec![],
            pipeline_state: NodePipelineState::Graphic {
                pipeline_handles: Vec::new(),
                job_functor: None,
            },
        });

        GraphicNodeBuilder {
            common: CommonNodeBuilder {
                node: &mut self.nodes[index],
                resources: &self.initial_resources,
                pipeline_cache: self.pipeline_cache,
                device: self.device,
            },
        }
    }

    #[must_use]
    pub fn add_lambda_node(&mut self, name: &str) -> LambdaNodeBuilder<'_> {
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
                pipeline_cache: self.pipeline_cache,
                device: self.device,
            }
        }
    }

    #[must_use]
    pub fn add_compute_node(&mut self, name: &str) -> ComputeNodeBuilder<'_> {
        let index = self.nodes.len();

        self.nodes.push(RenderGraphNode {
            name: name.to_string(),
            inputs: vec![],
            outputs: vec![],
            pipeline_state: NodePipelineState::Compute {
                pipeline_handles: Vec::new(),
                job_functor: None,
            },
        });

        ComputeNodeBuilder {
            common: CommonNodeBuilder {
                node: &mut self.nodes[index],
                resources: &self.initial_resources,
                pipeline_cache: self.pipeline_cache,
                device: self.device,
            },
        }
    }

    #[profiling::function]
    pub fn build(self) -> RenderGraph {
        RenderGraph {
            nodes: self.nodes,
            initial_resources: self.initial_resources,
        }
    }
}

pub struct CommonNodeBuilder<'builder> {
    node: &'builder mut RenderGraphNode,
    resources: &'builder Vec<InitialResourceStorage>,
    pipeline_cache: &'builder mut PipelineRegistry,
    device: &'builder RenderDevice,
}

impl<'builder> CommonNodeBuilder<'builder> {
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

        /// Read resource with pipeline stage hint.
        /// Stage hint will overwrite the shader reflection stages.
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

        /// Write resource with pipeline stage hint.
        /// Stage hint will overwrite the shader reflection stages.
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

pub struct GraphicNodeBuilder<'builder> {
    common: CommonNodeBuilder<'builder>,
}

impl<'builder> GraphicNodeBuilder<'builder> {
    inject_common_node_builder_methods!(Srv, Rt);

    pub fn register_pipeline(&mut self, desc: GraphicPipelineDesc) -> GraphPipelineHandle {
        if let NodePipelineState::Graphic { pipeline_handles, .. } = &mut self.common.node.pipeline_state {
            let cache_handle = self.common.pipeline_cache
                .register_graph_pipeline(self.common.device, &desc)
                .expect("Failed to register graphic pipeline");
            let index = pipeline_handles.len() as u32;
            pipeline_handles.push(cache_handle);
            GraphPipelineHandle(index)
        } else {
            unreachable!("register_pipeline called on non-graphic node");
        }
    }

    #[inline]
    pub fn execute<F>(self, node_job: F)
    where
        F: FnOnce(&mut GraphicNodeExecutionContext) -> anyhow::Result<()> + 'static
    {
        if let NodePipelineState::Graphic { job_functor, .. } = &mut self.common.node.pipeline_state {
            job_functor.replace(Box::new(node_job));
        } else {
            unreachable!("Use other node execution context in graphic node: {}", self.common.node.name());
        }
    }
}

pub struct LambdaNodeBuilder<'builder> {
    common: CommonNodeBuilder<'builder>,
}

impl<'builder> LambdaNodeBuilder<'builder> {
    inject_common_node_builder_methods!(Srv, Uav);

    #[inline]
    pub fn execute<F>(self, node_job: F)
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

pub struct ComputeNodeBuilder<'builder> {
    common: CommonNodeBuilder<'builder>,
}

impl<'builder> ComputeNodeBuilder<'builder> {
    inject_common_node_builder_methods!(Srv, Uav);

    pub fn register_pipeline(&mut self, desc: ComputePipelineDesc) -> GraphPipelineHandle {
        if let NodePipelineState::Compute { pipeline_handles, .. } = &mut self.common.node.pipeline_state {
            let cache_handle = self.common.pipeline_cache
                .register_compute_pipeline(self.common.device, &desc)
                .expect("Failed to register compute pipeline");
            let index = pipeline_handles.len() as u32;
            pipeline_handles.push(cache_handle);
            GraphPipelineHandle(index)
        } else {
            unreachable!("register_pipeline called on non-compute node");
        }
    }

    #[inline]
    pub fn execute<F>(self, node_job: F)
    where
        F: FnOnce(&mut ComputeNodeExecutionContext) -> anyhow::Result<()> + 'static
    {
        if let NodePipelineState::Compute { job_functor, .. } = &mut self.common.node.pipeline_state {
            job_functor.replace(Box::new(node_job));
        } else {
            unreachable!("Use other node execution context in compute node: {}", self.common.node.name());
        }
    }
}
