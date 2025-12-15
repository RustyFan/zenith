use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::Arc;
use log::warn;
use crate::node::{NodePipelineState, RenderGraphNode};
use crate::graph::{GraphicNodeExecutionContext, LambdaNodeExecutionContext, RenderGraph, ResourceStorage};
use crate::node::{DepthStencilInfo};
use crate::interface::{GraphResourceAccess, ResourceDescriptor, RenderResource, Texture};
use crate::resource::{
    ExportResourceStorage, ExportedRenderGraphResource, GraphImportExportResource,
    GraphResource, GraphResourceDescriptor, GraphResourceView,
    GraphResourceId, InitialResourceStorage,
    RenderGraphResource, RenderGraphResourceAccess, Rt, Srv, Uav};
use zenith_render::GraphicShader;
use crate::{ColorInfo, GraphicPipelineDescriptor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResourceAccessStorage {
    pub(crate) id: GraphResourceId,
    pub(crate) access: GraphResourceAccess,
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
        Self {
            ..Default::default()
        }
    }

    #[must_use]
    pub fn create<D: GraphResourceDescriptor>(
        &mut self,
        name: &str,
        desc: D,
    ) -> RenderGraphResource<D::Resource> {
        let id = self.initial_resources.len() as u32;
        let desc: ResourceDescriptor = desc.into();

        match desc {
            ResourceDescriptor::Buffer(desc) => {
                self.initial_resources.push((name.to_owned(), desc).into());
            }
            ResourceDescriptor::Texture(desc) => {
                self.initial_resources.push((name.to_owned(), desc).into());
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
        name: &str,
        import_resource: impl Into<RenderResource<R>>,
        access: impl Into<GraphResourceAccess>,
    ) -> RenderGraphResource<R> {
        GraphImportExportResource::import(import_resource, name, self, access)
    }

    #[must_use]
    pub fn export<R: GraphImportExportResource>(
        &mut self,
        resource: RenderGraphResource<R>,
        access: impl Into<GraphResourceAccess>,
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
                pipeline_desc: Default::default(),
                job_functor: None,
            },
        });

        GraphicNodeBuilder {
            common: CommonNodeBuilder {
                node: &mut self.nodes[index],
                resources: &self.initial_resources,
            }
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
    pub fn build(self, device: &wgpu::Device) -> RenderGraph {
        let resources = self.initial_resources
            .into_iter()
            .map(|res| {
                match res {
                    InitialResourceStorage::ManagedBuffer(name, desc) => {
                        let buffer = device.create_buffer(&desc);
                        ResourceStorage::ManagedBuffer {
                            name,
                            resource: buffer,
                            state_tracker: Cell::new(wgpu::BufferUses::empty()).into()
                        }
                    }
                    InitialResourceStorage::ManagedTexture(name, desc) => {
                        let tex = device.create_texture(&desc);
                        ResourceStorage::ManagedTexture {
                            name,
                            resource: tex,
                            state_tracker: Cell::new(wgpu::TextureUses::UNINITIALIZED).into()
                        }
                    }
                    InitialResourceStorage::ImportedBuffer(name, buffer, init_access) => ResourceStorage::ImportedBuffer {
                        name,
                        resource: buffer.into(),
                        state_tracker: Cell::new(init_access).into(),
                    },
                    InitialResourceStorage::ImportedTexture(name, tex, init_access) => ResourceStorage::ImportedTexture {
                        name,
                        resource: tex.into(),
                        state_tracker: Cell::new(init_access).into(),
                    },
                }
            })
            .collect();

        RenderGraph {
            nodes: self.nodes,
            resources
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
        access: impl Into<GraphResourceAccess>,
    ) -> RenderGraphResourceAccess<R, V> {
        let access = RenderGraphResourceAccess {
            id: resource.id,
            access: access.into(),
            _marker: PhantomData,
        };

        if let None = self.node.inputs.iter().find(|h| h.id == resource.id) {
            self.node.inputs.push(access.clone().into_untyped());
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
        access: impl Into<GraphResourceAccess>,
    ) -> RenderGraphResourceAccess<R, V>  {
        let access = RenderGraphResourceAccess {
            id: resource.id,
            access: access.into(),
            _marker: PhantomData,
        };

        if let None = self.node.outputs.iter().find(|h| h.id == resource.id) {
            self.node.outputs.push(access.clone().into_untyped());
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
            access: impl Into<GraphResourceAccess>
        ) -> RenderGraphResourceAccess<R, $read_view> {
            self.common.read(resource, access)
        }

        #[must_use]
        #[inline]
        pub fn write<R: GraphResource>(
            &mut self,
            resource: &mut RenderGraphResource<R>,
            access: impl Into<GraphResourceAccess>,
        ) -> RenderGraphResourceAccess<R, $write_view>  {
            self.common.write(resource, access)
        }
    };
}

pub struct GraphicNodeBuilder<'node, 'res> {
    common: CommonNodeBuilder<'node, 'res>,
}

impl<'node, 'res> Drop for GraphicNodeBuilder<'node, 'res> {
    fn drop(&mut self) {
        debug_assert!(self.common.node.pipeline_state.valid());
    }
}

impl<'node, 'res> GraphicNodeBuilder<'node, 'res> {
    inject_common_node_builder_methods!(Srv, Rt);

    #[inline]
    pub fn execute<F>(&mut self, node_job: F)
    where
        F: FnOnce(&mut GraphicNodeExecutionContext, &mut wgpu::CommandEncoder) + 'static
    {
        if let NodePipelineState::Graphic { job_functor, .. } = &mut self.common.node.pipeline_state {
            job_functor.replace(Box::new(node_job));
        } else {
            unreachable!("Use other node execution context in graphic node: {}", self.common.node.name());
        }
    }

    #[must_use]
    #[inline]
    pub fn setup_pipeline(&mut self) -> GraphicPipelineBuilder<'_> {
        let pipeline_desc = if let NodePipelineState::Graphic { pipeline_desc, .. } = &mut self.common.node.pipeline_state {
            pipeline_desc
        } else {
            unreachable!();
        };

        GraphicPipelineBuilder {
            pipeline_desc,
        }
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
        F: FnOnce(&mut LambdaNodeExecutionContext, &mut wgpu::CommandEncoder) + 'static
    {
        if let NodePipelineState::Lambda { job_functor } = &mut self.common.node.pipeline_state {
            job_functor.replace(Box::new(node_job));
        } else {
            unreachable!("Use other node execution context in lambda node: {}", self.common.node.name());
        }
    }
}

pub struct GraphicPipelineBuilder<'a> {
    pipeline_desc: &'a mut GraphicPipelineDescriptor,
}

impl<'a> GraphicPipelineBuilder<'a> {
    #[inline]
    pub fn with_shader(self, shader: Arc<GraphicShader>) -> Self {
        self.pipeline_desc.shader = Some(shader);
        self
    }

    #[inline]
    pub fn with_color(self, color: RenderGraphResourceAccess<Texture, Rt>, color_info: ColorInfo) -> Self {
        self.pipeline_desc.color_attachments.push((color, color_info));
        self
    }

    #[inline]
    pub fn with_depth_stencil(self, depth_stencil: RenderGraphResourceAccess<Texture, Rt>, depth_stencil_info: DepthStencilInfo) -> Self {
        self.pipeline_desc.depth_stencil_attachment = Some((depth_stencil, depth_stencil_info));
        self
    }

    // #[inline]
    // pub fn with_binding<R: GraphResource, V: GraphResourceView>(self, binding: u32, color: &RenderGraphResourceAccess<R, V>) -> Self {
    //     self.pipeline_desc.bindings.push((binding, color.id));
    //     self
    // }
}
