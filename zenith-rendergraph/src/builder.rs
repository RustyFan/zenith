use crate::graph::{GraphicNodeExecutionContext, LambdaNodeExecutionContext, RenderGraph};
use crate::interface::{ResourceDescriptor, ResourceState};
use crate::node::{ColorAttachmentDesc, DepthStencilInfo, GraphicPipelineDescriptor, NodePipelineState, RenderGraphNode, VertexAttributeDesc, VertexBindingDesc};
use crate::resource::{
    ExportResourceStorage, ExportedRenderGraphResource, GraphImportExportResource,
    GraphResource, GraphResourceDescriptor, GraphResourceId,
    GraphResourceView, InitialResourceStorage,
    RenderGraphResource, RenderGraphResourceAccess, Rt, Srv, Uav};
use log::warn;
use std::marker::PhantomData;
use std::sync::Arc;
use zenith_rhi::{vk, Shader, ShaderReflection, Texture};

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
        import_resource: Arc<R>,
        access: <R as GraphResource>::State,
    ) -> RenderGraphResource<R> {
        GraphImportExportResource::import(import_resource, name, self, access)
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
                pipeline_desc: Default::default(),
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
        F: FnOnce(&mut GraphicNodeExecutionContext) + 'static
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
        F: FnOnce(&mut LambdaNodeExecutionContext) + 'static
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
    fn update_reflection(&mut self) {
        let mut reflections: Vec<&ShaderReflection> = Vec::new();
        if let Some(vs) = &self.pipeline_desc.vertex_shader {
            reflections.push(vs.reflection());
        }
        if let Some(fs) = &self.pipeline_desc.fragment_shader {
            reflections.push(fs.reflection());
        }

        if !reflections.is_empty() {
            let merged = ShaderReflection::merge(&reflections);

            // Use cached layouts from shaders instead of creating new ones
            // Prefer fragment shader layouts as they typically have more bindings
            let layouts = if let Some(fs) = &self.pipeline_desc.fragment_shader {
                fs.descriptor_set_layouts().to_vec()
            } else if let Some(vs) = &self.pipeline_desc.vertex_shader {
                vs.descriptor_set_layouts().to_vec()
            } else {
                Vec::new()
            };

            self.pipeline_desc.descriptor_set_layouts = layouts;
            self.pipeline_desc.merged_reflection = Some(merged);
        }
    }

    #[inline]
    pub fn with_vertex_shader(mut self, shader: Arc<Shader>) -> Self {
        self.pipeline_desc.vertex_shader = Some(shader);
        self.update_reflection();
        self
    }

    #[inline]
    pub fn with_fragment_shader(mut self, shader: Arc<Shader>) -> Self {
        self.pipeline_desc.fragment_shader = Some(shader);
        self.update_reflection();
        self
    }

    #[inline]
    pub fn with_color(self, color: RenderGraphResourceAccess<Texture, Rt>, color_info: ColorAttachmentDesc) -> Self {
        self.pipeline_desc.color_attachments.push((color, color_info));
        self
    }

    #[inline]
    pub fn with_depth_stencil(self, depth_stencil: RenderGraphResourceAccess<Texture, Rt>, depth_stencil_info: DepthStencilInfo) -> Self {
        self.pipeline_desc.depth_stencil_attachment = Some((depth_stencil, depth_stencil_info));
        self
    }

    #[inline]
    pub fn with_vertex_binding(self, binding: u32, stride: u32, input_rate: vk::VertexInputRate) -> Self {
        self.pipeline_desc.vertex_bindings.push(VertexBindingDesc {
            binding,
            stride,
            input_rate,
        });
        self
    }

    #[inline]
    pub fn with_vertex_attribute(self, location: u32, binding: u32, format: vk::Format, offset: u32) -> Self {
        self.pipeline_desc.vertex_attributes.push(VertexAttributeDesc {
            location,
            binding,
            format,
            offset,
        });
        self
    }
}
