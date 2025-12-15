use crate::interface::RenderResource;
use std::cell::{Cell};
use bytemuck::NoUninit;
use derive_more::From;
use log::{warn};
use zenith_core::collections::SmallVec;
use zenith_render::PipelineCache;
use crate::node::{NodePipelineState, RenderGraphNode};
use crate::interface::{Buffer, BufferState, GraphResourceAccess, Texture, TextureState};
use crate::GraphicPipelineDescriptor;
use crate::resource::{GraphResourceId, GraphResourceView, GraphResourceState, RenderGraphResourceAccess};

pub(crate) enum ResourceStorage {
    ManagedBuffer {
        name: String,
        resource: Buffer,
        state_tracker: ResourceStateTracker<BufferState>
    },
    ManagedTexture {
        name: String,
        resource: Texture,
        state_tracker: ResourceStateTracker<TextureState>
    },
    ImportedBuffer {
        name: String,
        resource: RenderResource<Buffer>,
        state_tracker: ResourceStateTracker<BufferState>
    },
    ImportedTexture {
        name: String,
        resource: RenderResource<Texture>,
        state_tracker: ResourceStateTracker<TextureState>
    },
}

impl ResourceStorage {
    pub(crate) fn name(&self) -> &str {
        match self {
            ResourceStorage::ManagedBuffer { name, .. } => &name,
            ResourceStorage::ManagedTexture { name, .. } => &name,
            ResourceStorage::ImportedBuffer { name, .. } => &name,
            ResourceStorage::ImportedTexture { name, .. } => &name,
        }
    }

    pub(crate) fn as_buffer(&self) -> &Buffer {
        match self {
            ResourceStorage::ManagedBuffer { resource, .. } => { &resource }
            ResourceStorage::ImportedBuffer { resource, .. } => { &resource }
            ResourceStorage::ManagedTexture { .. } | ResourceStorage::ImportedTexture { .. } => {
                unreachable!("Expect buffer, but resource is a texture!");
            }
        }
    }

    pub(crate) fn as_texture(&self) -> &Texture {
        match self {
            ResourceStorage::ManagedTexture { resource, .. } => { &resource }
            ResourceStorage::ImportedTexture { resource, .. } => { &resource }
            ResourceStorage::ManagedBuffer { .. } | ResourceStorage::ImportedBuffer { .. } => {
                unreachable!("Expect texture, but resource is a buffer!");
            }
        }
    }
}

#[derive(From)]
pub(crate) struct ResourceStateTracker<T: GraphResourceState> {
    current_state: Cell<T>,
}

impl<T: GraphResourceState> ResourceStateTracker<T> {
    #[allow(dead_code)]
    pub(crate) fn current(&self) -> T {
        self.current_state.get()
    }

    pub(crate) fn should_transition_to(&self, next_state: T, skip_if_same: bool) -> bool {
        if skip_if_same {
            self.current_state.get() != next_state
        } else {
            true
        }
    }

    pub(crate) fn transition_to(&self, next_state: T) {
        self.current_state.set(next_state);
    }
}

pub struct RenderGraph {
    pub(crate) nodes: Vec<RenderGraphNode>,
    pub(crate) resources: Vec<ResourceStorage>,
}

impl RenderGraph {
    pub fn validate(&self) {

    }

    #[profiling::function]
    pub fn compile(
        self,
        device: &wgpu::Device,
        pipeline_cache: &mut PipelineCache,
    ) -> CompiledRenderGraph {
        let mut graphic_pipelines = vec![];
        let _compute_pipelines = vec![];

        for node in &self.nodes {
            match &node.pipeline_state {
                NodePipelineState::Graphic { pipeline_desc, .. } => {
                    let pipeline = self.create_graphic_pipeline(node.name(), device, pipeline_cache, pipeline_desc);
                    graphic_pipelines.push(pipeline);
                }
                NodePipelineState::Compute { .. } => { unimplemented!() }
                NodePipelineState::Lambda { .. } => {}
            }
        }

        CompiledRenderGraph {
            nodes: self.nodes,
            resources: self.resources,
            graphic_pipelines,
            _compute_pipelines,
        }
    }

    #[profiling::function]
    fn create_graphic_pipeline(
        &self,
        node_name: &str,
        device: &wgpu::Device,
        pipeline_cache: &mut PipelineCache,
        desc: &GraphicPipelineDescriptor,
    ) -> wgpu::RenderPipeline {
        let color_attachments = desc.color_attachments
            .iter()
            .map(|(resource, color_info)| {
                let storage = utility::resource_storage_ref(&self.resources, resource.id);

                match storage {
                    ResourceStorage::ManagedTexture { resource, .. } => {
                        wgpu::ColorTargetState {
                            format: resource.format(),
                            blend: color_info.blend,
                            write_mask: color_info.write_mask.unwrap_or(wgpu::ColorWrites::ALL),
                        }
                    }
                    ResourceStorage::ImportedTexture { resource, .. } => {
                        wgpu::ColorTargetState {
                            format: resource.format(),
                            blend: color_info.blend,
                            write_mask: color_info.write_mask.unwrap_or(wgpu::ColorWrites::ALL),
                        }
                    }
                    _ => unreachable!("Color attachment had bound to a non-texture resource!")
                }
            })
            .map(Some)
            .collect::<SmallVec<[Option<wgpu::ColorTargetState>; 8]>>();

        let depth_stencil_attachment = desc.depth_stencil_attachment
            .as_ref()
            .map(|(resource, depth)| {
                let storage = utility::resource_storage_ref(&self.resources, resource.id);

                match storage {
                    ResourceStorage::ManagedTexture { resource, .. } => {
                        wgpu::DepthStencilState {
                            format: resource.format(),
                            depth_write_enabled: depth.depth_write,
                            depth_compare: depth.compare,
                            stencil: depth.stencil.clone(),
                            bias: depth.bias,
                        }
                    }
                    ResourceStorage::ImportedTexture { resource, .. } => {
                        wgpu::DepthStencilState {
                            format: resource.format(),
                            depth_write_enabled: depth.depth_write,
                            depth_compare: depth.compare,
                            stencil: depth.stencil.clone(),
                            bias: depth.bias,
                        }
                    }
                    _ => unreachable!()
                }
            });

        let shader = desc
            .shader
            .as_ref()
            .expect(&format!("Missing raster shader for node {}", node_name));

        pipeline_cache
            .get_or_create_graphic_pipeline(
                device,
                shader,
                &color_attachments,
                depth_stencil_attachment)
            .expect(&format!("Failed to compile graphic pipeline: {}", shader.name()))
    }
}

pub struct CompiledRenderGraph {
    nodes: Vec<RenderGraphNode>,
    resources: Vec<ResourceStorage>,
    graphic_pipelines: Vec<wgpu::RenderPipeline>,
    _compute_pipelines: Vec<wgpu::ComputePipeline>,
}

impl CompiledRenderGraph {
    #[profiling::function]
    pub fn execute(self, device: &wgpu::Device, queue: &wgpu::Queue) -> PresentableRenderGraph {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render graph main command encoder"),
        });

        let mut graphic_pipe_index = 0u32;
        // let mut compute_pipe_index = 0u32;

        for node in self.nodes.into_iter() {
            {
                profiling::scope!("rendergraph::barriers");
                Self::transition_resources(
                    &mut encoder,
                    &self.resources,
                    node
                        .inputs
                        .iter()
                        .map(|access| (access.id, access.access))
                        .chain(node.outputs.iter().map(|access| (access.id, access.access)))
                );
            }

            {
                match node.pipeline_state {
                    NodePipelineState::Graphic { pipeline_desc, mut job_functor } => {
                        let name = node.name;
                        let pipeline = self.graphic_pipelines.get(graphic_pipe_index as usize).unwrap();
                        graphic_pipe_index += 1;

                        if let Some(record) = job_functor.take() {
                            let mut ctx = GraphicNodeExecutionContext {
                                name: name.as_str(),
                                pipeline_desc: &pipeline_desc,
                                device,
                                queue,
                                resources: &self.resources,
                                pipeline: pipeline.clone(),
                            };
                            let scope = || {
                                profiling::scope!(format!("rendergraph::node_recording [{}]", name));
                                record(&mut ctx, &mut encoder);
                            };
                            scope();
                        } else {
                            warn!("Missing job of graphic node {}!", name);
                        }
                    }
                    NodePipelineState::Compute{ .. } => {
                        // compute_pipe_index += 1;
                        unimplemented!()
                    }
                    NodePipelineState::Lambda{ mut job_functor } => {
                        let name = node.name;

                        if let Some(record) = job_functor.take() {
                            let mut ctx = LambdaNodeExecutionContext {
                                queue,
                                resources: &self.resources,
                            };
                            let scope = || {
                                profiling::scope!(format!("rendergraph::node_recording [{}]", name));
                                record(&mut ctx, &mut encoder);
                            };
                            scope();
                        } else {
                            warn!("Missing job of lambda node {}!", name);
                        }
                    }
                }
            }
        }

        queue.submit(Some(encoder.finish()));

        PresentableRenderGraph {
        }
    }

    fn transition_resources(
        encoder: &mut wgpu::CommandEncoder,
        resources: &Vec<ResourceStorage>,
        resources_to_transition: impl Iterator<Item = (GraphResourceId, GraphResourceAccess)>,
    ) {
        let mut buffer_transitions: SmallVec<[wgpu::BufferTransition<&Buffer>; 8]> = SmallVec::new();
        let mut texture_transitions: SmallVec<[wgpu::TextureTransition<&Texture>; 8]> = SmallVec::new();

        let mut add_buffer_transition = |next_state, buffer, state_tracker: &ResourceStateTracker<BufferState>| {
            if state_tracker.should_transition_to(next_state, true) {
                buffer_transitions.push(wgpu::BufferTransition {
                    buffer,
                    state: next_state,
                });
                state_tracker.transition_to(next_state);
            }
        };

        let mut add_texture_transition = |next_state, texture, state_tracker: &ResourceStateTracker<TextureState>| {
            if state_tracker.should_transition_to(next_state, true) {
                texture_transitions.push(wgpu::TextureTransition {
                    texture,
                    selector: None,
                    state: next_state,
                });
                state_tracker.transition_to(next_state);
            }
        };

        for (id, access) in resources_to_transition {
            let storage = utility::resource_storage_ref(resources, id);

            match access {
                GraphResourceAccess::Buffer(next_state) => {
                    match storage {
                        ResourceStorage::ManagedBuffer { resource, state_tracker, .. } => {
                            add_buffer_transition(next_state, &*resource, state_tracker);
                        }
                        ResourceStorage::ImportedBuffer { resource, state_tracker, .. } => {
                            add_buffer_transition(next_state, &*resource, state_tracker);
                        }
                        _ =>  {
                            unreachable!("Resource[{}] is a texture, but a non-texture state[{:?}] is provided when read/write!", storage.name(), next_state)
                        }
                    }
                }
                GraphResourceAccess::Texture(next_state) => {
                    match storage {
                        ResourceStorage::ManagedTexture { resource, state_tracker, .. } => {
                            add_texture_transition(next_state, &*resource, state_tracker);
                        }
                        ResourceStorage::ImportedTexture { resource, state_tracker, .. } => {
                            add_texture_transition(next_state, &*resource, state_tracker);
                        }
                        _ => {
                            unreachable!("Resource[{}] is a buffer, but a non-buffer state[{:?}] is provided when read/write!", storage.name(), next_state)
                        }
                    }
                }
            }
        }

        encoder.transition_resources(
            buffer_transitions.into_iter(),
            texture_transitions.into_iter()
        );
    }
}

pub struct GraphicNodeExecutionContext<'node> {
    name: &'node str,
    pipeline_desc: &'node GraphicPipelineDescriptor,
    device: &'node wgpu::Device,
    queue: &'node wgpu::Queue,
    resources: &'node Vec<ResourceStorage>,
    pipeline: wgpu::RenderPipeline,
}

impl<'node> GraphicNodeExecutionContext<'node> {
    #[inline]
    pub fn get_buffer<V: GraphResourceView>(&mut self, resource: &RenderGraphResourceAccess<Buffer, V>) -> Buffer {
        self.resources.get(resource.id as usize).expect("Graph resource index out of bound!").as_buffer().clone()
    }

    #[inline]
    pub fn get_texture<V: GraphResourceView>(&mut self, resource: &RenderGraphResourceAccess<Texture, V>) -> Texture {
        self.resources.get(resource.id as usize).expect("Graph resource index out of bound!").as_texture().clone()
    }

    #[inline]
    pub fn write_buffer<V: GraphResourceView, T: NoUninit>(&mut self, resource: &RenderGraphResourceAccess<Buffer, V>, offset: wgpu::BufferAddress, data: T) {
        let buffer = self.resources.get(resource.id as usize).expect("Graph resource index out of bound!").as_buffer();
        debug_assert_eq!(buffer.size() as usize - offset as usize, size_of::<T>());
        self.queue.write_buffer(buffer, offset, bytemuck::cast_slice(&[data]));
    }

    #[inline]
    pub fn bind_pipeline<'ctx, 'rp>(&'ctx mut self, render_pass: &'ctx mut wgpu::RenderPass<'rp>) -> PipelineBinder<'ctx, 'rp> {
        render_pass.set_pipeline(&self.pipeline);
        PipelineBinder {
            device: &self.device,
            render_pass,
            pipeline: &self.pipeline,
            pipeline_desc: &self.pipeline_desc,
            bind_group_entries: vec![],
        }
    }

    pub fn begin_render_pass<'encoder>(
        &mut self,
        encoder: &'encoder mut wgpu::CommandEncoder
    ) -> wgpu::RenderPass<'encoder> {
        let create_texture_view = |id| {
            let storage = utility::resource_storage_ref(self.resources, id);

            match storage {
                ResourceStorage::ManagedTexture { resource, .. } => {
                    resource.create_view(&wgpu::TextureViewDescriptor::default())
                }
                ResourceStorage::ImportedTexture { resource, .. } => {
                    resource.create_view(&wgpu::TextureViewDescriptor::default())
                }
                _ => unreachable!()
            }
        };

        // TODO: use iterator-valid container
        let color_views = self.pipeline_desc.color_attachments
            .iter()
            .map(|(res, _)| res.id)
            .map(create_texture_view)
            .collect::<SmallVec<[wgpu::TextureView; 8]>>();
        let depth_view = self.pipeline_desc.depth_stencil_attachment
            .as_ref()
            .map(|(res, _)| res.id)
            .map(create_texture_view);

        let (color_attachments, depth_stencil_attachment) = (
            self.pipeline_desc.color_attachments
                .iter()
                .zip(color_views.iter())
                .map(|((_, info), view)| {
                    Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: info.load_op,
                            store: info.store_op,
                        }
                    })
                })
                .collect::<SmallVec<[Option<wgpu::RenderPassColorAttachment>; 8]>>(),
            depth_view.as_ref().zip(self.pipeline_desc.depth_stencil_attachment.as_ref()).map(|(view, (_, depth_info))| {
                wgpu::RenderPassDepthStencilAttachment {
                    view: &view,
                    depth_ops: if depth_info.depth_write {
                        Some(wgpu::Operations {
                            load: depth_info.depth_load_op,
                            store: depth_info.depth_store_op,
                        })
                    } else {
                        None
                    },
                    stencil_ops: Some(wgpu::Operations {
                        load: depth_info.stencil_load_op,
                        store: depth_info.stencil_store_op,
                    })
                }
            })
        );

        encoder.begin_render_pass(
            &wgpu::RenderPassDescriptor {
                label: Some(self.name),
                color_attachments: &color_attachments,
                depth_stencil_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
            }
        )
    }
}

pub struct PipelineBinder<'ctx, 'rp> {
    device: &'ctx wgpu::Device,
    render_pass: &'ctx mut wgpu::RenderPass<'rp>,
    pipeline_desc: &'ctx GraphicPipelineDescriptor,
    pipeline: &'ctx wgpu::RenderPipeline,
    bind_group_entries: Vec<Vec<wgpu::BindGroupEntry<'ctx>>>,
}

impl<'ctx, 'rp> PipelineBinder<'ctx, 'rp> {
    pub fn with_binding(mut self, group: u32, binding: u32, resource: wgpu::BindingResource<'ctx>) -> Self {
        let shader = self.pipeline_desc.shader.as_ref().unwrap();
        debug_assert!(group < shader.num_bind_groups() as u32, "Invalid group index: {}, shader[{}] only have {} bind group(s)", group, shader.name(), shader.num_bind_groups());
        debug_assert!(binding < shader.num_bindings(group).unwrap() as u32, "Invalid binding index: {}, shader[{}] only have {} bind entry(s)", group, shader.name(), shader.num_bindings(group).unwrap());

        let non_allocated_groups = group as i32 - self.bind_group_entries.len() as i32 + 1;
        for _ in 0..non_allocated_groups {
            self.bind_group_entries.push(vec![]);
        }

        let bindings = self.bind_group_entries.get_mut(group as usize).unwrap();
        bindings.push(wgpu::BindGroupEntry {
            binding,
            resource,
        });

        self
    }

    pub fn bind(self) {
        let shader = self.pipeline_desc.shader.as_ref().unwrap();

        let bind_groups = self.bind_group_entries
            .into_iter()
            .enumerate()
            .map(|(group, group_entries)| {
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("{} BindGroup{}", shader.name(), group)),
                    layout: &shader.create_bind_group_layout(self.device, group as u32).unwrap(),
                    entries: &group_entries,
                }))
            })
            .collect::<SmallVec<[Option<wgpu::BindGroup>; 4]>>();

        self.render_pass.set_pipeline(self.pipeline);
        for (group, bind_group) in bind_groups.into_iter().enumerate() {
            self.render_pass.set_bind_group(group as u32, &bind_group, &[]);
        }
    }
}

pub struct LambdaNodeExecutionContext<'node> {
    queue: &'node wgpu::Queue,
    resources: &'node Vec<ResourceStorage>,
}

impl<'node> LambdaNodeExecutionContext<'node> {
    #[inline]
    #[allow(dead_code)]
    pub fn get_buffer<V: GraphResourceView>(&mut self, resource: &RenderGraphResourceAccess<Buffer, V>) -> Buffer {
        self.resources.get(resource.id as usize).expect("Graph resource index out of bound!").as_buffer().clone()
    }

    #[inline]
    #[allow(dead_code)]
    pub fn get_texture<V: GraphResourceView>(&mut self, resource: &RenderGraphResourceAccess<Texture, V>) -> Texture {
        self.resources.get(resource.id as usize).expect("Graph resource index out of bound!").as_texture().clone()
    }

    #[inline]
    #[allow(dead_code)]
    pub fn write_buffer<V: GraphResourceView>(&mut self, resource: &RenderGraphResourceAccess<Buffer, V>, offset: wgpu::BufferAddress, data: &[u8]) {
        let buffer = self.resources.get(resource.id as usize).expect("Graph resource index out of bound!").as_buffer();
        self.queue.write_buffer(buffer, offset, data);
    }
}

pub struct PresentableRenderGraph {}

impl PresentableRenderGraph {
    #[profiling::function]
    pub fn present(self, present_surface: wgpu::SurfaceTexture) -> Result<(), Box<anyhow::Error>> {
        present_surface.present();

        Ok(())
    }
}

pub(crate) mod utility {
    use crate::graph::ResourceStorage;
    use crate::resource::GraphResourceId;

    #[inline]
    pub(crate) fn resource_storage_ref(storage: &Vec<ResourceStorage>, id: GraphResourceId) -> &ResourceStorage {
        storage.get(id as usize).expect("Graph resource id out of bound!")
    }
}