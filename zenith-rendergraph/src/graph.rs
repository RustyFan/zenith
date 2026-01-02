//! Render graph execution and resource management.

use crate::interface::{Buffer, BufferState, ResourceState, Texture, TextureState};
use crate::node::{NodePipelineState, RenderGraphNode};
use crate::resource::{
    GraphResourceId, GraphResourceView, RenderGraphResourceAccess,
};
use crate::GraphicPipelineDescriptor;
use std::cell::Cell;
use std::sync::Arc;
use zenith_core::collections::SmallVec;
use zenith_rhi::{CommandEncoder};
use zenith_rhi::{buffer_barrier, texture_barrier, vk, ColorBlendAttachment, GraphicPipeline, GraphicPipelineKey, PipelineCache, RenderDevice, ShaderResourceBinder, Swapchain, VertexAttribute, VertexBinding};

pub(crate) enum ResourceStorage {
    ManagedBuffer {
        name: String,
        resource: Buffer,
        state_tracker: BufferStateTracker,
    },
    ManagedTexture {
        name: String,
        resource: Texture,
        state_tracker: TextureStateTracker,
    },
    ImportedBuffer {
        name: String,
        resource: Arc<Buffer>,
        state_tracker: BufferStateTracker,
    },
    ImportedTexture {
        name: String,
        resource: Arc<Texture>,
        state_tracker: TextureStateTracker,
    },
}

impl ResourceStorage {
    pub(crate) fn as_buffer(&self) -> &Buffer {
        match self {
            ResourceStorage::ManagedBuffer { resource, .. } => resource,
            ResourceStorage::ImportedBuffer { resource, .. } => resource,
            _ => unreachable!("Expected buffer, but resource is a texture!"),
        }
    }

    pub(crate) fn as_texture(&self) -> &Texture {
        match self {
            ResourceStorage::ManagedTexture { resource, .. } => resource,
            ResourceStorage::ImportedTexture { resource, .. } => resource,
            _ => unreachable!("Expected texture, but resource is a buffer!"),
        }
    }
}

pub(crate) struct BufferStateTracker {
    current_access: Cell<BufferState>,
    current_stage: Cell<vk::PipelineStageFlags2>,
}

impl BufferStateTracker {
    pub(crate) fn new(access: BufferState) -> Self {
        Self {
            current_access: Cell::new(access),
            current_stage: Cell::new(vk::PipelineStageFlags2::NONE),
        }
    }

    pub(crate) fn current_access(&self) -> BufferState {
        self.current_access.get()
    }

    pub(crate) fn current_stage(&self) -> vk::PipelineStageFlags2 {
        self.current_stage.get()
    }

    pub(crate) fn transition_to(&self, next_access: BufferState, next_stage: vk::PipelineStageFlags2) {
        self.current_access.set(next_access);
        self.current_stage.set(next_stage);
    }
}

pub(crate) struct TextureStateTracker {
    current_access: Cell<TextureState>,
    current_stage: Cell<vk::PipelineStageFlags2>,
}

impl TextureStateTracker {
    pub(crate) fn new(state: TextureState) -> Self {
        Self {
            current_access: Cell::new(state),
            current_stage: Cell::new(vk::PipelineStageFlags2::NONE),
        }
    }

    pub(crate) fn current_access(&self) -> TextureState {
        self.current_access.get()
    }

    pub(crate) fn current_stage(&self) -> vk::PipelineStageFlags2 {
        self.current_stage.get()
    }

    pub(crate) fn transition_to(&self, next_state: TextureState, next_stage: vk::PipelineStageFlags2) {
        self.current_access.set(next_state);
        self.current_stage.set(next_stage);
    }
}

use crate::resource::InitialResourceStorage;

pub struct RenderGraph {
    pub(crate) nodes: Vec<RenderGraphNode>,
    pub(crate) initial_resources: Vec<InitialResourceStorage>,
}

impl RenderGraph {
    #[profiling::function]
    pub fn compile(
        mut self,
        device: &RenderDevice,
        pipeline_cache: &mut PipelineCache,
    ) -> CompiledRenderGraph {
        // Create resources from initial resource descriptors
        let resources: Vec<ResourceStorage> = self.initial_resources
            .into_iter()
            .map(|res| {
                match res {
                    InitialResourceStorage::ManagedBuffer(name, desc) => {
                        let resource = Buffer::from_desc(
                            device.handle(),
                            device.memory_properties(),
                            &desc
                        ).expect("Failed to create buffer");
                        ResourceStorage::ManagedBuffer {
                            name,
                            resource,
                            state_tracker: BufferStateTracker::new(BufferState::Undefined),
                        }
                    }
                    InitialResourceStorage::ManagedTexture(name, desc) => {
                        let resource = Texture::from_desc(
                            device.handle(),
                            device.memory_properties(),
                            &desc
                        ).expect("Failed to create texture");
                        ResourceStorage::ManagedTexture {
                            name,
                            resource,
                            state_tracker: TextureStateTracker::new(TextureState::Undefined),
                        }
                    }
                    InitialResourceStorage::ImportedBuffer(name, buffer, initial_state) => ResourceStorage::ImportedBuffer {
                        name,
                        resource: buffer.clone(),
                        state_tracker: BufferStateTracker::new(initial_state),
                    },
                    InitialResourceStorage::ImportedTexture(name, tex, initial_state) => ResourceStorage::ImportedTexture {
                        name,
                        resource: tex.clone(),
                        state_tracker: TextureStateTracker::new(initial_state),
                    },
                }
            })
            .collect();

        let mut graphic_pipelines = vec![];

        for node in &self.nodes {
            if !node.pipeline_state.valid() {
                log::warn!("Incomplete information for render graph node [{}]. Skipped.", node.name);
                continue;
            }

            match &node.pipeline_state {
                NodePipelineState::Graphic { pipeline_desc, .. } => {
                    if pipeline_desc.valid() {
                        let pipeline = Self::create_graphic_pipeline(&resources, pipeline_cache, pipeline_desc);
                        graphic_pipelines.push(Some(pipeline));
                    } else {
                        graphic_pipelines.push(None);
                    }
                }
                NodePipelineState::Compute { .. } => unimplemented!(),
                NodePipelineState::Lambda { .. } => {}
            }
        }

        let mut swapchain_tex_id = GraphResourceId::MAX;
        // find the first present node (i.e. first node which outputs to swapchain texture)
        let first_present_node_index = self.nodes.iter()
            .position(|node| {
                node.outputs.iter()
                    .filter_map(|output| {
                        let res = utility::resource_storage_ref(&resources, output.id);
                        match res {
                            ResourceStorage::ImportedTexture { resource, .. } => {
                                Some((resource.as_ref(), output.id))
                            }
                            _ => None,
                        }
                    })
                    .any(|(tex, id)| {
                        swapchain_tex_id = id;
                        tex.is_swapchain_texture()
                    })
            });

        let (serial_nodes, present_nodes) = if let Some(present_node_index) = first_present_node_index {
            (self.nodes.drain(0..present_node_index).collect(), self.nodes)
        } else {
            (self.nodes, vec![])
        };

        CompiledRenderGraph {
            serial_nodes,
            present_nodes,
            resources,
            graphic_pipe_index: 0,
            graphic_pipelines,
            swapchain_tex_id,
        }
    }

    #[profiling::function]
    fn create_graphic_pipeline(
        resources: &Vec<ResourceStorage>,
        pipeline_cache: &mut PipelineCache,
        desc: &GraphicPipelineDescriptor,
    ) -> Arc<GraphicPipeline> {
        let vertex_shader = desc.vertex_shader.as_ref();

        let color_attachments: Vec<ColorBlendAttachment> = desc
            .color_attachments
            .iter()
            .map(|(_, color_info)| ColorBlendAttachment {
                blend_enable: color_info.blend_enable,
                src_color: color_info.src_color_blend,
                dst_color: color_info.dst_color_blend,
                color_op: color_info.color_blend_op,
                src_alpha: color_info.src_alpha_blend,
                dst_alpha: color_info.dst_alpha_blend,
                alpha_op: color_info.alpha_blend_op,
                write_mask: color_info.write_mask,
            })
            .collect();

        let color_formats: Vec<vk::Format> = desc
            .color_attachments
            .iter()
            .map(|(resource, _)| {
                utility::resource_storage_ref(resources, resource.id).as_texture().format()
            })
            .collect();

        let depth_format = desc.depth_stencil_attachment.as_ref().map(|(resource, _)| {
            utility::resource_storage_ref(resources, resource.id).as_texture().format()
        });

        let depth_info = desc.depth_stencil_attachment.as_ref().map(|(_, info)| info);

        // Convert vertex bindings from descriptor
        let vertex_bindings: Vec<VertexBinding> = desc
            .vertex_bindings
            .iter()
            .map(|b| VertexBinding {
                binding: b.binding,
                stride: b.stride,
                input_rate: b.input_rate,
            })
            .collect();

        // Convert vertex attributes from descriptor
        let vertex_attributes: Vec<VertexAttribute> = desc
            .vertex_attributes
            .iter()
            .map(|a| VertexAttribute {
                location: a.location,
                binding: a.binding,
                format: a.format,
                offset: a.offset,
            })
            .collect();

        let key = GraphicPipelineKey {
            vertex_shader: vertex_shader.unwrap().clone(),
            fragment_shader: desc.fragment_shader.clone(),
            vertex_bindings,
            vertex_attributes,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart: false,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_clamp: false,
            depth_bias_enable: false,
            depth_bias_constant: 0,
            depth_bias_slope: 0,
            line_width: f32::to_bits(1.0) as i32,
            samples: vk::SampleCountFlags::TYPE_1,
            sample_shading: false,
            min_sample_shading: 0,
            depth_test: depth_info.map(|d| d.depth_test_enable).unwrap_or(false),
            depth_write: depth_info.map(|d| d.depth_write_enable).unwrap_or(false),
            depth_compare_op: depth_info.map(|d| d.depth_compare_op).unwrap_or(vk::CompareOp::LESS),
            depth_bounds_test: false,
            stencil_test: depth_info.map(|d| d.stencil_test_enable).unwrap_or(false),
            stencil_front: Default::default(),
            stencil_back: Default::default(),
            color_attachments,
            blend_constants: [0; 4],
            dynamic_states: vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR],
            color_formats,
            depth_format,
            stencil_format: None,
            descriptor_set_layouts: desc.descriptor_set_layouts.iter().map(|l| l.handle()).collect(),
            push_constant_size: desc.merged_reflection.as_ref().map(|r| r.push_constant_size).unwrap_or(0),
        };

        pipeline_cache.get_or_create(&key).expect("Failed to create graphics pipeline")
    }
}

pub struct CompiledRenderGraph {
    serial_nodes: Vec<RenderGraphNode>,
    present_nodes: Vec<RenderGraphNode>,
    resources: Vec<ResourceStorage>,
    graphic_pipe_index: u32,
    graphic_pipelines: Vec<Option<Arc<GraphicPipeline>>>,
    swapchain_tex_id: GraphResourceId,
}

impl CompiledRenderGraph {
    #[profiling::function]
    pub fn execute(&mut self, device: &RenderDevice) {
        let cmd = device.execute_command_pool().allocate().expect("Failed to allocate command buffer");
        let encoder = CommandEncoder::new(device.handle(), cmd);
        encoder.begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT).unwrap();

        let nodes: Vec<RenderGraphNode> = self.serial_nodes.drain(..).collect();
        for node in nodes {
            match node.pipeline_state {
                NodePipelineState::Graphic { pipeline_desc, mut job_functor } => {
                    {
                        profiling::scope!("rendergraph::barriers");
                        // TODO: facilitate VkSubpassDependency2 for render targets barriers inside renderpass
                        let output_iter = node.outputs.iter()
                            .map(|res| (res.id, res.access, res.stage_hint));

                        Self::transition_resources(
                            device, &encoder, &self.resources,
                            node.inputs.iter()
                                .map(|res| (res.id, res.access, res.stage_hint))
                                .chain(output_iter),
                            Some(&pipeline_desc),
                        );
                    }

                    let name = node.name;
                    let pipeline = self.graphic_pipelines.get(self.graphic_pipe_index as usize).unwrap();
                    let pipeline = pipeline.as_ref().map(|pipe| pipe.as_ref());
                    self.graphic_pipe_index += 1;

                    if let Some(record) = job_functor.take() {
                        profiling::scope!("rendergraph::node_recording", &name);

                        let mut ctx = GraphicNodeExecutionContext {
                            pipeline_desc: &pipeline_desc,
                            device,
                            resources: &self.resources,
                            pipeline,
                            encoder: &encoder,
                        };
                        record(&mut ctx);
                    } else {
                        log::warn!("Missing job of graphic node {}!", name);
                    }
                }
                NodePipelineState::Compute { .. } => unimplemented!(),
                NodePipelineState::Lambda { mut job_functor } => {
                    {
                        profiling::scope!("rendergraph::barriers");
                        let output_iter = node.outputs.iter()
                            .map(|res| (res.id, res.access, res.stage_hint));

                        Self::transition_resources(
                            device, &encoder, &self.resources,
                            node.inputs.iter()
                                .map(|res| (res.id, res.access, res.stage_hint))
                                .chain(output_iter),
                            None,
                        );
                    }

                    let name = node.name;
                    if let Some(record) = job_functor.take() {
                        profiling::scope!("rendergraph::node_recording", &name);

                        let mut ctx = LambdaNodeExecutionContext { device, resources: &self.resources, encoder: &encoder };
                        record(&mut ctx);
                    } else {
                        log::warn!("Missing job of lambda node {}!", name);
                    }
                }
            }
        }

        encoder.end().unwrap();

        device.submit_commands(
            cmd,
            device.graphics_queue(),
            &[],
            vk::PipelineStageFlags2::NONE,
            &[],
            vk::PipelineStageFlags2::NONE,
            device.frame_resource_fence(),
        );
    }

    pub fn present(mut self, swapchain: &mut Swapchain, device: &mut RenderDevice) -> anyhow::Result<RetiredRenderGraph> {
        let (image_index, _) = swapchain.acquire_next_image(device.handle())?;
        swapchain.reset_current_fence(device.handle())?;

        // update the swapchain texture reference to the acquired image
        if self.swapchain_tex_id != GraphResourceId::MAX {
            let swapchain_tex = swapchain.swapchain_texture(image_index as usize);
            if let Some(ResourceStorage::ImportedTexture { resource, state_tracker, .. }) = self.resources.get_mut(self.swapchain_tex_id as usize) {
                *resource = swapchain_tex;
                // Reset state tracker since this is a newly acquired image
                *state_tracker = TextureStateTracker::new(TextureState::Undefined);
            }
        }

        device.reset_frame_resources();
        device.present_command_pool().reset()?;

        let cmd = device.present_command_pool().allocate().expect("Failed to allocate command buffer");
        let encoder = CommandEncoder::new(device.handle(), cmd);
        encoder.begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;

        for node in self.present_nodes {
            match node.pipeline_state {
                NodePipelineState::Graphic { pipeline_desc, mut job_functor } => {
                    {
                        profiling::scope!("rendergraph::barriers");
                        // TODO: facilitate VkSubpassDependency2 for render targets barriers inside renderpass
                        let output_iter = node.outputs.iter()
                            .map(|res| (res.id, res.access, res.stage_hint));

                        Self::transition_resources(
                            device, &encoder, &self.resources,
                            node.inputs.iter()
                                .map(|res| (res.id, res.access, res.stage_hint))
                                .chain(output_iter),
                            Some(&pipeline_desc),
                        );
                    }

                    let name = node.name;
                    let pipeline = self.graphic_pipelines.get(self.graphic_pipe_index as usize).unwrap();
                    let pipeline = pipeline.as_ref().map(|pipe| pipe.as_ref());
                    self.graphic_pipe_index += 1;

                    if let Some(record) = job_functor.take() {
                        profiling::scope!("rendergraph::node_recording", &name);

                        let mut ctx = GraphicNodeExecutionContext {
                            pipeline_desc: &pipeline_desc,
                            device,
                            resources: &self.resources,
                            pipeline,
                            encoder: &encoder,
                        };
                        record(&mut ctx);
                    } else {
                        log::warn!("Missing job of graphic node {}!", name);
                    }
                }
                NodePipelineState::Compute { .. } => unimplemented!(),
                NodePipelineState::Lambda { mut job_functor } => {
                    {
                        profiling::scope!("rendergraph::barriers");
                        let output_iter = node.outputs.iter()
                            .map(|res| (res.id, res.access, res.stage_hint));

                        Self::transition_resources(
                            device, &encoder, &self.resources,
                            node.inputs.iter()
                                .map(|res| (res.id, res.access, res.stage_hint))
                                .chain(output_iter),
                            None,
                        );
                    }

                    let name = node.name;
                    if let Some(record) = job_functor.take() {
                        profiling::scope!("rendergraph::node_recording", &name);

                        let mut ctx = LambdaNodeExecutionContext { device, resources: &self.resources, encoder: &encoder };
                        record(&mut ctx);
                    } else {
                        log::warn!("Missing job of lambda node {}!", name);
                    }
                }
            }
        }

        // make sure the swapchain texture has the right image layout for presentation
        Self::transition_resources(
            device, &encoder, &self.resources,
            [(self.swapchain_tex_id, TextureState::Present.into(), Some(vk::PipelineStageFlags2::BOTTOM_OF_PIPE))].into_iter(),
            None,
        );

        encoder.end()?;

        let frame_sync = swapchain.current_frame_sync();

        device.submit_commands(
            cmd,
            device.graphics_queue(),
            &[frame_sync.image_available],
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            &[frame_sync.render_finished],
            vk::PipelineStageFlags2::NONE,
            frame_sync.in_flight_fence,
        );

        swapchain.present(device.present_queue(), image_index)?;

        Ok(RetiredRenderGraph {
            resources: self.resources,
        })
    }

    fn transition_resources(
        device: &RenderDevice,
        encoder: &CommandEncoder,
        resource_storage: &Vec<ResourceStorage>,
        resources_to_transition: impl Iterator<Item = (GraphResourceId, ResourceState, Option<vk::PipelineStageFlags2>)>,
        pipeline_desc: Option<&GraphicPipelineDescriptor>,
    ) {
        let mut image_barriers: SmallVec<[vk::ImageMemoryBarrier2; 8]> = SmallVec::new();
        let mut buffer_barriers: SmallVec<[vk::BufferMemoryBarrier2; 8]> = SmallVec::new();

        // Get combined shader stages from all bindings in reflection
        let combined_shader_stage = pipeline_desc
            .and_then(|desc| desc.merged_reflection.as_ref())
            .map(|r| r.bindings.iter().fold(vk::ShaderStageFlags::empty(), |acc, b| acc | b.stage_flags))
            .map(shader_stage_to_pipeline_stage)
            .unwrap_or(vk::PipelineStageFlags2::ALL_COMMANDS);

        let mut add_buffer_barrier = |resource: &Buffer, access: ResourceState, state_tracker: &BufferStateTracker, stage_hint: Option<vk::PipelineStageFlags2>, name: &String| {
            let ResourceState::Buffer(next_state) = access else {
                return;
            };

            let prev_state = state_tracker.current_access();
            if prev_state == next_state {
                return;
            }

            let dst_stage = stage_hint.unwrap_or(combined_shader_stage);
            if dst_stage == vk::PipelineStageFlags2::ALL_COMMANDS {
                log::warn!("Render graph resource [{}] may cause serve pipeline stall due to unknown pipeline stage usage. Use read_hint() or write_hint() to get better performance.",
                name)
            }

            buffer_barriers.push(buffer_barrier(
                resource,
                state_tracker.current_stage(),
                prev_state,
                dst_stage,
                next_state,
                device.graphics_queue_family(),
                device.graphics_queue_family(),
                false,
            ));
            state_tracker.transition_to(next_state, next_state.into_pipeline_stage(dst_stage));
        };

        let mut add_texture_barrier = |resource: &Texture, access: ResourceState, state_tracker: &TextureStateTracker, stage_hint: Option<vk::PipelineStageFlags2>, name: &String| {
            let ResourceState::Texture(next_state) = access else {
                return;
            };

            let prev_state = state_tracker.current_access();
            if prev_state == next_state {
                return;
            }

            let dst_stage = stage_hint.unwrap_or(combined_shader_stage);
            if dst_stage == vk::PipelineStageFlags2::ALL_COMMANDS {
                log::warn!(r#"
                Render graph resource [{}] may cause serve pipeline stall due to unknown pipeline stage usage.
                Use read_hint() or write_hint() to get better performance."#,
                name)
            }

            image_barriers.push(texture_barrier(
                resource,
                state_tracker.current_stage(),
                prev_state,
                dst_stage,
                next_state,
                device.graphics_queue_family(),
                device.graphics_queue_family(),
                false,
                prev_state == TextureState::Undefined,
            ));
            state_tracker.transition_to(next_state, next_state.into_pipeline_stage(dst_stage));
        };

        for (id, access, stage_hint) in resources_to_transition {
            let storage = utility::resource_storage_ref(resource_storage, id);

            match storage {
                ResourceStorage::ManagedBuffer { resource, state_tracker, name } => {
                    add_buffer_barrier(resource, access, state_tracker, stage_hint, name);
                }
                ResourceStorage::ImportedBuffer { resource, state_tracker, name } => {
                    add_buffer_barrier(resource, access, state_tracker, stage_hint, name);
                }
                ResourceStorage::ManagedTexture { resource, state_tracker, name } => {
                    add_texture_barrier(resource, access, state_tracker, stage_hint, name);
                }
                ResourceStorage::ImportedTexture { resource, state_tracker, name } => {
                    add_texture_barrier(resource, access, state_tracker, stage_hint, name);
                }
            }
        }

        if !image_barriers.is_empty() || !buffer_barriers.is_empty() {
            let dependency_info = vk::DependencyInfo::default()
                .image_memory_barriers(&image_barriers)
                .buffer_memory_barriers(&buffer_barriers);

            encoder.pipeline_barrier(&dependency_info);
        }
    }
}

/// Convert shader stage flags to pipeline stage flags.
fn shader_stage_to_pipeline_stage(stage_flags: vk::ShaderStageFlags) -> vk::PipelineStageFlags2 {
    let mut result = vk::PipelineStageFlags2::NONE;
    if stage_flags.contains(vk::ShaderStageFlags::VERTEX) {
        result |= vk::PipelineStageFlags2::VERTEX_SHADER;
    }
    if stage_flags.contains(vk::ShaderStageFlags::FRAGMENT) {
        result |= vk::PipelineStageFlags2::FRAGMENT_SHADER;
    }
    if stage_flags.contains(vk::ShaderStageFlags::COMPUTE) {
        result |= vk::PipelineStageFlags2::COMPUTE_SHADER;
    }
    if stage_flags.contains(vk::ShaderStageFlags::GEOMETRY) {
        result |= vk::PipelineStageFlags2::GEOMETRY_SHADER;
    }
    if stage_flags.contains(vk::ShaderStageFlags::TESSELLATION_CONTROL) {
        result |= vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER;
    }
    if stage_flags.contains(vk::ShaderStageFlags::TESSELLATION_EVALUATION) {
        result |= vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER;
    }
    result
}

pub struct GraphicNodeExecutionContext<'node> {
    pipeline_desc: &'node GraphicPipelineDescriptor,
    device: &'node RenderDevice,
    resources: &'node Vec<ResourceStorage>,
    pipeline: Option<&'node GraphicPipeline>,
    encoder: &'node CommandEncoder<'node>,
}

impl<'node> GraphicNodeExecutionContext<'node> {
    #[inline]
    pub fn get_buffer<V: GraphResourceView>(&self, resource: &RenderGraphResourceAccess<Buffer, V>) -> &Buffer {
        self.resources.get(resource.id as usize).expect("Graph resource index out of bound!").as_buffer()
    }

    #[inline]
    pub fn get_texture<V: GraphResourceView>(&self, resource: &RenderGraphResourceAccess<Texture, V>) -> &Texture {
        self.resources.get(resource.id as usize).expect("Graph resource index out of bound!").as_texture()
    }

    pub fn bind_pipeline(&self) {
        if let Some(pipeline) = self.pipeline {
            self.encoder.bind_graphics_pipeline(pipeline.pipeline());
        }
    }

    #[inline]
    pub fn device(&self) -> &RenderDevice { self.device }

    #[inline]
    pub fn encoder(&self) -> &CommandEncoder<'node> { self.encoder }

    pub fn begin_rendering(&self, extent: vk::Extent2D) {
        let color_attachments: SmallVec<[vk::RenderingAttachmentInfo; 8]> = self.pipeline_desc.color_attachments.iter()
            .map(|(resource, color_info)| {
                let texture = utility::resource_storage_ref(self.resources, resource.id).as_texture();
                vk::RenderingAttachmentInfo::default()
                    .image_view(texture.view().expect("Texture view not created"))
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(color_info.load_op)
                    .store_op(color_info.store_op)
                    .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: color_info.clear_value } })
            })
            .collect();

        let depth_attachment = self.pipeline_desc.depth_stencil_attachment.as_ref().map(|(resource, depth_info)| {
            let texture = utility::resource_storage_ref(self.resources, resource.id).as_texture();
            vk::RenderingAttachmentInfo::default()
                .image_view(texture.view().expect("Texture view not created"))
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(depth_info.depth_load_op)
                .store_op(depth_info.depth_store_op)
                .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: depth_info.depth_clear_value, stencil: depth_info.stencil_clear_value } })
        });

        let mut rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
            .layer_count(1)
            .color_attachments(&color_attachments);

        if let Some(ref depth) = depth_attachment {
            rendering_info = rendering_info.depth_attachment(depth);
        }

        self.encoder.begin_rendering(&rendering_info);
    }

    pub fn end_rendering(&self) {
        self.encoder.end_rendering();
    }

    /// Create a shader resource binder for this node's pipeline.
    /// Returns None if the pipeline has no descriptor bindings.
    pub fn create_resource_binder(&self) -> Option<ShaderResourceBinder<'_>> {
        let reflection = self.pipeline_desc.merged_reflection.as_ref()?;
        let layouts = &self.pipeline_desc.descriptor_set_layouts;

        if layouts.is_empty() {
            return None;
        }

        ShaderResourceBinder::new(
            self.device.handle(),
            reflection,
            layouts,
            self.device.descriptor_pool(),
        ).ok()
    }

    /// Bind descriptor sets to the pipeline.
    pub fn bind_descriptor_sets(&self, sets: &[vk::DescriptorSet]) {
        if let Some(pipeline) = self.pipeline {
            if !sets.is_empty() {
                self.encoder.bind_descriptor_sets(
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.layout(),
                    0,
                    sets,
                    &[],
                );
            }
        }
    }
}

pub struct LambdaNodeExecutionContext<'node> {
    device: &'node RenderDevice,
    resources: &'node Vec<ResourceStorage>,
    encoder: &'node CommandEncoder<'node>,
}

impl<'node> LambdaNodeExecutionContext<'node> {
    #[inline]
    #[allow(dead_code)]
    pub fn get_buffer<V: GraphResourceView>(&self, resource: &RenderGraphResourceAccess<Buffer, V>) -> &Buffer {
        self.resources.get(resource.id as usize).expect("Graph resource index out of bound!").as_buffer()
    }

    #[inline]
    #[allow(dead_code)]
    pub fn get_texture<V: GraphResourceView>(&self, resource: &RenderGraphResourceAccess<Texture, V>) -> &Texture {
        self.resources.get(resource.id as usize).expect("Graph resource index out of bound!").as_texture()
    }

    #[inline]
    pub fn device(&self) -> &RenderDevice { self.device }

    #[inline]
    pub fn command_encoder(&self) -> &CommandEncoder<'node> { self.encoder }
}

pub struct RetiredRenderGraph {
    resources: Vec<ResourceStorage>,
}

impl RetiredRenderGraph {
    pub fn release_frame_resources(self, device: &mut RenderDevice) {
        for resource in self.resources.into_iter() {
            match resource {
                ResourceStorage::ManagedBuffer { resource, .. } => {
                    device.defer_release_buffer(resource);
                }
                ResourceStorage::ManagedTexture { resource, .. } => {
                    device.defer_release_texture(resource);
                }
                _ => {}
            }
        }
    }
}

pub(crate) mod utility {
    use super::ResourceStorage;
    use crate::resource::GraphResourceId;

    #[inline]
    pub(crate) fn resource_storage_ref(storage: &Vec<ResourceStorage>, id: GraphResourceId) -> &ResourceStorage {
        storage.get(id as usize).expect("Graph resource id out of bound!")
    }
}
