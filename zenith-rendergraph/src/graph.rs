//! Render graph execution and resource management.

use crate::interface::{Buffer, BufferState, ResourceState, Texture, TextureState};
use crate::node::{NodePipelineState, RenderGraphNode};
use crate::resource::{
    GraphResourceId, GraphResourceView, RenderGraphResourceAccess,
};
use crate::GraphicPipelineDescriptor;
use anyhow::anyhow;
use std::cell::Cell;
use std::sync::Arc;
use zenith_core::collections::SmallVec;
use zenith_rhi::swapchain::SwapchainWindow;
use zenith_rhi::{CommandPool, Semaphore};
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
    pub(crate) fn name(&self) -> &str {
        match self {
            ResourceStorage::ManagedBuffer { name, .. } => name,
            ResourceStorage::ManagedTexture { name, .. } => name,
            ResourceStorage::ImportedBuffer { name, .. } => name,
            ResourceStorage::ImportedTexture { name, .. } => name,
        }
    }

    pub(crate) fn is_buffer(&self) -> bool {
        match self {
            ResourceStorage::ManagedBuffer { .. } | ResourceStorage::ImportedBuffer { .. } => true,
            _ => false,
        }
    }

    pub(crate) fn is_texture(&self) -> bool {
        match self {
            ResourceStorage::ManagedTexture { .. } | ResourceStorage::ImportedTexture { .. } => true,
            _ => false,
        }
    }


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
}

impl BufferStateTracker {
    pub(crate) fn new(access: BufferState) -> Self {
        Self { current_access: Cell::new(access) }
    }

    pub(crate) fn current(&self) -> BufferState {
        self.current_access.get()
    }

    pub(crate) fn transition_to(&self, next_access: BufferState) {
        self.current_access.set(next_access);
    }
}

pub(crate) struct TextureStateTracker {
    current_access: Cell<TextureState>,
}

impl TextureStateTracker {
    pub(crate) fn new(state: TextureState) -> Self {
        Self { current_access: Cell::new(state) }
    }

    pub(crate) fn current(&self) -> TextureState {
        self.current_access.get()
    }

    pub(crate) fn transition_to(&self, next_state: TextureState) {
        self.current_access.set(next_state);
    }
}

pub struct RenderGraph {
    pub(crate) nodes: Vec<RenderGraphNode>,
    pub(crate) resources: Vec<ResourceStorage>,
}

impl RenderGraph {
    pub fn validate(&self) {
        // TODO: Validate graph structure
    }

    #[profiling::function]
    pub fn compile(
        mut self,
        device: &RenderDevice,
        pipeline_cache: &mut PipelineCache,
    ) -> CompiledRenderGraph {
        let mut graphic_pipelines = vec![];

        for node in &self.nodes {
            match &node.pipeline_state {
                NodePipelineState::Graphic { pipeline_desc, .. } => {
                    if pipeline_desc.valid() {
                        let pipeline = self.create_graphic_pipeline(pipeline_cache, pipeline_desc);
                        graphic_pipelines.push(Some(pipeline));
                    } else {
                        graphic_pipelines.push(None);
                    }
                }
                NodePipelineState::Compute { .. } => unimplemented!(),
                NodePipelineState::Lambda { .. } => {}
            }
        }

        // find the first present node (i.e. first node which outputs to swapchain texture)
        let first_present_node_index = self.nodes.iter()
            .position(|node| {
                node.outputs.iter()
                    .filter_map(|output| {
                        let res = utility::resource_storage_ref(&self.resources, output.id);
                        match res {
                            ResourceStorage::ManagedTexture { resource, .. }  => {
                                Some(resource)
                            }
                            ResourceStorage::ImportedTexture { resource, .. } => {
                                Some(resource.as_ref())
                            }
                            _ => None,
                        }
                    })
                    .any(|tex| tex.is_swapchain_texture())
            });

        let (serial_nodes, present_nodes) = if let Some(present_node_index) = first_present_node_index {
            (self.nodes.drain(0..present_node_index).collect(), self.nodes)
        } else {
            (self.nodes, vec![])
        };

        CompiledRenderGraph {
            serial_nodes,
            present_nodes,
            resources: self.resources,
            graphic_pipe_index: 0,
            graphic_pipelines,
        }
    }

    #[profiling::function]
    fn create_graphic_pipeline(
        &self,
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
                utility::resource_storage_ref(&self.resources, resource.id).as_texture().format()
            })
            .collect();

        let depth_format = desc.depth_stencil_attachment.as_ref().map(|(resource, _)| {
            utility::resource_storage_ref(&self.resources, resource.id).as_texture().format()
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
}

impl CompiledRenderGraph {
    #[profiling::function]
    pub fn execute(&mut self, device: &RenderDevice) {
        let cmd = device.execute_command_pool().allocate().expect("Failed to allocate command buffer");
        unsafe {
            device.handle().begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            ).unwrap();
        }

        let nodes: Vec<RenderGraphNode> = self.serial_nodes.drain(..).collect();
        for node in nodes {
            match node.pipeline_state {
                NodePipelineState::Graphic { pipeline_desc, mut job_functor } => {
                    {
                        profiling::scope!("rendergraph::barriers");
                        // TODO: facilitate VkSubpassDependency2 for render targets barriers inside renderpass
                        let output_iter = node.outputs.iter()
                            .map(|a| (a.id, a.access));

                        Self::transition_resources(
                            device, cmd, &self.resources,
                            node.inputs.iter()
                                .map(|a| (a.id, a.access))
                                .chain(output_iter),
                        );
                    }

                    let name = node.name;
                    let pipeline = self.graphic_pipelines.get(self.graphic_pipe_index as usize).unwrap();
                    let pipeline = pipeline.as_ref().map(|pipe| pipe.as_ref());
                    self.graphic_pipe_index += 1;

                    if let Some(record) = job_functor.take() {
                        profiling::scope!("rendergraph::node_recording", &name);

                        let mut ctx = GraphicNodeExecutionContext {
                            name: name.as_str(),
                            pipeline_desc: &pipeline_desc,
                            device,
                            resources: &self.resources,
                            pipeline,
                        };
                        record(&mut ctx, cmd);
                    } else {
                        log::warn!("Missing job of graphic node {}!", name);
                    }
                }
                NodePipelineState::Compute { .. } => unimplemented!(),
                NodePipelineState::Lambda { mut job_functor } => {
                    {
                        profiling::scope!("rendergraph::barriers");
                        let output_iter = node.outputs.iter()
                            .map(|a| (a.id, a.access));

                        Self::transition_resources(
                            device, cmd, &self.resources,
                            node.inputs.iter()
                                .map(|a| (a.id, a.access))
                                .chain(output_iter),
                        );
                    }

                    let name = node.name;
                    if let Some(record) = job_functor.take() {
                        profiling::scope!("rendergraph::node_recording", &name);

                        let mut ctx = LambdaNodeExecutionContext { device, resources: &self.resources };
                        record(&mut ctx, cmd);
                    } else {
                        log::warn!("Missing job of lambda node {}!", name);
                    }
                }
            }
        }

        unsafe {
            device.handle().end_command_buffer(cmd).unwrap();
        };

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

    // TODO: swapchain window should reference the swapchain
    pub fn present(mut self, swapchain: &mut Swapchain, device: &mut RenderDevice) -> anyhow::Result<RetiredRenderGraph> {
        let (image_index, _) = swapchain.acquire_next_image(device.handle())?;
        swapchain.reset_current_fence(device.handle())?;

        device.reset_frame_resources();
        device.present_command_pool().reset()?;

        let cmd = device.present_command_pool().allocate().expect("Failed to allocate command buffer");
        unsafe {
            device.handle().begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        for node in self.present_nodes {
            match node.pipeline_state {
                NodePipelineState::Graphic { pipeline_desc, mut job_functor } => {
                    {
                        profiling::scope!("rendergraph::barriers");
                        // TODO: facilitate VkSubpassDependency2 for render targets barriers inside renderpass
                        let output_iter = node.outputs.iter()
                            .map(|a| (a.id, a.access));

                        Self::transition_resources(
                            device, cmd, &self.resources,
                            node.inputs.iter()
                                .map(|a| (a.id, a.access))
                                .chain(output_iter),
                        );
                    }

                    let name = node.name;
                    let pipeline = self.graphic_pipelines.get(self.graphic_pipe_index as usize).unwrap();
                    let pipeline = pipeline.as_ref().map(|pipe| pipe.as_ref());
                    self.graphic_pipe_index += 1;

                    if let Some(record) = job_functor.take() {
                        profiling::scope!("rendergraph::node_recording", &name);

                        let mut ctx = GraphicNodeExecutionContext {
                            name: name.as_str(),
                            pipeline_desc: &pipeline_desc,
                            device,
                            resources: &self.resources,
                            pipeline,
                        };
                        record(&mut ctx, cmd);
                    } else {
                        log::warn!("Missing job of graphic node {}!", name);
                    }
                }
                NodePipelineState::Compute { .. } => unimplemented!(),
                NodePipelineState::Lambda { mut job_functor } => {
                    {
                        profiling::scope!("rendergraph::barriers");
                        let output_iter = node.outputs.iter()
                            .map(|a| (a.id, a.access));

                        Self::transition_resources(
                            device, cmd, &self.resources,
                            node.inputs.iter()
                                .map(|a| (a.id, a.access))
                                .chain(output_iter),
                        );
                    }

                    let name = node.name;
                    if let Some(record) = job_functor.take() {
                        profiling::scope!("rendergraph::node_recording", &name);

                        let mut ctx = LambdaNodeExecutionContext { device, resources: &self.resources };
                        record(&mut ctx, cmd);
                    } else {
                        log::warn!("Missing job of lambda node {}!", name);
                    }
                }
            }
        }

        // Transition swapchain image to present
        let present_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(swapchain.swapchain_texture(image_index as _).handle())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(std::slice::from_ref(&present_barrier));

        unsafe {
            device.handle().cmd_pipeline_barrier2(cmd, &dependency_info);
            device.handle().end_command_buffer(cmd)?;
        }

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
        cmd: vk::CommandBuffer,
        resources: &Vec<ResourceStorage>,
        resources_to_transition: impl Iterator<Item = (GraphResourceId, ResourceState)>,
    ) {
        let mut image_barriers: SmallVec<[vk::ImageMemoryBarrier2; 8]> = SmallVec::new();
        let mut buffer_barriers: SmallVec<[vk::BufferMemoryBarrier2; 8]> = SmallVec::new();

        let mut add_buffer_barrier_if_needed = |resource, access: ResourceState, state_tracker: &BufferStateTracker, name: &String| {
            let ResourceState::Buffer(next_state) = access else {
                log::warn!("Try to transition buffer [{}] into {:?}.", name, access);
                return;
            };

            let prev_state = state_tracker.current();
            if prev_state == next_state {
                return;
            }

            buffer_barriers.push(buffer_barrier(
                resource,
                vk::PipelineStageFlags2::NONE,
                prev_state,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                next_state,
                device.graphics_queue_family(),
                device.graphics_queue_family(),
                false,
            ));
            state_tracker.transition_to(next_state);
        };
        let mut add_texture_barrier_if_needed = |resource, access: ResourceState, state_tracker: &TextureStateTracker, name: &String| {
            let ResourceState::Texture(next_state) = access else {
                log::warn!("Try to transition texture [{}] into {:?}.", name, access);
                return;
            };

            let prev_state = state_tracker.current();
            if prev_state == next_state {
                return;
            }

            image_barriers.push(texture_barrier(
                resource,
                vk::PipelineStageFlags2::NONE,
                prev_state,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                next_state,
                device.graphics_queue_family(),
                device.graphics_queue_family(),
                false,
                false,
            ));
            state_tracker.transition_to(next_state);
        };

        for (id, access) in resources_to_transition {
            let storage = utility::resource_storage_ref(resources, id);

            match storage {
                ResourceStorage::ManagedBuffer { resource, state_tracker, name } => {
                    add_buffer_barrier_if_needed(resource, access, state_tracker, name);
                }
                ResourceStorage::ImportedBuffer { resource, state_tracker, name } => {
                    add_buffer_barrier_if_needed(resource, access, state_tracker, name);
                }
                ResourceStorage::ManagedTexture { resource, state_tracker, name } => {
                    add_texture_barrier_if_needed(resource, access, state_tracker, name);
                }
                ResourceStorage::ImportedTexture { resource, state_tracker, name } => {
                    add_texture_barrier_if_needed(resource, access, state_tracker, name);
                }
            }
        }

        if !image_barriers.is_empty() || !buffer_barriers.is_empty() {
            let dependency_info = vk::DependencyInfo::default()
                .image_memory_barriers(&image_barriers)
                .buffer_memory_barriers(&buffer_barriers);

            unsafe { device.handle().cmd_pipeline_barrier2(cmd, &dependency_info); }
        }
    }
}

pub struct GraphicNodeExecutionContext<'node> {
    name: &'node str,
    pipeline_desc: &'node GraphicPipelineDescriptor,
    device: &'node RenderDevice,
    resources: &'node Vec<ResourceStorage>,
    pipeline: Option<&'node GraphicPipeline>,
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

    pub fn bind_pipeline(&self, cmd: vk::CommandBuffer) {
        if let Some(pipeline) = self.pipeline {
            unsafe {
                self.device.handle().cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.pipeline(),
                );
            }
        }
    }

    #[inline]
    pub fn device(&self) -> &RenderDevice { self.device }

    pub fn begin_rendering(&self, cmd: vk::CommandBuffer, extent: vk::Extent2D) {
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

        unsafe { self.device.handle().cmd_begin_rendering(cmd, &rendering_info); }
    }

    pub fn end_rendering(&self, cmd: vk::CommandBuffer) {
        unsafe { self.device.handle().cmd_end_rendering(cmd); }
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
    pub fn bind_descriptor_sets(&self, cmd: vk::CommandBuffer, sets: &[vk::DescriptorSet]) {
        if let Some(pipeline) = self.pipeline {
            if !sets.is_empty() {
                unsafe {
                    self.device.handle().cmd_bind_descriptor_sets(
                        cmd,
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

    /// Get the pipeline layout for push constants.
    pub fn pipeline_layout(&self) -> Option<vk::PipelineLayout> {
        self.pipeline.map(|p| p.layout())
    }
}

pub struct LambdaNodeExecutionContext<'node> {
    device: &'node RenderDevice,
    resources: &'node Vec<ResourceStorage>,
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
