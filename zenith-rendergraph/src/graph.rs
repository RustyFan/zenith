//! Render graph execution and resource management.

use crate::interface::{Buffer, BufferState, ResourceState, Texture, TextureState};
use crate::node::{NodePipelineState, RenderGraphNode};
use crate::resource::{GraphResource, GraphResourceId, GraphResourceState, GraphResourceView, InitialResourceStorage, RenderGraphResourceAccess};
use std::cell::Cell;
use std::sync::Arc;
use zenith_core::collections::SmallVec;
use zenith_rhi::{CommandEncoder, BufferBarrier, TextureBarrier, PipelineStages, ShaderReflection};
use zenith_rhi::{
    vk, GraphicPipeline, GraphicPipelineDesc, PipelineCache, RenderDevice,
    DescriptorSetBinder, Swapchain,
};

pub enum ResourceStorage {
    ManagedBuffer {
        desc: zenith_rhi::BufferDesc,
        resource: Buffer,
        state_tracker: ResourceStateTracker<BufferState>,
    },
    ManagedTexture {
        desc: zenith_rhi::TextureDesc,
        resource: Texture,
        state_tracker: ResourceStateTracker<TextureState>,
    },
    ImportedBuffer {
        resource: Arc<Buffer>,
        state_tracker: ResourceStateTracker<BufferState>,
    },
    ImportedTexture {
        resource: Arc<Texture>,
        state_tracker: ResourceStateTracker<TextureState>,
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

pub struct ResourceStateTracker<S: GraphResourceState> {
    current_access: Cell<S>,
    current_stage: Cell<vk::PipelineStageFlags2>,
}

impl<S: GraphResourceState> ResourceStateTracker<S> {
    pub(crate) fn new(access: S) -> Self {
        Self {
            current_access: Cell::new(access),
            current_stage: Cell::new(vk::PipelineStageFlags2::NONE),
        }
    }

    pub(crate) fn current_access(&self) -> S {
        self.current_access.get()
    }

    pub(crate) fn current_stage(&self) -> vk::PipelineStageFlags2 {
        self.current_stage.get()
    }

    pub(crate) fn transition_to(&self, next_access: S, next_stage: vk::PipelineStageFlags2) {
        self.current_access.set(next_access);
        self.current_stage.set(next_stage);
    }
}

pub struct RenderGraph {
    pub(crate) nodes: Vec<RenderGraphNode>,
    pub(crate) initial_resources: Vec<InitialResourceStorage>,
}

impl RenderGraph {
    #[profiling::function]
    pub fn compile(
        mut self,
        device: &mut RenderDevice,
        pipeline_cache: &mut PipelineCache,
    ) -> CompiledRenderGraph {
        // Create resources from initial resource descriptors
        let resources: Vec<ResourceStorage> = self.initial_resources
            .into_iter()
            .map(|res| {
                match res {
                    InitialResourceStorage::ManagedBuffer(desc) => {
                        let resource = device
                            .acquire_buffer(&desc)
                            .expect("Failed to create buffer");
                        ResourceStorage::ManagedBuffer {
                            desc,
                            resource,
                            state_tracker: ResourceStateTracker::new(BufferState::Undefined),
                        }
                    }
                    InitialResourceStorage::ManagedTexture(desc) => {
                        let resource = device
                            .acquire_texture(&desc)
                            .expect("Failed to create texture");
                        ResourceStorage::ManagedTexture {
                            desc,
                            resource,
                            state_tracker: ResourceStateTracker::new(TextureState::Undefined),
                        }
                    }
                    InitialResourceStorage::ImportedBuffer(buffer, initial_state) => ResourceStorage::ImportedBuffer {
                        resource: buffer.clone(),
                        state_tracker: ResourceStateTracker::new(initial_state),
                    },
                    InitialResourceStorage::ImportedTexture(tex, initial_state) => ResourceStorage::ImportedTexture {
                        resource: tex.clone(),
                        state_tracker: ResourceStateTracker::new(initial_state),
                    },
                }
            })
            .collect();

        let mut graphic_pipelines = vec![];

        for node in &mut self.nodes {
            if !node.pipeline_state.valid() {
                log::warn!("Incomplete information for render graph node [{}]. Skipped.", node.name);
                continue;
            }

            match &mut node.pipeline_state {
                NodePipelineState::Graphic { pipeline_desc, .. } => {
                    let Some(pipeline_desc) = pipeline_desc.as_mut() else {
                        graphic_pipelines.push(None);
                        continue;
                    };

                    let pipeline = pipeline_cache
                        .get_or_create(pipeline_desc)
                        .expect("Failed to create graphics pipeline");

                    graphic_pipelines.push(Some(pipeline));
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
        
        let nodes = std::mem::take(&mut self.serial_nodes);
        self.record_nodes(device, &encoder, nodes);
        
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
        device.reset_frame_resources();
        device.present_command_pool().reset()?;

        // update the swapchain texture reference to the acquired image
        if self.swapchain_tex_id != GraphResourceId::MAX {
            let swapchain_tex = swapchain.swapchain_texture(image_index as usize);
            if let Some(ResourceStorage::ImportedTexture { resource, state_tracker, .. }) = self.resources.get_mut(self.swapchain_tex_id as usize) {
                *resource = swapchain_tex;
                // Reset state tracker since this is a newly acquired image
                *state_tracker = ResourceStateTracker::new(TextureState::Undefined);
            }
        }

        let cmd = device.present_command_pool().allocate().expect("Failed to allocate command buffer");
        let encoder = CommandEncoder::new(device.handle(), cmd);
        encoder.begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;

        let nodes = std::mem::take(&mut self.present_nodes);
        self.record_nodes(device, &encoder, nodes);

        // make sure the swapchain texture has the right image layout for presentation
        Self::transition_resources(
            device, &encoder, None, &self.resources,
            [(self.swapchain_tex_id, TextureState::Present.into(), Some(vk::PipelineStageFlags2::BOTTOM_OF_PIPE))].into_iter(),
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

    fn record_nodes(
        &mut self,
        device: &RenderDevice,
        encoder: &CommandEncoder,
        nodes: impl IntoIterator<Item = RenderGraphNode>,
    ) {
        for node in nodes {
            let transition_resources = |reflection| {
                profiling::scope!("rendergraph::barriers");
                let output_iter = node.outputs.iter()
                    .map(|res| (res.id, res.access, res.stage_hint));

                Self::transition_resources(
                    device, encoder, reflection, &self.resources,
                    node.inputs.iter()
                        .map(|res| (res.id, res.access, res.stage_hint))
                        .chain(output_iter),
                );
            };

            match node.pipeline_state {
                NodePipelineState::Graphic { pipeline_desc, color_attachments, depth_attachment, mut job_functor } => {
                    transition_resources(pipeline_desc.as_ref().map(|desc| &desc.shader.merged_reflection));

                    let pipeline_desc = pipeline_desc.as_ref();
                    let name = node.name;
                    let pipeline = self.graphic_pipelines.get(self.graphic_pipe_index as usize).unwrap();
                    let pipeline = pipeline.as_ref().map(|pipe| pipe.as_ref());

                    let color_attachment_ids: SmallVec<[GraphResourceId; 8]> =
                        color_attachments.iter().map(|(id, _)| *id).collect();
                    let depth_attachment_id: Option<GraphResourceId> =
                        depth_attachment.as_ref().map(|(id, _)| *id);
                    self.graphic_pipe_index += 1;

                    if let Some(record) = job_functor.take() {
                        profiling::scope!("rendergraph::node_recording", &name);

                        let Some(pipeline_desc) = pipeline_desc else {
                            continue;
                        };

                        let mut ctx = GraphicNodeExecutionContext {
                            pipeline_desc,
                            device,
                            resources: &self.resources,
                            pipeline,
                            encoder,
                            color_attachment_ids,
                            depth_attachment_id,
                        };
                        record(&mut ctx).expect("Failed to record graphic node.");
                    } else {
                        log::warn!("Missing job of graphic node {}!", name);
                    }
                }
                NodePipelineState::Compute { .. } => unimplemented!(),
                NodePipelineState::Lambda { mut job_functor } => {
                    transition_resources(None);

                    let name = node.name;
                    if let Some(record) = job_functor.take() {
                        profiling::scope!("rendergraph::node_recording", &name);

                        let mut ctx = LambdaNodeExecutionContext { device, resources: &self.resources, encoder };
                        record(&mut ctx).expect("Failed to record lambda node.");
                    } else {
                        log::warn!("Missing job of lambda node {}!", name);
                    }
                }
            }
        }
    }

    fn transition_resources(
        device: &RenderDevice,
        encoder: &CommandEncoder,
        merged_reflection: Option<&ShaderReflection>,
        resource_storage: &Vec<ResourceStorage>,
        resources_to_transition: impl Iterator<Item = (GraphResourceId, ResourceState, Option<vk::PipelineStageFlags2>)>,
    ) {
        let mut image_barriers: Vec<TextureBarrier> = Vec::new();
        let mut buffer_barriers: Vec<BufferBarrier> = Vec::new();

        let queue = device.graphics_queue();

        let combined_shader_stage = merged_reflection
            .map(|reflection| {
                reflection.bindings
                    .iter()
                    .fold(vk::ShaderStageFlags::empty(), |acc, bind| acc | bind.stage_flags)
            })
            .map(shader_stage_to_pipeline_stage)
            .unwrap_or(vk::PipelineStageFlags2::ALL_COMMANDS);

        for (id, access, stage_hint) in resources_to_transition {
            let storage = utility::resource_storage_ref(resource_storage, id);

            match storage {
                ResourceStorage::ManagedBuffer { resource, state_tracker, .. } => {
                    let ResourceState::Buffer(next_state) = access else { continue; };
                    let prev_state = state_tracker.current_access();
                    if prev_state == next_state { continue; }

                    let dst_stage_vk = stage_hint.unwrap_or(combined_shader_stage);
                    let src_stage = PipelineStages::from_vk(state_tracker.current_stage());
                    let dst_stage = PipelineStages::from_vk(dst_stage_vk);
                    if dst_stage_vk == vk::PipelineStageFlags2::ALL_COMMANDS {
                        log::warn!("Render graph buffer resource [{}] may cause serve pipeline stall due to unknown pipeline stage usage. Use read_hint() or write_hint() to get better performance.", resource.name())
                    }

                    buffer_barriers.push(BufferBarrier::new(
                        resource.as_range(..).unwrap(),
                        prev_state,
                        next_state,
                        src_stage,
                        dst_stage,
                        queue,
                        queue,
                        false,
                    ));
                    state_tracker.transition_to(next_state, next_state.into_pipeline_stage(dst_stage_vk));
                }
                ResourceStorage::ImportedBuffer { resource, state_tracker } => {
                    let ResourceState::Buffer(next_state) = access else { continue; };
                    let prev_state = state_tracker.current_access();
                    if prev_state == next_state { continue; }

                    let dst_stage_vk = stage_hint.unwrap_or(combined_shader_stage);
                    let src_stage = PipelineStages::from_vk(state_tracker.current_stage());
                    let dst_stage = PipelineStages::from_vk(dst_stage_vk);
                    if dst_stage_vk == vk::PipelineStageFlags2::ALL_COMMANDS {
                        log::warn!("Render graph buffer resource [{}] may cause serve pipeline stall due to unknown pipeline stage usage. Use read_hint() or write_hint() to get better performance.", resource.name())
                    }

                    buffer_barriers.push(BufferBarrier::new(
                        resource.as_range(..).unwrap(),
                        prev_state,
                        next_state,
                        src_stage,
                        dst_stage,
                        queue,
                        queue,
                        false,
                    ));
                    state_tracker.transition_to(next_state, next_state.into_pipeline_stage(dst_stage_vk));
                }
                ResourceStorage::ManagedTexture { resource, state_tracker, .. } => {
                    let ResourceState::Texture(next_state) = access else { continue; };
                    let prev_state = state_tracker.current_access();
                    if prev_state == next_state { continue; }

                    let dst_stage_vk = stage_hint.unwrap_or(combined_shader_stage);
                    let src_stage = PipelineStages::from_vk(state_tracker.current_stage());
                    let dst_stage = PipelineStages::from_vk(dst_stage_vk);
                    if dst_stage_vk == vk::PipelineStageFlags2::ALL_COMMANDS {
                        log::warn!("Render graph texture resource [{}] may cause serve pipeline stall due to unknown pipeline stage usage. Use read_hint() or write_hint() to get better performance.", resource.name())
                    }

                    image_barriers.push(TextureBarrier::new(
                        resource.as_range(.., ..).unwrap(),
                        prev_state,
                        next_state,
                        src_stage,
                        dst_stage,
                        queue,
                        queue,
                        false,
                        prev_state == TextureState::Undefined,
                    ));
                    state_tracker.transition_to(next_state, next_state.into_pipeline_stage(dst_stage_vk));
                }
                ResourceStorage::ImportedTexture { resource, state_tracker } => {
                    let ResourceState::Texture(next_state) = access else { continue; };
                    let prev_state = state_tracker.current_access();
                    if prev_state == next_state { continue; }

                    let dst_stage_vk = stage_hint.unwrap_or(combined_shader_stage);
                    let src_stage = PipelineStages::from_vk(state_tracker.current_stage());
                    let dst_stage = PipelineStages::from_vk(dst_stage_vk);
                    if dst_stage_vk == vk::PipelineStageFlags2::ALL_COMMANDS {
                        log::warn!("Render graph texture resource [{}] may cause serve pipeline stall due to unknown pipeline stage usage. Use read_hint() or write_hint() to get better performance.", resource.name())
                    }

                    image_barriers.push(TextureBarrier::new(
                        resource.as_range(.., ..).unwrap(),
                        prev_state,
                        next_state,
                        src_stage,
                        dst_stage,
                        queue,
                        queue,
                        false,
                        prev_state == TextureState::Undefined,
                    ));
                    state_tracker.transition_to(next_state, next_state.into_pipeline_stage(dst_stage_vk));
                }
            }
        }

        if !image_barriers.is_empty() {
            encoder.texture_barriers(&image_barriers);
        }
        if !buffer_barriers.is_empty() {
            encoder.buffer_barriers(&buffer_barriers);
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
    pipeline_desc: &'node GraphicPipelineDesc,
    device: &'node RenderDevice,
    resources: &'node Vec<ResourceStorage>,
    pipeline: Option<&'node GraphicPipeline>,
    encoder: &'node CommandEncoder<'node>,
    color_attachment_ids: SmallVec<[GraphResourceId; 8]>,
    depth_attachment_id: Option<GraphResourceId>,
}

impl<'node> GraphicNodeExecutionContext<'node> {
    #[inline]
    pub fn get<R: GraphResource, V: GraphResourceView>(&self, resource: &RenderGraphResourceAccess<R, V>) -> &R {
        let storage = self.resources.get(resource.id as usize)
            .expect("Graph resource index out of bound!");
        R::from_storage(storage)
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
        let color_infos = &self.pipeline_desc.state.color_blend.attachments;
        if self.color_attachment_ids.len() != color_infos.len() {
            panic!(
                "begin_rendering: node has {} color targets but pipeline state has {} color attachments",
                self.color_attachment_ids.len(),
                color_infos.len()
            );
        }

        let color_attachments: SmallVec<[vk::RenderingAttachmentInfo; 8]> = self
            .color_attachment_ids
            .iter()
            .zip(color_infos.iter())
            .map(|(id, info)| {
                let texture = utility::resource_storage_ref(self.resources, *id).as_texture();
                vk::RenderingAttachmentInfo::default()
                    .image_view(texture.as_range(.., ..).unwrap().view().expect("Texture view not created"))
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(info.load_op)
                    .store_op(info.store_op)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: info.clear_value,
                        },
                    })
            })
            .collect();

        let depth_attachment = match (
            self.depth_attachment_id,
            self.pipeline_desc.state.depth_stencil.as_ref(),
        ) {
            (Some(id), Some(info)) => {
                let texture = utility::resource_storage_ref(self.resources, id).as_texture();
                Some(
                    vk::RenderingAttachmentInfo::default()
                        .image_view(texture.as_range(.., ..).unwrap().view().expect("Texture view not created"))
                        .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .load_op(info.depth_load_op)
                        .store_op(info.depth_store_op)
                        .clear_value(vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: info.depth_clear_value,
                                stencil: info.stencil_clear_value,
                            },
                        }),
                )
            }
            _ => None,
        };

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
    pub fn create_binder(&self) -> DescriptorSetBinder<'_> {
        DescriptorSetBinder::new(self.device, &self.pipeline_desc.shader.merged_reflection).unwrap()
    }

    /// Bind descriptor sets to the pipeline.
    pub fn bind_descriptor_sets(&self, binder: DescriptorSetBinder) {
        let (pool, sets) = binder.finish(&self.pipeline_desc.shader.descriptor_set_layouts);
        if let Some(pipeline) = self.pipeline {
            if !sets.is_empty() {
                self.encoder.bind_descriptor_sets(
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.layout(),
                    0,
                    &sets,
                    &[],
                );
            }
        }
        self.device.defer_release(pool);
    }
}

pub struct LambdaNodeExecutionContext<'node> {
    device: &'node RenderDevice,
    resources: &'node Vec<ResourceStorage>,
    encoder: &'node CommandEncoder<'node>,
}

impl<'node> LambdaNodeExecutionContext<'node> {
    #[inline]
    pub fn get<R: GraphResource, V: GraphResourceView>(&self, resource: &RenderGraphResourceAccess<R, V>) -> &R {
        let storage = self.resources.get(resource.id as usize)
            .expect("Graph resource index out of bound!");
        R::from_storage(storage)
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
                ResourceStorage::ManagedBuffer { desc, resource, .. } => {
                    device.recycle_buffer(desc, resource);
                }
                ResourceStorage::ManagedTexture { desc, resource, .. } => {
                    device.recycle_texture(desc, resource);
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
