//! Render graph execution and resource management.

use crate::interface::{Buffer, BufferState, ResourceState, Texture, TextureState};
use crate::node::{NodePipelineState, RenderGraphNode};
use crate::resource::{GraphBindableAccess, GraphResource, GraphResourceId, GraphResourceState, GraphResourceView, InitialResourceStorage, RenderGraphResourceAccess, Rt, Srv};
use std::cell::Cell;
use std::collections::Bound;
use std::ops::RangeBounds;
use std::sync::{Arc};
use bytemuck::NoUninit;
use parking_lot::Mutex;
use zenith_core::collections::SmallVec;
use zenith_core::color;
use zenith_rhi::{BindlessPool, CommandEncoder, BufferBarrier, TextureBarrier, PipelineStages, ShaderReflection, CommandPool, TextureLayout, ColorAttachmentDesc, DepthStencilAttachmentDesc, GraphicPipeline, DescriptorBindingError, DescriptorSetBinder};
use zenith_rhi::{vk, RenderDevice, Swapchain};
use crate::GraphicPipelineHandle;

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
    /// Extracts a buffer reference from storage.
    ///
    /// # Panics
    /// Panics with unreachable!() if called on a texture variant.
    /// This is safe because the type system (via PhantomData<R> and sealed traits)
    /// ensures this is only called when the storage actually contains a buffer.
    #[allow(dead_code)]
    pub(crate) fn as_buffer(&self) -> &Buffer {
        match self {
            ResourceStorage::ManagedBuffer { resource, .. } => resource,
            ResourceStorage::ImportedBuffer { resource, .. } => resource,
            _ => unreachable!("Expected buffer, but resource is a texture!"),
        }
    }

    /// Extracts a texture reference from storage.
    ///
    /// # Panics
    /// Panics with unreachable!() if called on a buffer variant.
    /// This is safe because the type system (via PhantomData<R> and sealed traits)
    /// ensures this is only called when the storage actually contains a texture.
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
        pipeline_cache: &mut zenith_rhi::PipelineCache,
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

        // Validate nodes
        for node in &mut self.nodes {
            if !node.pipeline_state.valid() {
                log::warn!("Incomplete information for render graph node [{}]. Skipped.", node.name);
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

        let serial_nodes_count = serial_nodes.len();

        let node_pipelines = Self::create_node_pipelines(&serial_nodes, &present_nodes, device, pipeline_cache);

        CompiledRenderGraph {
            serial_nodes,
            present_nodes,
            resources,
            swapchain_tex_id,
            node_pipelines,
            serial_nodes_count,
        }
    }

    fn create_node_pipelines(
        serial_nodes: &[RenderGraphNode],
        present_nodes: &[RenderGraphNode],
        device: &RenderDevice,
        pipeline_cache: &mut zenith_rhi::PipelineCache,
    ) -> Vec<Vec<Arc<GraphicPipeline>>> {
        let total_nodes = serial_nodes.len() + present_nodes.len();
        let mut node_pipelines = Vec::with_capacity(total_nodes);

        for node in serial_nodes.iter().chain(present_nodes.iter()) {
            match &node.pipeline_state {
                NodePipelineState::Graphic { pipeline_descs, .. } => {
                    let mut pipelines = Vec::with_capacity(pipeline_descs.len());

                    for (handle_idx, desc) in pipeline_descs.iter().enumerate() {
                        let name = format!("{}.pipeline{}", node.name, handle_idx);
                        match pipeline_cache.get_or_create(&name, device, desc) {
                            Ok(pipeline) => {
                                pipelines.push(pipeline);
                            },
                            Err(e) => {
                                log::error!("Failed to create pipeline {} for node {}: {:?}", handle_idx, node.name, e);
                            }
                        }
                    }

                    node_pipelines.push(pipelines);
                }
                NodePipelineState::Compute { .. } => {
                    node_pipelines.push(Vec::new());  // No pipelines for compute yet
                }
                NodePipelineState::Lambda { .. } => {
                    node_pipelines.push(Vec::new());  // No pipelines for lambda nodes
                }
            }
        }

        node_pipelines
    }
}

pub struct CompiledRenderGraph {
    serial_nodes: Vec<RenderGraphNode>,
    present_nodes: Vec<RenderGraphNode>,
    resources: Vec<ResourceStorage>,
    swapchain_tex_id: GraphResourceId,
    node_pipelines: Vec<Vec<Arc<GraphicPipeline>>>,
    serial_nodes_count: usize,
}

impl CompiledRenderGraph {
    #[profiling::function]
    pub fn execute(&mut self, device: &RenderDevice, cmd_pool: &CommandPool) -> anyhow::Result<()>  {
        let encoder = CommandEncoder::new("cmd.rendergraph.execute", device, cmd_pool)?;

        encoder.begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;
        encoder.begin_debug_label("render_graph::execution", color::LIGHT_GREEN);

        let nodes = std::mem::take(&mut self.serial_nodes);
        self.record_nodes(device, &encoder, nodes, 0);

        encoder.end_debug_label();
        encoder.end()?;

        device.submit_commands(
            encoder,
            device.graphics_queue(),
            &[],
            vk::PipelineStageFlags2::NONE,
            &[],
            vk::PipelineStageFlags2::NONE,
            device.frame_resource_fence(),
        );

        Ok(())
    }

    pub fn present(mut self, device: &mut RenderDevice, cmd_pool: &CommandPool, swapchain: &mut Swapchain, image_index: u32) -> anyhow::Result<RetiredRenderGraph> {
        cmd_pool.reset()?;

        // update the swapchain texture reference to the acquired image
        if self.swapchain_tex_id != GraphResourceId::MAX {
            let swapchain_tex = swapchain.swapchain_texture(image_index as usize);
            if let Some(ResourceStorage::ImportedTexture { resource, state_tracker, .. }) = self.resources.get_mut(self.swapchain_tex_id as usize) {
                *resource = swapchain_tex;
                // Reset state tracker since this is a newly acquired image
                *state_tracker = ResourceStateTracker::new(TextureState::Undefined);
            }
        }

        let encoder = CommandEncoder::new("cmd.rendergraph.present", device, cmd_pool)?;
        encoder.begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;
        encoder.begin_debug_label("render_graph::present", color::LIGHT_YELLOW);

        let nodes = std::mem::take(&mut self.present_nodes);
        let serial_nodes_count = self.serial_nodes_count;
        self.record_nodes(device, &encoder, nodes, serial_nodes_count);

        // make sure the swapchain texture has the right image layout for presentation
        Self::transition_resources(
            device, &encoder, None, &self.resources,
            [(self.swapchain_tex_id, TextureState::Present.into(), Some(vk::PipelineStageFlags2::BOTTOM_OF_PIPE))].into_iter(),
        );

        encoder.end_debug_label();
        encoder.end()?;

        let frame_sync = swapchain.current_frame_sync();

        device.submit_commands(
            encoder,
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
        nodes: Vec<RenderGraphNode>,
        node_start_index: usize,
    ) {
        for (node_offset, node) in nodes.into_iter().enumerate() {
            let node_idx = node_start_index + node_offset;
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
                NodePipelineState::Graphic { ref pipeline_descs, mut job_functor } => {
                    let reflection = ShaderReflection::merge(&pipeline_descs.iter()
                        .map(|pipe| &pipe.shader.merged_reflection)
                        .collect::<Vec<_>>());
                    transition_resources(Some(&reflection));

                    let name = node.name;

                    if let Some(record) = job_functor.take() {
                        encoder.begin_debug_label(&name, color::LIGHT_BLUE);
                        profiling::scope!("rendergraph::node_recording", &name);

                        let mut ctx = GraphicNodeExecutionContext {
                            device,
                            resources: &self.resources,
                            encoder,
                            node_pipelines: &self.node_pipelines[node_idx],
                        };
                        let result = record(&mut ctx);
                        encoder.end_debug_label();
                        result.expect("Failed to record graphic node.");
                    } else {
                        log::warn!("Missing job of graphic node {}!", name);
                    }
                }
                NodePipelineState::Compute { .. } => unimplemented!(),
                NodePipelineState::Lambda { mut job_functor } => {
                    transition_resources(None);

                    let name = node.name;
                    if let Some(record) = job_functor.take() {
                        encoder.begin_debug_label(&name, color::LIGHT_PINK);
                        profiling::scope!("rendergraph::node_recording", &name);

                        let mut ctx = LambdaNodeExecutionContext {
                            device,
                            resources: &self.resources,
                            encoder,
                        };
                        let result = record(&mut ctx);
                        encoder.end_debug_label();
                        result.expect("Failed to record lambda node.");
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

        // merge all binding's stage together as a fallback stage
        let all_resource_stage_flags = merged_reflection
            .map(|reflection| {
                reflection.bindings
                    .iter()
                    .fold(vk::ShaderStageFlags::empty(), |acc, bind| acc | bind.stage_flags)
            })
            .map(shader_stage_to_pipeline_stage)
            .unwrap_or(vk::PipelineStageFlags2::ALL_COMMANDS);

        let queue = device.graphics_queue();

        for (id, access, stage_hint) in resources_to_transition {
            let storage = utility::resource_storage_ref(resource_storage, id);

            match storage {
                ResourceStorage::ManagedBuffer { resource, state_tracker, .. } => {
                    let stage = merged_reflection
                        .map(|reflection| reflection.bindings.iter()
                            .find(|binding| binding.name.as_str() == resource.name())
                            .map(|binding| binding.stage_flags)
                            .map(shader_stage_to_pipeline_stage)
                            .unwrap_or(all_resource_stage_flags)
                        ).unwrap_or(all_resource_stage_flags);

                    let ResourceState::Buffer(next_state) = access else { continue; };
                    let prev_state = state_tracker.current_access();
                    if prev_state == next_state { continue; }

                    let dst_stage_vk = stage_hint.unwrap_or(stage);
                    let src_stage = PipelineStages::from_vk(state_tracker.current_stage());
                    let dst_stage = PipelineStages::from_vk(dst_stage_vk);
                    if dst_stage_vk == vk::PipelineStageFlags2::ALL_COMMANDS {
                        log::warn!("Render graph buffer resource [{}] may cause serve pipeline stall due to unknown pipeline stage usage. Use read_hint() or write_hint() to get better performance.", resource.name())
                    }

                    buffer_barriers.push(BufferBarrier::new(
                        resource.as_range(..),
                        prev_state,
                        next_state,
                        src_stage,
                        dst_stage,
                        queue,
                        queue,
                    ));
                    state_tracker.transition_to(next_state, next_state.into_pipeline_stage(dst_stage_vk));
                }
                ResourceStorage::ImportedBuffer { resource, state_tracker } => {
                    let stage = merged_reflection
                        .map(|reflection| reflection.bindings.iter()
                            .find(|binding| binding.name.as_str() == resource.name())
                            .map(|binding| binding.stage_flags)
                            .map(shader_stage_to_pipeline_stage)
                            .unwrap_or(all_resource_stage_flags)
                        ).unwrap_or(all_resource_stage_flags);

                    let ResourceState::Buffer(next_state) = access else { continue; };
                    let prev_state = state_tracker.current_access();
                    if prev_state == next_state { continue; }

                    let dst_stage_vk = stage_hint.unwrap_or(stage);
                    let src_stage = PipelineStages::from_vk(state_tracker.current_stage());
                    let dst_stage = PipelineStages::from_vk(dst_stage_vk);
                    if dst_stage_vk == vk::PipelineStageFlags2::ALL_COMMANDS {
                        log::warn!("Render graph buffer resource [{}] may cause serve pipeline stall due to unknown pipeline stage usage. Use read_hint() or write_hint() to get better performance.", resource.name())
                    }

                    buffer_barriers.push(BufferBarrier::new(
                        resource.as_range(..),
                        prev_state,
                        next_state,
                        src_stage,
                        dst_stage,
                        queue,
                        queue,
                    ));
                    state_tracker.transition_to(next_state, next_state.into_pipeline_stage(dst_stage_vk));
                }
                ResourceStorage::ManagedTexture { resource, state_tracker, .. } => {
                    let stage = merged_reflection
                        .map(|reflection| reflection.bindings.iter()
                            .find(|binding| binding.name.as_str() == resource.name())
                            .map(|binding| binding.stage_flags)
                            .map(shader_stage_to_pipeline_stage)
                            .unwrap_or(all_resource_stage_flags)
                        ).unwrap_or(all_resource_stage_flags);

                    let ResourceState::Texture(next_state) = access else { continue; };
                    let prev_state = state_tracker.current_access();
                    if prev_state == next_state { continue; }

                    let dst_stage_vk = stage_hint.unwrap_or(stage);
                    let src_stage = PipelineStages::from_vk(state_tracker.current_stage());
                    let dst_stage = PipelineStages::from_vk(dst_stage_vk);
                    if dst_stage_vk == vk::PipelineStageFlags2::ALL_COMMANDS {
                        log::warn!("Render graph texture resource [{}] may cause serve pipeline stall due to unknown pipeline stage usage. Use read_hint() or write_hint() to get better performance.", resource.name())
                    }

                    image_barriers.push(TextureBarrier::new(
                        resource.as_range(TextureLayout::from(prev_state), .., ..),
                        prev_state,
                        next_state,
                        src_stage,
                        dst_stage,
                        queue,
                        queue,
                        prev_state == TextureState::Undefined,
                    ));
                    state_tracker.transition_to(next_state, next_state.into_pipeline_stage(dst_stage_vk));
                }
                ResourceStorage::ImportedTexture { resource, state_tracker } => {
                    let stage = merged_reflection
                        .map(|reflection| reflection.bindings.iter()
                            .find(|binding| binding.name.as_str() == resource.name())
                            .map(|binding| binding.stage_flags)
                            .map(shader_stage_to_pipeline_stage)
                            .unwrap_or(all_resource_stage_flags)
                        ).unwrap_or(all_resource_stage_flags);

                    let ResourceState::Texture(next_state) = access else { continue; };
                    let prev_state = state_tracker.current_access();
                    if prev_state == next_state { continue; }

                    let dst_stage_vk = stage_hint.unwrap_or(stage);
                    let src_stage = PipelineStages::from_vk(state_tracker.current_stage());
                    let dst_stage = PipelineStages::from_vk(dst_stage_vk);
                    if dst_stage_vk == vk::PipelineStageFlags2::ALL_COMMANDS {
                        log::warn!("Render graph texture resource [{}] may cause serve pipeline stall due to unknown pipeline stage usage. Use read_hint() or write_hint() to get better performance.", resource.name())
                    }

                    image_barriers.push(TextureBarrier::new(
                        resource.as_range(TextureLayout::from(prev_state), .., ..),
                        prev_state,
                        next_state,
                        src_stage,
                        dst_stage,
                        queue,
                        queue,
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

pub struct PipelineResourceBinder<'a> {
    device: &'a RenderDevice,
    ctx: &'a GraphicNodeExecutionContext<'a>,
    binder: Option<DescriptorSetBinder<'a>>,
    bind_point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
}

impl<'a> Drop for PipelineResourceBinder<'a> {
    fn drop(&mut self) {
        let (base_set, sets) = self.binder.take().unwrap().finish(self.device).unwrap();
        self.ctx.encoder.bind_descriptor_sets(
            self.bind_point,
            self.layout.clone(),
            base_set,
            &sets,
            &[],
        );
    }
}

impl<'a> PipelineResourceBinder<'a> {
    pub fn bind<A: GraphBindableAccess>(
        &mut self,
        name: &str,
        access: A,
    ) -> Result<&mut Self, DescriptorBindingError> {
        self.binder.as_mut().unwrap().bind(name, &access.into_bindable(&self.ctx))?;
        Ok(self)
    }

    pub fn bind_raw(
        &mut self,
        base_set: u32,
        sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) -> &mut Self {
        self.ctx.encoder.bind_descriptor_sets(
            self.bind_point,
            self.layout.clone(),
            base_set,
            sets,
            dynamic_offsets,
        );
        self
    }
}

/// Simple wrapper class to simplify user color render target binding
#[derive(Debug)]
pub struct ColorAttachment {
    pub color: RenderGraphResourceAccess<Texture, Rt>,
    pub desc: ColorAttachmentDesc,
}

impl ColorAttachment {
    pub fn new(color: RenderGraphResourceAccess<Texture, Rt>) -> Self {
        Self {
            color,
            desc: Default::default(),
        }
    }

    #[inline]
    pub fn discard_input(mut self) -> Self {
        self.desc.load_op = vk::AttachmentLoadOp::DONT_CARE;
        self
    }

    #[inline]
    pub fn clear_input(mut self) -> Self {
        self.desc.load_op = vk::AttachmentLoadOp::CLEAR;
        self
    }

    #[inline]
    pub fn discard_output(mut self) -> Self {
        self.desc.store_op = vk::AttachmentStoreOp::DONT_CARE;
        self
    }

    #[inline]
    pub fn clear_value(mut self, clear_value: [f32; 4]) -> Self {
        self.desc.clear_value = clear_value;
        self
    }
}

/// Simple wrapper class to simplify user depth/stencil render target binding
#[derive(Debug)]
pub struct DepthStencilAttachment {
    pub depth: RenderGraphResourceAccess<Texture, Rt>,
    pub desc: DepthStencilAttachmentDesc,
}

impl DepthStencilAttachment {
    pub fn new(depth: RenderGraphResourceAccess<Texture, Rt>) -> Self {
        Self {
            depth,
            desc: Default::default(),
        }
    }

    #[inline]
    pub fn discard_depth_input(mut self) -> Self {
        self.desc.depth_load_op = vk::AttachmentLoadOp::DONT_CARE;
        self
    }

    #[inline]
    pub fn discard_stencil_input(mut self) -> Self {
        self.desc.stencil_load_op = vk::AttachmentLoadOp::DONT_CARE;
        self
    }

    #[inline]
    pub fn clear_depth_input(mut self) -> Self {
        self.desc.depth_load_op = vk::AttachmentLoadOp::CLEAR;
        self
    }

    #[inline]
    pub fn clear_stencil_input(mut self) -> Self {
        self.desc.stencil_load_op = vk::AttachmentLoadOp::CLEAR;
        self
    }

    #[inline]
    pub fn discard_depth_output(mut self) -> Self {
        self.desc.depth_store_op = vk::AttachmentStoreOp::DONT_CARE;
        self
    }

    #[inline]
    pub fn discard_stencil_output(mut self) -> Self {
        self.desc.stencil_store_op = vk::AttachmentStoreOp::DONT_CARE;
        self
    }
}

pub struct GraphicNodeExecutionContext<'node> {
    device: &'node RenderDevice,
    resources: &'node Vec<ResourceStorage>,
    encoder: &'node CommandEncoder<'node>,
    node_pipelines: &'node [Arc<GraphicPipeline>],
}

impl<'node> GraphicNodeExecutionContext<'node> {
    /// Gets a reference to the concrete resource.
    ///
    /// # Safety
    /// This method uses transmute which is safe because:
    /// 1. The sealed trait ensures only Buffer and Texture implement GraphResource
    /// 2. PhantomData<R> in RenderGraphResourceAccess ensures the resource type matches the storage variant
    /// 3. The enum discriminant is checked before transmute
    #[inline]
    pub fn get<R: GraphResource, V: GraphResourceView>(&self, resource: &RenderGraphResourceAccess<R, V>) -> &R {
        match self.resources.get(resource.id as usize).expect("Graph resource index out of bound!") {
            ResourceStorage::ManagedBuffer { resource, .. } => unsafe { std::mem::transmute(resource) },
            ResourceStorage::ManagedTexture { resource, .. } => unsafe { std::mem::transmute(resource) },
            ResourceStorage::ImportedBuffer { resource, .. } => {
                let res: &Buffer = resource.as_ref();
                unsafe { std::mem::transmute(res) }
            },
            ResourceStorage::ImportedTexture { resource, .. } => {
                let res: &Texture = resource.as_ref();
                unsafe { std::mem::transmute(res) }
            },
        }
    }

    #[inline]
    pub fn device(&self) -> &RenderDevice { self.device }

    #[inline]
    pub fn encoder(&self) -> &CommandEncoder<'node> { self.encoder }

    pub fn bind_pipeline<'a>(&'a self, handle: GraphicPipelineHandle) -> PipelineResourceBinder<'a> {
        let pipeline = self.get_pipeline(handle);

        let binder = DescriptorSetBinder::new(
            self.device,
            &pipeline.desc().shader.merged_reflection,
            pipeline.descriptor_layouts(),
        );

        self.encoder.bind_pipeline(
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.handle(),
        );

        PipelineResourceBinder {
            device: self.device,
            ctx: self,
            binder: Some(binder),
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            layout: pipeline.layout(),
        }
    }

    pub fn push_constants<T: NoUninit>(&self, handle: GraphicPipelineHandle, offset: u32, data: &T) {
        let pipeline = self.get_pipeline(handle);

        self.encoder.push_constants(
            pipeline.layout(),
            vk::ShaderStageFlags::ALL_GRAPHICS,
            offset,
            data,
        );
    }

    pub fn begin_rendering(
        &self,
        extent: (u32, u32),
        color_attachments: &[ColorAttachment],
        depth_attachment: Option<DepthStencilAttachment>,
    ) {
        let color_attachments: SmallVec<[vk::RenderingAttachmentInfo; 8]> = color_attachments.iter()
            .map(|attachment| {
                let texture = utility::resource_storage_ref(self.resources, attachment.color.id).as_texture();

                vk::RenderingAttachmentInfo::default()
                    .image_view(texture.as_range(TextureLayout::Color, .., ..).view().expect("Texture view not created"))
                    .image_layout(TextureLayout::Color.to_vk())
                    .load_op(attachment.desc.load_op)
                    .store_op(attachment.desc.store_op)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: attachment.desc.clear_value,
                        },
                    })
            })
            .collect();

        let depth_attachment = depth_attachment.map(|attachment| {
            let texture = utility::resource_storage_ref(self.resources, attachment.depth.id).as_texture();

            vk::RenderingAttachmentInfo::default()
                .image_view(texture.as_range(TextureLayout::DepthStencil, .., ..).view().expect("Texture view not created"))
                .image_layout(TextureLayout::DepthStencil.to_vk())
                .load_op(attachment.desc.depth_load_op)
                .store_op(attachment.desc.depth_store_op)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: attachment.desc.depth_clear_value,
                        stencil: attachment.desc.stencil_clear_value,
                    },
                })
        });

        let extent = vk::Extent2D { width: extent.0, height: extent.1 };
        let mut rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
            .layer_count(1)
            .color_attachments(&color_attachments);

        if let Some(ref depth) = depth_attachment {
            rendering_info = rendering_info.depth_attachment(depth);
        }

        self.encoder.begin_rendering(&rendering_info);

        // TODO: multi-viewport and scissors is not support for now
        let viewport = vk::Viewport {
            x: 0.0,
            // Flip Y in rasterization (Vulkan supports negative viewport height).
            // This avoids baking Y-flip into the projection matrix.
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        self.encoder.set_viewport(0, &[viewport]);
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };
        self.encoder.set_scissor(0, &[scissor]);
    }

    #[inline]
    pub fn end_rendering(&self) {
        self.encoder.end_rendering();
    }

    #[inline]
    pub fn bind_vertex_buffers(&self, buf: RenderGraphResourceAccess<Buffer, Srv>, first_binding: u32, offsets: &[u64]) {
        self.encoder.bind_vertex_buffers(first_binding, &[self.get(&buf).handle()], offsets);
    }

    #[inline]
    pub fn bind_index_buffer(&self, buf: RenderGraphResourceAccess<Buffer, Srv>, offset: u64, index_ty: vk::IndexType) {
        self.encoder.bind_index_buffer(self.get(&buf).handle(), offset, index_ty);
    }

    pub fn draw_indexed<R: RangeBounds<u32>>(&self, index: R, instance: R, vertex_offset: i32) {
        let get_offset_and_size = |range: R| {
            let start = match range.start_bound() {
                Bound::Included(&v) => v,
                Bound::Excluded(&v) => v.checked_add(1).expect("Index overflow"),
                Bound::Unbounded => 0,
            };
            let end_exclusive = match range.end_bound() {
                Bound::Included(&v) => v.checked_add(1).expect("Index overflow"),
                Bound::Excluded(&v) => v,
                Bound::Unbounded => panic!("Range should have clear upper bound"),
            };

            if start > end_exclusive {
                panic!("Range should have clear upper bound");
            }

            (start, end_exclusive - start)
        };

        let (first_index, index_count) = get_offset_and_size(index);
        let (first_instance, instance_count) = get_offset_and_size(instance);

        self.encoder.draw_indexed(index_count, instance_count, first_index, vertex_offset, first_instance);
    }

    #[inline]
    fn get_pipeline(&self, handle: GraphicPipelineHandle) -> &GraphicPipeline {
        &self.node_pipelines.get(handle.0 as usize).unwrap()
    }
}

pub struct LambdaNodeExecutionContext<'node> {
    device: &'node RenderDevice,
    resources: &'node Vec<ResourceStorage>,
    encoder: &'node CommandEncoder<'node>,
}

impl<'node> LambdaNodeExecutionContext<'node> {
    /// Gets a reference to the concrete resource.
    ///
    /// # Safety
    /// This method uses transmute which is safe because:
    /// 1. The sealed trait ensures only Buffer and Texture implement GraphResource
    /// 2. PhantomData<R> in RenderGraphResourceAccess ensures the resource type matches the storage variant
    /// 3. The enum discriminant is checked before transmute
    #[inline]
    pub fn get<R: GraphResource, V: GraphResourceView>(&self, resource: &RenderGraphResourceAccess<R, V>) -> &R {
        match self.resources.get(resource.id as usize).expect("Graph resource index out of bound!") {
            ResourceStorage::ManagedBuffer { resource, .. } => unsafe { std::mem::transmute(resource) },
            ResourceStorage::ManagedTexture { resource, .. } => unsafe { std::mem::transmute(resource) },
            ResourceStorage::ImportedBuffer { resource, .. } => {
                let res: &Buffer = resource.as_ref();
                unsafe { std::mem::transmute(res) }
            },
            ResourceStorage::ImportedTexture { resource, .. } => {
                let res: &Texture = resource.as_ref();
                unsafe { std::mem::transmute(res) }
            },
        }
    }

    #[inline]
    pub fn device(&self) -> &RenderDevice { self.device }

    #[inline]
    pub fn command_encoder(&self) -> &CommandEncoder<'node> { self.encoder }

    #[inline]
    pub fn bindless_pool(&self) -> &Arc<Mutex<BindlessPool>> { self.device.bindless_pool() }
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
