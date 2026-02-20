//! Render graph execution and resource management.

use crate::interface::{Buffer, BufferDesc, BufferState, ResourceState, Texture, TextureDesc, TextureState};
use crate::node::{NodeState, RenderGraphNode};
use crate::resource::{GraphBindable, GraphResource, GraphResourceId, GraphResourceState, GraphResourceView, InitialResourceStorage, RenderGraphResourceAccess, Rt, Srv};
use std::cell::{Cell};
use std::sync::{Arc};
use bytemuck::NoUninit;
use zenith_core::collections::SmallVec;
use zenith_core::color;
use zenith_rhi::{vk, RenderDevice, Swapchain, CommandEncoder, BufferBarrier, TextureBarrier, PipelineStages, ShaderReflection, CommandPool, TextureLayout, ColorAttachmentDesc, DepthStencilAttachmentDesc, DescriptorBindingError, DescriptorSetBinder, PipelineRegistry, PipelineHandle, TransientResourceCache, Fence};
use zenith_rhi::defer_release::DeferReleaseQueue;
use crate::node::GraphPipelineHandle;

pub enum ResourceStorage {
    ManagedBuffer {
        desc: BufferDesc,
        resource: Buffer,
        state_tracker: ResourceStateTracker<BufferState>,
    },
    ManagedTexture {
        desc: TextureDesc,
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
        device: &Arc<RenderDevice>,
        resource_cache: &mut TransientResourceCache,
    ) -> CompiledRenderGraph {
        let resources: Vec<ResourceStorage> = self.initial_resources
            .into_iter()
            .map(|res| {
                match res {
                    InitialResourceStorage::ManagedBuffer(desc) => {
                        let resource = resource_cache
                            .acquire_buffer(device, &desc)
                            .expect("Failed to create buffer");
                        ResourceStorage::ManagedBuffer {
                            desc,
                            resource,
                            state_tracker: ResourceStateTracker::new(BufferState::Undefined),
                        }
                    }
                    InitialResourceStorage::ManagedTexture(desc) => {
                        let resource = resource_cache
                            .acquire_texture(device, &desc)
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

        let node_pipeline_handles: Vec<Vec<PipelineHandle>> = serial_nodes.iter()
            .chain(present_nodes.iter())
            .map(|node| node.pipeline_state.pipeline_handles().to_vec())
            .collect();

        let serial_nodes_count = serial_nodes.len();
        CompiledRenderGraph {
            serial_nodes,
            present_nodes,
            resources,
            swapchain_tex_id,
            node_pipeline_handles,
            serial_nodes_count,
        }
    }
}

pub struct CompiledRenderGraph {
    serial_nodes: Vec<RenderGraphNode>,
    present_nodes: Vec<RenderGraphNode>,
    resources: Vec<ResourceStorage>,
    swapchain_tex_id: GraphResourceId,
    node_pipeline_handles: Vec<Vec<PipelineHandle>>,
    serial_nodes_count: usize,
}

impl CompiledRenderGraph {
    #[profiling::function]
    pub fn execute(&mut self, device: &Arc<RenderDevice>, cmd_pool: &CommandPool, pipeline_cache: &PipelineRegistry, defer_release: &DeferReleaseQueue, fence: &Fence) -> anyhow::Result<()>  {
        let encoder = CommandEncoder::new("cmd.rendergraph.execute", device, cmd_pool)?;

        encoder.begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;
        encoder.begin_debug_label("render_graph::execution", color::LIGHT_GREEN);

        let nodes = std::mem::take(&mut self.serial_nodes);
        self.record_nodes(device, &encoder, pipeline_cache, defer_release, nodes, 0);

        encoder.end_debug_label();
        encoder.end()?;

        device.submit_commands(
            encoder,
            device.graphics_queue(),
            &[],
            vk::PipelineStageFlags2::NONE,
            &[],
            vk::PipelineStageFlags2::NONE,
            fence,
        );

        Ok(())
    }

    pub fn present(mut self, device: &Arc<RenderDevice>, cmd_pool: &CommandPool, pipeline_cache: &PipelineRegistry, defer_release: &DeferReleaseQueue, swapchain: &mut Swapchain, image_index: u32) -> anyhow::Result<RetiredRenderGraph> {
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
        self.record_nodes(device, &encoder, pipeline_cache, defer_release, nodes, serial_nodes_count);

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
        device: &Arc<RenderDevice>,
        encoder: &CommandEncoder,
        pipeline_cache: &PipelineRegistry,
        defer_release: &DeferReleaseQueue,
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
                NodeState::Graphic { ref pipeline_handles, mut job_functor } => {
                    let reflections: Vec<&ShaderReflection> = pipeline_handles.iter()
                        .filter_map(|h| pipeline_cache.try_get_pipeline(*h).map(|e| e.shader_reflection()))
                        .collect();
                    let reflection = ShaderReflection::merge(&reflections);
                    transition_resources(Some(&reflection));

                    let name = node.name;

                    if let Some(record) = job_functor.take() {
                        encoder.begin_debug_label(&name, color::LIGHT_BLUE);
                        profiling::scope!("rendergraph::node_recording", &name);

                        let pipeline_handles = self.node_pipeline_handles[node_idx].as_slice();
                        let mut ctx = GraphicNodeExecutionContext {
                            device,
                            encoder,
                            binding_ctx: NodeBindingContext {
                                resources: &self.resources,
                                device,
                                defer_release,
                                encoder,
                                pipeline_handles,
                                pipeline_cache,
                            },
                        };
                        let result = record(&mut ctx);
                        encoder.end_debug_label();
                        result.expect("Failed to record graphic node.");
                    } else {
                        log::warn!("Missing job of graphic node {}!", name);
                    }
                }
                NodeState::Compute { ref pipeline_handles, mut job_functor } => {
                    let reflections: Vec<&ShaderReflection> = pipeline_handles.iter()
                        .filter_map(|h| pipeline_cache.try_get_pipeline(*h).map(|e| e.shader_reflection()))
                        .collect();
                    let reflection = ShaderReflection::merge(&reflections);
                    transition_resources(Some(&reflection));

                    let name = node.name;

                    if let Some(record) = job_functor.take() {
                        encoder.begin_debug_label(&name, color::LIGHT_BLUE);
                        profiling::scope!("rendergraph::node_recording", &name);

                        let pipeline_handles = self.node_pipeline_handles[node_idx].as_slice();
                        let mut ctx = ComputeNodeExecutionContext {
                            device,
                            encoder,
                            binding_ctx: NodeBindingContext {
                                resources: &self.resources,
                                device,
                                defer_release,
                                encoder,
                                pipeline_handles,
                                pipeline_cache,
                            },
                        };
                        let result = record(&mut ctx);
                        encoder.end_debug_label();
                        result.expect("Failed to record compute node.");
                    } else {
                        log::warn!("Missing job of compute node {}!", name);
                    }
                }
                NodeState::Lambda { mut job_functor } => {
                    transition_resources(None);

                    let name = node.name;
                    if let Some(record) = job_functor.take() {
                        encoder.begin_debug_label(&name, color::LIGHT_PINK);
                        profiling::scope!("rendergraph::node_recording", &name);

                        let mut ctx = LambdaNodeExecutionContext {
                            device,
                            binding_ctx: NodeBindingContext {
                                resources: &self.resources,
                                device,
                                defer_release,
                                encoder,
                                pipeline_handles: &[],
                                pipeline_cache,
                            },
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

        if !buffer_barriers.is_empty() || !image_barriers.is_empty() {
            encoder.pipeline_barriers(&[], &image_barriers, &buffer_barriers);
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
    defer_release: &'a DeferReleaseQueue,
    encoder: &'a CommandEncoder<'a>,
    binding_ctx: &'a NodeBindingContext<'a>,
    binder: Option<DescriptorSetBinder<'a>>,
    bind_point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
}

impl<'a> Drop for PipelineResourceBinder<'a> {
    fn drop(&mut self) { self.finish_impl(); }
}

impl<'a> PipelineResourceBinder<'a> {
    pub fn bind<A: GraphBindable>(
        mut self,
        name: &str,
        access: A,
    ) -> Result<Self, DescriptorBindingError> {
        self.binder.as_mut().unwrap().bind(name, &access.into_bindable(self.binding_ctx))?;
        Ok(self)
    }

    pub fn bind_raw(
        self,
        base_set: u32,
        sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) -> Self {
        self.encoder.bind_descriptor_sets(
            self.bind_point,
            self.layout.clone(),
            base_set,
            sets,
            dynamic_offsets,
        );
        self
    }

    fn finish_impl(&mut self) {
        if let Some(binder) = self.binder.take() {
            let (base_set, pool, sets) = binder.finish().unwrap();
            self.defer_release.defer_release(pool);
            self.encoder.bind_descriptor_sets(
                self.bind_point,
                self.layout.clone(),
                base_set,
                &sets,
                &[],
            );
        }
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

pub struct NodeBindingContext<'node> {
    pub(crate) resources: &'node Vec<ResourceStorage>,
    pub(crate) device: &'node Arc<RenderDevice>,
    pub(crate) defer_release: &'node DeferReleaseQueue,
    pub(crate) encoder: &'node CommandEncoder<'node>,
    pub(crate) pipeline_handles: &'node [PipelineHandle],
    pub(crate) pipeline_cache: &'node PipelineRegistry,
}

impl<'node> NodeBindingContext<'node> {
    /// Gets a reference to the concrete resource.
    ///
    /// # Safety
    /// This method uses transmute which is safe because:
    /// 1. The sealed trait ensures only Buffer and Texture implement GraphResource
    /// 2. PhantomData<R> in RenderGraphResourceAccess ensures the resource type matches the storage variant
    /// 3. The enum discriminant is checked before transmute
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

    pub fn bind_pipeline(&self, handle: GraphPipelineHandle) -> Option<PipelineResourceBinder<'_>> {
        let cache_handle = *self.pipeline_handles.get(handle.0 as usize)?;
        let entry = self.pipeline_cache.try_get_pipeline(cache_handle)?;

        let binder = DescriptorSetBinder::new(
            self.device,
            entry.shader_reflection(),
            entry.descriptor_layouts(),
        );

        self.encoder.bind_pipeline(entry.bind_point(), entry.handle());

        Some(PipelineResourceBinder {
            defer_release: self.defer_release,
            encoder: self.encoder,
            binding_ctx: self,
            binder: Some(binder),
            bind_point: entry.bind_point(),
            layout: entry.layout(),
        })
    }

    /// Push constants for the pipeline at the given handle. No-op if pipeline is not found.
    pub fn push_constants<T: NoUninit>(&self, handle: GraphPipelineHandle, offset: u32, data: &T) {
        let cache_handle = match self.pipeline_handles.get(handle.0 as usize) {
            Some(h) => *h,
            None => return,
        };
        let entry = match self.pipeline_cache.try_get_pipeline(cache_handle) {
            Some(e) => e,
            None => return,
        };

        let stages = match entry.bind_point() {
            vk::PipelineBindPoint::GRAPHICS => vk::ShaderStageFlags::ALL_GRAPHICS,
            vk::PipelineBindPoint::COMPUTE => vk::ShaderStageFlags::COMPUTE,
            _ => return,
        };

        self.encoder.push_constants(entry.layout(), stages, offset, data);
    }
}

pub struct GraphicNodeExecutionContext<'node> {
    device: &'node RenderDevice,
    encoder: &'node CommandEncoder<'node>,
    binding_ctx: NodeBindingContext<'node>,
}

impl<'node> GraphicNodeExecutionContext<'node> {
    #[inline]
    pub fn get<R: GraphResource, V: GraphResourceView>(&self, resource: &RenderGraphResourceAccess<R, V>) -> &R {
        self.binding_ctx.get(resource)
    }

    #[inline]
    pub fn device(&self) -> &RenderDevice { self.device }

    #[inline]
    pub fn encoder(&self) -> &CommandEncoder<'node> { self.encoder }

    #[inline]
    pub fn bind_pipeline(&self, handle: GraphPipelineHandle) -> Option<PipelineResourceBinder<'_>> {
        self.binding_ctx.bind_pipeline(handle)
    }

    #[inline]
    pub fn push_constants<T: NoUninit>(&self, handle: GraphPipelineHandle, offset: u32, data: &T) {
        self.binding_ctx.push_constants(handle, offset, data);
    }

    pub fn begin_rendering(
        &self,
        extent: (u32, u32),
        color_attachments: &[ColorAttachment],
        depth_attachment: Option<DepthStencilAttachment>,
    ) {
        let color_attachments: SmallVec<[vk::RenderingAttachmentInfo; 8]> = color_attachments.iter()
            .map(|attachment| {
                let texture = utility::resource_storage_ref(self.binding_ctx.resources, attachment.color.id).as_texture();

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
            let texture = utility::resource_storage_ref(self.binding_ctx.resources, attachment.depth.id).as_texture();

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
        self.encoder.bind_vertex_buffers(first_binding, &[self.get(&buf)], offsets);
    }

    #[inline]
    pub fn bind_index_buffer(&self, buf: RenderGraphResourceAccess<Buffer, Srv>, offset: u64, index_ty: vk::IndexType) {
        self.encoder.bind_index_buffer(self.get(&buf), offset, index_ty);
    }
}

pub struct LambdaNodeExecutionContext<'node> {
    device: &'node RenderDevice,
    binding_ctx: NodeBindingContext<'node>,
    encoder: &'node CommandEncoder<'node>,
}

impl<'node> LambdaNodeExecutionContext<'node> {
    #[inline]
    pub fn get<R: GraphResource, V: GraphResourceView>(&self, resource: &RenderGraphResourceAccess<R, V>) -> &R {
        self.binding_ctx.get(resource)
    }

    #[inline]
    pub fn device(&self) -> &RenderDevice { self.device }

    #[inline]
    pub fn command_encoder(&self) -> &CommandEncoder<'node> { self.encoder }
}

pub struct ComputeNodeExecutionContext<'node> {
    device: &'node RenderDevice,
    encoder: &'node CommandEncoder<'node>,
    binding_ctx: NodeBindingContext<'node>,
}

impl<'node> ComputeNodeExecutionContext<'node> {
    #[inline]
    pub fn get<R: GraphResource, V: GraphResourceView>(&self, resource: &RenderGraphResourceAccess<R, V>) -> &R {
        self.binding_ctx.get(resource)
    }

    #[inline]
    pub fn device(&self) -> &RenderDevice { self.device }

    #[inline]
    pub fn encoder(&self) -> &CommandEncoder<'node> { self.encoder }

    #[inline]
    pub fn bind_pipeline(&self, handle: GraphPipelineHandle) -> Option<PipelineResourceBinder<'_>> {
        self.binding_ctx.bind_pipeline(handle)
    }

    #[inline]
    pub fn push_constants<T: NoUninit>(&self, handle: GraphPipelineHandle, offset: u32, data: &T) {
        self.binding_ctx.push_constants(handle, offset, data);
    }

    pub fn dispatch(&self, x: u32, y: u32, z: u32) {
        self.encoder.dispatch(x, y, z);
    }
}

pub struct RetiredRenderGraph {
    resources: Vec<ResourceStorage>,
}

impl RetiredRenderGraph {
    pub fn release_frame_resources(self, resource_cache: &mut TransientResourceCache) {
        for resource in self.resources.into_iter() {
            match resource {
                ResourceStorage::ManagedBuffer { desc, resource, .. } => {
                    resource_cache.recycle_buffer(desc, resource);
                }
                ResourceStorage::ManagedTexture { desc, resource, .. } => {
                    resource_cache.recycle_texture(desc, resource);
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
