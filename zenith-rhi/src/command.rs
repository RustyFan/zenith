//! Command buffer pool and recorder.

use std::cell::{Cell, RefCell};
use ash::{vk};
use zenith_rhi_derive::DeviceObject;
use crate::barrier::{BufferBarrier, TextureBarrier, MemoryBarrier};
use crate::{Queue, RenderDevice};
use crate::synchronization::Fence;
use crate::device::DebuggableObject;
use crate::device::set_debug_name_handle;

/// Command buffer pool for allocating command buffers.
#[DeviceObject]
pub struct CommandPool {
    name: String,
    pool: vk::CommandPool,
    buffers: RefCell<Vec<vk::CommandBuffer>>,
    next_index: Cell<usize>,
}

impl CommandPool {
    pub fn new(
        name: &str,
        device: &RenderDevice,
        queue_family: u32,
        flags: vk::CommandPoolCreateFlags,
    ) -> Result<Self, vk::Result> {
        let create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(flags);

        let pool = unsafe { device.handle().create_command_pool(&create_info, None)? };
        let pool = Self {
            name: name.to_string(),
            pool,
            buffers: RefCell::new(Vec::new()),
            next_index: Cell::new(0),
            device: device.handle().clone(),
        };
        device.set_debug_name(&pool);
        Ok(pool)
    }

    pub fn allocate(&self) -> Result<vk::CommandBuffer, vk::Result> {
        let index = self.next_index.get();
        self.next_index.set(index + 1);

        if let Some(buffer) = self.buffers.borrow().get(index) {
            return Ok(buffer.clone());
        }

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let buffers = unsafe { self.device.allocate_command_buffers(&alloc_info)? };
        let cmd = buffers[0];

        self.buffers.borrow_mut().push(cmd.clone());
        Ok(cmd)
    }

    pub fn reset(&self) -> Result<(), vk::Result> {
        self.next_index.set(0);
        unsafe { self.device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()) }
    }

    pub fn handle(&self) -> vk::CommandPool {
        self.pool
    }

    #[inline]
    pub fn name(&self) -> &str { &self.name }
}

impl DebuggableObject for CommandPool {
    fn set_debug_name(&self, device: &RenderDevice) {
        set_debug_name_handle(device, self.pool, vk::ObjectType::COMMAND_POOL, self.name());
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.pool, None);
        }
    }
}

/// Command encoder wrapping a command buffer with common graphics commands.
pub struct CommandEncoder<'a> {
    name: String,
    device: &'a RenderDevice,
    cmd: vk::CommandBuffer,
}

impl<'a> CommandEncoder<'a> {
    pub fn new(name: &str, device: &'a RenderDevice, pool: &CommandPool) -> anyhow::Result<Self> {
        let cmd = pool.allocate()?;
        let encoder = Self {
            name: name.to_owned(),
            device,
            cmd,
        };
        device.set_debug_name(&encoder);
        Ok(encoder)
    }

    pub fn begin(&self, flags: vk::CommandBufferUsageFlags) -> Result<(), vk::Result> {
        let begin_info = vk::CommandBufferBeginInfo::default().flags(flags);
        unsafe { self.device.handle().begin_command_buffer(self.cmd, &begin_info) }
    }

    pub fn end(&self) -> Result<(), vk::Result> {
        unsafe { self.device.handle().end_command_buffer(self.cmd) }
    }

    pub fn handle(&self) -> vk::CommandBuffer {
        self.cmd
    }

    // Pipeline commands
    pub fn bind_pipeline(&self, bind_point: vk::PipelineBindPoint, pipeline: vk::Pipeline) {
        unsafe { self.device.handle().cmd_bind_pipeline(self.cmd, bind_point, pipeline) }
    }

    pub fn bind_graphics_pipeline(&self, pipeline: vk::Pipeline) {
        self.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline);
    }

    pub fn bind_descriptor_sets(
        &self,
        bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        unsafe {
            self.device.handle().cmd_bind_descriptor_sets(
                self.cmd,
                bind_point,
                layout,
                first_set,
                descriptor_sets,
                dynamic_offsets,
            )
        }
    }

    // Vertex/Index buffer commands
    pub fn bind_vertex_buffers(&self, first_binding: u32, buffers: &[vk::Buffer], offsets: &[vk::DeviceSize]) {
        unsafe { self.device.handle().cmd_bind_vertex_buffers(self.cmd, first_binding, buffers, offsets) }
    }

    pub fn bind_index_buffer(&self, buffer: vk::Buffer, offset: vk::DeviceSize, index_type: vk::IndexType) {
        unsafe { self.device.handle().cmd_bind_index_buffer(self.cmd, buffer, offset, index_type) }
    }

    // Draw commands
    pub fn draw(&self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
        unsafe { self.device.handle().cmd_draw(self.cmd, vertex_count, instance_count, first_vertex, first_instance) }
    }

    pub fn draw_indexed(&self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32) {
        unsafe { self.device.handle().cmd_draw_indexed(self.cmd, index_count, instance_count, first_index, vertex_offset, first_instance) }
    }

    // Dynamic state commands
    pub fn set_viewport(&self, first: u32, viewports: &[vk::Viewport]) {
        unsafe { self.device.handle().cmd_set_viewport(self.cmd, first, viewports) }
    }

    pub fn set_scissor(&self, first: u32, scissors: &[vk::Rect2D]) {
        unsafe { self.device.handle().cmd_set_scissor(self.cmd, first, scissors) }
    }

    // Push constants
    pub fn push_constants<T: Copy>(&self, layout: vk::PipelineLayout, stages: vk::ShaderStageFlags, offset: u32, data: &T) {
        let bytes = unsafe {
            std::slice::from_raw_parts(data as *const T as *const u8, std::mem::size_of::<T>())
        };
        unsafe { self.device.handle().cmd_push_constants(self.cmd, layout, stages, offset, bytes) }
    }

    // Dynamic rendering (Vulkan 1.3)
    pub fn begin_rendering(&self, info: &vk::RenderingInfo) {
        unsafe { self.device.handle().cmd_begin_rendering(self.cmd, info) }
    }

    pub fn end_rendering(&self) {
        unsafe { self.device.handle().cmd_end_rendering(self.cmd) }
    }
    
    pub fn buffer_barriers<'b>(&self, barriers: &[BufferBarrier<'b>]) {
        if barriers.is_empty() {
            return;
        }
        let vk_barriers: Vec<vk::BufferMemoryBarrier2> = barriers.iter().map(|b| b.to_vk()).collect();
        let dep = vk::DependencyInfo::default().buffer_memory_barriers(&vk_barriers);
        unsafe { self.device.handle().cmd_pipeline_barrier2(self.cmd, &dep) }
    }

    pub fn texture_barriers<'b>(&self, barriers: &[TextureBarrier<'b>]) {
        if barriers.is_empty() {
            return;
        }
        let vk_barriers: Vec<vk::ImageMemoryBarrier2> = barriers.iter().map(|b| b.to_vk()).collect();
        let dep = vk::DependencyInfo::default().image_memory_barriers(&vk_barriers);
        unsafe { self.device.handle().cmd_pipeline_barrier2(self.cmd, &dep) }
    }

    pub fn memory_barrier(&self, barriers: &[MemoryBarrier]) {
        if barriers.is_empty() {
            return;
        }
        let vk_barriers: Vec<vk::MemoryBarrier2> = barriers.iter().map(|b| b.to_vk()).collect();
        let dep = vk::DependencyInfo::default().memory_barriers(&vk_barriers);
        unsafe { self.device.handle().cmd_pipeline_barrier2(self.cmd, &dep) }
    }

    // Copy commands
    pub fn copy_buffer(&self, src: vk::Buffer, dst: vk::Buffer, regions: &[vk::BufferCopy]) {
        unsafe { self.device.handle().cmd_copy_buffer(self.cmd, src, dst, regions) }
    }

    pub fn copy_buffer_to_image(&self, src: vk::Buffer, dst: vk::Image, layout: vk::ImageLayout, regions: &[vk::BufferImageCopy]) {
        unsafe { self.device.handle().cmd_copy_buffer_to_image(self.cmd, src, dst, layout, regions) }
    }

    // Blit
    pub fn blit_image(&self, src: vk::Image, src_layout: vk::ImageLayout, dst: vk::Image, dst_layout: vk::ImageLayout, regions: &[vk::ImageBlit], filter: vk::Filter) {
        unsafe { self.device.handle().cmd_blit_image(self.cmd, src, src_layout, dst, dst_layout, regions, filter) }
    }

    pub fn custom<F>(&self, func: F)
    where
        F: FnOnce(&RenderDevice, vk::CommandBuffer)
    {
        func(&self.device, self.cmd.clone());
    }
}

impl<'a> DebuggableObject for CommandEncoder<'a> {
    fn set_debug_name(&self, device: &RenderDevice) {
        set_debug_name_handle(device, self.cmd, vk::ObjectType::COMMAND_BUFFER, &self.name);
    }
}

/// An immediate encoder that can submit commands to a queue at any time and
/// block on a fence until completion.
pub struct ImmediateCommandEncoder<'a> {
    device: &'a RenderDevice,
    queue: Queue,
    pool: CommandPool,
    fence: Fence,
}

impl<'a> ImmediateCommandEncoder<'a> {
    pub fn new(device: &'a RenderDevice, queue: Queue) -> Result<Self, vk::Result> {
        let pool = CommandPool::new("command_pool.immediate", device, queue.family_index(), vk::CommandPoolCreateFlags::empty())?;
        let fence = Fence::new("fence.immediate", device, false)?;

        Ok(Self {
            device,
            queue,
            pool,
            fence,
        })
    }

    /// Record commands and submit immediately, blocking until the GPU finishes.
    pub fn submit_and_wait<F>(&self, record: F) -> Result<(), vk::Result>
    where
        F: FnOnce(&CommandEncoder),
    {
        self.pool.reset()?;

        let encoder = CommandEncoder::new("cmd.immediate", self.device, &self.pool)
            .map_err(|_| vk::Result::ERROR_UNKNOWN)?;

        encoder.begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;
        record(&encoder);
        encoder.end()?;

        let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(encoder.handle());
        let submit_info = vk::SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&cmd_info));

        unsafe {
            let fence = self.fence.handle();
            self.device.handle().queue_submit2(self.queue.handle(), &[submit_info], fence)?;
            self.device.handle().wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.handle().reset_fences(&[fence])?;
        }

        Ok(())
    }

    pub fn device(&self) -> &RenderDevice { &self.device }

    pub fn queue(&self) -> Queue { self.queue }
}
