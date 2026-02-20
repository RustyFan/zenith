//! Command buffer pool and recorder.

use std::cell::{Cell, RefCell};
use std::collections::Bound;
use std::ops::RangeBounds;
use std::sync::Arc;
use ash::{vk};
use bytemuck::NoUninit;
use glam::Vec4;
use zenith_core::collections::SmallVec;
use zenith_rhi_derive::DeviceObject;
use crate::barrier::{BufferBarrier, TextureBarrier, MemoryBarrier};
use crate::{Buffer, Queue, RenderDevice, Texture, TextureLayout};
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
        device: &Arc<RenderDevice>,
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
            device: device.clone(),
        };
        device.set_debug_name(&pool, name);
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

        let buffers = unsafe { self.device.handle().allocate_command_buffers(&alloc_info)? };
        let cmd = buffers[0];

        self.buffers.borrow_mut().push(cmd.clone());
        Ok(cmd)
    }

    pub fn reset(&self) -> Result<(), vk::Result> {
        self.next_index.set(0);
        unsafe { self.device.handle().reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()) }
    }

    pub fn handle(&self) -> vk::CommandPool {
        self.pool
    }

    #[inline]
    pub fn name(&self) -> &str { &self.name }
}

impl DebuggableObject for CommandPool {
    fn set_debug_name(&self, device: &ash::ext::debug_utils::Device, name: &str) {
        set_debug_name_handle(device, self.pool, vk::ObjectType::COMMAND_POOL, name);
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device.handle().destroy_command_pool(self.pool, None);
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
        device.set_debug_name(&encoder, name);
        Ok(encoder)
    }

    pub fn begin(&self, flags: vk::CommandBufferUsageFlags) -> Result<(), vk::Result> {
        let begin_info = vk::CommandBufferBeginInfo::default().flags(flags);
        unsafe { self.device.handle().begin_command_buffer(self.cmd, &begin_info) }
    }

    pub fn end(&self) -> Result<(), vk::Result> {
        unsafe { self.device.handle().end_command_buffer(self.cmd) }
    }

    #[inline]
    pub fn name(&self) -> &str { &self.name }

    #[inline]
    pub fn handle(&self) -> vk::CommandBuffer {
        self.cmd
    }

    // Debug label commands
    #[cfg(debug_assertions)]
    pub fn begin_debug_label(&self, label: &str, color: Vec4) {
        use std::ffi::CString;
        let c_label = CString::new(label).unwrap();
        let label_info = vk::DebugUtilsLabelEXT::default()
            .label_name(c_label.as_c_str())
            .color(color.to_array());
        unsafe {
            self.device
                .debug_utils
                .cmd_begin_debug_utils_label(self.cmd, &label_info);
        }
    }

    #[cfg(not(debug_assertions))]
    pub fn begin_debug_label(&self, _label: &str, _color: Vec4) {}

    #[cfg(debug_assertions)]
    pub fn end_debug_label(&self) {
        unsafe {
            self.device
                .debug_utils
                .cmd_end_debug_utils_label(self.cmd);
        }
    }

    #[cfg(not(debug_assertions))]
    pub fn end_debug_label(&self) {}

    pub fn bind_pipeline(&self, bind_point: vk::PipelineBindPoint, pipeline: vk::Pipeline) {
        unsafe { self.device.handle().cmd_bind_pipeline(self.cmd, bind_point, pipeline) }
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
    pub fn bind_vertex_buffers(&self, first_binding: u32, buffers: &[&Buffer], offsets: &[vk::DeviceSize]) {
        let buffers = buffers.iter().map(|buf| buf.handle()).collect::<SmallVec<[_; 4]>>();
        unsafe { self.device.handle().cmd_bind_vertex_buffers(self.cmd, first_binding, &buffers, offsets) }
    }

    pub fn bind_index_buffer(&self, buffer: &Buffer, offset: vk::DeviceSize, index_type: vk::IndexType) {
        unsafe { self.device.handle().cmd_bind_index_buffer(self.cmd, buffer.handle(), offset, index_type) }
    }

    // Draw commands
    pub fn draw<R: RangeBounds<u32>>(&self, vertex: R, instance: R) {
        let (first_vertex, vertex_count) = Self::extract_offset_and_size(vertex);
        let (first_instance, instance_count) = Self::extract_offset_and_size(instance);

        unsafe { self.device.handle().cmd_draw(self.cmd, vertex_count, instance_count, first_vertex, first_instance) }
    }

    pub fn draw_indexed<R: RangeBounds<u32>>(&self, index: R, instance: R, vertex_offset: i32) {
        let (first_index, index_count) = Self::extract_offset_and_size(index);
        let (first_instance, instance_count) = Self::extract_offset_and_size(instance);

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
    pub fn push_constants<T: NoUninit>(&self, layout: vk::PipelineLayout, stages: vk::ShaderStageFlags, offset: u32, data: &T) {
        let bytes = bytemuck::cast_slice(std::slice::from_ref(data));
        unsafe { self.device.handle().cmd_push_constants(self.cmd, layout, stages, offset, bytes) }
    }

    /// Dispatch a compute workload.
    pub fn dispatch(&self, x: u32, y: u32, z: u32) {
        unsafe { self.device.handle().cmd_dispatch(self.cmd, x, y, z) }
    }

    // Dynamic rendering (Vulkan 1.3)
    pub fn begin_rendering(&self, info: &vk::RenderingInfo) {
        unsafe { self.device.handle().cmd_begin_rendering(self.cmd, info) }
    }

    pub fn end_rendering(&self) {
        unsafe { self.device.handle().cmd_end_rendering(self.cmd) }
    }

    pub fn pipeline_barriers(&self,
                             mem_barriers: &[MemoryBarrier], 
                             tex_barriers: &[TextureBarrier], 
                             buf_barriers: &[BufferBarrier],
    ) {
        let vk_buf_barriers = buf_barriers.iter().map(|b| b.to_vk()).collect::<SmallVec<[_; 8]>>();
        let vk_tex_barriers = tex_barriers.iter().map(|b| b.to_vk()).collect::<SmallVec<[_; 8]>>();
        let vk_mem_barriers = mem_barriers.iter().map(|b| b.to_vk()).collect::<SmallVec<[_; 2]>>();
        
        let dep = vk::DependencyInfo::default()
            .buffer_memory_barriers(&vk_buf_barriers)
            .image_memory_barriers(&vk_tex_barriers)
            .memory_barriers(&vk_mem_barriers);
        
        unsafe { self.device.handle().cmd_pipeline_barrier2(self.cmd, &dep) }
    }

    // Copy commands
    pub fn copy_buffer(&self, src: &Buffer, dst: &Buffer, regions: &[vk::BufferCopy]) {
        unsafe { self.device.handle().cmd_copy_buffer(self.cmd, src.handle(), dst.handle(), regions) }
    }

    pub fn copy_buffer_to_image(&self, src: &Buffer, dst: &Texture, layout: TextureLayout, regions: &[vk::BufferImageCopy]) {
        unsafe { self.device.handle().cmd_copy_buffer_to_image(self.cmd, src.handle(), dst.handle(), layout.to_vk(), regions) }
    }

    // Blit
    pub fn blit_image(&self, 
                      src: &Texture, src_layout: TextureLayout,
                      dst: &Texture, dst_layout: TextureLayout,
                      regions: &[vk::ImageBlit], filter: vk::Filter
    ) {
        unsafe {
            self.device.handle().cmd_blit_image(self.cmd, 
                                                src.handle(), src_layout.to_vk(), 
                                                dst.handle(), dst_layout.to_vk(), 
                                                regions, filter) 
        }
    }

    pub fn custom<F>(&self, func: F)
    where
        F: FnOnce(&RenderDevice, vk::CommandBuffer)
    {
        func(&self.device, self.cmd.clone());
    }

    fn extract_offset_and_size<R: RangeBounds<u32>>(range: R) -> (u32, u32) {
        let start = match range.start_bound() {
            Bound::Included(&v) => v,
            Bound::Excluded(&v) => v.checked_add(1).expect("Index overflow"),
            Bound::Unbounded => 0,
        };
        let end_exclusive = match range.end_bound() {
            Bound::Included(&v) => v.checked_add(1).expect("Index overflow"),
            Bound::Excluded(&v) => v,
            Bound::Unbounded => panic!("Draw indexed range should have an upper bound!"),
        };

        if start > end_exclusive {
            panic!("Draw indexed do DOT support negative range length!");
        }

        (start, end_exclusive - start)
    }
}

impl<'a> DebuggableObject for CommandEncoder<'a> {
    fn set_debug_name(&self, device: &ash::ext::debug_utils::Device, name: &str) {
        set_debug_name_handle(device, self.cmd, vk::ObjectType::COMMAND_BUFFER, name);
    }
}

pub struct ImmediateCommandEncoder<'a> {
    device: &'a RenderDevice,
    queue: Queue,
    pool: CommandPool,
    fence: Fence,
}

impl<'a> ImmediateCommandEncoder<'a> {
    pub fn new(device: &'a Arc<RenderDevice>, queue: Queue) -> Result<Self, vk::Result> {
        let pool = CommandPool::new("command_pool.immediate", device, queue.family_index(), vk::CommandPoolCreateFlags::empty())?;
        let fence = Fence::new("fence.immediate", device, false)?;

        Ok(Self {
            device: &*device,
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
