//! Vulkan Device - logical device and queue management.

use crate::command::CommandPool;
use crate::core::PhysicalDevice;
use crate::defer_release::DeferReleaseQueue;
use crate::descriptor::{DescriptorPool, DescriptorSetLayout};
use crate::{Buffer, Texture};
use crate::queue::Queue;
use crate::synchronization::Fence;
use ash::{vk, Device, Instance};
use std::collections::HashSet;
use zenith_core::collections::SmallVec;

/// Get required device extensions.
fn get_required_device_extensions() -> Vec<*const i8> {
    vec![ash::khr::swapchain::NAME.as_ptr()]
}

/// Vulkan logical device with queues.
pub struct RenderDevice {
    parent_physical_device: PhysicalDevice,
    device: Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    execute_command_pools: Vec<CommandPool>,
    present_command_pools: Vec<CommandPool>,
    frame_resource_fences: Vec<vk::Fence>,
    defer_release_queues: Vec<DeferReleaseQueue>,
    descriptor_pools: Vec<DescriptorPool>,

    num_frames: u8,
    current_frame: u8,
}

impl RenderDevice {
    /// Create a new logical device from a physical device.
    pub fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        num_frames: u8,
    ) -> Result<Self, vk::Result> {
        // Collect unique queue families
        let unique_families: HashSet<u32> = [physical_device.graphics_queue_family(), physical_device.present_queue_family()]
            .into_iter()
            .collect();

        let queue_priority = 1.0f32;

        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = unique_families
            .iter()
            .map(|&family| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(std::slice::from_ref(&queue_priority))
            })
            .collect();

        let extensions = get_required_device_extensions();

        // Enable features
        let features = vk::PhysicalDeviceFeatures::default();
            // .sampler_anisotropy(true)
            // .fill_mode_non_solid(true);

        // Vulkan 1.2 features
        // let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
        //     .descriptor_indexing(true)
        //     .buffer_device_address(true)
        //     .timeline_semaphore(true);

        // Vulkan 1.3 features
        let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extensions)
            .enabled_features(&features)
            // .push_next(&mut vulkan_12_features)
            .push_next(&mut vulkan_13_features);

        let device = unsafe { instance.create_device(physical_device.handle(), &create_info, None)? };

        let graphics_queue = unsafe { device.get_device_queue(physical_device.graphics_queue_family(), 0) };
        let present_queue = unsafe { device.get_device_queue(physical_device.present_queue_family(), 0) };

        let mut execute_command_pools = Vec::with_capacity(num_frames as usize);
        let mut present_command_pools = Vec::with_capacity(num_frames as usize);
        let mut frame_resource_fences = Vec::with_capacity(num_frames as usize);
        let mut defer_release_queues = Vec::with_capacity(num_frames as usize);
        let mut descriptor_pools = Vec::with_capacity(num_frames as usize);
        for _ in 0..num_frames {
            execute_command_pools.push(
                CommandPool::new(&device, physical_device.graphics_queue_family(), vk::CommandPoolCreateFlags::empty())?
            );
            present_command_pools.push(
                CommandPool::new(&device, physical_device.graphics_queue_family(), vk::CommandPoolCreateFlags::empty())?
            );
            let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            frame_resource_fences.push(
                unsafe { device.create_fence(&fence_info, None)? }
            );
            defer_release_queues.push(
                DeferReleaseQueue::new()
            );
            // Create descriptor pool with common descriptor types
            let pool_sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 1000 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1000 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 1000 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: 500 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 500 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLER, descriptor_count: 100 },
            ];
            descriptor_pools.push(
                DescriptorPool::new(&device, 1000, &pool_sizes)?
            );
        }

        Ok(Self {
            parent_physical_device: physical_device.clone(),
            device,
            graphics_queue,
            present_queue,
            execute_command_pools,
            present_command_pools,
            frame_resource_fences,
            defer_release_queues,
            descriptor_pools,
            num_frames,
            current_frame: 0,
        })
    }

    /// Get a reference to the logical device.
    pub fn handle(&self) -> &Device {
        &self.device
    }

    pub fn execute_command_pool(&self) -> &CommandPool {
        &self.execute_command_pools[self.current_frame as usize]
    }

    pub fn present_command_pool(&self) -> &CommandPool {
        &self.present_command_pools[self.current_frame as usize]
    }

    pub fn begin_frame(&mut self) {
        // wait and reset until execution of current frame completes on GPU side
        unsafe {
            self.device.wait_for_fences(&[self.frame_resource_fences[self.current_frame as usize]], true, u64::MAX).unwrap();
            self.device.reset_fences(&[self.frame_resource_fences[self.current_frame as usize]]).unwrap();
        }
        self.execute_command_pools[self.current_frame as usize].reset().expect("Failed to reset execute command pool");
    }

    pub fn reset_frame_resources(&mut self) {
        self.descriptor_pools[self.current_frame as usize].reset().expect("Failed to reset descriptor pool");
        self.defer_release_queues[self.current_frame as usize].release_all();
    }

    pub fn defer_release_buffer(&mut self, buffer: Buffer) {
        self.defer_release_queues[self.current_frame as usize].add_buffer(buffer);
    }

    pub fn defer_release_texture(&mut self, texture: Texture) {
        self.defer_release_queues[self.current_frame as usize].add_texture(texture);
    }

    /// Get the current frame's descriptor pool.
    pub fn descriptor_pool(&self) -> &DescriptorPool {
        &self.descriptor_pools[self.current_frame as usize]
    }

    /// Allocate a descriptor set from the current frame's pool.
    pub fn allocate_descriptor_set(&self, layout: &DescriptorSetLayout) -> Result<vk::DescriptorSet, vk::Result> {
        self.descriptor_pools[self.current_frame as usize].allocate(layout)
    }

    pub fn end_frame(&mut self) {
        self.current_frame = (self.current_frame + 1) % self.num_frames;
    }

    pub fn frame_index(&self) -> usize { self.current_frame as _ }

    pub fn frame_resource_fence(&self) -> Fence {
        Fence::new(self.frame_resource_fences[self.current_frame as usize])
    }

    /// Get the total number of deferred buffers for this frame.
    pub fn deferred_buffer_count(&self) -> usize {
        self.defer_release_queues[self.current_frame as usize].buffer_count()
    }

    /// Get the total number of deferred textures for this frame.
    pub fn deferred_texture_count(&self) -> usize {
        self.defer_release_queues[self.current_frame as usize].texture_count()
    }

    /// Get the physical device properties.
    pub fn properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.parent_physical_device.properties()
    }

    /// Get the physical device memory properties.
    pub fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.parent_physical_device.memory_properties()
    }

    pub fn graphics_queue(&self) -> Queue {
        Queue::new(self.graphics_queue, self.parent_physical_device.graphics_queue_family())
    }

    pub fn present_queue(&self) -> Queue {
        Queue::new(self.present_queue, self.parent_physical_device.present_queue_family())
    }

    pub fn wait_until_idle(&self) -> Result<(), vk::Result> {
        unsafe { self.device.device_wait_idle() }
    }

    pub fn parent_physical_device(&self) -> &PhysicalDevice {
        &self.parent_physical_device
    }

    pub fn submit_commands<'a>(
        &self,
        command: vk::CommandBuffer,
        queue: Queue,
        wait_semaphores: &'a [vk::Semaphore],
        wait_stage: vk::PipelineStageFlags2,
        signal_semaphores: &'a [vk::Semaphore],
        signal_stage: vk::PipelineStageFlags2,
        fence: Fence,
    ) {
        let command_submit_info = vk::CommandBufferSubmitInfo::default()
            .command_buffer(command);

        let wait_semaphore_infos = wait_semaphores.iter()
            .map(|semaphore| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(*semaphore)
                    .stage_mask(signal_stage)
            })
            .collect::<SmallVec<[vk::SemaphoreSubmitInfo; 4]>>();

        let signal_semaphore_infos = signal_semaphores.iter()
            .map(|semaphore| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(*semaphore)
                    .stage_mask(wait_stage)
            })
            .collect::<SmallVec<[vk::SemaphoreSubmitInfo; 4]>>();

        let submit_info = vk::SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&command_submit_info))
            .wait_semaphore_infos(&wait_semaphore_infos)
            .signal_semaphore_infos(&signal_semaphore_infos);

        unsafe {
            self.device.queue_submit2(
                queue.handle(),
                &[submit_info],
                fence.handle()
            ).unwrap();
        }
    }
}

impl Drop for RenderDevice {
    fn drop(&mut self) {
        self.execute_command_pools.clear();
        self.present_command_pools.clear();
        self.descriptor_pools.clear();

        for queue in &mut self.defer_release_queues {
            queue.release_all();
        }

        for fence in &mut self.frame_resource_fences {
            unsafe { self.device.destroy_fence(*fence, None); }
        }

        unsafe {
            self.device.destroy_device(None);
        }
    }
}
