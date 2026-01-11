//! Vulkan Device - logical device and queue management.

use crate::core::PhysicalDevice;
use crate::defer_release::{DeferRelease, DeferReleaseQueue};
use crate::resource_cache::ResourceCache;
use crate::queue::Queue;
use crate::synchronization::{Fence, Semaphore};
use ash::{vk, Device, Instance};
use std::cell::RefCell;
#[cfg(feature = "validation")]
use std::ffi::CString;
use std::default::Default;
use zenith_core::collections::{SmallVec, hashset::HashSet};
use crate::CommandEncoder;

#[cfg(feature = "validation")]
fn set_debug_name_raw(
    debug_utils: &ash::ext::debug_utils::Device,
    object_handle: u64,
    object_type: vk::ObjectType,
    name: &str,
) {
    if name.is_empty() {
        return;
    }

    let Ok(c_name) = CString::new(name) else {
        return;
    };

    let info = vk::DebugUtilsObjectNameInfoEXT {
        object_type,
        object_handle,
        p_object_name: c_name.as_ptr(),
        ..Default::default()
    };

    unsafe {
        debug_utils.set_debug_utils_object_name(&info).unwrap();
    }
}

#[cfg(not(feature = "validation"))]
#[inline]
fn set_debug_name_raw(
    _debug_utils: &ash::ext::debug_utils::Device,
    _object_handle: u64,
    _object_type: vk::ObjectType,
    _name: &str,
) {
}

/// Set debug name for a raw Vulkan handle (best-effort, no-op without validation).
pub(crate) fn set_debug_name_handle<T: vk::Handle>(
    device: &RenderDevice,
    object: T,
    object_type: vk::ObjectType,
    name: &str,
) {
    #[cfg(feature = "validation")]
    {
        set_debug_name_raw(&device.debug_utils, object.as_raw(), object_type, name);
    }
    #[cfg(not(feature = "validation"))]
    {
        let _ = (device, object, object_type, name);
    }
}

/// Get required device extensions.
fn get_required_device_extensions() -> Vec<*const i8> {
    vec![ash::khr::swapchain::NAME.as_ptr()]
}

/// Vulkan logical device with queues.
pub struct RenderDevice {
    parent_physical_device: PhysicalDevice,
    device: Device,
    #[cfg(feature = "validation")]
    debug_utils: ash::ext::debug_utils::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    frame_resource_fences: Vec<Fence>,
    defer_release_queues: RefCell<Vec<DeferReleaseQueue>>,
    resource_caches: Vec<ResourceCache>,

    current_frame: u8,
}

impl RenderDevice {
    /// Create a new logical device from a physical device.
    pub fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        num_frames: u32,
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
        #[cfg(feature = "validation")]
        let debug_utils = ash::ext::debug_utils::Device::new(instance, &device);

        let graphics_queue = unsafe { device.get_device_queue(physical_device.graphics_queue_family(), 0) };
        let present_queue = unsafe { device.get_device_queue(physical_device.present_queue_family(), 0) };
        
        let resource_caches: Vec<ResourceCache> =
            (0..num_frames as usize).map(|_| ResourceCache::default()).collect();

        let mut device = Self {
            parent_physical_device: physical_device.clone(),
            device,
            #[cfg(feature = "validation")]
            debug_utils,
            graphics_queue,
            present_queue,
            frame_resource_fences: Vec::with_capacity(num_frames as usize),
            defer_release_queues: RefCell::new(Vec::with_capacity(num_frames as usize)),
            resource_caches,
            current_frame: 0,
        };

        for _ in 0..num_frames {
            device.frame_resource_fences.push(Fence::new("fence.execution", &device, true)?);
            device.defer_release_queues.borrow_mut().push(
                DeferReleaseQueue::new()
            );
        }

        set_debug_name_handle(&device, device.handle().handle(), vk::ObjectType::DEVICE, "device.main");
        Ok(device)
    }

    /// Get a reference to the logical device.
    #[inline]
    pub fn handle(&self) -> &Device {
        &self.device
    }

    /// Set debug name for a zenith-rhi wrapper object (best-effort, no-op without validation).
    #[inline]
    pub(crate) fn set_debug_name<T: DebuggableObject>(&self, obj: &T) {
        obj.set_debug_name(self)
    }

    pub fn begin_frame(&mut self) -> usize {
        // wait and reset until execution of current frame completes on GPU side
        unsafe {
            let fence = self.frame_resource_fences[self.current_frame as usize].handle();
            self.device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            self.device.reset_fences(&[fence]).unwrap();
        }
        self.current_frame as _
    }

    #[inline]
    pub fn reset_frame_resources(&self) {
        self.defer_release_queues.borrow_mut()[self.current_frame as usize].release_all();
    }

    #[inline]
    pub fn defer_release<T: DeferRelease>(&self, value: T) {
        self.defer_release_queues.borrow_mut()[self.current_frame as usize].push(value);
    }

    #[inline]
    pub fn last_defer_release_stats(&self) -> crate::LastFreedStats {
        self.defer_release_queues.borrow()[self.current_frame as usize]
            .last_freed()
            .clone()
    }

    #[inline]
    pub fn end_frame(&mut self) {
        self.current_frame = (self.current_frame + 1) % (self.defer_release_queues.borrow().len() as u8);
    }

    #[inline]
    pub fn current_frame_index(&self) -> usize { self.current_frame as _ }

    #[inline]
    pub fn num_frames(&self) -> usize { self.defer_release_queues.borrow().len() as _ }

    pub fn acquire_buffer(&mut self, desc: &crate::BufferDesc) -> Result<crate::Buffer, vk::Result> {
        let frame = self.current_frame as usize;
        {
            let cache = &mut self.resource_caches[frame];
            if let Some(buf) = cache.pop_buffer(desc) {
                return Ok(buf);
            }
        }
        crate::Buffer::new(self, desc)
    }

    #[inline]
    pub fn recycle_buffer(&mut self, desc: crate::BufferDesc, buffer: crate::Buffer) {
        let frame = self.current_frame as usize;
        self.resource_caches[frame].recycle_buffer(desc, buffer);
    }

    pub fn acquire_texture(&mut self, desc: &crate::TextureDesc) -> Result<crate::Texture, vk::Result> {
        let frame = self.current_frame as usize;
        {
            let cache = &mut self.resource_caches[frame];
            if let Some(tex) = cache.pop_texture(desc) {
                return Ok(tex);
            }
        }
        crate::Texture::new(self, desc)
    }

    #[inline]
    pub fn recycle_texture(&mut self, desc: crate::TextureDesc, texture: crate::Texture) {
        let frame = self.current_frame as usize;
        self.resource_caches[frame].recycle_texture(desc, texture);
    }

    #[inline]
    pub fn resource_cache(&self) -> &ResourceCache {
        &self.resource_caches[self.current_frame as usize]
    }

    #[inline]
    pub fn resource_cache_mut(&mut self) -> &mut ResourceCache {
        &mut self.resource_caches[self.current_frame as usize]
    }

    pub fn frame_resource_fence(&self) -> &Fence {
        &self.frame_resource_fences[self.current_frame as usize]
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
        encoder: CommandEncoder<'a>,
        queue: Queue,
        wait_semaphores: &'a [&Semaphore],
        wait_stage: vk::PipelineStageFlags2,
        signal_semaphores: &'a [&Semaphore],
        signal_stage: vk::PipelineStageFlags2,
        fence: &Fence,
    ) {
        let command_submit_info = vk::CommandBufferSubmitInfo::default()
            .command_buffer(encoder.handle());

        let wait_semaphore_infos = wait_semaphores.iter()
            .map(|semaphore| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(semaphore.handle())
                    .stage_mask(wait_stage)
            })
            .collect::<SmallVec<[vk::SemaphoreSubmitInfo; 4]>>();

        let signal_semaphore_infos = signal_semaphores.iter()
            .map(|semaphore| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(semaphore.handle())
                    .stage_mask(signal_stage)
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
        unsafe { self.device.device_wait_idle().unwrap(); }

        for queue in self.defer_release_queues.get_mut() {
            queue.release_all();
        }
        // Cached resources may still hold Buffers/Textures that require `Device` to destroy.
        for cache in &mut self.resource_caches {
            cache.clear();
        }
        self.resource_caches.clear();
        self.frame_resource_fences.clear();

        unsafe {
            self.device.destroy_device(None);
        }
    }
}

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// Crate-only trait for objects that own an `ash::Device` used for destruction and device calls.
///
/// This trait is sealed and not visible to users of `zenith-rhi`.
pub trait DeviceObject: sealed::Sealed {
    fn device(&self) -> &Device;
}

/// Crate-only trait for objects that can name their Vulkan resources.
///
/// This is used by `RenderDevice::set_debug_name(&T)` to delegate debug-name work to the object
/// itself (so it can also name sub-resources like `vk::DeviceMemory`).
pub(crate) trait DebuggableObject {
    fn set_debug_name(&self, device: &RenderDevice);
}
