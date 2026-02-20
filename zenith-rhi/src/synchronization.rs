use std::sync::Arc;
use ash::{vk};
use zenith_rhi_derive::DeviceObject;
use crate::{RenderDevice};
use crate::device::DebuggableObject;
use crate::device::set_debug_name_handle;

#[DeviceObject]
pub struct Fence {
    name: String,
    fence: vk::Fence,
}

impl Fence {
    pub fn new(name: &str, device: &Arc<RenderDevice>, signaled: bool) -> Result<Self, vk::Result> {
        let fence_info = vk::FenceCreateInfo::default().flags(if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        });
        let fence = unsafe { device.handle().create_fence(&fence_info, None)? };

        let fence = Self {
            name: name.to_string(),
            fence,
            device: device.clone(),
        };
        device.set_debug_name(&fence, name);
        Ok(fence)
    }

    #[inline]
    pub fn name(&self) -> &str { &self.name }

    #[inline]
    pub fn handle(&self) -> vk::Fence {
        self.fence
    }
}

impl DebuggableObject for Fence {
    fn set_debug_name(&self, device: &ash::ext::debug_utils::Device, name: &str) {
        set_debug_name_handle(device, self.fence, vk::ObjectType::FENCE, name);
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.device.handle().destroy_fence(self.fence, None);
        }
    }
}

#[DeviceObject]
pub struct Semaphore {
    name: String,
    semaphore: vk::Semaphore,
}

impl Semaphore {
    pub fn new(name: &str, device: &Arc<RenderDevice>) -> Result<Self, vk::Result> {
        let semaphore = unsafe {
            device.handle().create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
        };

        let semaphore = Self {
            name: name.to_string(),
            semaphore,
            device: device.clone(),
        };
        device.set_debug_name(&semaphore, name);
        Ok(semaphore)
    }

    #[inline]
    pub fn name(&self) -> &str { &self.name }
    
    #[inline]
    pub fn handle(&self) -> vk::Semaphore {
        self.semaphore
    }
}

impl DebuggableObject for Semaphore {
    fn set_debug_name(&self, device: &ash::ext::debug_utils::Device, name: &str) {
        set_debug_name_handle(device, self.semaphore, vk::ObjectType::SEMAPHORE, name);
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.handle().destroy_semaphore(self.semaphore, None);
        }
    }
}