use ash::{vk, Device};
use zenith_rhi_derive::DeviceObject;

/// An owning Vulkan fence.
#[DeviceObject]
pub struct Fence {
    fence: vk::Fence,
}

impl Fence {
    pub fn new(device: &Device, signaled: bool) -> Result<Self, vk::Result> {
        let fence_info = vk::FenceCreateInfo::default().flags(if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        });
        let fence = unsafe { device.create_fence(&fence_info, None)? };

        Ok(Self {
            fence,
            device: device.clone(),
        })
    }

    pub fn handle(&self) -> vk::Fence {
        self.fence
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
        }
}
}

/// An owning Vulkan semaphore.
#[DeviceObject]
pub struct Semaphore {
    semaphore: vk::Semaphore,
}

impl Semaphore {
    pub fn new(device: &Device) -> Result<Self, vk::Result> {
        let semaphore = unsafe {
            device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
        };

        Ok(Self {
            semaphore,
            device: device.clone(),
        })
    }

    pub fn handle(&self) -> vk::Semaphore {
        self.semaphore
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.semaphore, None);
        }
    }
}