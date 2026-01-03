use ash::vk;
use crate::RenderDevice;

/// A non-owning fence handle wrapper for public APIs.
///
/// This type does NOT destroy the underlying Vulkan fence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fence {
    fence: vk::Fence,
}

impl Fence {
    pub fn new(fence: vk::Fence) -> Self { Self { fence } }
    pub fn handle(&self) -> vk::Fence { self.fence }
}

pub struct Semaphore {
    device: ash::Device,
    semaphore: vk::Semaphore,
}

impl Semaphore {
    pub fn new(device: &RenderDevice) -> anyhow::Result<Self> {
        let semaphore = unsafe {
            device.handle().create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
        };

        Ok(Self {
            device: device.handle().clone(),
            semaphore,
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