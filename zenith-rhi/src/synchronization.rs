use ash::vk;
use crate::RenderDevice;

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