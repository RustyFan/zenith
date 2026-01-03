//! Immediate command submission helpers.

use ash::{vk, Device};

use crate::{CommandEncoder, CommandPool};
use crate::Queue;

/// An immediate encoder that can submit commands to a queue at any time and
/// block on a fence until completion.
pub struct ImmediateCommandEncoder {
    device: Device,
    queue: Queue,
    pool: CommandPool,
    fence: vk::Fence,
}

impl ImmediateCommandEncoder {
    pub fn new(device: &Device, queue: Queue) -> Result<Self, vk::Result> {
        let pool = CommandPool::new(device, queue.family_index(), vk::CommandPoolCreateFlags::empty())?;
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { device.create_fence(&fence_info, None)? };

        Ok(Self {
            device: device.clone(),
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
        let cmd = self.pool.allocate()?;

        let encoder = CommandEncoder::new(&self.device, cmd);
        encoder.begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;
        record(&encoder);
        encoder.end()?;

        let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
        let submit_info = vk::SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&cmd_info));

        unsafe {
            self.device.queue_submit2(self.queue.handle(), &[submit_info], self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            self.device.reset_fences(&[self.fence])?;
        }

        Ok(())
    }

    pub fn device(&self) -> &Device { &self.device }

    pub fn queue(&self) -> Queue { self.queue }
}

impl Drop for ImmediateCommandEncoder {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
        }
    }
}


