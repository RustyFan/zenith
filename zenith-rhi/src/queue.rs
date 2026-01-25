use ash::vk;

#[derive(Clone, Copy, Debug)]
pub struct Queue {
    handle: vk::Queue,
    family_index: u32,
}

impl Queue {
    pub fn new(handle: vk::Queue, family_index: u32) -> Self {
        Self { handle, family_index }
    }

    #[inline]
    pub fn handle(&self) -> vk::Queue { self.handle }

    #[inline]
    pub fn family_index(&self) -> u32 { self.family_index }
}
