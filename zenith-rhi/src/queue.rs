use ash::vk;

/// A queue wrapper that carries its family index.
#[derive(Clone, Copy, Debug)]
pub struct Queue {
    handle: vk::Queue,
    family_index: u32,
}

impl Queue {
    pub fn new(handle: vk::Queue, family_index: u32) -> Self {
        Self { handle, family_index }
    }

    pub fn handle(&self) -> vk::Queue { self.handle }

    pub fn family_index(&self) -> u32 { self.family_index }
}
