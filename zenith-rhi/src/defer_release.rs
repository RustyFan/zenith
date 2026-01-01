use crate::{Buffer, Texture};

pub struct DeferReleaseQueue {
    buffers: Vec<Buffer>,
    textures: Vec<Texture>,
}

impl DeferReleaseQueue {
    pub fn new() -> Self {
        Self {
            buffers: Default::default(),
            textures: Default::default(),
        }
    }

    pub fn add_buffer(&mut self, buffer: Buffer) { 
        self.buffers.push(buffer); 
    }

    pub fn add_texture(&mut self, texture: Texture) {
        self.textures.push(texture);
    }

    pub fn release_all(&mut self) {
        self.buffers.clear();
        self.textures.clear();
    }

    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }
}