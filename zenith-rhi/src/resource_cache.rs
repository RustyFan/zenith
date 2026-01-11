use crate::{vk, Buffer, BufferDesc, RenderDevice, Texture, TextureDesc};
use std::collections::HashMap;

#[derive(Default)]
pub struct ResourceCache {
    available_buffers: HashMap<BufferDesc, Vec<Buffer>>,
    available_textures: HashMap<TextureDesc, Vec<Texture>>,
}

impl ResourceCache {
    pub(crate) fn pop_buffer(&mut self, desc: &BufferDesc) -> Option<Buffer> {
        self.available_buffers.get_mut(desc).and_then(|list| list.pop())
    }

    pub(crate) fn pop_texture(&mut self, desc: &TextureDesc) -> Option<Texture> {
        self.available_textures.get_mut(desc).and_then(|list| list.pop())
    }

    pub fn acquire_buffer(
        &mut self,
        device: &RenderDevice,
        desc: &BufferDesc,
    ) -> Result<Buffer, vk::Result> {
        if let Some(buf) = self.pop_buffer(desc) {
            return Ok(buf);
        }
        Buffer::new(device, desc)
    }

    pub fn recycle_buffer(&mut self, desc: BufferDesc, buffer: Buffer) {
        self.available_buffers.entry(desc).or_default().push(buffer);
    }

    pub fn acquire_texture(
        &mut self,
        device: &RenderDevice,
        desc: &TextureDesc,
    ) -> Result<Texture, vk::Result> {
        if let Some(tex) = self.pop_texture(desc) {
            return Ok(tex);
        }
        Texture::new(device, desc)
    }

    pub fn recycle_texture(&mut self, desc: TextureDesc, texture: Texture) {
        self.available_textures.entry(desc).or_default().push(texture);
    }

    pub fn clear_buffers(&mut self) {
        self.available_buffers.clear();
    }

    pub fn clear_textures(&mut self) {
        self.available_textures.clear();
    }

    pub fn clear(&mut self) {
        self.clear_buffers();
        self.clear_textures();
    }

    pub fn stats(&self) -> ResourceCacheStats {
        let available_buffer_count = self.available_buffers.values().map(|v| v.len()).sum();
        let available_texture_count = self.available_textures.values().map(|v| v.len()).sum();

        ResourceCacheStats {
            available_buffer_count,
            available_texture_count,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ResourceCacheStats {
    pub available_buffer_count: usize,
    pub available_texture_count: usize,
}


