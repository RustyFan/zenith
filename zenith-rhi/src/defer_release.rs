use crate::{Buffer, DescriptorPool, Texture};

pub(crate) mod sealed {
    pub trait Sealed {}
}

pub trait DeferRelease: sealed::Sealed {
    fn enqueue(self, q: &mut DeferReleaseQueue);
}

impl sealed::Sealed for Buffer {}
impl sealed::Sealed for Texture {}
impl sealed::Sealed for DescriptorPool {}

impl DeferRelease for Buffer {
    #[inline]
    fn enqueue(self, q: &mut DeferReleaseQueue) {
        q.add_buffer(self)
    }
}

impl DeferRelease for Texture {
    #[inline]
    fn enqueue(self, q: &mut DeferReleaseQueue) {
        q.add_texture(self)
    }
}

impl DeferRelease for DescriptorPool {
    #[inline]
    fn enqueue(self, q: &mut DeferReleaseQueue) {
        q.add_pool(self)
    }
}

#[derive(Default, Clone, Debug)]
pub struct LastFreedStats {
    pub buffer_count: usize,
    pub texture_count: usize,
    pub pool_count: usize,
    pub total_count: usize,

    pub buffer_names: Vec<String>,
    pub texture_names: Vec<String>,
    pub pool_names: Vec<String>,
}

pub struct DeferReleaseQueue {
    buffers: Vec<Buffer>,
    textures: Vec<Texture>,
    pools: Vec<DescriptorPool>,
    last_freed: LastFreedStats,
}

impl DeferReleaseQueue {
    pub fn new() -> Self {
        Self {
            buffers: Default::default(),
            textures: Default::default(),
            pools: Default::default(),
            last_freed: LastFreedStats::default(),
        }
    }

    pub fn add_buffer(&mut self, buffer: Buffer) { 
        self.buffers.push(buffer); 
    }

    pub fn add_texture(&mut self, texture: Texture) {
        self.textures.push(texture);
    }

    pub fn add_pool(&mut self, pool: DescriptorPool) {
        self.pools.push(pool);
    }

    pub(crate) fn push<T: DeferRelease>(&mut self, value: T) {
        value.enqueue(self)
    }

    pub fn release_all(&mut self) {
        // Update last freed stats before actual drops.
        self.last_freed.buffer_count = self.buffers.len();
        self.last_freed.texture_count = self.textures.len();
        self.last_freed.pool_count = self.pools.len();
        self.last_freed.total_count =
            self.last_freed.buffer_count + self.last_freed.texture_count + self.last_freed.pool_count;

        self.last_freed.buffer_names = self.buffers.iter().map(|b| b.name().to_owned()).collect();
        self.last_freed.texture_names = self.textures.iter().map(|t| t.name().to_owned()).collect();
        self.last_freed.pool_names = self.pools.iter().map(|p| p.name().to_owned()).collect();

        self.buffers.clear();
        self.textures.clear();
        self.pools.clear();
    }

    pub fn last_freed(&self) -> &LastFreedStats {
        &self.last_freed
    }

    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }
    
    pub fn pool_count(&self) -> usize {
        self.pools.len()
    }
}