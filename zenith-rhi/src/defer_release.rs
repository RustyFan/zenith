use std::cell::RefCell;
use crate::{Buffer, DescriptorPool, Texture};

pub(crate) mod sealed {
    pub trait Sealed {}
}

pub trait DeferRelease: sealed::Sealed {
    fn enqueue(self, defer_release: &DeferReleaseQueue);
}

impl sealed::Sealed for Buffer {}
impl sealed::Sealed for Texture {}
impl sealed::Sealed for DescriptorPool {}

impl DeferRelease for Buffer {
    #[inline]
    fn enqueue(self, defer_release: &DeferReleaseQueue) {
        defer_release.add_buffer(self)
    }
}

impl DeferRelease for Texture {
    #[inline]
    fn enqueue(self, defer_release: &DeferReleaseQueue) {
        defer_release.add_texture(self)
    }
}

impl DeferRelease for DescriptorPool {
    #[inline]
    fn enqueue(self, defer_release: &DeferReleaseQueue) {
        defer_release.add_pool(self)
    }
}


pub struct DeferReleaseQueue {
    buffers: RefCell<Vec<Buffer>>,
    textures: RefCell<Vec<Texture>>,
    pools: RefCell<Vec<DescriptorPool>>,
}

impl DeferReleaseQueue {
    pub fn new() -> Self {
        Self {
            buffers: Default::default(),
            textures: Default::default(),
            pools: Default::default(),
        }
    }

    pub fn defer_release<T: DeferRelease>(&self, value: T) {
        value.enqueue(self)
    }

    pub fn release_all(&self) {
        self.buffers.borrow_mut().clear();
        self.textures.borrow_mut().clear();
        self.pools.borrow_mut().clear();
    }

    #[inline]
    pub fn buffer_count(&self) -> usize {
        self.buffers.borrow().len()
    }

    #[inline]
    pub fn texture_count(&self) -> usize {
        self.textures.borrow().len()
    }

    #[inline]
    pub fn pool_count(&self) -> usize {
        self.pools.borrow().len()
    }

    pub(crate) fn add_buffer(&self, buffer: Buffer) {
        self.buffers.borrow_mut().push(buffer);
    }

    pub(crate) fn add_texture(&self, texture: Texture) {
        self.textures.borrow_mut().push(texture);
    }

    pub(crate) fn add_pool(&self, pool: DescriptorPool) {
        self.pools.borrow_mut().push(pool);
    }
}