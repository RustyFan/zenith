//! Bindless resource manager.
//!
//! Option A (non-owning): upload functions accept `&Texture` / `&Buffer` and we store only the
//! raw Vulkan handle (as `u64`) -> bindless index mapping. The caller is responsible for keeping
//! the resources alive while GPU work may reference them.

use std::cell::RefCell;
use ash::vk;
use ash::vk::Handle as _;
use std::collections::HashMap;
use std::sync::Arc;

use crate::buffer::BufferRange;
use crate::descriptor::LayoutBinding;
use crate::texture::TextureRange;
use crate::{CommandEncoder, DescriptorPool, DescriptorSetLayout, RenderDevice};

#[repr(u8)]
#[derive(Debug, Copy, Clone)]
pub enum ResourceType {
    Texture2D, // binding 0
    Buffer,    // binding 1
    Sampler,   // binding 2
}

// Used to index the resource in shader using bindless pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindlessResourceHandle {
    packed: u32,
}

impl BindlessResourceHandle {
    const TY_BITS: u32 = 3;
    const INDEX_BITS: u32 = 32 - Self::TY_BITS;
    const INDEX_MASK: u32 = (1u32 << Self::INDEX_BITS) - 1;

    const INVALID: Self = Self { packed: u32::MAX };

    pub fn new(ty: ResourceType, index: u32) -> Self {
        // encode ty to the most-significant bit use 3 bit.
        // Remaining 29 bits is used to encode index.
        // Add safety check and helper functions
        debug_assert!(index <= Self::INDEX_MASK, "bindless index out of range");
        let ty_bits = (ty as u32) & ((1u32 << Self::TY_BITS) - 1);
        let packed = (ty_bits << Self::INDEX_BITS) | (index & Self::INDEX_MASK);
        Self { packed }
    }

    #[inline]
    pub fn ty(&self) -> ResourceType {
        match (self.packed >> Self::INDEX_BITS) as u8 {
            0 => ResourceType::Texture2D,
            1 => ResourceType::Buffer,
            2 => ResourceType::Sampler,
            _ => ResourceType::Texture2D, // unreachable with proper construction
        }
    }

    #[inline]
    pub fn index(&self) -> u32 {
        self.packed & Self::INDEX_MASK
    }

    #[inline]
    pub fn raw(self) -> u32 {
        self.packed
    }

    #[inline]
    pub fn invalid(&self) -> bool {
        *self == Self::INVALID
    }
}

#[allow(dead_code)]
#[derive(Debug)]
enum PendingWrite {
    Texture2D { index: u32, view: vk::ImageView },
    Buffer { index: u32, buffer: vk::Buffer, offset: vk::DeviceSize, range: vk::DeviceSize },
    #[allow(dead_code)]
    Sampler { index: u32, sampler: vk::Sampler },
}

#[derive(Default)]
struct SlotMap {
    by_key: HashMap<u64, u32>,
    keys_by_index: Vec<Option<u64>>,
    free_list: Vec<u32>,
}

impl SlotMap {
    fn get_or_alloc(&mut self, key: u64, max: u32) -> Option<(u32, bool)> {
        if let Some(&idx) = self.by_key.get(&key) {
            return Some((idx, false));
        }

        let idx = if let Some(i) = self.free_list.pop() {
            i
        } else {
            let next = self.keys_by_index.len() as u32;
            if next >= max {
                return None;
            }
            self.keys_by_index.push(None);
            next
        };

        if (idx as usize) >= self.keys_by_index.len() {
            self.keys_by_index.resize((idx as usize) + 1, None);
        }
        self.keys_by_index[idx as usize] = Some(key);
        self.by_key.insert(key, idx);
        Some((idx, true))
    }

    fn free_by_index(&mut self, idx: u32) -> Option<u64> {
        let Some(slot) = self.keys_by_index.get_mut(idx as usize) else { return None };
        let key = slot.take()?;
        self.by_key.remove(&key);
        self.free_list.push(idx);
        Some(key)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BindlessCaps {
    pub max_textures: u32,
    pub max_uniform_buffers: u32,
    pub max_storage_buffers: u32,
    pub max_samplers: u32,
}

#[derive(Default)]
struct BindlessPoolState {
    textures: SlotMap,
    buffers: SlotMap,
    _samplers: SlotMap,
    pending: Vec<PendingWrite>,
}

// TODO: support texture 2d and buffers for now; sampler API can be added later.
pub struct BindlessPool {
    device: ash::Device,
    _pool: DescriptorPool,
    set_layout: Arc<DescriptorSetLayout>,
    set: vk::DescriptorSet,

    caps: BindlessCaps,

    state: RefCell<BindlessPoolState>,
}

impl BindlessPool {
    pub fn new(device: &RenderDevice) -> Result<Self, vk::Result> {
        let caps = device.bindless_caps();

        // Bindless descriptor set layout: set 0 with 3 bindings.
        // - binding 0: textures (sampled)
        // - binding 1: typeless buffers (ByteAddressBuffer) -> STORAGE_BUFFER
        // - binding 3: samplers
        let bindings = [
            LayoutBinding {
                binding: ResourceType::Texture2D as u32,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                count: caps.max_textures,
                stage_flags: vk::ShaderStageFlags::ALL,
            },
            LayoutBinding {
                binding: ResourceType::Buffer as u32,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                count: caps.max_storage_buffers,
                stage_flags: vk::ShaderStageFlags::ALL,
            },
            LayoutBinding {
                binding: ResourceType::Sampler as u32,
                descriptor_type: vk::DescriptorType::SAMPLER,
                count: caps.max_samplers,
                stage_flags: vk::ShaderStageFlags::ALL,
            },
        ];

        let binding_flags = [
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
        ];

        let set_layout = DescriptorSetLayout::new_with_flags(
            "bindless.set_layout",
            device,
            &bindings,
            vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
            &binding_flags,
        )?;
        let set_layout = Arc::new(set_layout);

        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: caps.max_textures },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: caps.max_storage_buffers },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLER, descriptor_count: caps.max_samplers },
        ];

        let pool = DescriptorPool::new_with_flags(
            "bindless.pool",
            device,
            1,
            &pool_sizes,
            vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND,
        )?;

        let set = pool.allocate(&set_layout)?;

        Ok(Self {
            device: device.handle().clone(),
            _pool: pool,
            set_layout,
            set,
            caps: *caps,
            state: RefCell::new(BindlessPoolState::default()),
        })
    }
    
    #[inline]
    pub fn set_layout(&self) -> &Arc<DescriptorSetLayout> { &self.set_layout }

    #[inline]
    pub fn set(&self) -> vk::DescriptorSet { self.set }

    #[inline]
    pub fn caps(&self) -> BindlessCaps { self.caps }

    pub fn bind_texture(&self, texture: TextureRange<'_>) -> BindlessResourceHandle {
        // add a pending upload
        let key = texture.texture().handle().as_raw();
        let mut state = self.state.borrow_mut();
        let Some((index, is_new)) = state.textures.get_or_alloc(key, self.caps.max_textures) else {
            panic!("bindless texture capacity exceeded (max={})", self.caps.max_textures);
        };
        if is_new {
            let view = texture.view().expect("Invalid texture view creation");
            state.pending.push(PendingWrite::Texture2D { index, view });
        }
        BindlessResourceHandle::new(ResourceType::Texture2D, index)
    }

    pub fn free_texture(&self, handle: BindlessResourceHandle) {
        // Restore the index of current resources.
        debug_assert_eq!(handle.ty() as u8, ResourceType::Texture2D as u8);
        let mut state = self.state.borrow_mut();
        let _ = state.textures.free_by_index(handle.index());
    }

    pub fn bind_buffer(&self, buffer: BufferRange<'_>) -> BindlessResourceHandle {
        let key = buffer.buffer().handle().as_raw();
        let mut state = self.state.borrow_mut();
        let Some((index, is_new)) = state.buffers.get_or_alloc(key, self.caps.max_storage_buffers) else {
            panic!("bindless buffer capacity exceeded (max={})", self.caps.max_storage_buffers);
        };
        if is_new {
            let info = buffer.to_binding();
            state.pending.push(PendingWrite::Buffer {
                index,
                buffer: info.buffer,
                offset: info.offset,
                range: info.range,
            });
        }

        BindlessResourceHandle::new(ResourceType::Buffer, index)
    }

    pub fn free_buffer(&self, handle: BindlessResourceHandle) {
        // Restore the index of current resources.
        let mut state = self.state.borrow_mut();
        match handle.ty() {
            ResourceType::Buffer => { let _ = state.buffers.free_by_index(handle.index()); }
            _ => {}
        }
    }

    pub fn update(&self, _encoder: &CommandEncoder<'_>) {
        // update all pending uploads by vkWriteDescriptorSet.
        let mut state = self.state.borrow_mut();
        if state.pending.is_empty() {
            return;
        }

        enum Kind { Tex, Buf, Samp }
        struct Resolved { kind: Kind, index: u32, info_index: usize }

        let mut resolved: Vec<Resolved> = Vec::new();
        let mut image_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        let mut sampler_infos: Vec<vk::DescriptorImageInfo> = Vec::new();

        for p in state.pending.drain(..) {
            match p {
                PendingWrite::Texture2D { index, view } => {
                    image_infos.push(
                        vk::DescriptorImageInfo::default()
                            .image_view(view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    );
                    resolved.push(Resolved { kind: Kind::Tex, index, info_index: image_infos.len() - 1 });
                }
                PendingWrite::Buffer { index, buffer, offset, range } => {
                    buffer_infos.push(
                        vk::DescriptorBufferInfo::default()
                            .buffer(buffer)
                            .offset(offset)
                            .range(range)
                    );
                    resolved.push(Resolved { kind: Kind::Buf, index, info_index: buffer_infos.len() - 1 });
                }
                PendingWrite::Sampler { index, sampler } => {
                    sampler_infos.push(vk::DescriptorImageInfo::default().sampler(sampler));
                    resolved.push(Resolved { kind: Kind::Samp, index, info_index: sampler_infos.len() - 1 });
                }
            }
        }

        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::with_capacity(resolved.len());
        for r in resolved {
            match r.kind {
                Kind::Tex => {
                    let info_ref = &image_infos[r.info_index];
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.set)
                            .dst_binding(ResourceType::Texture2D as u32)
                            .dst_array_element(r.index)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .image_info(std::slice::from_ref(info_ref))
                    );
                }
                Kind::Buf => {
                    let info_ref = &buffer_infos[r.info_index];
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.set)
                            .dst_binding(ResourceType::Buffer as u32)
                            .dst_array_element(r.index)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(std::slice::from_ref(info_ref))
                    );
                }
                Kind::Samp => {
                    let info_ref = &sampler_infos[r.info_index];
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.set)
                            .dst_binding(ResourceType::Sampler as u32)
                            .dst_array_element(r.index)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .image_info(std::slice::from_ref(info_ref))
                    );
                }
            }
        }

        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }
    }
}