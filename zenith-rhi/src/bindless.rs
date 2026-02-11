//! Bindless resource manager.
//!
//! Option A (non-owning): upload functions accept `&Texture` / `&Buffer` and we store only the
//! raw Vulkan handle (as `u64`) -> bindless index mapping. The caller is responsible for keeping
//! the resources alive while GPU work may reference them.

use ash::vk;
use ash::vk::Handle;
use std::collections::HashMap;
use std::ops::Deref;

use crate::descriptor::{BindableResource, BindableResourceType, DescriptorBindLocation, DescriptorWriter, LayoutBinding};
use crate::{DescriptorBindingError, DescriptorPool, DescriptorSetLayout, RenderDevice, ShaderBinding};
use std::sync::Arc;

#[repr(u8)]
#[derive(Debug, Copy, Clone)]
pub enum ResourceType {
    Texture2D, // binding 0
    Buffer,    // binding 1
    Sampler,   // binding 2
}

impl ResourceType {
    pub fn binding_index(&self) -> u32 {
        *self as u32
    }
}

// Used to index the resource in shader using bindless pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindlessHandle {
    packed: u32,
}

impl Deref for BindlessHandle {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.packed
    }
}

impl BindlessHandle {
    const TY_BITS: u32 = 3;
    const INDEX_BITS: u32 = 32 - Self::TY_BITS;
    const INDEX_MASK: u32 = (1u32 << Self::INDEX_BITS) - 1;

    pub const INVALID: Self = Self { packed: u32::MAX };

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
            _ => unimplemented!(),
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
}

pub struct BindlessPool {
    writer: DescriptorWriter,
    _pool: DescriptorPool,
    set_layout: Arc<DescriptorSetLayout>,
    set: vk::DescriptorSet,
    state: BindlessPoolState,
    caps: BindlessCaps,
}

impl BindlessPool {
    pub const SET_INDEX: u32 = 0;

    /// Shader variable name prefix for bindless resources.
    /// During reflection, any set containing a binding whose name starts
    /// with this prefix will have its layout replaced by the bindless pool layout.
    pub const BINDING_PREFIX: &'static str = "bindless_";
    
    pub fn canonical_shader_bindings(caps: &BindlessCaps) -> [ShaderBinding; 3] {
        let binding_flags = vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND;
        [
            ShaderBinding {
                name: "bindless_texture_heap".to_string(),
                set: Self::SET_INDEX,
                binding: ResourceType::Texture2D as u32,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                count: caps.max_textures,
                binding_flags,
            },
            ShaderBinding {
                name: "bindless_buffer_heap".to_string(),
                set: Self::SET_INDEX,
                binding: ResourceType::Buffer as u32,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                count: caps.max_storage_buffers,
                binding_flags,
            },
            ShaderBinding {
                name: "bindless_sampler_heap".to_string(),
                set: Self::SET_INDEX,
                binding: ResourceType::Sampler as u32,
                descriptor_type: vk::DescriptorType::SAMPLER,
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                count: caps.max_samplers,
                binding_flags,
            },
        ]
    }

    pub fn descriptor_layout_bindings(caps: &BindlessCaps) -> Vec<LayoutBinding> {
        let canonical = Self::canonical_shader_bindings(caps);
        let bindings = canonical
            .into_iter()
            .map(|binding| {
                let count = match binding.binding {
                    x if x == ResourceType::Texture2D as u32 => caps.max_textures,
                    x if x == ResourceType::Buffer as u32 => caps.max_storage_buffers,
                    x if x == ResourceType::Sampler as u32 => caps.max_samplers,
                    _ => 1,
                };
                LayoutBinding {
                    binding: binding.binding,
                    descriptor_type: binding.descriptor_type,
                    count,
                    stage_flags: binding.stage_flags,
                    binding_flags: vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                }
            })
            .collect::<Vec<_>>();
        bindings
    }

    pub fn new(device: &RenderDevice) -> Result<Self, vk::Result> {
        let caps = device.bindless_caps();
        let bindings = Self::descriptor_layout_bindings(caps);
        let set_layout = Arc::new(DescriptorSetLayout::new(
            "layout.bindless",
            device,
            &bindings,
            vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
        )?);
        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: caps.max_textures },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: caps.max_storage_buffers },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLER, descriptor_count: caps.max_samplers },
        ];
        let pool = DescriptorPool::new(
            "pool.bindless",
            device,
            1,
            &pool_sizes,
            vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND,
        )?;
        let set = pool.allocate(&set_layout)?;

        Ok(Self {
            writer: Default::default(),
            _pool: pool,
            set_layout,
            set,
            caps: *caps,
            state: Default::default(),
        })
    }

    #[inline]
    pub fn set(&self) -> vk::DescriptorSet { self.set }

    #[inline]
    pub fn set_layout(&self) -> &Arc<DescriptorSetLayout> { &self.set_layout }

    #[inline]
    pub fn caps(&self) -> BindlessCaps { self.caps }

    pub fn bind<T: BindableResource>(&mut self, res: T) -> anyhow::Result<BindlessHandle, DescriptorBindingError> {
        let key = res.bind_key();

        let (index, ty) = match res.ty() {
            BindableResourceType::Buffer => {
                let Some((index, is_new)) = self.state.buffers.get_or_alloc(key, self.caps.max_storage_buffers) else {
                    panic!("bindless buffer capacity exceeded (max={})", self.caps.max_storage_buffers);
                };

                if is_new {
                    self.writer.begin_binding("__bindless_buffers".to_string(), DescriptorBindLocation {
                        set: 0,
                        binding: ResourceType::Buffer.binding_index(),
                        array_index: index,
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                    });
                    self.writer.end_binding();
                    res.bind_to(&mut self.writer)?;
                }

                (index, ResourceType::Buffer)
            }
            BindableResourceType::Texture => {
                let Some((index, is_new)) = self.state.textures.get_or_alloc(key, self.caps.max_textures) else {
                    panic!("bindless texture capacity exceeded (max={})", self.caps.max_textures);
                };

                if is_new {
                    self.writer.begin_binding("__bindless_textures".to_string(), DescriptorBindLocation {
                        set: 0,
                        binding: ResourceType::Texture2D.binding_index(),
                        array_index: index,
                        ty: vk::DescriptorType::SAMPLED_IMAGE,
                    });
                    res.bind_to(&mut self.writer)?;
                    self.writer.end_binding();
                }

                (index, ResourceType::Texture2D)
            }
        };

        Ok(BindlessHandle::new(ty, index))
    }

    /// Bind a sampler into the bindless sampler heap at a **fixed index**.
    ///
    /// Intended usage: define a small shader-side enum (e.g. LinearRepeat=0...) and
    /// keep the mapping stable across the engine.
    pub fn bind_sampler_at(&mut self, index: u32, sampler: vk::Sampler) -> anyhow::Result<(), DescriptorBindingError> {
        if index >= self.caps.max_samplers {
            panic!("bindless sampler index out of range (index={}, max={})", index, self.caps.max_samplers);
        }

        let key = sampler.as_raw() as u64;
        // Track the last key written into this slot to avoid redundant descriptor writes.
        if (index as usize) >= self.state._samplers.keys_by_index.len() {
            self.state._samplers.keys_by_index.resize((index as usize) + 1, None);
        }
        if self.state._samplers.keys_by_index[index as usize] == Some(key) {
            return Ok(());
        }
        self.state._samplers.keys_by_index[index as usize] = Some(key);

        self.writer.begin_binding("__bindless_samplers".to_string(), DescriptorBindLocation {
            set: 0,
            binding: ResourceType::Sampler.binding_index(),
            array_index: index,
            ty: vk::DescriptorType::SAMPLER,
        });
        // Note: a SAMPLER descriptor uses only `sampler`; image_view/layout are ignored.
        self.writer.bind_texture(
            vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(vk::ImageView::null())
                .image_layout(vk::ImageLayout::UNDEFINED)
        )?;
        self.writer.end_binding();

        Ok(())
    }

    pub fn unbind(&mut self, handle: BindlessHandle) {
        match handle.ty() {
            ResourceType::Texture2D => {
                let _ = self.state.textures.free_by_index(handle.index());
            }
            ResourceType::Buffer => {
                let _ = self.state.buffers.free_by_index(handle.index());
            }
            ResourceType::Sampler => unimplemented!(),
        }
    }

    /// Flush pending descriptor writes into the pool's internal set.
    pub fn flush(&mut self, device: &RenderDevice) {
        if self.writer.num_bindings() == 0 {
            return;
        }
        let writes = self.writer.write_to(0, std::slice::from_ref(&self.set));
        unsafe {
            device.handle().update_descriptor_sets(&writes, &[]);
        }
        self.writer.clear();
    }
}