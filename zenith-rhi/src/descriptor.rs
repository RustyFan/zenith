//! Vulkan Descriptor - descriptor pool, layout, and resource binding.

use ash::{vk};
use zenith_core::collections::HashMap;
use std::default::Default;
use std::hash::{Hash, Hasher};
use std::ops::{Range};
use std::sync::Arc;
use ash::vk::Handle;
use zenith_rhi_derive::DeviceObject;
use crate::{RenderDevice};
use crate::device::DebuggableObject;
use crate::device::set_debug_name_handle;
use crate::shader::{ShaderBinding, ShaderReflection};

/// Layout binding information.
#[derive(Debug, Clone)]
pub struct LayoutBinding {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub count: u32,
    pub stage_flags: vk::ShaderStageFlags,
    pub binding_flags: vk::DescriptorBindingFlags,
}

/// Descriptor set layout with binding metadata for validation.
#[DeviceObject]
pub struct DescriptorSetLayout {
    name: String,
    layout: vk::DescriptorSetLayout,
    bindings: Vec<LayoutBinding>,
    binding_map: HashMap<u32, usize>,
}

impl DescriptorSetLayout {
    pub fn new(
        name: &str,
        device: &RenderDevice,
        bindings: &[LayoutBinding],
        layout_flags: vk::DescriptorSetLayoutCreateFlags,
    ) -> Result<Self, vk::Result> {
        let (vk_bindings, binding_flags) = bindings
            .iter()
            .map(|b| {
                (
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b.binding)
                        .descriptor_type(b.descriptor_type)
                        .descriptor_count(b.count)
                        .stage_flags(b.stage_flags),
                    b.binding_flags
                )
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
            .binding_flags(&binding_flags);

        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&vk_bindings)
            .flags(layout_flags)
            .push_next(&mut flags_info);

        let layout = unsafe { device.handle().create_descriptor_set_layout(&create_info, None)? };

        let binding_map: HashMap<u32, usize> = bindings
            .iter()
            .enumerate()
            .map(|(i, b)| (b.binding, i))
            .collect();

        let layout = Self {
            name: name.to_owned(),
            layout,
            bindings: bindings.to_vec(),
            binding_map,
            device: device.handle().clone(),
        };
        device.set_debug_name(&layout, name);
        Ok(layout)
    }

    /// Create a descriptor set layout from shader reflection for a specific set index.
    pub fn from_reflection(
        name: &str,
        device: &RenderDevice,
        shader_bindings: &[ShaderBinding],
        set_index: u32,
    ) -> Result<Self, vk::Result> {
        let bindings: Vec<LayoutBinding> = shader_bindings
            .iter()
            .filter(|b| b.set == set_index)
            .map(|b| {
                LayoutBinding {
                    binding: b.binding,
                    descriptor_type: b.descriptor_type,
                    count: b.count,
                    stage_flags: b.stage_flags,
                    binding_flags: b.binding_flags,
                }
            })
            .collect();

        // If any binding requires UPDATE_AFTER_BIND, the layout must use the corresponding flag.
        let layout_flags = if bindings.iter().any(|b| b.binding_flags.contains(vk::DescriptorBindingFlags::UPDATE_AFTER_BIND)) {
            vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL
        } else {
            vk::DescriptorSetLayoutCreateFlags::empty()
        };

        let layout = Self::new(name, device, &bindings, layout_flags)?;
        device.set_debug_name(&layout, name);
        Ok(layout)
    }

    #[inline]
    pub fn name(&self) -> &str { &self.name }

    #[inline]
    pub fn handle(&self) -> vk::DescriptorSetLayout { self.layout }

    pub fn get_binding(&self, binding: u32) -> Option<&LayoutBinding> {
        self.binding_map.get(&binding).map(|&i| &self.bindings[i])
    }

    pub fn bindings(&self) -> &[LayoutBinding] {
        &self.bindings
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

impl DebuggableObject for DescriptorSetLayout {
    fn set_debug_name(&self, device: &ash::ext::debug_utils::Device, name: &str) {
        set_debug_name_handle(
            device,
            self.layout,
            vk::ObjectType::DESCRIPTOR_SET_LAYOUT,
            name,
        );
    }
}

/// Descriptor pool for allocating descriptor sets.
#[DeviceObject]
pub struct DescriptorPool {
    name: String,
    pool: vk::DescriptorPool,
    max_sets: u32,
}

impl DescriptorPool {
    pub fn new(
        name: &str,
        device: &RenderDevice,
        max_sets: u32,
        sizes: &[vk::DescriptorPoolSize],
        flags: vk::DescriptorPoolCreateFlags,
    ) -> Result<Self, vk::Result> {
        let create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(flags)
            .max_sets(max_sets)
            .pool_sizes(sizes);

        let pool = unsafe { device.handle().create_descriptor_pool(&create_info, None)? };

        let pool = Self {
            name: name.to_owned(),
            pool,
            max_sets,
            device: device.handle().clone(),
        };
        device.set_debug_name(&pool, name);
        Ok(pool)
    }

    #[inline]
    pub fn name(&self) -> &str { &self.name }

    pub fn allocate(&self, layout: &DescriptorSetLayout) -> Result<vk::DescriptorSet, vk::Result> {
        let layouts = [layout.handle()];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);

        let sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info)? };
        Ok(sets[0])
    }

    #[inline]
    pub fn reset(&self) -> Result<(), vk::Result> {
        unsafe { self.device.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty()) }
    }

    #[inline]
    pub fn handle(&self) -> vk::DescriptorPool { self.pool }

    #[inline]
    pub fn max_sets(&self) -> u32 { self.max_sets }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

impl DebuggableObject for DescriptorPool {
    fn set_debug_name(&self, device: &ash::ext::debug_utils::Device, name: &str) {
        set_debug_name_handle(device, self.pool, vk::ObjectType::DESCRIPTOR_POOL, name);
    }
}

/// Error type for shader resource binding.
#[derive(Debug)]
pub enum DescriptorBindingError {
    BindingNotFound(String),
    MissingTextureView(vk::Result, String),
    AllocationFailed(vk::Result),
}

impl std::fmt::Display for DescriptorBindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DescriptorBindingError::BindingNotFound(name) => write!(f, "Binding '{}' not found in shader reflection", name),
            DescriptorBindingError::MissingTextureView(e, name) => { write!(f, "Missing texture view for: {:?}. {:?}", name, e) }
            DescriptorBindingError::AllocationFailed(e) => write!(f, "Descriptor set allocation failed: {:?}", e),
        }
    }
}

impl std::error::Error for DescriptorBindingError {}

/// Shader resource binder that binds resources by name using shader reflection.
pub struct DescriptorSetBinder<'a> {
    device: &'a RenderDevice,
    reflection: &'a ShaderReflection,
    layouts: &'a [Arc<DescriptorSetLayout>],
    resource_ty_sizes: HashMap<vk::DescriptorType, u32>,
    writer: DescriptorWriter,
    set_range: Range<usize>,
}

impl<'a> DescriptorSetBinder<'a> {
    pub fn new(
        device: &'a RenderDevice,
        reflection: &'a ShaderReflection,
        layouts: &'a [Arc<DescriptorSetLayout>],
    ) -> Self {
        Self {
            device,
            reflection,
            layouts,
            resource_ty_sizes: Default::default(),
            writer: Default::default(),
            set_range: usize::MAX..0,
        }
    }

    #[inline]
    pub fn num_bindings(&self) -> usize {
        self.writer.num_bindings()
    }

    pub fn bind<T: BindableResource>(
        &mut self,
        name: &str,
        res: &T,
    ) -> Result<&mut Self, DescriptorBindingError> {
        let binding = self.reflection.find_binding(name)
            .ok_or_else(|| DescriptorBindingError::BindingNotFound(name.to_string()))?;

        self.writer.add_binding(DescriptorBindLocation {
            set: binding.set,
            binding: binding.binding,
            expected_ty: binding.descriptor_type,
        }, res);

        *self.resource_ty_sizes.entry(binding.descriptor_type).or_insert(0) += 1;
        self.set_range.start = self.set_range.start.min(binding.set as usize);
        self.set_range.end = self.set_range.end.max((binding.set + 1) as usize);
        Ok(self)
    }

    pub fn finish(self, device: &RenderDevice) -> anyhow::Result<(u32, Vec<vk::DescriptorSet>), DescriptorBindingError> {
         let pool_sizes = self.resource_ty_sizes.into_iter()
            .map(|(ty, descriptor_count)| vk::DescriptorPoolSize {
                ty,
                descriptor_count,
            })
            .collect::<Vec<_>>();

        let base_set = self.set_range.start as u32;
        let layouts = &self.layouts[self.set_range];
        let pool = DescriptorPool::new("descriptor_pool", self.device, layouts.len() as u32, &pool_sizes, vk::DescriptorPoolCreateFlags::empty())
            .map_err(|e| DescriptorBindingError::AllocationFailed(e))?;
        let sets: Result<Vec<_>, _> = layouts.iter()
            .map(|layout| {
                pool.allocate(layout).map_err(DescriptorBindingError::AllocationFailed)
            })
            .collect();
        device.defer_release(pool);
        let sets = sets?;

        let writes = self.writer.write_to(base_set, &sets);
        if !writes.is_empty() {
            unsafe {
                self.device.handle().update_descriptor_sets(&writes, &[]);
            }
        }

        Ok((base_set, sets))
    }
}

#[derive(Debug, Default, PartialEq, Eq, Hash, Clone, Copy)]
pub struct DescriptorBindLocation {
    pub set: u32,
    pub binding: u32,
    pub expected_ty: vk::DescriptorType,
}

#[derive(Debug, Default)]
pub struct DescriptorWriter {
    bindings: Vec<(DescriptorBindLocation, ResourceBinding)>,
}

impl DescriptorWriter {
    pub fn add_binding<T: BindableResource>(&mut self, location: DescriptorBindLocation, res: &T) {
        let binding = res.as_binding();
        self.add_binding_raw(location, binding);
    }

    pub fn add_binding_raw(&mut self, location: DescriptorBindLocation, binding: ResourceBinding) {
        self.bindings.push((location, binding));
    }

    #[inline]
    pub fn num_bindings(&self) -> usize {
        self.bindings.len()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.bindings.clear();
    }

    pub fn write_to(&'_ self, base_set: u32, descriptor_sets: &[vk::DescriptorSet]) -> Vec<vk::WriteDescriptorSet<'_>> {
        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::with_capacity(self.num_bindings());

        for (location, info) in &self.bindings {
            debug_assert!(location.set - base_set < descriptor_sets.len() as _);

            match info {
                ResourceBinding::Buffer(buf_info) => {
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_sets[(location.set - base_set) as usize])
                        .dst_binding(location.binding)
                        .dst_array_element(0)
                        .descriptor_type(location.expected_ty)
                        .buffer_info(std::slice::from_ref(&buf_info));
                    writes.push(write);
                }
                ResourceBinding::Texture(tex_info) |
                ResourceBinding::Sampler(tex_info) => {
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_sets[(location.set - base_set) as usize])
                        .dst_binding(location.binding)
                        .dst_array_element(0)
                        .descriptor_type(location.expected_ty)
                        .image_info(std::slice::from_ref(&tex_info));
                    writes.push(write);
                }
                ResourceBinding::BufferArray(base, infos) => {
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_sets[(location.set - base_set) as usize])
                        .dst_binding(location.binding)
                        .dst_array_element(*base)
                        .descriptor_type(location.expected_ty)
                        .buffer_info(&infos);
                    writes.push(write);
                }
                ResourceBinding::TextureArray(base, infos) => {
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_sets[(location.set - base_set) as usize])
                        .dst_binding(location.binding)
                        .dst_array_element(*base)
                        .descriptor_type(location.expected_ty)
                        .image_info(&infos);
                    writes.push(write);
                }
            }
        }

        writes
    }
}

#[derive(Debug, Clone)]
pub enum ResourceBinding {
    Buffer(vk::DescriptorBufferInfo),
    /// (base array index, array of resources)
    BufferArray(u32, Vec<vk::DescriptorBufferInfo>),
    Texture(vk::DescriptorImageInfo),
    /// (base array index, array of resources)
    TextureArray(u32, Vec<vk::DescriptorImageInfo>),
    Sampler(vk::DescriptorImageInfo)
}

impl Hash for ResourceBinding {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            ResourceBinding::Buffer(buf_info) => {
                state.write_u64(buf_info.buffer.as_raw());
                buf_info.range.hash(state);
                buf_info.offset.hash(state);
            }
            ResourceBinding::BufferArray(base, infos) => {
                state.write_u32(*base);
                for info in infos {
                    state.write_u64(info.buffer.as_raw());
                    info.range.hash(state);
                    info.offset.hash(state);
                }
            }
            ResourceBinding::Texture(tex_info) |
            ResourceBinding::Sampler(tex_info) => {
                state.write_u64(tex_info.sampler.as_raw());
                state.write_u64(tex_info.image_view.as_raw());
                tex_info.image_layout.hash(state);
            }
            ResourceBinding::TextureArray(base, infos) => {
                state.write_u32(*base);
                for info in infos {
                    state.write_u64(info.sampler.as_raw());
                    state.write_u64(info.image_view.as_raw());
                    info.image_layout.hash(state);
                }
            }
        }
    }
}

pub trait BindableResource {
    fn as_binding(&self) -> ResourceBinding;
}
