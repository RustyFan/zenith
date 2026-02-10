//! Vulkan Descriptor - descriptor pool, layout, and resource binding.

use ash::{vk};
use std::collections::HashMap;
use std::default::Default;
use zenith_rhi_derive::DeviceObject;
use crate::{GraphicPipeline, RenderDevice};
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
        device.set_debug_name(&layout);
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
        device.set_debug_name(&layout);
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
    fn set_debug_name(&self, device: &RenderDevice) {
        set_debug_name_handle(
            device,
            self.layout,
            vk::ObjectType::DESCRIPTOR_SET_LAYOUT,
            self.name(),
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
        device.set_debug_name(&pool);
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

    pub fn reset(&self) -> Result<(), vk::Result> {
        unsafe {
            self.device
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
        }
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
    fn set_debug_name(&self, device: &RenderDevice) {
        set_debug_name_handle(device, self.pool, vk::ObjectType::DESCRIPTOR_POOL, self.name());
    }
}

/// Error type for shader resource binding.
#[derive(Debug)]
pub enum DescriptorBindingError {
    BindingNotFound(String),
    TypeMismatch { name: String, expected: vk::DescriptorType },
    MissingTextureView(vk::Result, String),
    AllocationFailed(vk::Result),
}

impl std::fmt::Display for DescriptorBindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DescriptorBindingError::BindingNotFound(name) => write!(f, "Binding '{}' not found in shader reflection", name),
            DescriptorBindingError::TypeMismatch { name, expected } => {
                write!(f, "Type mismatch for '{}': expected {:?}", name, expected)
            }
            DescriptorBindingError::MissingTextureView(e, name) => { write!(f, "Missing texture view for: {:?}. {:?}", name, e) }
            DescriptorBindingError::AllocationFailed(e) => write!(f, "Descriptor set allocation failed: {:?}", e),
        }
    }
}

impl std::error::Error for DescriptorBindingError {}

/// Shader resource binder that binds resources by name using shader reflection.
pub struct DescriptorSetBinder<'a> {
    device: &'a RenderDevice,
    pipeline: &'a GraphicPipeline,
    reflection: &'a ShaderReflection,
    resource_ty_sizes: HashMap<vk::DescriptorType, u32>,
    collector: DescriptorBindingCollector,
}

impl<'a> DescriptorSetBinder<'a> {
    pub fn new(
        device: &'a RenderDevice,
        pipeline: &'a GraphicPipeline,
        reflection: &'a ShaderReflection,
    ) -> Result<Self, DescriptorBindingError> {
        Ok(Self {
            device,
            pipeline,
            reflection,
            resource_ty_sizes: Default::default(),
            collector: Default::default(),
        })
    }

    pub fn bind<T: BindableResource>(
        &mut self,
        name: &str,
        res: T,
    ) -> Result<&mut Self, DescriptorBindingError> {
        let binding = self.reflection.find_binding(name)
            .ok_or_else(|| DescriptorBindingError::BindingNotFound(name.to_string()))?;

        // TODO: every binding will clone the name, try to avoid frequently cloning
        self.collector.begin_binding(binding.name.clone(), DescriptorBindLocation {
            set: binding.set,
            binding: binding.binding,
            // TODO: support fixed-size array binding
            array_index: 0,
            ty: binding.descriptor_type,
        });
        res.bind_to(&mut self.collector)?;
        self.collector.end_binding();

        *self.resource_ty_sizes.entry(binding.descriptor_type).or_insert(0) += 1;
        Ok(self)
    }

    pub fn finish(self, device: &RenderDevice, first_set: u32) -> anyhow::Result<Vec<vk::DescriptorSet>, DescriptorBindingError> {
         let pool_sizes = self.resource_ty_sizes.into_iter()
            .map(|(ty, descriptor_count)| vk::DescriptorPoolSize {
                ty,
                descriptor_count,
            })
            .collect::<Vec<_>>();

        let layouts = &self.pipeline.descriptor_layouts[first_set as usize..];
        let pool = DescriptorPool::new("descriptor_pool", self.device, layouts.len() as u32, &pool_sizes, vk::DescriptorPoolCreateFlags::empty())
            .map_err(|e| DescriptorBindingError::AllocationFailed(e))?;
        let descriptor_sets: Result<Vec<_>, _> = layouts.iter()
            .map(|layout| {
                pool.allocate(layout).map_err(DescriptorBindingError::AllocationFailed)
            })
            .collect();
        device.defer_release(pool);
        let descriptor_sets = descriptor_sets?;

        let writes = self.collector.write_to(first_set, &descriptor_sets);
        if !writes.is_empty() {
            unsafe {
                self.device.handle().update_descriptor_sets(&writes, &[]);
            }
        }

        Ok(descriptor_sets)
    }
}

#[derive(Debug, Default, PartialEq, Eq, Hash, Clone, Copy)]
pub(crate) struct DescriptorBindLocation {
    pub(crate) set: u32,
    pub(crate) binding: u32,
    pub(crate) array_index: u32,
    pub(crate) ty: vk::DescriptorType,
}

#[derive(Debug, Default)]
pub struct DescriptorBindingCollector {
    current_binding: Option<(String, DescriptorBindLocation)>,
    buffer_bindings: HashMap<DescriptorBindLocation, vk::DescriptorBufferInfo>,
    texture_bindings: HashMap<DescriptorBindLocation, vk::DescriptorImageInfo>,
}

impl DescriptorBindingCollector {
    #[inline]
    pub(crate) fn begin_binding(&mut self, debug_name: String, location: DescriptorBindLocation) {
        self.current_binding = Some((debug_name, location));
    }

    #[inline]
    pub(crate) fn end_binding(&mut self) {
        self.current_binding = None;
    }

    pub fn bind_buffer(&mut self, info: vk::DescriptorBufferInfo) -> anyhow::Result<(), DescriptorBindingError> {
        if let Some((name, context)) = &self.current_binding {
            if !is_buffer(context.ty) {
                return Err(DescriptorBindingError::TypeMismatch {
                    name: name.clone(),
                    expected: context.ty,
                });
            }

            *self.buffer_bindings.entry(context.clone()).or_insert(info) = info;
        }
        Ok(())
    }

    pub fn bind_texture(&mut self, info: vk::DescriptorImageInfo) -> anyhow::Result<(), DescriptorBindingError> {
        if let Some((name, context)) = &self.current_binding {
            if !is_texture(context.ty) {
                return Err(DescriptorBindingError::TypeMismatch {
                    name: name.clone(),
                    expected: context.ty,
                });
            }

            *self.texture_bindings.entry(context.clone()).or_insert(info) = info;
        }
        Ok(())
    }

    pub fn num_bindings(&self) -> usize {
        self.buffer_bindings.len() + self.texture_bindings.len()
    }

    pub fn clear(&mut self) {
        self.end_binding();
        self.buffer_bindings.clear();
        self.texture_bindings.clear();
    }

    // TODO: buffer descriptor infos and texture descriptor infos will have N-to-1 relationship for fixed-sized descriptor array
    pub fn write_to(&self, base_set: u32, descriptor_sets: &[vk::DescriptorSet]) -> Vec<vk::WriteDescriptorSet> {
        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::with_capacity(self.buffer_bindings.len() + self.texture_bindings.len());

        for (location, info) in &self.buffer_bindings {
            assert!(location.set - base_set < descriptor_sets.len() as _);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[(location.set - base_set) as usize])
                .dst_binding(location.binding)
                .dst_array_element(location.array_index)
                .descriptor_type(location.ty)
                .buffer_info(std::slice::from_ref(&info));

            writes.push(write);
        }

        for (location, info) in &self.texture_bindings {
            assert!(location.set - base_set < descriptor_sets.len() as _);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[(location.set - base_set) as usize])
                .dst_binding(location.binding)
                .dst_array_element(location.array_index)
                .descriptor_type(location.ty)
                .image_info(std::slice::from_ref(&info));

            writes.push(write);
        }

        writes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindableResourceType {
    Buffer,
    Texture,
}

pub trait BindableResource {
    fn bind_to(&self, collector: &mut DescriptorBindingCollector) -> anyhow::Result<(), DescriptorBindingError>;
    fn bind_key(&self) -> u64;
    fn ty(&self) -> BindableResourceType;
}

fn is_buffer(ty: vk::DescriptorType) -> bool {
    matches!(
        ty,
        vk::DescriptorType::UNIFORM_BUFFER
            | vk::DescriptorType::STORAGE_BUFFER
            | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
            | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
    )
}

fn is_texture(ty: vk::DescriptorType) -> bool {
    matches!(
        ty,
        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            | vk::DescriptorType::SAMPLED_IMAGE
            | vk::DescriptorType::STORAGE_IMAGE
            | vk::DescriptorType::SAMPLER
    )
}