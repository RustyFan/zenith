//! Vulkan Descriptor - descriptor pool, layout, and resource binding.

use ash::{vk, Device};
use std::collections::HashMap;
use std::sync::Arc;
use zenith_core::collections::SmallVec;
use crate::shader::{ShaderBinding, ShaderReflection};

/// Descriptor binding validation error.
#[derive(Debug)]
pub enum BindingError {
    /// Binding index not found in layout.
    BindingNotFound(u32),
    /// Descriptor type mismatch.
    TypeMismatch {
        binding: u32,
        expected: vk::DescriptorType,
        got: vk::DescriptorType,
    },
    /// Array index out of bounds.
    ArrayIndexOutOfBounds { binding: u32, index: u32, max: u32 },
}

impl std::fmt::Display for BindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BindingError::BindingNotFound(b) => write!(f, "Binding {} not found in layout", b),
            BindingError::TypeMismatch {
                binding,
                expected,
                got,
            } => write!(
                f,
                "Type mismatch at binding {}: expected {:?}, got {:?}",
                binding, expected, got
            ),
            BindingError::ArrayIndexOutOfBounds { binding, index, max } => write!(
                f,
                "Array index {} out of bounds at binding {} (max: {})",
                index, binding, max
            ),
        }
    }
}

impl std::error::Error for BindingError {}

/// Layout binding information.
#[derive(Debug, Clone)]
pub struct LayoutBinding {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub count: u32,
    pub stage_flags: vk::ShaderStageFlags,
}

/// Descriptor set layout with binding metadata for validation.
pub struct DescriptorSetLayout {
    device: Device,
    layout: vk::DescriptorSetLayout,
    bindings: Vec<LayoutBinding>,
    binding_map: HashMap<u32, usize>,
}

impl DescriptorSetLayout {
    /// Create a new descriptor set layout from binding descriptions.
    pub fn new(device: &Device, bindings: &[LayoutBinding]) -> Result<Self, vk::Result> {
        let vk_bindings: Vec<vk::DescriptorSetLayoutBinding> = bindings
            .iter()
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(b.descriptor_type)
                    .descriptor_count(b.count)
                    .stage_flags(b.stage_flags)
            })
            .collect();

        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&vk_bindings);

        let layout = unsafe { device.create_descriptor_set_layout(&create_info, None)? };

        let binding_map: HashMap<u32, usize> = bindings
            .iter()
            .enumerate()
            .map(|(i, b)| (b.binding, i))
            .collect();

        Ok(Self {
            device: device.clone(),
            layout,
            bindings: bindings.to_vec(),
            binding_map,
        })
    }

    /// Create a descriptor set layout from shader reflection for a specific set index.
    pub fn from_reflection(
        device: &Device,
        shader_bindings: &[ShaderBinding],
        set_index: u32,
    ) -> Result<Self, vk::Result> {
        let bindings: Vec<LayoutBinding> = shader_bindings
            .iter()
            .filter(|b| b.set == set_index)
            .map(|b| LayoutBinding {
                binding: b.binding,
                descriptor_type: b.descriptor_type,
                count: b.count,
                stage_flags: b.stage_flags,
            })
            .collect();

        Self::new(device, &bindings)
    }

    /// Get the raw Vulkan descriptor set layout handle.
    pub fn handle(&self) -> vk::DescriptorSetLayout {
        self.layout
    }

    /// Get binding information by binding index.
    pub fn get_binding(&self, binding: u32) -> Option<&LayoutBinding> {
        self.binding_map.get(&binding).map(|&i| &self.bindings[i])
    }

    /// Get all bindings.
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

/// Descriptor pool for allocating descriptor sets.
pub struct DescriptorPool {
    device: Device,
    pool: vk::DescriptorPool,
    max_sets: u32,
}

impl DescriptorPool {
    /// Create a new descriptor pool.
    pub fn new(
        device: &Device,
        max_sets: u32,
        pool_sizes: &[vk::DescriptorPoolSize],
    ) -> Result<Self, vk::Result> {
        let create_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        let pool = unsafe { device.create_descriptor_pool(&create_info, None)? };

        Ok(Self {
            device: device.clone(),
            pool,
            max_sets,
        })
    }

    /// Allocate a single descriptor set.
    pub fn allocate(&self, layout: &DescriptorSetLayout) -> Result<vk::DescriptorSet, vk::Result> {
        let layouts = [layout.handle()];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);

        let sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info)? };
        Ok(sets[0])
    }

    /// Allocate multiple descriptor sets with the same layout.
    pub fn allocate_many(
        &self,
        layout: &DescriptorSetLayout,
        count: u32,
    ) -> Result<Vec<vk::DescriptorSet>, vk::Result> {
        let layouts: Vec<vk::DescriptorSetLayout> = (0..count).map(|_| layout.handle()).collect();
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);

        unsafe { self.device.allocate_descriptor_sets(&alloc_info) }
    }

    /// Free a descriptor set back to the pool.
    pub fn free(&self, set: vk::DescriptorSet) -> Result<(), vk::Result> {
        unsafe { self.device.free_descriptor_sets(self.pool, &[set]) }
    }

    /// Reset the pool, freeing all allocated descriptor sets.
    pub fn reset(&self) -> Result<(), vk::Result> {
        unsafe {
            self.device
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
        }
    }

    /// Get the raw Vulkan descriptor pool handle.
    pub fn handle(&self) -> vk::DescriptorPool {
        self.pool
    }

    /// Get the maximum number of sets this pool can allocate.
    pub fn max_sets(&self) -> u32 {
        self.max_sets
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

/// Resource binder for safely binding resources to descriptor sets.
pub struct ResourceBinder<'a> {
    device: &'a Device,
    descriptor_set: vk::DescriptorSet,
    layout: &'a DescriptorSetLayout,
    buffer_infos: Vec<vk::DescriptorBufferInfo>,
    image_infos: Vec<vk::DescriptorImageInfo>,
    writes: Vec<WriteInfo>,
}

struct WriteInfo {
    binding: u32,
    array_element: u32,
    descriptor_type: vk::DescriptorType,
    buffer_info_index: Option<usize>,
    image_info_index: Option<usize>,
}

impl<'a> ResourceBinder<'a> {
    /// Create a new resource binder for a descriptor set.
    pub fn new(
        device: &'a Device,
        descriptor_set: vk::DescriptorSet,
        layout: &'a DescriptorSetLayout,
    ) -> Self {
        Self {
            device,
            descriptor_set,
            layout,
            buffer_infos: Vec::new(),
            image_infos: Vec::new(),
            writes: Vec::new(),
        }
    }

    /// Bind a buffer to a binding slot.
    pub fn bind_buffer(
        mut self,
        binding: u32,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) -> Result<Self, BindingError> {
        let layout_binding = self
            .layout
            .get_binding(binding)
            .ok_or(BindingError::BindingNotFound(binding))?;

        // Validate descriptor type is a buffer type
        let is_buffer_type = matches!(
            layout_binding.descriptor_type,
            vk::DescriptorType::UNIFORM_BUFFER
                | vk::DescriptorType::STORAGE_BUFFER
                | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
        );

        if !is_buffer_type {
            return Err(BindingError::TypeMismatch {
                binding,
                expected: layout_binding.descriptor_type,
                got: vk::DescriptorType::UNIFORM_BUFFER,
            });
        }

        let buffer_info_index = self.buffer_infos.len();
        self.buffer_infos.push(
            vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .offset(offset)
                .range(range),
        );

        self.writes.push(WriteInfo {
            binding,
            array_element: 0,
            descriptor_type: layout_binding.descriptor_type,
            buffer_info_index: Some(buffer_info_index),
            image_info_index: None,
        });

        Ok(self)
    }

    /// Bind an image view with sampler to a binding slot.
    pub fn bind_image(
        mut self,
        binding: u32,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
        image_layout: vk::ImageLayout,
    ) -> Result<Self, BindingError> {
        let layout_binding = self
            .layout
            .get_binding(binding)
            .ok_or(BindingError::BindingNotFound(binding))?;

        // Validate descriptor type is an image type
        let is_image_type = matches!(
            layout_binding.descriptor_type,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                | vk::DescriptorType::SAMPLED_IMAGE
                | vk::DescriptorType::STORAGE_IMAGE
        );

        if !is_image_type {
            return Err(BindingError::TypeMismatch {
                binding,
                expected: layout_binding.descriptor_type,
                got: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            });
        }

        let image_info_index = self.image_infos.len();
        self.image_infos.push(
            vk::DescriptorImageInfo::default()
                .image_view(image_view)
                .sampler(sampler)
                .image_layout(image_layout),
        );

        self.writes.push(WriteInfo {
            binding,
            array_element: 0,
            descriptor_type: layout_binding.descriptor_type,
            buffer_info_index: None,
            image_info_index: Some(image_info_index),
        });

        Ok(self)
    }

    /// Bind a standalone sampler to a binding slot.
    pub fn bind_sampler(mut self, binding: u32, sampler: vk::Sampler) -> Result<Self, BindingError> {
        let layout_binding = self
            .layout
            .get_binding(binding)
            .ok_or(BindingError::BindingNotFound(binding))?;

        if layout_binding.descriptor_type != vk::DescriptorType::SAMPLER {
            return Err(BindingError::TypeMismatch {
                binding,
                expected: layout_binding.descriptor_type,
                got: vk::DescriptorType::SAMPLER,
            });
        }

        let image_info_index = self.image_infos.len();
        self.image_infos.push(
            vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(vk::ImageView::null())
                .image_layout(vk::ImageLayout::UNDEFINED),
        );

        self.writes.push(WriteInfo {
            binding,
            array_element: 0,
            descriptor_type: vk::DescriptorType::SAMPLER,
            buffer_info_index: None,
            image_info_index: Some(image_info_index),
        });

        Ok(self)
    }

    /// Finish binding and write all descriptors to the set.
    pub fn finish(self) {
        let writes: Vec<vk::WriteDescriptorSet> = self
            .writes
            .iter()
            .map(|w| {
                let mut write = vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(w.binding)
                    .dst_array_element(w.array_element)
                    .descriptor_type(w.descriptor_type);

                if let Some(idx) = w.buffer_info_index {
                    write = write.buffer_info(std::slice::from_ref(&self.buffer_infos[idx]));
                }
                if let Some(idx) = w.image_info_index {
                    write = write.image_info(std::slice::from_ref(&self.image_infos[idx]));
                }

                write
            })
            .collect();

        if !writes.is_empty() {
            unsafe {
                self.device.update_descriptor_sets(&writes, &[]);
            }
        }
    }
}

/// Create all descriptor set layouts from shader reflection.
pub fn create_layouts_from_reflection(
    device: &Device,
    reflection: &ShaderReflection,
) -> Result<Vec<Arc<DescriptorSetLayout>>, vk::Result> {
    let max_set = reflection.max_set().unwrap_or(0);
    let mut layouts = Vec::with_capacity((max_set + 1) as usize);

    for set_index in 0..=max_set {
        let layout = DescriptorSetLayout::from_reflection(device, &reflection.bindings, set_index)?;
        layouts.push(Arc::new(layout));
    }

    Ok(layouts)
}

/// Error type for shader resource binding.
#[derive(Debug)]
pub enum ShaderBindingError {
    BindingNotFound(String),
    TypeMismatch { name: String, expected: vk::DescriptorType, got: vk::DescriptorType },
    AllocationFailed(vk::Result),
}

impl std::fmt::Display for ShaderBindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShaderBindingError::BindingNotFound(name) => write!(f, "Binding '{}' not found in shader reflection", name),
            ShaderBindingError::TypeMismatch { name, expected, got } => {
                write!(f, "Type mismatch for '{}': expected {:?}, got {:?}", name, expected, got)
            }
            ShaderBindingError::AllocationFailed(e) => write!(f, "Descriptor set allocation failed: {:?}", e),
        }
    }
}

impl std::error::Error for ShaderBindingError {}

struct PendingWrite {
    set_index: u32,
    binding: u32,
    descriptor_type: vk::DescriptorType,
    buffer_info: Option<vk::DescriptorBufferInfo>,
    image_info: Option<vk::DescriptorImageInfo>,
}

/// Shader resource binder that binds resources by name using shader reflection.
pub struct ShaderResourceBinder<'a> {
    device: &'a Device,
    reflection: &'a ShaderReflection,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pending_writes: Vec<PendingWrite>,
}

impl<'a> ShaderResourceBinder<'a> {
    /// Create a new shader resource binder.
    pub fn new(
        device: &'a Device,
        reflection: &'a ShaderReflection,
        layouts: &[Arc<DescriptorSetLayout>],
        pool: &DescriptorPool,
    ) -> Result<Self, ShaderBindingError> {
        let mut descriptor_sets = Vec::with_capacity(layouts.len());
        for layout in layouts {
            let set = pool.allocate(layout).map_err(ShaderBindingError::AllocationFailed)?;
            descriptor_sets.push(set);
        }

        Ok(Self {
            device,
            reflection,
            descriptor_sets,
            pending_writes: Vec::new(),
        })
    }

    /// Bind a buffer by name.
    pub fn bind_buffer(
        &mut self,
        name: &str,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) -> Result<&mut Self, ShaderBindingError> {
        let binding = self.reflection.find_binding(name)
            .ok_or_else(|| ShaderBindingError::BindingNotFound(name.to_string()))?;

        let is_buffer_type = matches!(
            binding.descriptor_type,
            vk::DescriptorType::UNIFORM_BUFFER
                | vk::DescriptorType::STORAGE_BUFFER
                | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
        );

        if !is_buffer_type {
            return Err(ShaderBindingError::TypeMismatch {
                name: name.to_string(),
                expected: binding.descriptor_type,
                got: vk::DescriptorType::UNIFORM_BUFFER,
            });
        }

        self.pending_writes.push(PendingWrite {
            set_index: binding.set,
            binding: binding.binding,
            descriptor_type: binding.descriptor_type,
            buffer_info: Some(vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .offset(offset)
                .range(range)),
            image_info: None,
        });

        Ok(self)
    }

    /// Bind a combined image sampler by name.
    pub fn bind_texture(
        &mut self,
        name: &str,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
        layout: vk::ImageLayout,
    ) -> Result<&mut Self, ShaderBindingError> {
        let binding = self.reflection.find_binding(name)
            .ok_or_else(|| ShaderBindingError::BindingNotFound(name.to_string()))?;

        let is_image_type = matches!(
            binding.descriptor_type,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                | vk::DescriptorType::SAMPLED_IMAGE
                | vk::DescriptorType::STORAGE_IMAGE
        );

        if !is_image_type {
            return Err(ShaderBindingError::TypeMismatch {
                name: name.to_string(),
                expected: binding.descriptor_type,
                got: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            });
        }

        self.pending_writes.push(PendingWrite {
            set_index: binding.set,
            binding: binding.binding,
            descriptor_type: binding.descriptor_type,
            buffer_info: None,
            image_info: Some(vk::DescriptorImageInfo::default()
                .image_view(image_view)
                .sampler(sampler)
                .image_layout(layout)),
        });

        Ok(self)
    }

    /// Finish binding and return the descriptor sets for binding to the pipeline.
    pub fn finish(self) -> Vec<vk::DescriptorSet> {
        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();
        let mut buffer_infos: SmallVec<[vk::DescriptorBufferInfo; 8]> = SmallVec::new();
        let mut image_infos: SmallVec<[vk::DescriptorImageInfo; 8]> = SmallVec::new();

        for pending in &self.pending_writes {
            if let Some(buf_info) = pending.buffer_info {
                buffer_infos.push(buf_info);
            }
            if let Some(img_info) = pending.image_info {
                image_infos.push(img_info);
            }
        }

        let mut buf_idx = 0;
        let mut img_idx = 0;

        for pending in &self.pending_writes {
            let set = self.descriptor_sets[pending.set_index as usize];
            let mut write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(pending.binding)
                .dst_array_element(0)
                .descriptor_type(pending.descriptor_type);

            if pending.buffer_info.is_some() {
                write = write.buffer_info(std::slice::from_ref(&buffer_infos[buf_idx]));
                buf_idx += 1;
            }
            if pending.image_info.is_some() {
                write = write.image_info(std::slice::from_ref(&image_infos[img_idx]));
                img_idx += 1;
            }

            writes.push(write);
        }

        if !writes.is_empty() {
            unsafe {
                self.device.update_descriptor_sets(&writes, &[]);
            }
        }

        self.descriptor_sets
    }
}
