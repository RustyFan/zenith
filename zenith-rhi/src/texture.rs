//! Vulkan Texture - GPU texture resource management.

use ash::{vk, Device};
use std::default::Default;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::ops::RangeBounds;
use zenith_core::collections::hashmap::HashMap;
use zenith_rhi_derive::DeviceObject;
use crate::{Sampler};
use crate::utility::{find_memory_type, normalize_range_u32};

/// Texture descriptor for creating GPU textures.
#[derive(Debug, Clone)]
pub struct TextureDesc {
    pub name: String,
    /// Image format.
    pub format: vk::Format,
    /// Image extent (width, height, depth).
    pub extent: vk::Extent3D,
    /// Image usage flags.
    pub usage: vk::ImageUsageFlags,
    /// Memory property flags for allocation.
    pub memory_flags: vk::MemoryPropertyFlags,
    /// Image type (1D, 2D, 3D).
    pub image_type: vk::ImageType,
    /// Image view type.
    pub view_type: vk::ImageViewType,
    /// Number of mip levels.
    pub mip_levels: u32,
    /// Number of array layers.
    pub array_layers: u32,
    /// Sample count for multisampling.
    pub samples: vk::SampleCountFlags,
    /// Image tiling mode.
    pub tiling: vk::ImageTiling,
}

impl Default for TextureDesc {
    fn default() -> Self {
        Self {
            name: String::new(),
            format: vk::Format::R8G8B8A8_UNORM,
            extent: vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            usage: vk::ImageUsageFlags::SAMPLED,
            memory_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
        }
    }
}

impl TextureDesc {
    /// Create a new 2D texture descriptor.
    pub fn new_2d(width: u32, height: u32, format: vk::Format) -> Self {
        Self {
            name: String::new(),
            format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            ..Default::default()
        }
    }

    /// Create a new 3D texture descriptor.
    pub fn new_3d(width: u32, height: u32, depth: u32, format: vk::Format) -> Self {
        Self {
            name: String::new(),
            format,
            extent: vk::Extent3D {
                width,
                height,
                depth,
            },
            image_type: vk::ImageType::TYPE_3D,
            view_type: vk::ImageViewType::TYPE_3D,
            ..Default::default()
        }
    }

    /// Create a cube map texture descriptor.
    pub fn new_cube(size: u32, format: vk::Format) -> Self {
        Self {
            name: String::new(),
            format,
            extent: vk::Extent3D {
                width: size,
                height: size,
                depth: 1,
            },
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::CUBE,
            array_layers: 6,
            ..Default::default()
        }
    }

    /// Create a 2D texture array descriptor.
    pub fn new_2d_array(width: u32, height: u32, layers: u32, format: vk::Format) -> Self {
        Self {
            name: String::new(),
            format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D_ARRAY,
            array_layers: layers,
            ..Default::default()
        }
    }

    /// Create a color attachment descriptor.
    pub fn new_color_attachment(width: u32, height: u32, format: vk::Format) -> Self {
        Self {
            name: String::new(),
            format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            ..Default::default()
        }
    }

    /// Create a depth attachment descriptor.
    pub fn new_depth_attachment(width: u32, height: u32) -> Self {
        Self {
            name: String::new(),
            format: vk::Format::D32_SFLOAT,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            ..Default::default()
        }
    }

    /// Create a depth-stencil attachment descriptor.
    pub fn new_depth_stencil_attachment(width: u32, height: u32) -> Self {
        Self {
            name: String::new(),
            format: vk::Format::D24_UNORM_S8_UINT,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            image_type: vk::ImageType::TYPE_2D,
            view_type: vk::ImageViewType::TYPE_2D,
            ..Default::default()
        }
    }

    /// Set the texture format.
    pub fn with_format(mut self, format: vk::Format) -> Self {
        self.format = format;
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the texture extent.
    pub fn with_extent(mut self, width: u32, height: u32, depth: u32) -> Self {
        self.extent = vk::Extent3D {
            width,
            height,
            depth,
        };
        self
    }

    /// Set the texture usage flags.
    pub fn with_usage(mut self, usage: vk::ImageUsageFlags) -> Self {
        self.usage = usage;
        self
    }

    /// Add additional usage flags.
    pub fn with_additional_usage(mut self, usage: vk::ImageUsageFlags) -> Self {
        self.usage |= usage;
        self
    }

    /// Set memory property flags.
    pub fn with_memory_flags(mut self, flags: vk::MemoryPropertyFlags) -> Self {
        self.memory_flags = flags;
        self
    }

    /// Set the number of mip levels.
    pub fn with_mip_levels(mut self, levels: u32) -> Self {
        self.mip_levels = levels;
        self
    }

    /// Set the number of array layers.
    pub fn with_array_layers(mut self, layers: u32) -> Self {
        self.array_layers = layers;
        self
    }

    /// Set the sample count.
    pub fn with_samples(mut self, samples: vk::SampleCountFlags) -> Self {
        self.samples = samples;
        self
    }

    /// Set the tiling mode.
    pub fn with_tiling(mut self, tiling: vk::ImageTiling) -> Self {
        self.tiling = tiling;
        self
    }

    /// Enable transfer source usage.
    pub fn with_transfer_src_usage(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        self
    }

    /// Enable transfer destination usage.
    pub fn with_transfer_dst_usage(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::TRANSFER_DST;
        self
    }
}

impl PartialEq for TextureDesc {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.format == other.format
            && self.extent.width == other.extent.width
            && self.extent.height == other.extent.height
            && self.extent.depth == other.extent.depth
            && self.usage.as_raw() == other.usage.as_raw()
            && self.memory_flags.as_raw() == other.memory_flags.as_raw()
            && self.image_type == other.image_type
            && self.view_type == other.view_type
            && self.mip_levels == other.mip_levels
            && self.array_layers == other.array_layers
            && self.samples.as_raw() == other.samples.as_raw()
            && self.tiling == other.tiling
    }
}

impl Eq for TextureDesc {}

impl Hash for TextureDesc {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        (self.format.as_raw() as i32).hash(state);
        self.extent.width.hash(state);
        self.extent.height.hash(state);
        self.extent.depth.hash(state);
        self.usage.as_raw().hash(state);
        self.memory_flags.as_raw().hash(state);
        (self.image_type.as_raw() as i32).hash(state);
        (self.view_type.as_raw() as i32).hash(state);
        self.mip_levels.hash(state);
        self.array_layers.hash(state);
        self.samples.as_raw().hash(state);
        (self.tiling.as_raw() as i32).hash(state);
    }
}

/// GPU texture with memory allocation and optional image view.
#[DeviceObject]
pub struct Texture {
    desc: TextureDesc,
    image: vk::Image,
    /// If memory is null, it is a swapchain texture
    memory: vk::DeviceMemory,
    views: RefCell<HashMap<TextureSubresource, vk::ImageView>>,
}

impl Texture {
    /// Create a new texture from a descriptor (view is not created).
    pub fn new(
        device: &crate::RenderDevice,
        desc: &TextureDesc,
    ) -> Result<Self, vk::Result> {
        let memory_properties = device.memory_properties();
        let device = device.handle();
        // Create image
        let image_info = vk::ImageCreateInfo::default()
            .image_type(desc.image_type)
            .format(desc.format)
            .extent(desc.extent)
            .mip_levels(desc.mip_levels)
            .array_layers(desc.array_layers)
            .samples(desc.samples)
            .tiling(desc.tiling)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.create_image(&image_info, None)? };

        // Get memory requirements
        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };

        // Find suitable memory type
        let memory_type_index = find_memory_type(memory_properties, mem_requirements.memory_type_bits, desc.memory_flags)
            .ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;

        // Allocate memory
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };

        // Bind memory to image
        unsafe { device.bind_image_memory(image, memory, 0)? };

        let texture = Self {
            desc: desc.clone(),
            image,
            memory,
            views: RefCell::new(Default::default()),
            device: device.clone(),
        };
        Ok(texture)
    }

    /// Create a texture wrapper for a swapchain image (does not own the image or memory).
    pub(crate) fn from_swapchain_image(
        device: &Device,
        image: vk::Image,
        format: vk::Format,
        extent: vk::Extent2D,
    ) -> Self {
        let desc = TextureDesc {
            name: "swapchain_back_buffer".to_owned(),
            format,
            extent: vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            memory_flags: Default::default(),
            image_type: Default::default(),
            view_type: vk::ImageViewType::TYPE_2D,
            mip_levels: 1,
            array_layers: 1,
            samples: Default::default(),
            tiling: Default::default(),
        };
        let texture = Self {
            desc,
            image,
            memory: vk::DeviceMemory::null(),
            views: RefCell::new(Default::default()),
            device: device.clone(),
        };
        texture
    }

    pub fn as_range<R: RangeBounds<u32>>(&self, mipmaps: R, levels: R) -> Result<TextureRange<'_>, vk::Result> {
        let (base_mip, num_mips) = normalize_range_u32(mipmaps, self.desc.mip_levels)?;
        let (base_layer, num_layers) = normalize_range_u32(levels, self.desc.array_layers)?;

        Ok(TextureRange {
            texture: self,
            subresource: TextureSubresource {
                base_mip,
                num_mips,
                base_layer,
                num_layers,
            },
        })
    }

    /// Get the raw Vulkan image handle.
    pub fn handle(&self) -> vk::Image {
        self.image
    }

    #[inline]
    pub fn name(&self) -> &str {
        &self.desc.name
    }

    #[inline]
    pub fn desc(&self) -> &TextureDesc {
        &self.desc
    }

    /// Get the texture format.
    #[inline]
    pub fn format(&self) -> vk::Format {
        self.desc.format
    }

    /// Get the texture extent.
    #[inline]
    pub fn extent(&self) -> vk::Extent3D {
        self.desc.extent
    }

    /// Get the texture width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.desc.extent.width
    }

    /// Get the texture height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.desc.extent.height
    }

    /// Get the texture usage flags.
    #[inline]
    pub fn usage(&self) -> vk::ImageUsageFlags {
        self.desc.usage
    }

    /// Get the aspect flags for this texture based on its format.
    #[inline]
    pub fn aspect(&self) -> vk::ImageAspectFlags {
        format_to_aspect_mask(self.desc.format)
    }
    
    pub fn is_swapchain_texture(&self) -> bool {
        self.memory == vk::DeviceMemory::null() 
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        unsafe {
            for view in self.views.borrow().values() {
                self.device.destroy_image_view(*view, None);
            }

            if self.memory != vk::DeviceMemory::null() {
                self.device.destroy_image(self.image, None);
                self.device.free_memory(self.memory, None);
            }
        }
    }
}

/// Get the appropriate aspect mask for an image format.
fn format_to_aspect_mask(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
            vk::ImageAspectFlags::DEPTH
        }
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        _ => vk::ImageAspectFlags::COLOR,
    }
}

pub struct TextureRange<'a> {
    texture: &'a Texture,
    subresource: TextureSubresource
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
struct TextureSubresource {
    base_mip: u32,
    num_mips: u32,
    base_layer: u32,
    num_layers: u32,
}

impl TextureSubresource {
    fn to_vk(&self, aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: self.base_mip,
            level_count: self.num_mips,
            base_array_layer: self.base_layer,
            layer_count: self.num_layers,
        }
    }
}

impl<'a> TextureRange<'a> {
    #[inline]
    pub fn texture(&self) -> &'a Texture { self.texture }

    pub fn view(&self) -> Result<vk::ImageView, vk::Result> {
        // Cached per-subresource view.
        if let Some(v) = { self.texture.views.borrow().get(&self.subresource).copied() } {
            return Ok(v);
        }

        let aspect_mask = format_to_aspect_mask(self.texture.desc.format);
        let view_info = vk::ImageViewCreateInfo::default()
            .image(self.texture.image)
            .view_type(self.texture.desc.view_type)
            .format(self.texture.desc.format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(self.subresource.to_vk(aspect_mask));

        let view = unsafe { self.texture.device.create_image_view(&view_info, None)? };
        self.texture.views.borrow_mut().insert(self.subresource, view);
        Ok(view)
    }

    pub fn to_binding(&self, sampler: &Sampler, layout: vk::ImageLayout) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo::default()
            .image_view(self.view().expect("Invalid texture view creation."))
            .sampler(sampler.handle())
            .image_layout(layout)
    }
}
