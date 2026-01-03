//! Vulkan Texture - GPU texture resource management.

use ash::{vk, Device};
use std::cell::RefCell;
use crate::buffer::find_memory_type;

/// Texture descriptor for creating GPU textures.
#[derive(Debug, Clone)]
pub struct TextureDesc {
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
    /// Sharing mode between queue families.
    pub sharing_mode: vk::SharingMode,
    /// Initial image layout.
    pub initial_layout: vk::ImageLayout,
}

impl Default for TextureDesc {
    fn default() -> Self {
        Self {
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
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            initial_layout: vk::ImageLayout::UNDEFINED,
        }
    }
}

impl TextureDesc {
    /// Create a new 2D texture descriptor.
    pub fn new_2d(width: u32, height: u32, format: vk::Format) -> Self {
        Self {
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
    pub fn color_attachment(width: u32, height: u32, format: vk::Format) -> Self {
        Self {
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
    pub fn depth_attachment(width: u32, height: u32) -> Self {
        Self {
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
    pub fn depth_stencil_attachment(width: u32, height: u32) -> Self {
        Self {
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

    /// Create a storage texture descriptor.
    pub fn storage(width: u32, height: u32, format: vk::Format) -> Self {
        Self {
            format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
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

    /// Set the sharing mode.
    pub fn with_sharing_mode(mut self, mode: vk::SharingMode) -> Self {
        self.sharing_mode = mode;
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

/// GPU texture with memory allocation and optional image view.
pub struct Texture {
    device: Device,
    image: vk::Image,
    /// If memory is null, it is a swapchain texture
    memory: vk::DeviceMemory,
    view: RefCell<Option<vk::ImageView>>,
    format: vk::Format,
    extent: vk::Extent3D,
    usage: vk::ImageUsageFlags,
    mip_levels: u32,
    array_layers: u32,
    view_type: vk::ImageViewType,
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
            .sharing_mode(desc.sharing_mode)
            .initial_layout(desc.initial_layout);

        let image = unsafe { device.create_image(&image_info, None)? };

        // Get memory requirements
        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };

        // Find suitable memory type
        let memory_type_index =
            find_memory_type(memory_properties, mem_requirements.memory_type_bits, desc.memory_flags)
                .ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;

        // Allocate memory
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };

        // Bind memory to image
        unsafe { device.bind_image_memory(image, memory, 0)? };

        let texture = Self {
            device: device.clone(),
            image,
            memory,
            view: RefCell::new(None),
            format: desc.format,
            extent: desc.extent,
            usage: desc.usage,
            mip_levels: desc.mip_levels,
            array_layers: desc.array_layers,
            view_type: desc.view_type,
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
        let texture = Self {
            device: device.clone(),
            image,
            memory: vk::DeviceMemory::null(),
            view: RefCell::new(None),
            format,
            extent: vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
            mip_levels: 1,
            array_layers: 1,
            view_type: vk::ImageViewType::TYPE_2D,
        };
        texture
    }

    /// Get or create the texture view (lazily creates the view on first call).
    pub fn view(&self) -> Result<vk::ImageView, vk::Result> {
        if let Some(view) = *self.view.borrow() {
            return Ok(view);
        }

        let aspect_mask = format_to_aspect_mask(self.format);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(self.image)
            .view_type(self.view_type)
            .format(self.format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: self.mip_levels,
                base_array_layer: 0,
                layer_count: self.array_layers,
            });

        let view = unsafe { self.device.create_image_view(&view_info, None)? };
        *self.view.borrow_mut() = Some(view);
        Ok(view)
    }

    /// Get the raw Vulkan image handle.
    pub fn handle(&self) -> vk::Image {
        self.image
    }

    /// Get the texture format.
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Get the texture extent.
    pub fn extent(&self) -> vk::Extent3D {
        self.extent
    }

    /// Get the texture width.
    pub fn width(&self) -> u32 {
        self.extent.width
    }

    /// Get the texture height.
    pub fn height(&self) -> u32 {
        self.extent.height
    }

    /// Get the texture usage flags.
    pub fn usage(&self) -> vk::ImageUsageFlags {
        self.usage
    }

    /// Get the aspect flags for this texture based on its format.
    pub fn aspect(&self) -> vk::ImageAspectFlags {
        format_to_aspect_mask(self.format)
    }
    
    pub fn is_swapchain_texture(&self) -> bool {
        self.memory == vk::DeviceMemory::null() 
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        unsafe {
            if let Some(view) = *self.view.borrow() {
                self.device.destroy_image_view(view, None);
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
