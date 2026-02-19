//! Vulkan Sampler - texture sampling configuration.

use ash::{vk};
use zenith_rhi_derive::DeviceObject;
use crate::{RenderDevice};
use crate::descriptor::{BindableResource, ResourceBinding};
use crate::device::DebuggableObject;
use crate::device::set_debug_name_handle;

#[derive(Debug, Clone)]
pub struct SamplerDesc {
    pub name: String,
    pub mag_filter: vk::Filter,
    pub min_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode_u: vk::SamplerAddressMode,
    pub address_mode_v: vk::SamplerAddressMode,
    pub address_mode_w: vk::SamplerAddressMode,
    pub mip_lod_bias: f32,
    pub anisotropy_enable: bool,
    pub max_anisotropy: f32,
    pub compare_enable: bool,
    pub compare_op: vk::CompareOp,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border_color: vk::BorderColor,
    pub unnormalized_coordinates: bool,
}

impl Default for SamplerDesc {
    fn default() -> Self {
        Self {
            name: "unnamed_sampler".to_owned(),
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
            mip_lod_bias: 0.0,
            anisotropy_enable: false,
            max_anisotropy: 1.0,
            compare_enable: false,
            compare_op: vk::CompareOp::ALWAYS,
            min_lod: 0.0,
            max_lod: vk::LOD_CLAMP_NONE,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: false,
        }
    }
}

impl SamplerDesc {
    #[inline]
    pub fn linear() -> Self {
        Self::default()
    }

    pub fn nearest() -> Self {
        Self {
            mag_filter: vk::Filter::NEAREST,
            min_filter: vk::Filter::NEAREST,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            ..Default::default()
        }
    }

    pub fn anisotropic(max_anisotropy: f32) -> Self {
        Self {
            anisotropy_enable: true,
            max_anisotropy,
            ..Default::default()
        }
    }

    pub fn with_address_mode(mut self, mode: vk::SamplerAddressMode) -> Self {
        self.address_mode_u = mode;
        self.address_mode_v = mode;
        self.address_mode_w = mode;
        self
    }
}

/// Vulkan sampler for texture sampling.
#[DeviceObject]
pub struct Sampler {
    desc: SamplerDesc,
    sampler: vk::Sampler,
}

impl Sampler {
    /// Create a new sampler with the given configuration.
    pub fn new(device: &RenderDevice, desc: &SamplerDesc) -> Result<Self, vk::Result> {
        let create_info = vk::SamplerCreateInfo::default()
            .mag_filter(desc.mag_filter)
            .min_filter(desc.min_filter)
            .mipmap_mode(desc.mipmap_mode)
            .address_mode_u(desc.address_mode_u)
            .address_mode_v(desc.address_mode_v)
            .address_mode_w(desc.address_mode_w)
            .mip_lod_bias(desc.mip_lod_bias)
            .anisotropy_enable(desc.anisotropy_enable)
            .max_anisotropy(desc.max_anisotropy)
            .compare_enable(desc.compare_enable)
            .compare_op(desc.compare_op)
            .min_lod(desc.min_lod)
            .max_lod(desc.max_lod)
            .border_color(desc.border_color)
            .unnormalized_coordinates(desc.unnormalized_coordinates);

        let sampler = unsafe { device.handle().create_sampler(&create_info, None)? };

        Ok(Self {
            desc: desc.clone(),
            sampler,
            device: device.handle().clone(),
        })
    }

    #[inline]
    pub fn name(&self) -> &str { &self.desc.name }

    #[inline]
    pub fn handle(&self) -> vk::Sampler {
        self.sampler
    }

    #[inline]
    pub fn desc(&self) -> &SamplerDesc {
        &self.desc
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
        }
    }
}

impl BindableResource for Sampler {
    fn as_binding(&self) -> ResourceBinding {
        ResourceBinding::Sampler(
            vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(vk::ImageView::null())
                .image_layout(vk::ImageLayout::UNDEFINED)
        )
    }
}

impl DebuggableObject for Sampler {
    fn set_debug_name(&self, device: &ash::ext::debug_utils::Device, name: &str) {
        set_debug_name_handle(device, self.sampler, vk::ObjectType::SAMPLER, name);
    }
}
