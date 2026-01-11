//! Vulkan Sampler - texture sampling configuration.

use ash::{vk};
use zenith_rhi_derive::DeviceObject;
use crate::{RenderDevice};
use crate::device::DebuggableObject;
use crate::device::set_debug_name_handle;

/// Sampler configuration.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
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

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
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

impl SamplerConfig {
    /// Create a linear filtering sampler config.
    pub fn linear() -> Self {
        Self::default()
    }

    /// Create a nearest (point) filtering sampler config.
    pub fn nearest() -> Self {
        Self {
            mag_filter: vk::Filter::NEAREST,
            min_filter: vk::Filter::NEAREST,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            ..Default::default()
        }
    }

    /// Create a sampler config with anisotropic filtering.
    pub fn anisotropic(max_anisotropy: f32) -> Self {
        Self {
            anisotropy_enable: true,
            max_anisotropy,
            ..Default::default()
        }
    }

    /// Set address mode for all axes.
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
    name: String,
    sampler: vk::Sampler,
}

impl Sampler {
    /// Create a new sampler with the given configuration.
    pub fn new(name: &str, device: &ash::Device, config: &SamplerConfig) -> Result<Self, vk::Result> {
        let create_info = vk::SamplerCreateInfo::default()
            .mag_filter(config.mag_filter)
            .min_filter(config.min_filter)
            .mipmap_mode(config.mipmap_mode)
            .address_mode_u(config.address_mode_u)
            .address_mode_v(config.address_mode_v)
            .address_mode_w(config.address_mode_w)
            .mip_lod_bias(config.mip_lod_bias)
            .anisotropy_enable(config.anisotropy_enable)
            .max_anisotropy(config.max_anisotropy)
            .compare_enable(config.compare_enable)
            .compare_op(config.compare_op)
            .min_lod(config.min_lod)
            .max_lod(config.max_lod)
            .border_color(config.border_color)
            .unnormalized_coordinates(config.unnormalized_coordinates);

        let sampler = unsafe { device.create_sampler(&create_info, None)? };

        Ok(Self {
            name: name.to_owned(),
            sampler,
            device: device.clone(),
        })
    }

    #[inline]
    pub fn name(&self) -> &str { &self.name }

    /// Get the raw Vulkan sampler handle.
    pub fn handle(&self) -> vk::Sampler {
        self.sampler
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
        }
    }
}

impl DebuggableObject for Sampler {
    fn set_debug_name(&self, device: &RenderDevice) {
        set_debug_name_handle(device, self.sampler, vk::ObjectType::SAMPLER, self.name());
    }
}
