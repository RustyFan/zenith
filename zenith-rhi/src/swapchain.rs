//! Vulkan Swapchain - surface, swapchain, and frame synchronization management.

use std::sync::{Arc, Weak};
use ash::{vk, Device};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;
use zenith_core::log::info;
use anyhow::{anyhow, Result};
use zenith_rhi_derive::DeviceObject;
use crate::{RhiCore, RenderDevice, Texture, Queue, Fence, Semaphore, NUM_BACK_BUFFERS};
use crate::device::DebuggableObject;
use crate::device::set_debug_name_handle;

#[derive(Clone)]
pub struct SwapchainWindow {
    window: Weak<Window>,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
}

impl Drop for SwapchainWindow {
    fn drop(&mut self) {
        unsafe { self.surface_loader.destroy_surface(self.surface, None); }
    }
}

impl SwapchainWindow {
    pub fn new(window: &Arc<Window>, core: &RhiCore) -> Result<Self> {
        let display_handle = window
            .display_handle()
            .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?
            .as_raw();
        let window_handle = window
            .window_handle()
            .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?
            .as_raw();

        let surface_loader = ash::khr::surface::Instance::new(core.entry(), core.instance());
        let surface = unsafe {
            ash_window::create_surface(core.entry(), core.instance(), display_handle, window_handle, None)?
        };

        Ok(Self {
            window: Arc::downgrade(window),
            surface_loader,
            surface,
        })
    }

    pub fn window(&self) -> &Weak<Window> { &self.window }

    pub fn surface_loader(&self) -> &ash::khr::surface::Instance {
        &self.surface_loader
    }

    pub fn surface(&self) -> vk::SurfaceKHR {
        self.surface
    }
}

/// Swapchain configuration parameters.
pub struct SwapchainConfig {
    pub preferred_format: vk::Format,
    pub preferred_color_space: vk::ColorSpaceKHR,
    pub preferred_present_mode: vk::PresentModeKHR,
    pub num_back_buffers: u32,
}

impl Default for SwapchainConfig {
    fn default() -> Self {
        Self {
            preferred_format: vk::Format::B8G8R8A8_SRGB,
            preferred_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            preferred_present_mode: vk::PresentModeKHR::MAILBOX,
            num_back_buffers: NUM_BACK_BUFFERS,
        }
    }
}

/// Synchronization objects for a single frame.
pub struct FrameSync<'a> {
    pub image_available: &'a Semaphore,
    pub render_finished: &'a Semaphore,
    pub in_flight_fence: &'a Fence,
}

/// Vulkan swapchain management.
#[DeviceObject]
pub struct Swapchain {
    name: String,
    physical_device: vk::PhysicalDevice,
    window: SwapchainWindow,

    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,

    textures: Vec<Arc<Texture>>,
    extent: vk::Extent2D,

    image_available_semaphores: Vec<Semaphore>,
    render_finished_semaphores: Vec<Semaphore>,
    in_flight_fences: Vec<Fence>,

    format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,

    current_frame: usize,
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe { self.device.device_wait_idle().unwrap(); }
        self.clean_up_render_resources();

        unsafe {
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
        }
    }
}

impl Swapchain {
    #[profiling::function]
    pub fn new(
        name: &str,
        core: &RhiCore,
        device: &RenderDevice,
        window: SwapchainWindow,
        config: SwapchainConfig,
    ) -> Result<Self> {
        if device.graphics_queue().family_index() != device.present_queue().family_index() {
            return Err(anyhow!("Graphic queue and present queue should be the same!"));
        }

        let physical_device = device.parent_physical_device();
        let capabilities = unsafe {
            window.surface_loader.get_physical_device_surface_capabilities(physical_device.handle(), window.surface)?
        };
        let formats = unsafe {
            window.surface_loader.get_physical_device_surface_formats(physical_device.handle(), window.surface)?
        };
        let format = choose_surface_format(&formats, &config);

        let present_modes = unsafe {
            window.surface_loader.get_physical_device_surface_present_modes(physical_device.handle(), window.surface)?
        };
        let present_mode = choose_present_mode(&present_modes, &config);

        let os_window = window.window.upgrade().ok_or(anyhow!("Try to create a swapchain with invalid window instance."))?;
        let extent = vk::Extent2D {
            width: os_window.inner_size().width,
            height: os_window.inner_size().height,
        };
        let extent = get_swapchain_extent(&capabilities, extent);

        let swapchain_loader = ash::khr::swapchain::Device::new(core.instance(), device.handle());
        let swapchain = Swapchain::create_or_recreate(
            &swapchain_loader,
            window.surface,
            capabilities,
            format,
            present_mode,
            config.num_back_buffers,
            extent,
            vk::SwapchainKHR::null(),
        )?;

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let mut textures = Vec::with_capacity(images.len());

        for (idx, image) in images.iter().enumerate() {
            let texture = Texture::from_swapchain_image(
                device,
                format!("swapchain.backbuffer.f{idx}"),
                image.clone(),
                format.format,
                extent.clone(),
            );
            textures.push(Arc::new(texture));
        }

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            create_sync_objects(device, images.len())?;

        set_debug_name_handle(device, swapchain, vk::ObjectType::SWAPCHAIN_KHR, name);
        Ok(Swapchain {
            name: name.to_owned(),
            physical_device: physical_device.handle(),
            window,
            swapchain_loader,
            swapchain,
            textures,
            format,
            extent,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            current_frame: 0,
            present_mode,
            device: device.handle().clone(),
        })
    }

    #[inline]
    pub fn name(&self) -> &str { &self.name }

    #[inline]
    pub fn handle(&self) -> vk::SwapchainKHR { self.swapchain }

    #[profiling::function]
    pub fn acquire_next_image(&mut self, device: &Device) -> Result<(u32, bool), vk::Result> {
        // Wait for the fence of the current frame
        unsafe {
            device.wait_for_fences(
                &[self.in_flight_fences[self.current_frame].handle()],
                true,
                u64::MAX,
            )?;
        }

        // Acquire next image
        let (image_index, suboptimal) = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available_semaphores[self.current_frame].handle(),
                vk::Fence::null(),
            )?
        };

        Ok((image_index, suboptimal))
    }

    /// Reset the fence for the current frame before submitting work.
    pub fn reset_current_fence(&self, device: &Device) -> Result<(), vk::Result> {
        unsafe {
            device.reset_fences(&[self.in_flight_fences[self.current_frame].handle()])?;
        }
        Ok(())
    }

    /// Present the rendered image.
    /// Returns whether the swapchain is suboptimal.
    #[profiling::function]
    pub fn present(&mut self, present_queue: Queue, image_index: u32) -> Result<bool, vk::Result> {
        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let wait_semaphores = [self.render_finished_semaphores[self.current_frame].handle()];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        self.window.window.upgrade().unwrap().pre_present_notify();
        let result = unsafe { self.swapchain_loader.queue_present(present_queue.handle(), &present_info) };

        self.current_frame = (self.current_frame + 1) % self.textures.len();

        match result {
            Ok(suboptimal) => Ok(suboptimal),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(true),
            Err(e) => Err(e),
        }
    }

    /// Get current frame synchronization objects.
    pub fn current_frame_sync(&self) -> FrameSync<'_> {
        FrameSync {
            image_available: &self.image_available_semaphores[self.current_frame],
            render_finished: &self.render_finished_semaphores[self.current_frame],
            in_flight_fence: &self.in_flight_fences[self.current_frame],
        }
    }

    pub fn resize(&mut self, device: &RenderDevice, extent: vk::Extent2D) -> Result<()> {
        device.wait_until_idle()?;

        // re-query surface capabilities as they may have changed
        let capabilities = unsafe {
            self.window.surface_loader.get_physical_device_surface_capabilities(self.physical_device, self.window.surface)?
        };
        let extent = get_swapchain_extent(&capabilities, extent);

        let config = SwapchainConfig::default();
        let swapchain = Swapchain::create_or_recreate(
            &self.swapchain_loader,
            self.window.surface,
            capabilities,
            self.format,
            self.present_mode,
            config.num_back_buffers,
            extent,
            self.swapchain,
        )?;

        self.clean_up_render_resources();

        let images = unsafe { self.swapchain_loader.get_swapchain_images(swapchain)? };
        let mut textures = Vec::with_capacity(images.len());

        for (idx, image) in images.iter().enumerate() {
            let texture = Texture::from_swapchain_image(
                &device,
                format!("swapchain.backbuffer.f{idx}"),
                image.clone(),
                self.format.format,
                extent.clone(),
            );
            textures.push(Arc::new(texture));
        }

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            create_sync_objects(&device, images.len())?;

        self.textures = textures;
        self.image_available_semaphores = image_available_semaphores;
        self.render_finished_semaphores = render_finished_semaphores;
        self.in_flight_fences = in_flight_fences;

        self.extent = extent;
        self.swapchain = swapchain;

        Ok(())
    }

    /// Recreate the swapchain with a new extent.
    #[profiling::function]
    fn create_or_recreate(
        swapchain_loader: &ash::khr::swapchain::Device,
        surface: vk::SurfaceKHR,
        capabilities: vk::SurfaceCapabilitiesKHR,
        format: vk::SurfaceFormatKHR,
        present_mode: vk::PresentModeKHR,
        num_back_buffers: u32,
        extent: vk::Extent2D,
        old_swapchain: vk::SwapchainKHR,
    ) -> Result<vk::SwapchainKHR> {
        let mut image_count = num_back_buffers;
        image_count = image_count.max(capabilities.min_image_count);
        if capabilities.max_image_count > 0 {
            image_count = image_count.min(capabilities.max_image_count);
        }

        info!(
            "Creating new swapchain: {:?} {:?}, {}x{}, {} images, {:?}",
            format.format,
            format.color_space,
            extent.width,
            extent.height,
            image_count,
            present_mode
        );

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&[][..])
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        let swapchain = unsafe { swapchain_loader.create_swapchain(&create_info, None)? };

        if old_swapchain != vk::SwapchainKHR::null() {
            unsafe {
                swapchain_loader.destroy_swapchain(old_swapchain, None);
            }
        }

        Ok(swapchain)
    }

    fn clean_up_render_resources(&mut self) {
        // Fence/Semaphore drop will clean up.
        self.image_available_semaphores.clear();
        self.render_finished_semaphores.clear();
        self.in_flight_fences.clear();
    }
    
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }
    
    pub fn format(&self) -> vk::Format {
        self.format.format
    }
    
    pub fn num_back_buffers(&self) -> u32 { self.textures.len() as u32 }

    pub fn swapchain_texture(&self, frame_index: usize) -> Arc<Texture> {
        self.textures[frame_index].clone()
    }

    pub fn window(&self) -> &SwapchainWindow {
        &self.window
    }
}

impl DebuggableObject for Swapchain {
    fn set_debug_name(&self, device: &RenderDevice) {
        set_debug_name_handle(device, self.swapchain, vk::ObjectType::SWAPCHAIN_KHR, self.name());
    }
}

fn choose_surface_format(
    formats: &[vk::SurfaceFormatKHR],
    config: &SwapchainConfig,
) -> vk::SurfaceFormatKHR {
    // Try to find preferred format
    formats
        .iter()
        .find(|f| {
            f.format == config.preferred_format && f.color_space == config.preferred_color_space
        })
        .copied()
        .unwrap_or(formats[0])
}

fn choose_present_mode(
    modes: &[vk::PresentModeKHR],
    config: &SwapchainConfig,
) -> vk::PresentModeKHR {
    // Prefer requested mode, fallback to mailbox, then FIFO (always available)
    if modes.contains(&config.preferred_present_mode) {
        config.preferred_present_mode
    } else if modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else {
        vk::PresentModeKHR::FIFO // Guaranteed to be available
    }
}

fn get_swapchain_extent(
    capabilities: &vk::SurfaceCapabilitiesKHR,
    window_extent: vk::Extent2D,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: window_extent.width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: window_extent.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }
}

fn create_sync_objects(
    device: &RenderDevice,
    count: usize,
) -> Result<(Vec<Semaphore>, Vec<Semaphore>, Vec<Fence>), vk::Result> {
    let mut image_available = Vec::with_capacity(count);
    let mut render_finished = Vec::with_capacity(count);
    let mut in_flight = Vec::with_capacity(count);

    for _ in 0..count {
        image_available.push(Semaphore::new("semaphore.image_available", device)?);
        render_finished.push(Semaphore::new("semaphore.render_finish", device)?);
        in_flight.push(Fence::new("fence.swapchain", device, true)?);
    }

    Ok((image_available, render_finished, in_flight))
}