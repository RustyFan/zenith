//! Vulkan Core - instance and physical device selection.

use ash::{vk, Entry, Instance};
use raw_window_handle::{HasDisplayHandle, RawDisplayHandle};
use std::ffi::{CStr, CString};
use anyhow::anyhow;
use winit::window::Window;
use zenith_core::log;

use crate::device::RenderDevice;
use crate::swapchain::SwapchainWindow;

/// Validation layers to enable in debug builds.
#[cfg(feature = "validation")]
const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];

/// Scoring weights for physical device selection.
const SCORE_DISCRETE_GPU: u32 = 10000;
const SCORE_INTEGRATED_GPU: u32 = 1000;
const SCORE_PER_GB_VRAM: u32 = 100;
const SCORE_VULKAN_1_4: u32 = 600;
const SCORE_VULKAN_1_3: u32 = 400;
const SCORE_VULKAN_1_2: u32 = 200;

#[derive(Clone)]
pub struct PhysicalDevice {
    handle: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    memory_properties: vk::PhysicalDeviceMemoryProperties,

    graphics_queue_family: u32,
    present_queue_family: u32,
}

impl PhysicalDevice {
    /// Get the physical device.
    pub fn handle(&self) -> vk::PhysicalDevice {
        self.handle
    }

    /// Get the physical device properties.
    pub fn properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.properties
    }

    /// Get the physical device memory properties.
    pub fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.memory_properties
    }

    pub fn graphics_queue_family(&self) -> u32 { self.graphics_queue_family }

    pub fn present_queue_family(&self) -> u32 { self.present_queue_family }
}

/// This is the global entry point for Vulkan initialization.
pub struct RhiCore {
    entry: Entry,
    instance: Instance,

    /// Debug messenger (only in debug builds with validation).
    #[cfg(feature = "validation")]
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    #[cfg(feature = "validation")]
    debug_utils: Option<ash::ext::debug_utils::Instance>,
}

impl RhiCore {
    /// Create a new Vulkan core with instance and physical device.
    #[profiling::function]
    pub fn new(window: &Window) -> Result<Self, anyhow::Error> {
        // Load Vulkan dynamically
        let entry = unsafe { Entry::load()? };

        // Get display handle for platform-specific extensions
        let display_handle = window.display_handle()?.as_raw();

        // Create instance
        let instance = create_instance(&entry, display_handle)?;

        // Setup debug messenger (validation only)
        #[cfg(feature = "validation")]
        let (debug_utils, debug_messenger) = setup_debug_messenger(&entry, &instance)?;

        Ok(Self {
            entry,
            instance,
            #[cfg(feature = "validation")]
            debug_messenger,
            #[cfg(feature = "validation")]
            debug_utils,
        })
    }

    /// Create a logical device from this core.
    pub fn create_render_device(&self, physical_device: &PhysicalDevice) -> Result<RenderDevice, vk::Result> {
        RenderDevice::new(
            &self.instance,
            physical_device,
            3,
        )
    }

    /// Get the entry point.
    pub fn entry(&self) -> &Entry {
        &self.entry
    }

    /// Get a reference to the Vulkan instance.
    pub fn instance(&self) -> &Instance {
        &self.instance
    }
}

impl Drop for RhiCore {
    fn drop(&mut self) {
        unsafe {
            #[cfg(feature = "validation")]
            if let (Some(debug_utils), Some(messenger)) = (&self.debug_utils, self.debug_messenger) {
                debug_utils.destroy_debug_utils_messenger(messenger, None);
            }

            self.instance.destroy_instance(None);
        }
    }
}

/// Get required instance extensions based on platform.
fn get_required_instance_extensions(display_handle: RawDisplayHandle) -> Vec<*const i8> {
    let mut extensions = vec![
        // Surface extension (always needed)
        ash::khr::surface::NAME.as_ptr(),
    ];

    // Platform-specific surface extension
    #[cfg(target_os = "windows")]
    {
        let _ = display_handle; // Suppress unused warning
        extensions.push(ash::khr::win32_surface::NAME.as_ptr());
    }

    #[cfg(target_os = "linux")]
    {
        match display_handle {
            RawDisplayHandle::Xlib(_) => {
                extensions.push(ash::khr::xlib_surface::NAME.as_ptr());
            }
            RawDisplayHandle::Xcb(_) => {
                extensions.push(ash::khr::xcb_surface::NAME.as_ptr());
            }
            RawDisplayHandle::Wayland(_) => {
                extensions.push(ash::khr::wayland_surface::NAME.as_ptr());
            }
            _ => {}
        }
    }

    // Debug utils (for validation layers)
    #[cfg(feature = "validation")]
    extensions.push(ash::ext::debug_utils::NAME.as_ptr());

    extensions
}

/// Create Vulkan instance with required extensions and validation layers.
fn create_instance(entry: &Entry, display_handle: RawDisplayHandle) -> Result<Instance, vk::Result> {
    let app_name = CString::new("Zenith Engine").unwrap();
    let engine_name = CString::new("Zenith").unwrap();

    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::API_VERSION_1_3);

    let extensions = get_required_instance_extensions(display_handle);

    #[cfg(feature = "validation")]
    let layer_names: Vec<CString> = VALIDATION_LAYERS
        .iter()
        .map(|&s| CString::new(s).unwrap())
        .collect();

    #[cfg(feature = "validation")]
    let layer_pointers: Vec<*const i8> = layer_names.iter().map(|s| s.as_ptr()).collect();

    let mut create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&extensions);

    #[cfg(feature = "validation")]
    {
        create_info = create_info.enabled_layer_names(&layer_pointers);
    }

    unsafe { entry.create_instance(&create_info, None) }
}

/// Setup debug messenger for validation layers.
#[cfg(feature = "validation")]
fn setup_debug_messenger(
    entry: &Entry,
    instance: &Instance,
) -> Result<(Option<ash::ext::debug_utils::Instance>, Option<vk::DebugUtilsMessengerEXT>), vk::Result> {
    let debug_utils = ash::ext::debug_utils::Instance::new(entry, instance);

    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));

    let messenger = unsafe { debug_utils.create_debug_utils_messenger(&create_info, None)? };

    Ok((Some(debug_utils), Some(messenger)))
}

/// Vulkan debug callback function.
#[cfg(feature = "validation")]
unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let callback_data = unsafe { *p_callback_data };
    let message = unsafe { CStr::from_ptr(callback_data.p_message) }.to_string_lossy();

    let type_str = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        _ => "[Unknown]",
    };

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            log::error!("Vulkan {}: {}", type_str, message);
            // TODO: break point
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            log::warn!("Vulkan {}: {}", type_str, message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            log::info!("Vulkan {}: {}", type_str, message);
        }
        _ => {
            log::debug!("Vulkan {}: {}", type_str, message);
        }
    }

    vk::FALSE
}

/// Find queue families that support graphics and present.
fn find_queue_families(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    swapchain_window: &SwapchainWindow,
) -> (Option<u32>, Option<u32>) {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    let mut graphics_family = None;
    let mut present_family = None;

    for (index, family) in queue_families.iter().enumerate() {
        let index = index as u32;

        // Check for graphics support
        if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            graphics_family = Some(index);
        }

        // Check for present support
        let present_support = unsafe {
            swapchain_window.surface_loader()
                .get_physical_device_surface_support(physical_device, index, swapchain_window.surface())
                .unwrap_or(false)
        };

        if present_support {
            present_family = Some(index);
        }

        // Prefer queue family that supports both
        if graphics_family.is_some() && present_family.is_some() {
            break;
        }
    }

    (graphics_family, present_family)
}

/// Calculate a score for the physical device (higher is better).
fn score_physical_device(
    properties: &vk::PhysicalDeviceProperties,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    has_required_queues: bool,
) -> u32 {
    if !has_required_queues {
        return 0; // Unusable device
    }

    let mut score = 0u32;

    // Device type scoring (discrete > integrated > others)
    match properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => score += SCORE_DISCRETE_GPU,
        vk::PhysicalDeviceType::INTEGRATED_GPU => score += SCORE_INTEGRATED_GPU,
        vk::PhysicalDeviceType::VIRTUAL_GPU => score += 500,
        vk::PhysicalDeviceType::CPU => score += 100,
        _ => score += 10,
    }

    // API version scoring
    let api_version = properties.api_version;
    if api_version >= vk::make_api_version(0, 1, 4, 0) {
        score += SCORE_VULKAN_1_4;
    } else if api_version >= vk::API_VERSION_1_3 {
        score += SCORE_VULKAN_1_3;
    } else if api_version >= vk::API_VERSION_1_2 {
        score += SCORE_VULKAN_1_2;
    }

    // VRAM scoring (calculate total device-local memory)
    let vram_bytes: u64 = memory_properties.memory_heaps
        [..memory_properties.memory_heap_count as usize]
        .iter()
        .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|heap| heap.size)
        .sum();

    let vram_gb = (vram_bytes / (1024 * 1024 * 1024)) as u32;
    score += vram_gb * SCORE_PER_GB_VRAM;

    score
}

/// Select the best physical device.
pub fn select_physical_device(
    instance: &Instance,
    swapchain_window: &SwapchainWindow,
) -> Result<PhysicalDevice, anyhow::Error> {
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };

    if physical_devices.is_empty() {
        return Err(anyhow::anyhow!("No Vulkan-capable GPU found"));
    }

    let mut best_device = None;
    let mut best_device_score = 0u32;

    for device in physical_devices {
        let properties = unsafe { instance.get_physical_device_properties(device) };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(device) };

        let (graphics_family, present_family) = find_queue_families(instance, device, swapchain_window);
        let graphics_queue_family = graphics_family.ok_or(anyhow!("Invalid graphic queue family."))?;
        let present_queue_family = present_family.ok_or(anyhow!("Invalid graphic queue family."))?;

        let has_required_queues = graphics_family.is_some() && present_family.is_some();
        let score = score_physical_device(&properties, &memory_properties, has_required_queues);

        let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy() };

        log::info!(
            "Found GPU: {} (score: {}, type: {:?})",
            device_name, score, properties.device_type
        );

        if score > best_device_score {
            best_device = Some(PhysicalDevice {
                handle: device,
                properties,
                memory_properties,
                graphics_queue_family,
                present_queue_family,
            });
            best_device_score = score;
        }
    }

    best_device.ok_or_else(|| anyhow::anyhow!("No suitable GPU found"))
}

