use std::sync::Arc;
use winit::window::Window;
use zenith_core::log::info;

/// Render device to maintain and dispatch all rendering instructions.
pub struct RenderDevice {
    #[allow(dead_code)]
    instance: wgpu::Instance,
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
}

impl RenderDevice {
    #[profiling::function]
    pub fn new(window: Arc<Window>) -> Result<Self, anyhow::Error> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::VALIDATION,
            ..Default::default()
        });

        let adapter = pollster::block_on(async {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await
                .unwrap()
        });
        let adapter_info = adapter.get_info();
        info!("Selected adapter: {} ({:?})\n\tDriver {}: {}",
            adapter_info.name,
            adapter_info.backend,
            adapter_info.driver,
            adapter_info.driver_info);

        let (device, queue) = pollster::block_on(async {
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("zenith rhi device"),
                        ..Default::default()
                    },
                )
                .await
                .unwrap()
        });

        let window_size = window.inner_size();
        let width = window_size.width.max(1);
        let height = window_size.height.max(1);
        let surface = instance.create_surface(window)?;

        let mut surface_config = surface
            .get_default_config(&adapter, width, height)
            .expect("Surface isn't supported by the adapter.");
        surface_config.usage |= wgpu::TextureUsages::COPY_DST;

        let view_format = surface_config.format.add_srgb_suffix();
        surface_config.view_formats.push(view_format);

        info!("Picked surface pixel format: {:?}, resolution({}x{})", surface_config.format, width, height);

        surface.configure(&device, &surface_config);

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface,
            surface_config,
        })
    }

    /// Return the inner render device (wgpu).
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Return the main submit queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Acquire next frame from swapchain.
    /// If acquire fails, this function will panic.
    #[profiling::function]
    pub fn acquire_next_frame(&self) -> wgpu::SurfaceTexture {
        match self.surface.get_current_texture() {
            Ok(frame) => frame,
            // If we timed out, just try again
            Err(wgpu::SurfaceError::Timeout) => self.surface
                .get_current_texture()
                .expect("Failed to acquire next surface texture!"),
            Err(
                // If the surface is outdated, or was lost, reconfigure it.
                wgpu::SurfaceError::Outdated
                | wgpu::SurfaceError::Lost
                | wgpu::SurfaceError::Other
                // If OutOfMemory happens, reconfiguring may not help, but we might as well try
                | wgpu::SurfaceError::OutOfMemory,
            ) => {
                self.surface.configure(&self.device, &self.surface_config);
                self.surface
                    .get_current_texture()
                    .expect("Failed to acquire next surface texture!")
            }
        }
    }

    /// Resize the swapchain with specific width and height.
    #[profiling::function]
    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface.configure(&self.device, &self.surface_config);
    }
}

