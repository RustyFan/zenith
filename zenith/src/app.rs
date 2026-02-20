use std::sync::Arc;
use winit::event::{DeviceEvent, WindowEvent};
use winit::window::Window;
use zenith_core::cli::EngineArgs;
use zenith_rhi::{RenderDevice, Swapchain, Texture, vk, BindlessPool};
use zenith_rendergraph::{RenderGraphBuilder};

pub trait App: Sized + 'static {
    fn new(args: &EngineArgs) -> anyhow::Result<Self>;
    #[allow(unused)]
    fn on_window_event(&mut self, event: &WindowEvent, window: &Window) {}
    #[allow(unused)]
    fn on_device_event(&mut self, event: &DeviceEvent) {}
    #[allow(unused)]
    fn tick(&mut self, delta_time: f32) {}
}

pub struct RenderContext<'a> {
    pub bindless_pool: &'a mut BindlessPool,
    swapchain: &'a Swapchain,
    frame_index: usize,
}

impl<'a> RenderContext<'a> {
    pub fn new(
        bindless_pool: &'a mut BindlessPool,
        swapchain: &'a Swapchain,
        frame_index: usize,
    ) -> Self {
        Self {
            bindless_pool,
            swapchain,
            frame_index,
        }
    }

    #[inline]
    pub fn swapchain_texture(&self) -> Arc<Texture> { self.swapchain.swapchain_texture(self.frame_index).clone() }

    #[inline]
    pub fn extent(&self) -> vk::Extent2D { self.swapchain.extent() }

    #[inline]
    pub fn frame_index(&self) -> usize { self.frame_index }
}

pub trait RenderableApp: App {
    #[allow(unused)]
    fn prepare(&mut self, render_device: &Arc<RenderDevice>, bindless_pool: &mut BindlessPool, window: Arc<Window>) -> anyhow::Result<()> { Ok(()) }
    #[allow(unused)]
    fn resize(&mut self, width: u32, height: u32) {}
    fn render<'a>(&mut self, builder: &mut RenderGraphBuilder, context: RenderContext<'a>);
}
