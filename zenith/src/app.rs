use std::sync::Arc;
use winit::event::{DeviceEvent, WindowEvent};
use winit::window::Window;
use zenith_core::cli::EngineArgs;
use zenith_rhi::{RenderDevice, Swapchain, Texture};
use zenith_rendergraph::{RenderGraphBuilder};
use crate::rhi::vk;

pub trait App: Sized + 'static {
    fn new(args: &EngineArgs) -> anyhow::Result<Self>;
    fn on_window_event(&mut self, _event: &WindowEvent, _window: &Window) {}
    fn on_device_event(&mut self, _event: &DeviceEvent) {}
    fn tick(&mut self, _delta_time: f32) {}
}

/// All contexts needed to render.
pub struct RenderContext<'a> {
    builder: &'a mut RenderGraphBuilder,
    swapchain: &'a Swapchain,
    frame_index: usize,
}

impl<'a> RenderContext<'a> {
    pub fn new(
        builder: &'a mut RenderGraphBuilder,
        swapchain: &'a Swapchain,
        frame_index: usize
    ) -> Self {
        Self {
            builder,
            swapchain,
            frame_index,
        }
    }

    #[inline]
    pub fn builder(&mut self) -> &mut RenderGraphBuilder { self.builder }

    #[inline]
    pub fn swapchain_texture(&self) -> Arc<Texture> { self.swapchain.swapchain_texture(self.frame_index).clone() }

    #[inline]
    pub fn extent(&self) -> vk::Extent2D { self.swapchain.extent() }

    #[inline]
    pub fn frame_index(&self) -> usize { self.frame_index }
}

pub trait RenderableApp: App {
    fn prepare(&mut self, _render_device: &RenderDevice, _window: Arc<Window>) -> anyhow::Result<()> { Ok(()) }
    fn resize(&mut self, _width: u32, _height: u32) {}
    fn render(&mut self, context: RenderContext);
}
