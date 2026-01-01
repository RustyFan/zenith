use std::sync::Arc;
use winit::window::Window;
use zenith::{launch, App, Args, RenderableApp, RenderContext};
use zenith::rhi::{RenderDevice, TextureState};
use zenith::renderer::TriangleRenderer;

static mut FIRST_FRAME_COUNTER: u8 = 3;

pub struct TriangleApp {
    triangle_renderer: Option<TriangleRenderer>,
}

impl App for TriangleApp {
    fn new(_args: &Args) -> Result<Self, anyhow::Error> {
        Ok(Self {
            triangle_renderer: None,
        })
    }
}

impl RenderableApp for TriangleApp {
    fn prepare(&mut self, render_device: &RenderDevice, _window: Arc<Window>) -> Result<(), anyhow::Error> {
        self.triangle_renderer = Some(TriangleRenderer::new(render_device));
        Ok(())
    }

    fn render(&mut self, mut context: RenderContext) {
        let extent = context.extent();
        if extent.width == 0 || extent.height == 0 {
            return;
        }

        let output = context.swapchain_texture();
        let builder = context.builder();

        let initial_state = if unsafe { FIRST_FRAME_COUNTER > 0 } {
            TextureState::Undefined
        } else {
            TextureState::Present
        };

        let mut output = builder.import("back_buffer", output, initial_state);

        self.triangle_renderer.as_ref().unwrap().render_to(
            builder,
            &mut output,
            extent.width,
            extent.height,
        );

        unsafe { if FIRST_FRAME_COUNTER > 0 { FIRST_FRAME_COUNTER -= 1 }; }
    }
}

fn main() {
    launch::<TriangleApp>().expect("Failed to launch zenith engine loop!");
}
