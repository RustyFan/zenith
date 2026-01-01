#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use zenith::rhi::{vk, TextureState};
use zenith::{launch, App, Args, RenderContext, RenderableApp};
use zenith::rendergraph::ColorInfo;

static mut FIRST_FRAME_COUNTER: u8 = 3;

pub struct SimpleApp;

impl App for SimpleApp {
    fn new(_args: &Args) -> anyhow::Result<Self> {
        Ok(Self {})
    }
}

impl RenderableApp for SimpleApp {
    fn render(&mut self, mut context: RenderContext) {
        let extent = context.extent();
        let (width, height) = (extent.width, extent.height);

        if width == 0 || height == 0 {
            return;
        }

        let output = context.swapchain_texture();

        let builder = context.builder();
        let mut output = builder.import("back_buffer", output, if unsafe { FIRST_FRAME_COUNTER > 0 } { TextureState::Undefined } else { TextureState::Present });

        let mut node = builder.add_graphic_node("clear");
        let output_access = node.write(&mut output, TextureState::Color);

        let mut output_color_info = ColorInfo::default();
        output_color_info.load_op = vk::AttachmentLoadOp::CLEAR;
        output_color_info.store_op = vk::AttachmentStoreOp::STORE;
        output_color_info.clear_value = [0.2, 0.3, 0.8, 1.0]; // Blue-ish color

        node.setup_pipeline()
            .with_color(output_access, output_color_info);

        node.execute(move |ctx, cmd| {
            let extent = vk::Extent2D { width, height };

            ctx.begin_rendering(cmd, extent);
            ctx.end_rendering(cmd);
        });

        unsafe { if FIRST_FRAME_COUNTER > 0 { FIRST_FRAME_COUNTER -= 1 }; }
    }
}

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    launch::<SimpleApp>().expect("Failed to launch zenith engine loop!");
}
