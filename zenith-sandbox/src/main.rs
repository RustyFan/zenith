use zenith::rhi::{vk, TextureState};
use zenith::{launch, App, Args, RenderContext, RenderableApp};

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
        let mut output = builder.import("back_buffer", output, TextureState::Undefined);

        let mut node = builder.add_lambda_node("clear");

        let output_access = node.write_hint(&mut output, TextureState::General, vk::PipelineStageFlags2::TRANSFER);
        node.execute(move |ctx| {
            let rt = ctx.get_texture(&output_access);
            let encoder = ctx.command_encoder();

            encoder.custom(|device, cmd| {
                unsafe {
                    device.cmd_clear_color_image(
                        cmd,
                        rt.handle(),
                        vk::ImageLayout::GENERAL,
                        &vk::ClearColorValue { float32: [0.2, 0.3, 0.8, 1.0] },
                        &[
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1)
                        ]);
                }
            });
        });
    }
}

fn main() {
    launch::<SimpleApp>().expect("Failed to launch zenith engine loop!");
}
