use zenith_rendergraph::RenderGraphBuilder;
use zenith_rhi::{vk, Texture, TextureDesc};

pub struct GBuffer {
    width: u32,
    height: u32,
    base_color_desc: TextureDesc,
    normal_mra_desc: TextureDesc,
}

impl GBuffer {
    pub fn new(width: u32, height: u32) -> Self {
        let base_color_desc = TextureDesc::new_color(
            "gbuffer.base_color",
            width,
            height,
            vk::Format::R8G8B8A8_UNORM,
        );
        let normal_mra_desc = TextureDesc::new_color(
            "gbuffer.normal_mra",
            width,
            height,
            vk::Format::R8G8B8A8_UNORM,
        );

        Self {
            width,
            height,
            base_color_desc,
            normal_mra_desc,
        }
    }

    pub fn ensure_size(&mut self, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }

        self.width = width;
        self.height = height;
        self.base_color_desc = TextureDesc::new_color(
            "gbuffer.base_color",
            width,
            height,
            vk::Format::R8G8B8A8_UNORM,
        );
        self.normal_mra_desc = TextureDesc::new_color(
            "gbuffer.normal_mra",
            width,
            height,
            vk::Format::R8G8B8A8_UNORM,
        );
    }

    pub fn create(
        &self,
        builder: &mut RenderGraphBuilder,
    ) -> (zenith_rendergraph::RenderGraphResource<Texture>, zenith_rendergraph::RenderGraphResource<Texture>) {
        (
            builder.create(self.base_color_desc.clone()),
            builder.create(self.normal_mra_desc.clone()),
        )
    }
}
