use zenith_rendergraph::{RenderGraphBuilder, RenderGraphResource};
use zenith_rhi::{vk, Texture, TextureDesc};

pub struct SceneTextures {
    pub base_color: RenderGraphResource<Texture>,
    pub normal_mra: RenderGraphResource<Texture>,
    pub depth: RenderGraphResource<Texture>,
}

impl SceneTextures {
    pub fn new(builder: &mut RenderGraphBuilder, width: u32, height: u32) -> Self {
        let base_color_desc = TextureDesc::new_color(
            "scene.gbuffer.base_color",
            width,
            height,
            vk::Format::R8G8B8A8_UNORM,
        );
        let normal_mra_desc = TextureDesc::new_color(
            "scene.gbuffer.normal_mra",
            width,
            height,
            vk::Format::R8G8B8A8_UNORM,
        );
        let depth_desc = TextureDesc::new_depth("scene.depth", width, height)
            .with_additional_usage(vk::ImageUsageFlags::SAMPLED);

        let base_color = builder.create(base_color_desc);
        let normal_mra = builder.create(normal_mra_desc);
        let depth = builder.create(depth_desc);

        Self {
            base_color,
            normal_mra,
            depth,
        }
    }
}
