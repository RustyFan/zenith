use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use enumflags2::BitFlags;
use zenith_rendergraph::{ColorAttachment, RenderGraphBuilder, RenderGraphResource, VertexLayout};
use zenith_rhi::{vk, Buffer, BufferState, GraphicPipelineStateBuilder, GraphicShaderInputBuilder, RenderDevice, Shader, Texture, TextureState, GraphicPipelineDesc, ShaderStage, BindlessPool, TextureDesc, TextureLayout, UploadPool, ImmediateCommandEncoder};
use zenith_rhi::pipeline::{GraphicPipelineAttachmentsBuilder, RasterizationStateBuilder};
use crate::{DEFAULT_RENDER_RESOURCES};
use crate::defer_shading::SceneTextures;
use crate::world::DebugMode;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, VertexLayout)]
struct ScreenVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

pub struct DirectLightingRenderer {
    width: u32,
    height: u32,
    lighting_fragment_shader: Arc<Shader>,
    default_cubemap: Arc<Texture>,
}

impl DirectLightingRenderer {
    pub fn new(device: &Arc<RenderDevice>, width: u32, height: u32) -> anyhow::Result<Self> {
        let lighting_fragment_shader = Arc::new(Shader::from_file(
            "shader.lighting.ps",
            device,
            "content/shaders/lighting.slang",
            ShaderStage::Fragment,
        )?);

        // Create a default 1x1 black cubemap for when no skybox is set
        let default_cubemap = {
            let cubemap_tex = Texture::new(
                device,
                &TextureDesc::new_cube(
                    "default_cubemap",
                    1,
                    vk::Format::R8G8B8A8_UNORM,
                )
                .with_usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            )?;

            // Upload black pixels for all 6 faces (1x1x6 = 6 pixels, 4 bytes each = 24 bytes)
            let black_pixels = vec![0u8; 24];  // 6 faces * 1 pixel * 4 channels
            {
                let mut upload_pool = UploadPool::new()?;
                upload_pool.enqueue_texture_upload(
                    device,
                    cubemap_tex.as_range(TextureLayout::Undefined, .., ..),
                    &black_pixels,
                    TextureState::Sampled,
                )?;

                let immediate = ImmediateCommandEncoder::new(device, device.graphics_queue())?;
                upload_pool.flush(&immediate, device)?;
            }

            Arc::new(cubemap_tex)
        };

        Ok(Self {
            width,
            height,
            lighting_fragment_shader,
            default_cubemap,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        if width == self.width && height == self.height {
            return;
        }

        self.width = width;
        self.height = height;
    }

    pub fn render(
        &self,
        builder: &mut RenderGraphBuilder,
        bindless_pool: &mut BindlessPool,
        scene_texture: SceneTextures,
        cubemap: Option<Arc<Texture>>,
        view: RenderGraphResource<Buffer>,
        ibl_sh_diffuse: RenderGraphResource<Buffer>,
        debug_mode: BitFlags<DebugMode>,
        output: &mut RenderGraphResource<Texture>,
    ) {
        let default_res = DEFAULT_RENDER_RESOURCES.get().unwrap().lock();
        let default_res = default_res.as_ref().unwrap();

        let pipeline_desc = GraphicPipelineDesc::new(
            "lighting",
            GraphicShaderInputBuilder::default()
                .vertex_shader(default_res.screen_vertex_shader.clone())
                .fragment_shader(self.lighting_fragment_shader.clone())
                .vertex_layout::<ScreenVertex>()
                .build().unwrap(),
            GraphicPipelineStateBuilder::default()
                .rasterization(RasterizationStateBuilder::default().no_culling().build().unwrap())
                .build().unwrap(),
            GraphicPipelineAttachmentsBuilder::default()
                .color_no_blending(output.desc(builder).format)
                .build().unwrap()
        );

        let vb = builder.import(default_res.screen_vb.clone(), BufferState::Vertex);
        let ib = builder.import(default_res.screen_ib.clone(), BufferState::Index);

        // Import skybox texture (or default cubemap)
        let skybox_to_bind = cubemap.as_ref().unwrap_or(&self.default_cubemap);
        let skybox_res = builder.import(skybox_to_bind.clone(), TextureState::Sampled);

        let mut node = builder.add_graphic_node("lighting");

        let vb = node.read(&vb, BufferState::Vertex);
        let ib = node.read(&ib, BufferState::Index);
        let gbuffer_base = node.read(&scene_texture.base_color, TextureState::Sampled);
        let gbuffer_nmr = node.read(&scene_texture.normal_mra, TextureState::Sampled);
        let scene_depth = node.read(&scene_texture.depth, TextureState::Sampled);
        let view = node.read(&view, BufferState::Uniform);
        let skybox_handle = node.read(&skybox_res, TextureState::Sampled);
        let ibl_sh_diffuse = node.read(&ibl_sh_diffuse, BufferState::StorageRead);

        let output_rt = node.write(output, TextureState::Color);

        let width = self.width;
        let height = self.height;
        let bindless_set = bindless_pool.set();
        let pipeline_handle = node.register_pipeline(pipeline_desc);
        let debug_mode = debug_mode.bits() as u32;
        node.execute(move |ctx| {
            if let Some(pipeline) = ctx.bind_pipeline(pipeline_handle) {
                pipeline.bind_raw(BindlessPool::SET_INDEX, &[bindless_set], &[])
                    .bind("view", view)?
                    .bind("sh_diffuse", ibl_sh_diffuse)?
                    .bind("base_color_tex", gbuffer_base)?
                    .bind("normal_mra_tex", gbuffer_nmr)?
                    .bind("depth_tex", scene_depth)?
                    .bind("skybox_cubemap", skybox_handle)?;

                ctx.begin_rendering(
                    (width, height),
                    &[ColorAttachment::new(output_rt).clear_input().clear_value([0.02, 0.02, 0.02, 1.0])],
                    None
                );

                // TODO: make it a PipelineResourceBinder's api
                ctx.push_constants(pipeline_handle, 0, &debug_mode);

                ctx.bind_vertex_buffers(vb, 0, &[0]);
                ctx.bind_index_buffer(ib, 0, vk::IndexType::UINT16);
                ctx.encoder().draw_indexed(0..6, 0..1, 0);

                ctx.end_rendering();
            }

            Ok(())
        });
    }
}