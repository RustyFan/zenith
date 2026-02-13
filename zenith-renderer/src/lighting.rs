use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use zenith_rendergraph::{ColorAttachment, RenderGraphBuilder, RenderGraphResource, VertexLayout};
use zenith_rhi::{vk, Buffer, BufferState, GraphicPipelineStateBuilder, GraphicShaderInputBuilder, RenderDevice, Shader, Texture, TextureState, GraphicPipelineDesc, ShaderStage, BindlessPool};
use zenith_rhi::pipeline::{GraphicPipelineAttachmentsBuilder, RasterizationStateBuilder};
use crate::{DEFAULT_RENDER_RESOURCES};
use crate::defer_shading::SceneTextures;

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
}

impl DirectLightingRenderer {
    pub fn new(device: &RenderDevice, width: u32, height: u32) -> anyhow::Result<Self> {
        let lighting_fragment_shader = Arc::new(Shader::from_file(
            "shader.lighting.ps",
            device,
            "content/shaders/lighting.slang",
            ShaderStage::Fragment,
        )?);

        Ok(Self {
            width,
            height,
            lighting_fragment_shader,
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
        scene_texture: SceneTextures,
        view: &RenderGraphResource<Buffer>,
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
                .rasterization(RasterizationStateBuilder::default().cull_mode(vk::CullModeFlags::NONE).build().unwrap())
                .build().unwrap(),
            GraphicPipelineAttachmentsBuilder::default()
                .color_no_blending(output.desc(builder).format)
                .build().unwrap()
        );

        let vb = builder.import(default_res.screen_vb.clone(), BufferState::Vertex);
        let ib = builder.import(default_res.screen_ib.clone(), BufferState::Index);

        let mut node = builder.add_graphic_node("lighting");

        let pipeline_handle = node.register_pipeline(pipeline_desc.clone());

        let vb = node.read(&vb, BufferState::Vertex);
        let ib = node.read(&ib, BufferState::Index);
        let gbuffer_base = node.read(&scene_texture.base_color, TextureState::Sampled);
        let gbuffer_nmr = node.read(&scene_texture.normal_mra, TextureState::Sampled);
        let scene_depth = node.read(&scene_texture.depth, TextureState::Sampled);
        let view = node.read(view, BufferState::Uniform);
        let output_rt = node.write(output, TextureState::Color);

        let width = self.width;
        let height = self.height;
        node.execute(move |ctx| {
            ctx.bind_pipeline(pipeline_handle)
                .bind_raw(BindlessPool::SET_INDEX, &[ctx.device().bindless_pool().lock().set()], &[])
                .bind("view", view)?
                .bind("base_color_tex", gbuffer_base)?
                .bind("normal_mra_tex", gbuffer_nmr)?
                .bind("depth_tex", scene_depth)?;

            ctx.begin_rendering(
                (width, height),
                &[ColorAttachment::new(output_rt).clear_input().clear_value([0.02, 0.02, 0.02, 1.0])],
                None
            );

            ctx.bind_vertex_buffers(vb, 0, &[0]);
            ctx.bind_index_buffer(ib, 0, vk::IndexType::UINT16);
            ctx.encoder().draw_indexed(0..6, 0..1, 0);

            ctx.end_rendering();
            Ok(())
        });
    }
}