use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use zenith_rendergraph::{RenderGraphBuilder, RenderGraphResource, VertexLayout};
use zenith_rhi::{vk, Buffer, BufferDesc, BufferState, ColorAttachmentDescBuilder, GraphicPipelineStateBuilder, GraphicShaderInputBuilder, ImmediateCommandEncoder, RenderDevice, Shader, Texture, TextureLayout, TextureState, UploadPool, GraphicPipelineDesc, GraphicPipelineAttachments};
use zenith_rhi::pipeline::{BlendStateBuilder, RasterizationStateBuilder};
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
    screen_vertex_shader: Arc<Shader>,
    lighting_fragment_shader: Arc<Shader>,
    lighting_vertex_buffer: Arc<Buffer>,
    lighting_index_buffer: Arc<Buffer>,
}

impl DirectLightingRenderer {
    pub fn new(device: &RenderDevice, width: u32, height: u32) -> anyhow::Result<Self> {
        let lighting_vertex_shader = Shader::from_file(
            "shader.lighting.vs",
            device,
            "content/shaders/lighting.slang",
            zenith_rhi::ShaderStage::Vertex,
        )?;

        let lighting_fragment_shader = Shader::from_file(
            "shader.lighting.ps",
            device,
            "content/shaders/lighting.slang",
            zenith_rhi::ShaderStage::Fragment,
        )?;

        let lighting_vertices = [
            ScreenVertex { position: [-1.0,  1.0], uv: [0.0, 0.0] },
            ScreenVertex { position: [ 1.0,  1.0], uv: [1.0, 0.0] },
            ScreenVertex { position: [-1.0, -1.0], uv: [0.0, 1.0] },
            ScreenVertex { position: [ 1.0, -1.0], uv: [1.0, 1.0] },
        ];
        let lighting_indices: [u16; 6] = [0, 1, 2, 1, 2, 3];

        let lighting_vertex_buffer = Arc::new(Buffer::new(
            device,
            &BufferDesc::vertex("screen.vertex", (lighting_vertices.len() * size_of::<ScreenVertex>()) as u64),
        )?);
        let lighting_index_buffer = Arc::new(Buffer::new(
            device,
            &BufferDesc::index("screen.index", (lighting_indices.len() * size_of::<u16>()) as u64),
        )?);

        {
            let vertex_data = bytemuck::cast_slice(&lighting_vertices);
            let index_data = bytemuck::cast_slice(&lighting_indices);
            let mut upload_pool = UploadPool::new()?;
            upload_pool.enqueue_copy_buffer(device, lighting_vertex_buffer.as_range(..), vertex_data, BufferState::Vertex)?;
            upload_pool.enqueue_copy_buffer(device, lighting_index_buffer.as_range(..), index_data, BufferState::Index)?;

            let immediate = ImmediateCommandEncoder::new(device, device.graphics_queue())?;
            upload_pool.flush(&immediate, device)?;
        }

        Ok(Self {
            width,
            height,
            screen_vertex_shader: Arc::new(lighting_vertex_shader),
            lighting_fragment_shader: Arc::new(lighting_fragment_shader),
            lighting_vertex_buffer,
            lighting_index_buffer,
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
        // Create pipeline descriptor
        let shader = GraphicShaderInputBuilder::default()
            .vertex_shader(self.screen_vertex_shader.clone())
            .fragment_shader(self.lighting_fragment_shader.clone())
            .vertex_layout::<ScreenVertex>()
            .build()
            .unwrap();

        let state = GraphicPipelineStateBuilder::default()
            .rasterization(RasterizationStateBuilder::default().cull_mode(vk::CullModeFlags::NONE).build().unwrap())
            .push_blend_state(BlendStateBuilder::default().build().unwrap())
            .build();

        let attachments = GraphicPipelineAttachments {
            color_formats: vec![output.desc(builder).format],
            depth_format: None,
            stencil_format: None,
        };

        let pipeline_desc = GraphicPipelineDesc::new("lighting", shader, state, attachments);

        let vb = builder.import(self.lighting_vertex_buffer.clone(), BufferState::Vertex);
        let ib = builder.import(self.lighting_index_buffer.clone(), BufferState::Index);

        let mut node = builder.add_graphic_node("lighting");

        let pipeline_handle = node.register_pipeline(pipeline_desc.clone());

        let vb = node.read(&vb, BufferState::Vertex);
        let ib = node.read(&ib, BufferState::Index);
        let gbuffer_base = node.read(&scene_texture.base_color, TextureState::Sampled);
        let gbuffer_nmr = node.read(&scene_texture.normal_mra, TextureState::Sampled);
        let scene_depth = node.read(&scene_texture.depth, TextureState::Sampled);
        let view = node.read(view, BufferState::StorageRead);
        let output_rt = node.write(output, TextureState::Color);

        let width = self.width;
        let height = self.height;
        node.execute(move |ctx| {
            let view_range = ctx.get(&view).as_range(..);
            let base_range = ctx.get(&gbuffer_base).as_range(TextureLayout::ShaderReadOnly, .., ..);
            let nmr_range = ctx.get(&gbuffer_nmr).as_range(TextureLayout::ShaderReadOnly, .., ..);
            let depth_range = ctx.get(&scene_depth).as_range(TextureLayout::ShaderReadOnly, .., ..);

            ctx.bind_pipeline(pipeline_handle)
                .bind_raw(0, &[ctx.device().bindless_pool().lock().set()], &[])
                .bind("view", view_range)?
                .bind("base_color_tex", base_range)?
                .bind("normal_mra_tex", nmr_range)?
                .bind("depth_tex", depth_range)?;

            ctx.begin_rendering(
                (width, height),
                &[(output_rt, ColorAttachmentDescBuilder::default()
                    .clear_input().clear_value([0.02, 0.02, 0.02, 1.0])
                    .build().unwrap())],
                None
            );

            let encoder = ctx.encoder();
            encoder.bind_vertex_buffers(0, &[ctx.get(&vb).handle()], &[0]);
            encoder.bind_index_buffer(ctx.get(&ib).handle(), 0, vk::IndexType::UINT16);
            encoder.draw_indexed(6, 1, 0, 0, 0);

            ctx.end_rendering();
            Ok(())
        });
    }
}