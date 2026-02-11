use std::sync::Arc;
use anyhow::anyhow;
use bytemuck::{Pod, Zeroable};
use zenith_rendergraph::{RenderGraphBuilder, RenderGraphResource, VertexLayout};
use zenith_rhi::{vk, Buffer, BufferDesc, BufferState, ColorAttachmentDescBuilder, GraphicPipelineStateBuilder, GraphicShaderInputBuilder, ImmediateCommandEncoder, RenderDevice, Shader, Texture, TextureLayout, TextureState, UploadPool};
use zenith_rhi::pipeline::RasterizationStateBuilder;
use crate::defer_shading::GBuffer;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, VertexLayout)]
struct ScreenVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PushConstants {
    gbuffer_base: u32,
    gbuffer_nmr: u32,
    scene_depth: u32,
}

pub struct DirectLightingRenderer {
    width: u32,
    height: u32,
    lighting_vertex_shader: Arc<Shader>,
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
            upload_pool.enqueue_copy_buffer(device, lighting_vertex_buffer.as_range(..)?, vertex_data, BufferState::Vertex)?;
            upload_pool.enqueue_copy_buffer(device, lighting_index_buffer.as_range(..)?, index_data, BufferState::Index)?;

            let immediate = ImmediateCommandEncoder::new(device, device.graphics_queue())?;
            upload_pool.flush(&immediate, device)?;
        }

        Ok(Self {
            width,
            height,
            lighting_vertex_shader: Arc::new(lighting_vertex_shader),
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
        gbuffer: GBuffer,
        scene_depth: RenderGraphResource<Texture>,
        view: &RenderGraphResource<Buffer>,
        output: &mut RenderGraphResource<Texture>,
    ) {
        let vb = builder.import(self.lighting_vertex_buffer.clone(), BufferState::Vertex);
        let ib = builder.import(self.lighting_index_buffer.clone(), BufferState::Index);

        let mut node = builder.add_graphic_node("lighting");
        let vb = node.read(&vb, BufferState::Vertex);
        let ib = node.read(&ib, BufferState::Index);
        let gbuffer_base = node.read(&gbuffer.base_color, TextureState::Sampled);
        let gbuffer_nmr = node.read(&gbuffer.normal_mra, TextureState::Sampled);
        let scene_depth = node.read(&scene_depth, TextureState::Sampled);
        let view = node.read(view, BufferState::StorageRead);
        let output_rt = node.write(output, TextureState::Color);

        let shader = GraphicShaderInputBuilder::default()
            .vertex_shader(self.lighting_vertex_shader.clone())
            .fragment_shader(self.lighting_fragment_shader.clone())
            .vertex_layout::<ScreenVertex>()
            .build()
            .unwrap();

        let color_info = ColorAttachmentDescBuilder::default()
            .clear_input()
            .clear_value([0.0, 0.0, 0.0, 1.0])
            .build()
            .unwrap();

        let state = GraphicPipelineStateBuilder::default()
            .rasterization(RasterizationStateBuilder::default().cull_mode(vk::CullModeFlags::NONE).build().unwrap())
            .build();

        {
            let mut binder = node.pipeline(shader, state);
            binder.push_color(output_rt, color_info);
            binder.finish();
        }

        let width = self.width;
        let height = self.height;
        node.execute(move |ctx| {
            let extent = vk::Extent2D { width, height };
            let encoder = ctx.encoder();

            // Bind all bindless resources via direct pool access (set0 bound at command buffer start).
            let base_handle;
            let nmr_handle;
            let depth_handle;
            {
                let mut pool = ctx.device().bindless_pool().lock();
                let base_range = ctx.get(&gbuffer_base)
                    .as_range(TextureLayout::ShaderReadOnly, .., ..)
                    .map_err(|e| anyhow!("failed to create gbuffer base range: {e:?}"))?;
                let nmr_range = ctx.get(&gbuffer_nmr)
                    .as_range(TextureLayout::ShaderReadOnly, .., ..)
                    .map_err(|e| anyhow!("failed to create gbuffer nmr range: {e:?}"))?;
                let depth_range = ctx.get(&scene_depth)
                    .as_range(TextureLayout::ShaderReadOnly, .., ..)
                    .map_err(|e| anyhow!("failed to create scene depth range: {e:?}"))?;

                base_handle = pool.bind(base_range)?;
                nmr_handle = pool.bind(nmr_range)?;
                depth_handle = pool.bind(depth_range)?;

                pool.flush(ctx.device());
                ctx.bind_descriptor_sets(0, std::slice::from_ref(&pool.set()));
            }

            // Bind the view buffer (already written by render_meshes).
            let view_range = ctx.get(&view)
                .as_range(..)
                .map_err(|e| anyhow!("failed to create lighting view buffer range: {e:?}"))?;
            let mut binder = ctx.create_binder();
            binder.bind("view", view_range)?;
            let sets = binder.finish(ctx.device(), 1).map_err(|e| anyhow::anyhow!("descriptor set finish failed: {e}"))?;
            ctx.bind_descriptor_sets(1, &sets);

            ctx.begin_rendering(extent);
            ctx.bind_pipeline();

            let pc = PushConstants {
                gbuffer_base: *base_handle,
                gbuffer_nmr: *nmr_handle,
                scene_depth: *depth_handle,
            };
            ctx.push_constants(0, &pc);

            let viewport = vk::Viewport {
                x: 0.0,
                // Flip Y in rasterization (Vulkan supports negative viewport height).
                // This avoids baking Y-flip into the projection matrix.
                y: height as f32,
                width: width as f32,
                height: -(height as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            };
            encoder.set_viewport(0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            };
            encoder.set_scissor(0, &[scissor]);

            encoder.bind_vertex_buffers(0, &[ctx.get(&vb).handle()], &[0]);
            encoder.bind_index_buffer(ctx.get(&ib).handle(), 0, vk::IndexType::UINT16);
            encoder.draw_indexed(6, 1, 0, 0, 0);

            ctx.end_rendering();
            Ok(())
        });
    }
}