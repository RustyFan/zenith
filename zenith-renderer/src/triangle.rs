use std::sync::Arc;
use std::time::Instant;
use anyhow::anyhow;
use bytemuck::{Pod, Zeroable};
use zenith_rhi::{vk, RenderDevice, Buffer, BufferDesc, Shader, TextureState, BufferState, Texture, ImmediateCommandEncoder, UploadPool, GraphicPipelineDesc, GraphicPipelineAttachments};
use zenith_rendergraph::{RenderGraphBuilder, RenderGraphResource, VertexLayout, GraphicShaderInputBuilder, GraphicPipelineStateBuilder, ColorAttachment};
use zenith_rhi::pipeline::{RasterizationStateBuilder, BlendStateBuilder};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, VertexLayout)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

pub struct TriangleRenderer {
    vertex_buffer: Arc<Buffer>,
    index_buffer: Arc<Buffer>,
    vertex_shader: Arc<Shader>,
    fragment_shader: Arc<Shader>,
    start_time: Instant,
}

impl TriangleRenderer {
    pub fn new(device: &RenderDevice) -> anyhow::Result<Self> {
        let vertices = [
            Vertex { position: [0.0, 0.5, 0.0], color: [1.0, 0.0, 0.0] },
            Vertex { position: [-0.5, -0.5, 0.0], color: [0.0, 1.0, 0.0] },
            Vertex { position: [0.5, -0.5, 0.0], color: [0.0, 0.0, 1.0] },
        ];
        let indices: [u16; 3] = [0, 1, 2];

        let vertex_data = bytemuck::cast_slice(&vertices);
        let index_data = bytemuck::cast_slice(&indices);

        let vertex_buffer = Arc::new(Buffer::new(device, &BufferDesc::vertex("triangle.vertex", vertex_data.len() as u64))?);
        let index_buffer = Arc::new(Buffer::new(device, &BufferDesc::index("triangle.index", index_data.len() as u64))?);

        {
            let mut upload_pool = UploadPool::new()?;
            upload_pool.enqueue_copy_buffer(device, vertex_buffer.as_range(..), vertex_data, BufferState::Vertex)?;
            upload_pool.enqueue_copy_buffer(device, index_buffer.as_range(..), index_data, BufferState::Index)?;

            let immediate = ImmediateCommandEncoder::new(device, device.graphics_queue())?;
            upload_pool.flush(&immediate, device)?;
        }

        let vertex_shader = Shader::from_file(
            "shader.triangle.vs",
            &device,
            "content/shaders/triangle.slang",
            zenith_rhi::ShaderStage::Vertex,
        )?;

        let fragment_shader = Shader::from_file(
            "shader.triangle.ps",
            &device,
            "content/shaders/triangle.slang",
            zenith_rhi::ShaderStage::Fragment,
        )?;

        Ok(Self {
            vertex_buffer,
            index_buffer,
            vertex_shader: Arc::new(vertex_shader),
            fragment_shader: Arc::new(fragment_shader),
            start_time: Instant::now(),
        })
    }

    /// Render the triangle directly to the provided output texture.
    pub fn render(
        &self,
        builder: &mut RenderGraphBuilder,
        output: &mut RenderGraphResource<Texture>,
        width: u32,
        height: u32,
    ) {
        // Create pipeline descriptor
        let shader = GraphicShaderInputBuilder::default()
            .vertex_shader(self.vertex_shader.clone())
            .fragment_shader(self.fragment_shader.clone())
            .vertex_layout::<Vertex>()
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

        let pipeline_desc = GraphicPipelineDesc::new("triangle", shader, state, attachments);

        let vb = builder.import(self.vertex_buffer.clone(), BufferState::Vertex);
        let ib = builder.import(self.index_buffer.clone(), BufferState::Index);
        let tb = builder.create(BufferDesc::uniform("triangle.time", size_of::<f32>() as _));

        let mut node = builder.add_graphic_node("triangle");

        let pipeline_handle = node.register_pipeline(pipeline_desc.clone());

        let vb = node.read(&vb, BufferState::Vertex);
        let ib = node.read(&ib, BufferState::Index);
        let tb = node.read(&tb, BufferState::Uniform);
        let output_rt = node.write(output, TextureState::Color);

        let elapsed = self.start_time.elapsed().as_secs_f32();
        node.execute(move |ctx| {
            ctx.get(&tb)
                .as_range(..)
                .write(bytemuck::bytes_of(&elapsed))
                .map_err(|e| anyhow!("Failed to write time buffer: {e:?}"))?;

            ctx.bind_pipeline(pipeline_handle)
                .bind("time", tb)?;

            ctx.begin_rendering(
                (width, height),
                &[ColorAttachment::new(output_rt).clear_input().clear_value([0.1, 0.1, 0.1, 1.0])],
                None
            );

            ctx.bind_vertex_buffers(vb, 0, &[0]);
            ctx.bind_index_buffer(ib, 0, vk::IndexType::UINT16);
            ctx.draw_indexed(0..3, 0..1, 0);

            ctx.end_rendering();
            Ok(())
        });
    }
}
