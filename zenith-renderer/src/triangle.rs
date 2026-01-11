use std::sync::Arc;
use std::time::Instant;
use bytemuck::{Pod, Zeroable};
use zenith_rhi::{vk, RenderDevice, Buffer, BufferDesc, Shader, TextureState, BufferState, Texture, ImmediateCommandEncoder, UploadPool};
use zenith_rendergraph::{
    ColorAttachmentDescBuilder, RenderGraphBuilder, RenderGraphResource, VertexLayout,
    GraphicShaderInputBuilder, GraphicPipelineStateBuilder,
};
use zenith_rhi::pipeline::RasterizationStateBuilder;
use zenith_rhi::shader::ShaderModel;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, VertexLayout)]
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
            let total_size = vertex_data.len() + index_data.len();
            let mut upload_pool = UploadPool::new(device, total_size as _)?;
            upload_pool.enqueue_copy(vertex_buffer.as_range(..)?, vertex_data, BufferState::Vertex)?;
            upload_pool.enqueue_copy(index_buffer.as_range(..)?, index_data, BufferState::Index)?;

            let immediate = ImmediateCommandEncoder::new(device, device.graphics_queue())?;
            upload_pool.flush(&immediate, device)?;
        }

        // Load HLSL shaders from files
        let vertex_shader = Shader::from_hlsl_file(
            &device,
            "content/shaders/triangle.vs.hlsl",
            "main",
            zenith_rhi::ShaderStage::Vertex,
            ShaderModel::SM6,
        )?;

        let fragment_shader = Shader::from_hlsl_file(
            &device,
            "content/shaders/triangle.ps.hlsl",
            "main",
            zenith_rhi::ShaderStage::Fragment,
            ShaderModel::SM6,
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
    pub fn render_to(
        &self,
        builder: &mut RenderGraphBuilder,
        output: &mut RenderGraphResource<Texture>,
        width: u32,
        height: u32,
    ) {
        let vb = builder.import(
            self.vertex_buffer.clone(),
            BufferState::Undefined,
        );
        let ib = builder.import(
            self.index_buffer.clone(),
            BufferState::Undefined,
        );
        let tb = builder.create(
            BufferDesc::uniform("triangle.time", size_of::<f32>() as _),
        );

        let mut node = builder.add_graphic_node("triangle");

        let vb = node.read(&vb, BufferState::Vertex);
        let ib = node.read(&ib, BufferState::Index);
        let tb = node.read(&tb, BufferState::Uniform);
        let output_rt = node.write(output, TextureState::Color);

        let shader = GraphicShaderInputBuilder::default()
            .vertex_shader(self.vertex_shader.clone())
            .fragment_shader(self.fragment_shader.clone())
            .vertex_layout::<Vertex>()
            .build().unwrap();

        let color_info = ColorAttachmentDescBuilder::default()
            .clear_input()
            .clear_value([0.1, 0.1, 0.1, 1.0])
            .build().unwrap();

        let state = GraphicPipelineStateBuilder::default()
            .rasterization(RasterizationStateBuilder::default().cull_mode(vk::CullModeFlags::NONE).build().unwrap())
            .build();

        {
            let mut binder = node.pipeline(shader, state);
            binder.push_color(output_rt, color_info);
            binder.finish();
        }

        let elapsed = self.start_time.elapsed().as_secs_f32();

        node.execute(move |ctx| {
            let extent = vk::Extent2D { width, height };
            let encoder = ctx.encoder();

            // Update time buffer
            let elapsed_bytes = bytemuck::bytes_of(&elapsed);
            let time_buffer = ctx.get(&tb)
                .as_range(0..(elapsed_bytes.len() as u64))
                .map_err(|e| anyhow::anyhow!("failed to create time buffer range: {:?}", e))?;

            time_buffer.write(elapsed_bytes)
                .map_err(|e| anyhow::anyhow!("failed to write time buffer: {:?}", e))?;

            // Bind uniform buffer using shader resource binder
            let mut binder = ctx.create_binder();
            match binder.bind_buffer("TimeData", time_buffer) {
                Ok(_) => {
                    ctx.bind_descriptor_sets(binder);
                }
                Err(e) => {
                    log::warn!("Failed to bind time buffer: {:?}", e);
                }
            }

            ctx.begin_rendering(extent);
            ctx.bind_pipeline();

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
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

            encoder.draw_indexed(3, 1, 0, 0, 0);

            ctx.end_rendering();

            Ok(())
        });
    }
}
