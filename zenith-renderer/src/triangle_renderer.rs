use std::sync::Arc;
use std::time::Instant;
use bytemuck::{Pod, Zeroable};
use zenith_rhi::{vk, RenderDevice, Buffer, BufferDesc, Shader, TextureState, BufferState, Texture, ImmediateCommandEncoder, UploadPool};
use zenith_rendergraph::{RenderGraphBuilder, RenderGraphResource, ColorAttachmentDescBuilder};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
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
    pub fn new(device: &RenderDevice) -> Self {
        let vertices = [
            Vertex { position: [0.0, 0.5, 0.0], color: [1.0, 0.0, 0.0] },
            Vertex { position: [-0.5, -0.5, 0.0], color: [0.0, 1.0, 0.0] },
            Vertex { position: [0.5, -0.5, 0.0], color: [0.0, 0.0, 1.0] },
        ];
        let indices: [u16; 3] = [0, 1, 2];

        let vertex_data = bytemuck::cast_slice(&vertices);
        let index_data = bytemuck::cast_slice(&indices);

        // Device-local GPU buffers (uploaded via staging)
        let vertex_buffer = Arc::new(
            Buffer::new(device, &BufferDesc::vertex(vertex_data.len() as u64))
                .expect("Failed to create vertex buffer")
        );
        let index_buffer = Arc::new(
            Buffer::new(device, &BufferDesc::index(index_data.len() as u64))
                .expect("Failed to create index buffer")
        );

        // Upload via staging buffer (blocking for now)
        let immediate = ImmediateCommandEncoder::new(device.handle(), device.graphics_queue())
            .expect("Failed to create ImmediateCommandEncoder");
        let mut upload_pool = UploadPool::new(device, 1024 * 1024).expect("Failed to create UploadPool");
        upload_pool.enqueue_copy(vertex_buffer.clone(), 0, vertex_data, BufferState::Vertex)
            .expect("Failed to enqueue vertex upload");
        upload_pool.enqueue_copy(index_buffer.clone(), 0, index_data, BufferState::Index)
            .expect("Failed to enqueue index upload");
        upload_pool.flush(&immediate, device).expect("Failed to flush uploads");

        // Load HLSL shaders from files
        let vertex_shader = match Shader::from_hlsl_file(
            device.handle(),
            "content/shaders/triangle.vs.hlsl",
            "main",
            zenith_rhi::ShaderStage::Vertex,
            "6_0",
        ) {
            Ok(shader) => shader,
            Err(e) => {
                log::warn!("HLSL compilation failed: {:?}. DXC may not be installed.", e);
                log::warn!("Please ensure dxcompiler.dll and dxil.dll are in your PATH or install the Vulkan SDK.");
                panic!("Shader compilation failed. See warnings above for details.");
            }
        };

        let fragment_shader = match Shader::from_hlsl_file(
            device.handle(),
            "content/shaders/triangle.ps.hlsl",
            "main",
            zenith_rhi::ShaderStage::Fragment,
            "6_0",
        ) {
            Ok(shader) => shader,
            Err(e) => {
                log::warn!("HLSL compilation failed: {:?}. DXC may not be installed.", e);
                panic!("Shader compilation failed. See warnings above for details.");
            }
        };

        Self {
            vertex_buffer,
            index_buffer,
            vertex_shader: Arc::new(vertex_shader),
            fragment_shader: Arc::new(fragment_shader),
            start_time: Instant::now(),
        }
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
            "triangle.vertex",
            self.vertex_buffer.clone(),
            BufferState::Undefined,
        );
        let ib = builder.import(
            "triangle.index",
            self.index_buffer.clone(),
            BufferState::Undefined,
        );
        let tb = builder.create(
            "triangle.time",
            BufferDesc {
                size: size_of::<f32>() as u64,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                memory_flags: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
            }
        );

        let mut node = builder.add_graphic_node("triangle");

        let vb = node.read(&vb, BufferState::Vertex);
        let ib = node.read(&ib, BufferState::Index);
        let tb = node.read(&tb, BufferState::Uniform);
        let output_rt = node.write(output, TextureState::Color);

        let color_info = ColorAttachmentDescBuilder::default()
            .clear_input()
            .clear_value([0.1, 0.1, 0.1, 1.0])
            .build().unwrap();

        node.setup_pipeline()
            .with_vertex_shader(self.vertex_shader.clone())
            .with_fragment_shader(self.fragment_shader.clone())
            .with_color(output_rt, color_info)
            .with_vertex_binding(0, size_of::<Vertex>() as u32, vk::VertexInputRate::VERTEX)
            .with_vertex_attribute(0, 0, vk::Format::R32G32B32_SFLOAT, 0)
            .with_vertex_attribute(1, 0, vk::Format::R32G32B32_SFLOAT, 12);

        let elapsed = self.start_time.elapsed().as_secs_f32();

        node.execute(move |ctx| {
            let extent = vk::Extent2D { width, height };
            let encoder = ctx.encoder();

            // Update time buffer
            let time_buffer = ctx.get(&tb);
            time_buffer.map_and_write(bytemuck::bytes_of(&elapsed));

            // Bind uniform buffer using shader resource binder
            if let Some(mut binder) = ctx.create_resource_binder() {
                match binder.bind_buffer("TimeData", time_buffer.handle(), 0, size_of_val(&elapsed) as _) {
                    Ok(_) => {
                        let sets = binder.finish();
                        ctx.bind_descriptor_sets(&sets);
                    }
                    Err(e) => {
                        log::warn!("Failed to bind time buffer: {:?}", e);
                    }
                }
            } else {
                log::warn!("No resource binder available");
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
        });
    }
}
