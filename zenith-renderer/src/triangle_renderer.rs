use std::sync::Arc;
use std::time::Instant;
use bytemuck::{Pod, Zeroable};
use zenith_rhi::{vk, RenderDevice, Buffer, BufferDesc, Shader, TextureState, BufferState, Texture};
use zenith_rendergraph::{
    RenderGraphBuilder, RenderGraphResource,
    ColorInfo,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

pub struct TriangleRenderer {
    vertex_buffer: Arc<Buffer>,
    index_buffer: Arc<Buffer>,
    time_buffer: Arc<Buffer>,
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

        let vertex_buffer = Buffer::from_desc(
            device.handle(),
            device.memory_properties(),
            &BufferDesc {
                size: vertex_data.len() as u64,
                usage: vk::BufferUsageFlags::VERTEX_BUFFER,
                memory_flags: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
            },
        ).expect("Failed to create vertex buffer");

        // Copy vertex data
        vertex_buffer.map_and_write(vertex_data);

        let index_buffer = Buffer::from_desc(
            device.handle(),
            device.memory_properties(),
            &BufferDesc {
                size: index_data.len() as u64,
                usage: vk::BufferUsageFlags::INDEX_BUFFER,
                memory_flags: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
            },
        ).expect("Failed to create index buffer");

        // Copy index data
        index_buffer.map_and_write(index_data);

        // Create uniform buffer for time
        let time_buffer = Buffer::from_desc(
            device.handle(),
            device.memory_properties(),
            &BufferDesc {
                size: std::mem::size_of::<f32>() as u64,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                memory_flags: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
            },
        ).expect("Failed to create time buffer");

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
            vertex_buffer: Arc::new(vertex_buffer),
            index_buffer: Arc::new(index_buffer),
            time_buffer: Arc::new(time_buffer),
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
        let tb = builder.import(
            "triangle.time",
            self.time_buffer.clone(),
            BufferState::Undefined,
        );

        let mut node = builder.add_graphic_node("triangle");

        let vb = node.read(&vb, BufferState::Vertex);
        let ib = node.read(&ib, BufferState::Index);
        let tb = node.read(&tb, BufferState::Uniform);
        let output_rt = node.write(output, TextureState::Color);

        let color_info = ColorInfo {
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: [0.1, 0.1, 0.1, 1.0],
            blend_enable: false,
            src_color_blend: vk::BlendFactor::ONE,
            dst_color_blend: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend: vk::BlendFactor::ONE,
            dst_alpha_blend: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            write_mask: vk::ColorComponentFlags::RGBA,
        };

        node.setup_pipeline()
            .with_vertex_shader(self.vertex_shader.clone())
            .with_fragment_shader(self.fragment_shader.clone())
            .with_color(output_rt, color_info)
            .with_vertex_binding(0, std::mem::size_of::<Vertex>() as u32, vk::VertexInputRate::VERTEX)
            .with_vertex_attribute(0, 0, vk::Format::R32G32B32_SFLOAT, 0)
            .with_vertex_attribute(1, 0, vk::Format::R32G32B32_SFLOAT, 12);

        let elapsed = self.start_time.elapsed().as_secs_f32();

        node.execute(move |ctx, cmd| {
            let device = ctx.device();
            let extent = vk::Extent2D { width, height };

            // Update time buffer
            let time_buffer = ctx.get_buffer(&tb);
            time_buffer.map_and_write(bytemuck::bytes_of(&elapsed));

            // Bind uniform buffer using shader resource binder
            if let Some(mut binder) = ctx.create_resource_binder() {
                match binder.bind_buffer("TimeData", time_buffer.handle(), 0, 4) {
                    Ok(_) => {
                        let sets = binder.finish();
                        ctx.bind_descriptor_sets(cmd, &sets);
                    }
                    Err(e) => {
                        log::warn!("Failed to bind time buffer: {:?}", e);
                    }
                }
            } else {
                log::warn!("No resource binder available");
            }

            ctx.begin_rendering(cmd, extent);
            ctx.bind_pipeline(cmd);

            unsafe {
                let viewport = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: width as f32,
                    height: height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                device.handle().cmd_set_viewport(cmd, 0, &[viewport]);

                let scissor = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                };
                device.handle().cmd_set_scissor(cmd, 0, &[scissor]);

                device.handle().cmd_bind_vertex_buffers(cmd, 0, &[ctx.get_buffer(&vb).handle()], &[0]);
                device.handle().cmd_bind_index_buffer(cmd, ctx.get_buffer(&ib).handle(), 0, vk::IndexType::UINT16);

                device.handle().cmd_draw_indexed(cmd, 3, 1, 0, 0, 0);
            }

            ctx.end_rendering(cmd);
        });
    }
}
