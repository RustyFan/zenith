use std::sync::{Arc};
use bytemuck::{Pod, Zeroable};
use zenith_rhi::{Buffer, BufferDesc, BufferState, ImmediateCommandEncoder, RenderDevice, Shader, ShaderStage, UploadPool, VertexLayout};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, VertexLayout)]
pub struct ScreenVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

pub struct DefaultRenderResources {
    /// Vertex buffer of screen quad
    pub screen_vb: Arc<Buffer>,
    /// Index buffer of screen quad
    pub screen_ib: Arc<Buffer>,
    /// Vertex shader to render a screen quad
    pub screen_vertex_shader: Arc<Shader>,
}

impl DefaultRenderResources {
    pub fn new(device: &RenderDevice) -> anyhow::Result<Self> {
        let screen_vertices = get_screen_vertices();
        let screen_indices = get_screen_indices();

        let screen_vb = Arc::new(Buffer::new(
            device,
            &BufferDesc::vertex("screen.vertex", (screen_vertices.len() * size_of::<ScreenVertex>()) as u64),
        )?);
        let screen_ib = Arc::new(Buffer::new(
            device,
            &BufferDesc::index("screen.index", (screen_indices.len() * size_of::<u16>()) as u64),
        )?);

        {
            let vertex_data = bytemuck::cast_slice(&screen_vertices);
            let index_data = bytemuck::cast_slice(&screen_indices);
            let mut upload_pool = UploadPool::new()?;
            upload_pool.enqueue_copy_buffer(device, screen_vb.as_range(..), vertex_data, BufferState::Vertex)?;
            upload_pool.enqueue_copy_buffer(device, screen_ib.as_range(..), index_data, BufferState::Index)?;

            let immediate = ImmediateCommandEncoder::new(device, device.graphics_queue())?;
            upload_pool.flush(&immediate, device)?;
        }

        let screen_vertex_shader = Arc::new(Shader::from_file(
            "shader.screen_quad.vs",
            device,
            "content/shaders/screen_quad.slang",
            ShaderStage::Vertex,
        )?);

        Ok(Self {
            screen_vb,
            screen_ib,
            screen_vertex_shader,
        })
    }
}

// pub fn add_screen_node<'a>(
//     builder: &'a mut RenderGraphBuilder,
//     node_name: &str,
//     extent: (u32, u32),
//     fragment_shader: Arc<Shader>,
//     colors: &[ColorAttachment],
//     depth: Option<DepthStencilAttachment>,
// ) -> GraphicNodeBuilder<'a> {
//     let default_res = DEFAULT_RENDER_RESOURCES.get().unwrap();
//
//     let shader = GraphicShaderInputBuilder::default()
//         .vertex_shader(default_res.screen_vertex_shader.clone())
//         .fragment_shader(fragment_shader)
//         .vertex_layout::<ScreenVertex>()
//         .build().unwrap();
//
//     let mut state = GraphicPipelineStateBuilder::default()
//         .rasterization(RasterizationStateBuilder::default().cull_mode(vk::CullModeFlags::NONE).build().unwrap());
//
//     let mut attachments = GraphicPipelineAttachments::default();
//     for color in colors {
//         attachments.color_formats.push(color.color.desc(builder).format);
//     }
//     if let Some(depth) = &depth {
//         attachments.depth_format = Some(depth.depth.desc(builder).format);
//         // TODO: what about stencil format?
//         // attachments.stencil_format = Some(depth.depth.desc(builder).format);
//     }
//
//     for _ in &attachments.color_formats {
//         state = state.push_blend_state(BlendStateBuilder::default().build().unwrap());
//     }
//     let state = state.build();
//
//     let pipeline_desc = GraphicPipelineDesc::new("screen", shader, state, attachments);
//
//     let vb = builder.import(default_res.screen_vb.clone(), BufferState::Vertex);
//     let ib = builder.import(default_res.screen_ib.clone(), BufferState::Index);
//
//     let mut node = builder.add_graphic_node(node_name);
//
//     let pipeline_handle = node.register_pipeline(pipeline_desc);
//
//     let vb = node.read(&vb, BufferState::Vertex);
//     let ib = node.read(&ib, BufferState::Index);
//
//     let width = extent.0;
//     let height = extent.1;
//     node.execute(move |ctx| {
//         // let view_range = ctx.get(&view).as_range(..);
//         // let base_range = ctx.get(&gbuffer_base).as_range(TextureLayout::ShaderReadOnly, .., ..);
//         // let nmr_range = ctx.get(&gbuffer_nmr).as_range(TextureLayout::ShaderReadOnly, .., ..);
//         // let depth_range = ctx.get(&scene_depth).as_range(TextureLayout::ShaderReadOnly, .., ..);
//
//         ctx.bind_pipeline(pipeline_handle)
//             .bind_raw(0, &[ctx.device().bindless_pool().lock().set()], &[])
//             .bind("view", view)?
//             .bind("base_color_tex", gbuffer_base)?
//             .bind("normal_mra_tex", gbuffer_nmr)?
//             .bind("depth_tex", scene_depth)?;
//
//         ctx.begin_rendering(
//             (width, height),
//             colors,
//             depth
//         );
//
//         let encoder = ctx.encoder();
//         encoder.bind_vertex_buffers(0, &[ctx.get(&vb).handle()], &[0]);
//         encoder.bind_index_buffer(ctx.get(&ib).handle(), 0, vk::IndexType::UINT16);
//         encoder.draw_indexed(6, 1, 0, 0, 0);
//
//         ctx.end_rendering();
//         Ok(())
//     });
//
//     node
// }

pub fn get_screen_vertices() -> [ScreenVertex; 4] {
    [
        ScreenVertex { position: [-1.0,  1.0], uv: [0.0, 0.0] },
        ScreenVertex { position: [ 1.0,  1.0], uv: [1.0, 0.0] },
        ScreenVertex { position: [-1.0, -1.0], uv: [0.0, 1.0] },
        ScreenVertex { position: [ 1.0, -1.0], uv: [1.0, 1.0] },
    ]
}

pub fn get_screen_indices() -> [u16; 6] {
    [0, 1, 2, 1, 2, 3]
}
