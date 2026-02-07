use std::sync::Arc;
use anyhow::anyhow;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4};
use zenith_asset::{Asset, AssetHandle};
use zenith_asset::render::{MeshCollection, Mesh, Material, TextureFormat};
use zenith_core::camera::{Camera, WORLD_SPACE_UP};
use zenith_core::math::{Degree};
use zenith_rhi::{
    vk, RenderDevice,
    Buffer, BufferDesc, BufferState,
    Shader, Texture, TextureDesc, TextureLayout, TextureState,
    ImmediateCommandEncoder, UploadPool,
    Sampler, SamplerDesc,
};
use zenith_rhi::pipeline::RasterizationStateBuilder;
use zenith_rendergraph::{
    ColorAttachmentDescBuilder, DepthStencilDesc,
    RenderGraphBuilder, RenderGraphResource, VertexLayout,
    GraphicShaderInputBuilder, GraphicPipelineStateBuilder,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, VertexLayout)]
pub struct WorldVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coord: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PushConstants {
    // Column-major 4x4 (matches glam::Mat4::to_cols_array()).
    mvp: [f32; 16],
    base_color_factor: [f32; 4],
    base_color_tex: u32,
    sampler_kind: u32,
    flags: u32,
    keep_normal_scale: f32,
}

const INVALID_BINDLESS: u32 = u32::MAX;
const FLAG_HAS_BASE_COLOR_TEX: u32 = 1 << 0;

#[derive(Clone)]
struct GpuMaterial {
    base_color_factor: [f32; 4],
    base_color_tex: Option<Arc<Texture>>,
    mra_tex: Option<Arc<Texture>>,
    normal_tex: Option<Arc<Texture>>,
    emissive_tex: Option<Arc<Texture>>,
}

#[derive(Clone)]
struct GpuMesh {
    vertex_buffer: Arc<Buffer>,
    index_buffer: Arc<Buffer>,
    index_count: u32,
    model: Mat4,
    material: GpuMaterial,
}

/// A simple world renderer: upload meshes/textures synchronously and render base_color.
///
/// Coordinate system: **right-handed, Z-up**.
/// Input assets (glTF) are assumed **right-handed, Y-up** and are rotated +90° around X to Z-up.
pub struct WorldRenderer {
    vertex_shader: Arc<Shader>,
    fragment_shader: Arc<Shader>,
    meshes: Vec<GpuMesh>,
    // Fixed bindless sampler heap indices:
    // 0=LinearRepeat, 1=LinearClamp, 2=NearestRepeat, 3=NearestClamp
    samplers: [Arc<Sampler>; 4],
}

impl WorldRenderer {
    pub fn new(device: &RenderDevice) -> anyhow::Result<Self> {
        let vertex_shader = Shader::from_file(
            "shader.world.vs",
            device,
            "content/shaders/world.slang",
            "vsmain",
            zenith_rhi::ShaderStage::Vertex,
        )?;

        let fragment_shader = Shader::from_file(
            "shader.world.ps",
            device,
            "content/shaders/world.slang",
            "psmain",
            zenith_rhi::ShaderStage::Fragment,
        )?;

        let linear_repeat = Arc::new(Sampler::new(device, &SamplerDesc::linear())?);
        let linear_clamp = Arc::new(Sampler::new(device, &SamplerDesc::linear().with_address_mode(vk::SamplerAddressMode::CLAMP_TO_EDGE))?);
        let nearest_repeat = Arc::new(Sampler::new(device, &SamplerDesc::nearest())?);
        let nearest_clamp = Arc::new(Sampler::new(device, &SamplerDesc::nearest().with_address_mode(vk::SamplerAddressMode::CLAMP_TO_EDGE))?);

        Ok(Self {
            vertex_shader: Arc::new(vertex_shader),
            fragment_shader: Arc::new(fragment_shader),
            meshes: Vec::new(),
            samplers: [linear_repeat, linear_clamp, nearest_repeat, nearest_clamp],
        })
    }

    pub fn add_mesh(&mut self, device: &RenderDevice,  mesh_collection: AssetHandle<MeshCollection>) -> anyhow::Result<()> {
        let collection = mesh_collection
            .get()
            .ok_or_else(|| anyhow!("MeshCollection not loaded/registered (call AssetManager::request_load first)"))?;

        // 1. Precalculate the size of gpu upload pool
        //------------------------------------------------------------------------------------------------

        let mut total_upload_bytes: usize = 0;

        for (mesh_url, mat_url) in collection.iter()? {
            let mesh_handle = AssetHandle::<Mesh>::new(mesh_url.clone());
            let mesh = mesh_handle
                .get()
                .ok_or_else(|| anyhow!("Mesh not loaded: {:?}", mesh_url.as_ref()))?;
            total_upload_bytes += mesh.gpu_size_in_bytes();

            let mat_handle = AssetHandle::<Material>::new(mat_url.clone());
            let mat = mat_handle
                .get()
                .ok_or_else(|| anyhow!("Material not loaded: {:?}", mat_url.as_ref()))?;

            let tex_urls = [
                mat.base_color_tex.as_ref(),
                mat.mra_tex.as_ref(),
                mat.normal_tex.as_ref(),
                mat.emissive_tex.as_ref(),
            ];
            for url in tex_urls.into_iter().flatten() {
                let tex_handle = AssetHandle::<zenith_asset::render::Texture>::new(url.clone());
                let tex = tex_handle
                    .get()
                    .ok_or_else(|| anyhow!("Texture not loaded: {:?}", url.as_ref()))?;
                total_upload_bytes += tex.gpu_size_in_bytes();
            }
        }

        // 2. Populate all upload data
        //------------------------------------------------------------------------------------------------

        struct PendingMeshUpload {
            gpu: GpuMesh,
            mesh_url: zenith_asset::AssetUrl,
            tex_urls: [Option<zenith_asset::AssetUrl>; 4],
        }

        let mut pending: Vec<PendingMeshUpload> = Vec::with_capacity(collection.iter()?.size_hint().1.unwrap_or(0));

        for (mesh_url, mat_url) in collection.iter()? {
            let mesh_handle = AssetHandle::<Mesh>::new(mesh_url.clone());
            let mesh = mesh_handle
                .get()
                .ok_or_else(|| anyhow!("Mesh not loaded: {:?}", mesh_url.as_ref()))?;

            let mat_handle = AssetHandle::<Material>::new(mat_url.clone());
            let mat = mat_handle
                .get()
                .ok_or_else(|| anyhow!("Material not loaded: {:?}", mat_url.as_ref()))?;

            let vertex_buffer = Arc::new(Buffer::new(
                device,
                &BufferDesc::vertex("world.mesh.vertex", mesh.vertices_bytes().len() as u64),
            )?);
            let index_buffer = Arc::new(Buffer::new(
                device,
                &BufferDesc::index("world.mesh.index", mesh.indices_bytes().len() as u64),
            )?);

            let create_texture = |name: &str, tex_url: Option<&zenith_asset::AssetUrl>| -> anyhow::Result<Option<Arc<Texture>>> {
                if let Some(tex_url) = tex_url {
                    let handle = AssetHandle::<zenith_asset::render::Texture>::new(tex_url.clone());
                    let tex = handle
                        .get()
                        .ok_or_else(|| anyhow!("Texture not loaded: {:?}", tex_url.as_ref()))?;

                    let format = texture_format_to_vk(&tex.format);
                    let desc = TextureDesc::new_2d(
                        &format!("world.tex.{name}"),
                        tex.width,
                        tex.height,
                        format,
                    ).with_transfer_dst_usage();

                    Ok(Some(Arc::new(Texture::new(device, &desc)?)))
                } else {
                    Ok(None)
                }
            };

            let tex_urls = [
                mat.base_color_tex.clone(),
                mat.mra_tex.clone(),
                mat.normal_tex.clone(),
                mat.emissive_tex.clone(),
            ];

            let material = GpuMaterial {
                base_color_factor: mat.base_color,
                base_color_tex: create_texture("base_color", tex_urls[0].as_ref())?,
                mra_tex: create_texture("mra", tex_urls[1].as_ref())?,
                normal_tex: create_texture("normal", tex_urls[2].as_ref())?,
                emissive_tex: create_texture("emissive", tex_urls[3].as_ref())?,
            };

            pending.push(PendingMeshUpload {
                gpu: GpuMesh {
                    vertex_buffer,
                    index_buffer,
                    index_count: mesh.indices.len() as u32,
                    model: Mat4::from_axis_angle(WORLD_SPACE_UP, Degree::new(90.0).to_radians().into()),
                    material,
                },
                mesh_url: mesh_url.clone(),
                tex_urls,
            });
        }

        // 3. Upload data to gpu
        //------------------------------------------------------------------------------------------------

        let immediate = ImmediateCommandEncoder::new(device, device.graphics_queue())?;
        let mut upload_pool = UploadPool::new(device, total_upload_bytes.max(1) as _)?;

        for p in &pending {
            let mesh_handle = AssetHandle::<Mesh>::new(p.mesh_url.clone());
            let mesh = mesh_handle
                .get()
                .ok_or_else(|| anyhow!("Mesh not loaded: {:?}", p.mesh_url.as_ref()))?;

            upload_pool.enqueue_copy_buffer(p.gpu.vertex_buffer.as_range(..)?, mesh.vertices_bytes(), BufferState::Vertex)?;
            upload_pool.enqueue_copy_buffer(p.gpu.index_buffer.as_range(..)?, mesh.indices_bytes(), BufferState::Index)?;

            if let (Some(gpu_tex), Some(url)) = (&p.gpu.material.base_color_tex, p.tex_urls[0].as_ref()) {
                let tex_handle = AssetHandle::<zenith_asset::render::Texture>::new(url.clone());
                let tex = tex_handle
                    .get()
                    .ok_or_else(|| anyhow!("Texture not loaded: {:?}", url.as_ref()))?;

                upload_pool.enqueue_upload_texture(
                    gpu_tex.as_range(TextureLayout::Undefined, .., ..)?,
                    tex.pixels.as_slice(),
                    TextureState::Sampled,
                )?;
            }
            if let (Some(gpu_tex), Some(url)) = (&p.gpu.material.mra_tex, p.tex_urls[1].as_ref()) {
                let tex_handle = AssetHandle::<zenith_asset::render::Texture>::new(url.clone());
                let tex = tex_handle
                    .get()
                    .ok_or_else(|| anyhow!("Texture not loaded: {:?}", url.as_ref()))?;

                upload_pool.enqueue_upload_texture(
                    gpu_tex.as_range(TextureLayout::Undefined, .., ..)?,
                    tex.pixels.as_slice(),
                    TextureState::Sampled,
                )?;
            }
            if let (Some(gpu_tex), Some(url)) = (&p.gpu.material.normal_tex, p.tex_urls[2].as_ref()) {
                let tex_handle = AssetHandle::<zenith_asset::render::Texture>::new(url.clone());
                let tex = tex_handle
                    .get()
                    .ok_or_else(|| anyhow!("Texture not loaded: {:?}", url.as_ref()))?;

                upload_pool.enqueue_upload_texture(
                    gpu_tex.as_range(TextureLayout::Undefined, .., ..)?,
                    tex.pixels.as_slice(),
                    TextureState::Sampled,
                )?;
            }
            if let (Some(gpu_tex), Some(url)) = (&p.gpu.material.emissive_tex, p.tex_urls[3].as_ref()) {
                let tex_handle = AssetHandle::<zenith_asset::render::Texture>::new(url.clone());
                let tex = tex_handle
                    .get()
                    .ok_or_else(|| anyhow!("Texture not loaded: {:?}", url.as_ref()))?;

                upload_pool.enqueue_upload_texture(
                    gpu_tex.as_range(TextureLayout::Undefined, .., ..)?,
                    tex.pixels.as_slice(),
                    TextureState::Sampled,
                )?;
            }
        }

        upload_pool.flush(&immediate, device)?;
        self.meshes.extend(pending.into_iter().map(|p| p.gpu));

        Ok(())
    }

    pub fn render(
        &self,
        builder: &mut RenderGraphBuilder,
        output: &mut RenderGraphResource<Texture>,
        width: u32,
        height: u32,
        camera: &Camera,
    ) {
        if self.meshes.is_empty() {
            return;
        }

        let mut depth = builder.create(TextureDesc::new_depth("world.depth", width, height));

        // Import buffers first (RenderGraphBuilder is mutably borrowed when a node builder exists).
        let mut imported = Vec::with_capacity(self.meshes.len());
        for m in &self.meshes {
            let vb = builder.import(m.vertex_buffer.clone(), BufferState::Vertex);
            let ib = builder.import(m.index_buffer.clone(), BufferState::Index);
            imported.push((vb, ib, m.index_count, m.model, m.material.clone()));
        }

        let mut node = builder.add_graphic_node("world");

        let output_rt = node.write(output, TextureState::Color);
        let depth_rt = node.write(&mut depth, TextureState::DepthStencil);

        // Import buffers and capture per-mesh draw info.
        let mut draws = Vec::with_capacity(self.meshes.len());

        for (vb_res, ib_res, index_count, model, material) in imported {
            let vb = node.read(&vb_res, BufferState::Vertex);
            let ib = node.read(&ib_res, BufferState::Index);
            draws.push((vb, ib, index_count, model, material));
        }

        let shader = GraphicShaderInputBuilder::default()
            .vertex_shader(self.vertex_shader.clone())
            .fragment_shader(self.fragment_shader.clone())
            .vertex_layout::<WorldVertex>()
            .build()
            .unwrap();

        let color_info = ColorAttachmentDescBuilder::default()
            .clear_input()
            .clear_value([0.05, 0.05, 0.05, 1.0])
            .build()
            .unwrap();

        let depth_info = DepthStencilDesc {
            depth_test_enable: true,
            depth_write_enable: true,
            // zenith-core camera uses reverse-Z projection -> GREATER with clear=0
            depth_compare_op: vk::CompareOp::GREATER,
            depth_load_op: vk::AttachmentLoadOp::CLEAR,
            depth_store_op: vk::AttachmentStoreOp::STORE,
            depth_clear_value: 0.0,
            ..Default::default()
        };

        let state = GraphicPipelineStateBuilder::default()
            .rasterization(RasterizationStateBuilder::default().build().unwrap())
            .depth_stencil(depth_info.clone())
            .build();

        {
            let mut binder = node.pipeline(shader, state);
            binder.push_color(output_rt, color_info);
            binder.depth(depth_rt, depth_info);
            binder.finish();
        }

        let samplers = self.samplers.clone();

        let view_proj_mat = camera.view_projection();
        node.execute(move |ctx| {
            let extent = vk::Extent2D { width, height };
            let encoder = ctx.encoder();

            // Bindless: ensure our fixed samplers exist in heap[0..3].
            let mut bindless = ctx.create_bindless_binder();
            bindless.bind_sampler_at(0, samplers[0].handle())?;
            bindless.bind_sampler_at(1, samplers[1].handle())?;
            bindless.bind_sampler_at(2, samplers[2].handle())?;
            bindless.bind_sampler_at(3, samplers[3].handle())?;

            // Bind textures (idempotent) and then bind the bindless set.
            // Note: we still call bind() for all textures to satisfy \"all textures bound\" requirement.
            let mut draw_state: Vec<(u32, Mat4, [f32; 4], u32, u32)> = Vec::with_capacity(draws.len());
            for (_vb, _ib, index_count, model, material) in draws.iter() {
                if let Some(t) = &material.base_color_tex {
                    let handle = bindless.bind(
                        t.as_range(TextureLayout::ShaderReadOnly, .., ..)
                            .map_err(|e| anyhow!("failed to create base_color texture range: {e:?}"))?,
                    )?;
                    draw_state.push((*index_count, *model, material.base_color_factor, *handle, FLAG_HAS_BASE_COLOR_TEX));
                } else {
                    draw_state.push((*index_count, *model, material.base_color_factor, INVALID_BINDLESS, 0));
                }

                if let Some(t) = &material.mra_tex {
                    let _ = bindless.bind(t.as_range(TextureLayout::ShaderReadOnly, .., ..).map_err(|e| anyhow!("failed to create mra texture range: {e:?}"))?)?;
                }
                if let Some(t) = &material.normal_tex {
                    let _ = bindless.bind(t.as_range(TextureLayout::ShaderReadOnly, .., ..).map_err(|e| anyhow!("failed to create normal texture range: {e:?}"))?)?;
                }
                if let Some(t) = &material.emissive_tex {
                    let _ = bindless.bind(t.as_range(TextureLayout::ShaderReadOnly, .., ..).map_err(|e| anyhow!("failed to create emissive texture range: {e:?}"))?)?;
                }
            }
            bindless.finish();

            ctx.begin_rendering(extent);
            ctx.bind_pipeline();

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

            for (i, (index_count, model, base_color_factor, base_color_tex, flags)) in draw_state.into_iter().enumerate() {
                let (vb_acc, ib_acc, _, _, _) = &draws[i];

                let mvp = view_proj_mat * model;
                let pc = PushConstants {
                    mvp: mvp.to_cols_array(),
                    base_color_factor,
                    base_color_tex,
                    sampler_kind: 0, // LinearRepeat
                    flags,
                    keep_normal_scale: 0.0,
                };
                ctx.push_constants(0, &pc);

                encoder.bind_vertex_buffers(0, &[ctx.get(vb_acc).handle()], &[0]);
                encoder.bind_index_buffer(ctx.get(ib_acc).handle(), 0, vk::IndexType::UINT32);
                encoder.draw_indexed(index_count, 1, 0, 0, 0);
            }

            ctx.end_rendering();
            Ok(())
        });
    }
}

fn texture_format_to_vk(format: &TextureFormat) -> vk::Format {
    match *format {
        TextureFormat::R8 => vk::Format::R8_UNORM,
        TextureFormat::R8G8 => vk::Format::R8G8_UNORM,
        TextureFormat::R8G8B8A8 => vk::Format::R8G8B8A8_SRGB,
        TextureFormat::R16 => vk::Format::R16_UNORM,
        TextureFormat::R16G16 => vk::Format::R16G16_UNORM,
        TextureFormat::R16G16B16A16 => vk::Format::R16G16B16A16_UNORM,
        TextureFormat::R32G32B32A32Float => vk::Format::R32G32B32A32_SFLOAT,
    }
}
