use std::sync::Arc;
use anyhow::anyhow;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec4};
use zenith_asset::{AssetHandle};
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
use crate::defer_shading::GBuffer;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, VertexLayout)]
pub struct WorldVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coord: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, VertexLayout)]
struct LightingVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PushConstants {
    // Column-major 4x4 (matches glam::Mat4::to_cols_array()).
    model: [f32; 16],
    base_color_factor: [f32; 4],
    base_color_tex: u32,
    mra_tex: u32,
    normal_tex: u32,
    sampler_kind: u32,
    flags: u32,
    keep_normal_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightingPushConstants {
    gbuffer_base: u32,
    gbuffer_nmr: u32,
    scene_depth: u32,
}

const INVALID_BINDLESS: u32 = u32::MAX;
const FLAG_HAS_BASE_COLOR_TEX: u32 = 1 << 0;
const FLAG_HAS_MRA_TEX: u32 = 1 << 1;
const FLAG_HAS_NORMAL_TEX: u32 = 1 << 2;

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
    lighting_vertex_shader: Arc<Shader>,
    lighting_fragment_shader: Arc<Shader>,
    lighting_vertex_buffer: Arc<Buffer>,
    lighting_index_buffer: Arc<Buffer>,
    meshes: Vec<GpuMesh>,
    gbuffer: Option<GBuffer>,
    width: u32,
    height: u32,
    // Fixed bindless sampler heap indices:
    // 0=LinearRepeat, 1=LinearClamp, 2=NearestRepeat, 3=NearestClamp
    samplers: [Arc<Sampler>; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ViewData {
    view_proj: [f32; 16],
    inv_view_proj: [f32; 16],
    view: [f32; 16],
    inv_view: [f32; 16],
    proj: [f32; 16],
    inv_proj: [f32; 16],
    position: [f32; 4],
}

impl ViewData {
    pub fn from_camera(cam: &Camera) -> Self {
        let vp = cam.view_projection();
        let v = cam.view();
        let p = cam.projection();
        Self {
            view_proj: vp.to_cols_array(),
            inv_view_proj: vp.inverse().to_cols_array(),
            view: v.to_cols_array(),
            inv_view: v.inverse().to_cols_array(),
            proj: p.to_cols_array(),
            inv_proj: p.inverse().to_cols_array(),
            position: Vec4::new(cam.position().x, cam.position().y, cam.position().z, 1.0).to_array(),
        }
    }
}

impl WorldRenderer {
    pub fn new(device: &RenderDevice) -> anyhow::Result<Self> {
        let vertex_shader = Shader::from_file(
            "shader.defer_shading.vs",
            device,
            "content/shaders/defer_shading.slang",
            zenith_rhi::ShaderStage::Vertex,
        )?;

        let fragment_shader = Shader::from_file(
            "shader.defer_shading.ps",
            device,
            "content/shaders/defer_shading.slang",
            zenith_rhi::ShaderStage::Fragment,
        )?;

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
            LightingVertex { position: [-1.0,  1.0], uv: [0.0, 0.0] },
            LightingVertex { position: [ 1.0,  1.0], uv: [1.0, 0.0] },
            LightingVertex { position: [-1.0, -1.0], uv: [0.0, 1.0] },
            LightingVertex { position: [ 1.0, -1.0], uv: [1.0, 1.0] },
        ];
        let lighting_indices: [u16; 6] = [0, 1, 2, 1, 2, 3];

        let lighting_vertex_buffer = Arc::new(Buffer::new(
            device,
            &BufferDesc::vertex("world.lighting.vertex", (lighting_vertices.len() * size_of::<LightingVertex>()) as u64),
        )?);
        let lighting_index_buffer = Arc::new(Buffer::new(
            device,
            &BufferDesc::index("world.lighting.index", (lighting_indices.len() * size_of::<u16>()) as u64),
        )?);

        {
            let vertex_data = bytemuck::cast_slice(&lighting_vertices);
            let index_data = bytemuck::cast_slice(&lighting_indices);
            let total_size = vertex_data.len() + index_data.len();
            let mut upload_pool = UploadPool::new(device, total_size as _)?;
            upload_pool.enqueue_copy_buffer(device, lighting_vertex_buffer.as_range(..)?, vertex_data, BufferState::Vertex)?;
            upload_pool.enqueue_copy_buffer(device, lighting_index_buffer.as_range(..)?, index_data, BufferState::Index)?;

            let immediate = ImmediateCommandEncoder::new(device, device.graphics_queue())?;
            upload_pool.flush(&immediate, device)?;
        }

        let linear_repeat = Arc::new(Sampler::new(device, &SamplerDesc::linear())?);
        let linear_clamp = Arc::new(Sampler::new(device, &SamplerDesc::linear().with_address_mode(vk::SamplerAddressMode::CLAMP_TO_EDGE))?);
        let nearest_repeat = Arc::new(Sampler::new(device, &SamplerDesc::nearest())?);
        let nearest_clamp = Arc::new(Sampler::new(device, &SamplerDesc::nearest().with_address_mode(vk::SamplerAddressMode::CLAMP_TO_EDGE))?);

        Ok(Self {
            vertex_shader: Arc::new(vertex_shader),
            fragment_shader: Arc::new(fragment_shader),
            lighting_vertex_shader: Arc::new(lighting_vertex_shader),
            lighting_fragment_shader: Arc::new(lighting_fragment_shader),
            lighting_vertex_buffer,
            lighting_index_buffer,
            meshes: Vec::new(),
            gbuffer: None,
            width: 0,
            height: 0,
            samplers: [linear_repeat, linear_clamp, nearest_repeat, nearest_clamp],
        })
    }

    pub fn add_mesh(&mut self, device: &RenderDevice,  mesh_collection: AssetHandle<MeshCollection>) -> anyhow::Result<()> {
        let collection = mesh_collection
            .get()
            .ok_or_else(|| anyhow!("MeshCollection not loaded/registered (call AssetManager::request_load first)"))?;

        // 1. Populate all upload data
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

        // 2. Upload data to gpu
        //------------------------------------------------------------------------------------------------

        let immediate = ImmediateCommandEncoder::new(device, device.graphics_queue())?;
        let mut upload_pool = UploadPool::new(device, 64 * 1024)?;

        let validate_texture_data_size = |tex: &zenith_asset::render::Texture| -> anyhow::Result<()> {
            let expected = tex.format.data_size_in_bytes(tex.width, tex.height);
            if tex.pixels.len() != expected {
                return Err(anyhow!(
                    "Texture data size mismatch: {} ({}x{}, {:?}) expected {} bytes, got {} bytes",
                    tex.width * tex.height,
                    tex.width,
                    tex.height,
                    tex.format,
                    expected,
                    tex.pixels.len()
                ));
            }
            Ok(())
        };

        for p in &pending {
            let mesh_handle = AssetHandle::<Mesh>::new(p.mesh_url.clone());
            let mesh = mesh_handle
                .get()
                .ok_or_else(|| anyhow!("Mesh not loaded: {:?}", p.mesh_url.as_ref()))?;

            upload_pool.enqueue_copy_buffer(device, p.gpu.vertex_buffer.as_range(..)?, mesh.vertices_bytes(), BufferState::Vertex)?;
            upload_pool.enqueue_copy_buffer(device, p.gpu.index_buffer.as_range(..)?, mesh.indices_bytes(), BufferState::Index)?;

            if let (Some(gpu_tex), Some(url)) = (&p.gpu.material.base_color_tex, p.tex_urls[0].as_ref()) {
                let tex_handle = AssetHandle::<zenith_asset::render::Texture>::new(url.clone());
                let tex = tex_handle
                    .get()
                    .ok_or_else(|| anyhow!("Texture not loaded: {:?}", url.as_ref()))?;

                validate_texture_data_size(&tex)?;
                upload_pool.enqueue_upload_texture(
                    device,
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

                validate_texture_data_size(&tex)?;
                upload_pool.enqueue_upload_texture(
                    device,
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

                validate_texture_data_size(&tex)?;
                upload_pool.enqueue_upload_texture(
                    device,
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

                validate_texture_data_size(&tex)?;
                upload_pool.enqueue_upload_texture(
                    device,
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
        &mut self,
        builder: &mut RenderGraphBuilder,
        output: &mut RenderGraphResource<Texture>,
        width: u32,
        height: u32,
        camera: &Camera,
    ) {
        if self.meshes.is_empty() || width == 0 || height == 0 {
            return;
        }

        self.width = width;
        self.height = height;

        let view = builder.create(
            BufferDesc::uniform("view", size_of::<ViewData>() as _)
        );
        let (gbuffer_base, gbuffer_nmr, scene_depth) = self.render_meshes(builder, camera, &view);
        self.render_lighting(builder, output, gbuffer_base, gbuffer_nmr, scene_depth, &view);
    }

    fn render_meshes(
        &mut self,
        builder: &mut RenderGraphBuilder,
        camera: &Camera,
        view: &RenderGraphResource<Buffer>,
    ) -> (RenderGraphResource<Texture>, RenderGraphResource<Texture>, RenderGraphResource<Texture>) {
        let width = self.width;
        let height = self.height;

        let mut depth = builder.create(
            TextureDesc::new_depth("scene.depth", width, height)
                .with_additional_usage(vk::ImageUsageFlags::SAMPLED)
        );

        let gbuffer = self.gbuffer.get_or_insert_with(|| GBuffer::new(width, height));
        gbuffer.ensure_size(width, height);
        let (mut gbuffer_base_color, mut gbuffer_normal_mra) = gbuffer.create(builder);

        // Import buffers first (RenderGraphBuilder is mutably borrowed when a node builder exists).
        let mut imported = Vec::with_capacity(self.meshes.len());
        for m in &self.meshes {
            let vb = builder.import(m.vertex_buffer.clone(), BufferState::Vertex);
            let ib = builder.import(m.index_buffer.clone(), BufferState::Index);
            imported.push((vb, ib, m.index_count, m.model, m.material.clone()));
        }

        let mut node = builder.add_graphic_node("gbuffer");

        let view = node.read(&view, BufferState::Uniform);
        let gbuffer_base_rt = node.write(&mut gbuffer_base_color, TextureState::Color);
        let gbuffer_nmr_rt = node.write(&mut gbuffer_normal_mra, TextureState::Color);
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

        let base_color_info = ColorAttachmentDescBuilder::default()
            .clear_input()
            .clear_value([0.05, 0.05, 0.05, 1.0])
            .build()
            .unwrap();
        let normal_info = ColorAttachmentDescBuilder::default()
            .clear_input()
            .clear_value([0.5, 0.5, 1.0, 1.0])
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
            binder.push_color(gbuffer_base_rt, base_color_info);
            binder.push_color(gbuffer_nmr_rt, normal_info);
            binder.depth(depth_rt, depth_info);
            binder.finish();
        }

        let samplers = self.samplers.clone();

        let view_data = ViewData::from_camera(camera);
        node.execute(move |ctx| {
            let extent = vk::Extent2D { width, height };
            let encoder = ctx.encoder();

            // Bind all bindless resources via direct pool access (set0 bound at command buffer start).
            let mut draw_state: Vec<(u32, Mat4, [f32; 4], u32, u32, u32, u32)> = Vec::with_capacity(draws.len());
            {
                let mut pool = ctx.device().bindless_pool().lock();

                // Ensure our fixed samplers exist in heap[0..3].
                pool.bind_sampler_at(0, samplers[0].handle())?;
                pool.bind_sampler_at(1, samplers[1].handle())?;
                pool.bind_sampler_at(2, samplers[2].handle())?;
                pool.bind_sampler_at(3, samplers[3].handle())?;

                // Bind textures (idempotent).
                for (_vb, _ib, index_count, model, material) in draws.iter() {
                    let mut flags = 0;
                    let mut base_color_tex = INVALID_BINDLESS;
                    let mut mra_tex = INVALID_BINDLESS;
                    let mut normal_tex = INVALID_BINDLESS;

                    if let Some(t) = &material.base_color_tex {
                        let handle = pool.bind(
                            t.as_range(TextureLayout::ShaderReadOnly, .., ..)
                                .map_err(|e| anyhow!("failed to create base_color texture range: {e:?}"))?,
                        )?;
                        base_color_tex = *handle;
                        flags |= FLAG_HAS_BASE_COLOR_TEX;
                    }

                    if let Some(t) = &material.mra_tex {
                        let handle = pool.bind(
                            t.as_range(TextureLayout::ShaderReadOnly, .., ..)
                                .map_err(|e| anyhow!("failed to create mra texture range: {e:?}"))?,
                        )?;
                        mra_tex = *handle;
                        flags |= FLAG_HAS_MRA_TEX;
                    }
                    if let Some(t) = &material.normal_tex {
                        let handle = pool.bind(
                            t.as_range(TextureLayout::ShaderReadOnly, .., ..)
                                .map_err(|e| anyhow!("failed to create normal texture range: {e:?}"))?,
                        )?;
                        normal_tex = *handle;
                        flags |= FLAG_HAS_NORMAL_TEX;
                    }
                    if let Some(t) = &material.emissive_tex {
                        let _ = pool.bind(
                            t.as_range(TextureLayout::ShaderReadOnly, .., ..)
                                .map_err(|e| anyhow!("failed to create emissive texture range: {e:?}"))?,
                        )?;
                    }

                    draw_state.push((
                        *index_count,
                        *model,
                        material.base_color_factor,
                        base_color_tex,
                        mra_tex,
                        normal_tex,
                        flags,
                    ));
                }

                // Flush all pending descriptor writes into the pool-owned bindless set.
                pool.flush(ctx.device());
                ctx.bind_descriptor_sets(0, std::slice::from_ref(&pool.set()));
            }

            // Write and bind view buffer.
            let view = ctx.get(&view)
                .as_range(..)
                .map_err(|e| anyhow::anyhow!("failed to create view buffer range: {:?}", e))?;
            view.write(bytemuck::bytes_of(&view_data))
                .map_err(|e| anyhow::anyhow!("failed to write view buffer: {:?}", e))?;

            let mut binder = ctx.create_binder();
            binder.bind("view", view)?;
            // first_set=1: skip set 0 (bindless, managed by BindlessPool)
            let sets = binder.finish(ctx.device(), 1).map_err(|e| anyhow::anyhow!("descriptor set finish failed: {e}"))?;
            ctx.bind_descriptor_sets(1, &sets);

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

            for (i, (index_count, model, base_color_factor, base_color_tex, mra_tex, normal_tex, flags)) in draw_state.into_iter().enumerate() {
                let (vb_acc, ib_acc, _, _, _) = &draws[i];

                let pc = PushConstants {
                    model: model.to_cols_array(),
                    base_color_factor,
                    base_color_tex,
                    mra_tex,
                    normal_tex,
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

        (gbuffer_base_color, gbuffer_normal_mra, depth)
    }

    fn render_lighting(
        &self,
        builder: &mut RenderGraphBuilder,
        output: &mut RenderGraphResource<Texture>,
        gbuffer_base: RenderGraphResource<Texture>,
        gbuffer_nmr: RenderGraphResource<Texture>,
        scene_depth: RenderGraphResource<Texture>,
        view: &RenderGraphResource<Buffer>,
    ) {
        let width = self.width;
        let height = self.height;

        let vb = builder.import(self.lighting_vertex_buffer.clone(), BufferState::Vertex);
        let ib = builder.import(self.lighting_index_buffer.clone(), BufferState::Index);

        let mut node = builder.add_graphic_node("lighting");
        let vb = node.read(&vb, BufferState::Vertex);
        let ib = node.read(&ib, BufferState::Index);
        let gbuffer_base = node.read(&gbuffer_base, TextureState::Sampled);
        let gbuffer_nmr = node.read(&gbuffer_nmr, TextureState::Sampled);
        let scene_depth = node.read(&scene_depth, TextureState::Sampled);
        let view = node.read(view, BufferState::StorageRead);
        let output_rt = node.write(output, TextureState::Color);

        let shader = GraphicShaderInputBuilder::default()
            .vertex_shader(self.lighting_vertex_shader.clone())
            .fragment_shader(self.lighting_fragment_shader.clone())
            .vertex_layout::<LightingVertex>()
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

        let samplers = self.samplers.clone();
        node.execute(move |ctx| {
            let extent = vk::Extent2D { width, height };
            let encoder = ctx.encoder();

            // Bind all bindless resources via direct pool access (set0 bound at command buffer start).
            let base_handle;
            let nmr_handle;
            let depth_handle;
            {
                let mut pool = ctx.device().bindless_pool().lock();
                pool.bind_sampler_at(0, samplers[0].handle())?;
                pool.bind_sampler_at(1, samplers[1].handle())?;
                pool.bind_sampler_at(2, samplers[2].handle())?;
                pool.bind_sampler_at(3, samplers[3].handle())?;

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

            let pc = LightingPushConstants {
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

fn texture_format_to_vk(format: &TextureFormat) -> vk::Format {
    match *format {
        TextureFormat::R8 => vk::Format::R8_UNORM,
        TextureFormat::R8G8 => vk::Format::R8G8_UNORM,
        TextureFormat::R8G8B8A8 => vk::Format::R8G8B8A8_SRGB,
        TextureFormat::R16 => vk::Format::R16_UNORM,
        TextureFormat::R16G16 => vk::Format::R16G16_UNORM,
        TextureFormat::R16G16B16A16 => vk::Format::R16G16B16A16_UNORM,
        TextureFormat::R32G32B32A32Float => vk::Format::R32G32B32A32_SFLOAT,
        TextureFormat::Bc5Unorm => vk::Format::BC5_UNORM_BLOCK,
        TextureFormat::Bc7Unorm => vk::Format::BC7_UNORM_BLOCK,
        TextureFormat::Bc7Srgb => vk::Format::BC7_SRGB_BLOCK,
    }
}

