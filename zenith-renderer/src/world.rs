use std::sync::Arc;
use anyhow::anyhow;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4};
use zenith_asset::{AssetHandle};
use zenith_asset::material::Material;
use zenith_asset::mesh::{Scene, Mesh};
use zenith_core::camera::{Camera, ViewData, WORLD_SPACE_UP};
use zenith_core::math::{Degree};
use zenith_rhi::{vk, RenderDevice, Buffer, BufferDesc, BufferState, Shader, Texture, TextureDesc, TextureLayout, TextureState, ImmediateCommandEncoder, UploadPool, Sampler, SamplerDesc, GraphicPipelineDesc, DepthStencilStateBuilder, BindlessHandle, BindlessPool, GraphicPipelineState};
use zenith_rhi::pipeline::{GraphicPipelineAttachmentsBuilder};
use zenith_rendergraph::{RenderGraphBuilder, RenderGraphResource, VertexLayout, GraphicShaderInputBuilder, ColorAttachment, DepthStencilAttachment};
use crate::defer_shading::SceneTextures;
use crate::lighting::DirectLightingRenderer;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, VertexLayout)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coord: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PushConstants {
    model: [f32; 16],
    base_color_factor: [f32; 4],
    base_color_tex: u32,
    mra_tex: u32,
    normal_tex: u32,
}

#[derive(Clone)]
struct GpuMaterial {
    base_color_factor: [f32; 4],
    base_color_tex: Option<Arc<Texture>>,
    mra_tex: Option<Arc<Texture>>,
    normal_tex: Option<Arc<Texture>>,
    emissive_tex: Option<Arc<Texture>>,
    base_color_tex_handle: BindlessHandle,
    mra_tex_handle: BindlessHandle,
    normal_tex_handle: BindlessHandle,
    _emissive_tex_handle: BindlessHandle,
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
    pub(crate) width: u32,
    pub(crate) height: u32,
    // 0=LinearRepeat, 1=LinearClamp, 2=NearestRepeat, 3=NearestClamp
    _samplers: Vec<Arc<Sampler>>,

    direct_lighting_renderer: DirectLightingRenderer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuViewData {
    view_proj: [f32; 16],
    inv_view_proj: [f32; 16],
    view: [f32; 16],
    inv_view: [f32; 16],
    proj: [f32; 16],
    inv_proj: [f32; 16],
    position: [f32; 4],
}

impl GpuViewData {
    pub fn from_view_data(view: &ViewData) -> Self {
        Self {
            view_proj: view.view_proj.to_cols_array(),
            inv_view_proj: view.inv_view_proj.to_cols_array(),
            view: view.view.to_cols_array(),
            inv_view: view.inv_view.inverse().to_cols_array(),
            proj: view.proj.to_cols_array(),
            inv_proj: view.inv_proj.inverse().to_cols_array(),
            position: view.position.to_array(),
        }
    }
}

impl WorldRenderer {
    pub fn new(device: &RenderDevice, width: u32, height: u32) -> anyhow::Result<Self> {
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

        let linear_repeat = Arc::new(Sampler::new(device, &SamplerDesc::linear())?);
        let linear_clamp = Arc::new(Sampler::new(device, &SamplerDesc::linear().with_address_mode(vk::SamplerAddressMode::CLAMP_TO_EDGE))?);
        let nearest_repeat = Arc::new(Sampler::new(device, &SamplerDesc::nearest())?);
        let nearest_clamp = Arc::new(Sampler::new(device, &SamplerDesc::nearest().with_address_mode(vk::SamplerAddressMode::CLAMP_TO_EDGE))?);

        let mut pool = device.bindless_pool().lock();
        pool.upload(&*linear_repeat)?;
        pool.upload(&*linear_clamp)?;
        pool.upload(&*nearest_repeat)?;
        pool.upload(&*nearest_clamp)?;
        pool.flush(device);

        Ok(Self {
            vertex_shader: Arc::new(vertex_shader),
            fragment_shader: Arc::new(fragment_shader),
            meshes: Vec::new(),
            width,
            height,
            _samplers: vec![linear_repeat, linear_clamp, nearest_repeat, nearest_clamp],
            direct_lighting_renderer: DirectLightingRenderer::new(device, width, height)?,
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

        self.direct_lighting_renderer.resize(width, height);
    }

    pub fn set_skybox(&mut self, device: &RenderDevice, skybox: &zenith_asset::texture::Texture) -> anyhow::Result<()> {
        self.direct_lighting_renderer.set_skybox(device, skybox)
    }

    pub fn add_scene(&mut self, device: &RenderDevice, scene: AssetHandle<Scene>) -> anyhow::Result<()> {
        let collection = scene
            .get()
            .ok_or_else(|| anyhow!("MeshCollection not loaded/registered (call AssetManager::request_load first)"))?;

        // 1. Populate all upload data
        // TODO: render resource creation should NOT be done here
        //------------------------------------------------------------------------------------------------

        struct PendingMeshUpload {
            gpu: GpuMesh,
            mesh_url: zenith_asset::AssetUrl,
            tex_urls: [Option<zenith_asset::AssetUrl>; 4],
        }

        let mut pending: Vec<PendingMeshUpload> = Vec::with_capacity(collection.iter().size_hint().1.unwrap_or(0));

        for mesh_url in collection.iter() {
            let mesh_handle = AssetHandle::<Mesh>::new(mesh_url.clone());
            let mesh = mesh_handle
                .get()
                .ok_or_else(|| anyhow!("Mesh not loaded: {:?}", mesh_url.as_ref()))?;

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
                    let handle = AssetHandle::<zenith_asset::texture::Texture>::new(tex_url.clone());
                    let tex = handle
                        .get()
                        .ok_or_else(|| anyhow!("Texture not loaded: {:?}", tex_url.as_ref()))?;

                    let desc = TextureDesc::new_2d(
                        &format!("world.tex.{name}"),
                        tex.width,
                        tex.height,
                        tex.format.to_vk(),
                    ).with_transfer_dst_usage();

                    Ok(Some(Arc::new(Texture::new(device, &desc)?)))
                } else {
                    Ok(None)
                }
            };

            let handle = mesh.material.as_ref()
                .map(|url| AssetHandle::<Material>::new(url.clone()));

            if let Some(handle) = &handle {
                let mut bindless_pool = device.bindless_pool().lock();

                let mat = handle.get()
                    .ok_or_else(|| anyhow!("Material not loaded: {:?}", handle.url().as_ref()))?;

                let tex_urls = [
                    mat.base_color_tex.clone(),
                    mat.mra_tex.clone(),
                    mat.normal_tex.clone(),
                    mat.emissive_tex.clone(),
                ];

                let base_color_tex = create_texture("base_color", tex_urls[0].as_ref())?;
                let mra_tex = create_texture("mra", tex_urls[1].as_ref())?;
                let normal_tex = create_texture("normal", tex_urls[2].as_ref())?;
                let emissive_tex = create_texture("emissive", tex_urls[3].as_ref())?;

                let base_color_tex_handle = if let Some(tex) = &base_color_tex {
                    bindless_pool.upload(&tex.as_range(TextureLayout::ShaderReadOnly, .., ..))?
                } else {
                    BindlessHandle::INVALID
                };

                let mra_tex_handle = if let Some(tex) = &mra_tex {
                    bindless_pool.upload(&tex.as_range(TextureLayout::ShaderReadOnly, .., ..))?
                } else {
                    BindlessHandle::INVALID
                };

                let normal_tex_handle = if let Some(tex) = &normal_tex {
                    bindless_pool.upload(&tex.as_range(TextureLayout::ShaderReadOnly, .., ..))?
                } else {
                    BindlessHandle::INVALID
                };

                let emissive_tex_handle = if let Some(tex) = &emissive_tex {
                    bindless_pool.upload(&tex.as_range(TextureLayout::ShaderReadOnly, .., ..))?
                } else {
                    BindlessHandle::INVALID
                };

                bindless_pool.flush(device);

                let material = GpuMaterial {
                    base_color_factor: mat.base_color,
                    base_color_tex,
                    mra_tex,
                    normal_tex,
                    emissive_tex,
                    base_color_tex_handle,
                    mra_tex_handle,
                    normal_tex_handle,
                    _emissive_tex_handle: emissive_tex_handle,
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
        }

        // 2. Upload data to gpu
        //------------------------------------------------------------------------------------------------

        let mut upload_pool = UploadPool::new()?;

        let validate_texture_data_size = |tex: &zenith_asset::texture::Texture| -> anyhow::Result<()> {
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

            upload_pool.enqueue_copy_buffer(device, p.gpu.vertex_buffer.as_range(..), mesh.vertices_bytes(), BufferState::Vertex)?;
            upload_pool.enqueue_copy_buffer(device, p.gpu.index_buffer.as_range(..), mesh.indices_bytes(), BufferState::Index)?;

            if let (Some(gpu_tex), Some(url)) = (&p.gpu.material.base_color_tex, p.tex_urls[0].as_ref()) {
                let tex_handle = AssetHandle::<zenith_asset::texture::Texture>::new(url.clone());
                let tex = tex_handle
                    .get()
                    .ok_or_else(|| anyhow!("Texture not loaded: {:?}", url.as_ref()))?;

                validate_texture_data_size(&tex)?;
                upload_pool.enqueue_upload_texture(
                    device,
                    gpu_tex.as_range(TextureLayout::Undefined, .., ..),
                    tex.pixels.as_slice(),
                    TextureState::Sampled,
                )?;
            }
            if let (Some(gpu_tex), Some(url)) = (&p.gpu.material.mra_tex, p.tex_urls[1].as_ref()) {
                let tex_handle = AssetHandle::<zenith_asset::texture::Texture>::new(url.clone());
                let tex = tex_handle
                    .get()
                    .ok_or_else(|| anyhow!("Texture not loaded: {:?}", url.as_ref()))?;

                validate_texture_data_size(&tex)?;
                upload_pool.enqueue_upload_texture(
                    device,
                    gpu_tex.as_range(TextureLayout::Undefined, .., ..),
                    tex.pixels.as_slice(),
                    TextureState::Sampled,
                )?;
            }
            if let (Some(gpu_tex), Some(url)) = (&p.gpu.material.normal_tex, p.tex_urls[2].as_ref()) {
                let tex_handle = AssetHandle::<zenith_asset::texture::Texture>::new(url.clone());
                let tex = tex_handle
                    .get()
                    .ok_or_else(|| anyhow!("Texture not loaded: {:?}", url.as_ref()))?;

                validate_texture_data_size(&tex)?;
                upload_pool.enqueue_upload_texture(
                    device,
                    gpu_tex.as_range(TextureLayout::Undefined, .., ..),
                    tex.pixels.as_slice(),
                    TextureState::Sampled,
                )?;
            }
            if let (Some(gpu_tex), Some(url)) = (&p.gpu.material.emissive_tex, p.tex_urls[3].as_ref()) {
                let tex_handle = AssetHandle::<zenith_asset::texture::Texture>::new(url.clone());
                let tex = tex_handle
                    .get()
                    .ok_or_else(|| anyhow!("Texture not loaded: {:?}", url.as_ref()))?;

                validate_texture_data_size(&tex)?;
                upload_pool.enqueue_upload_texture(
                    device,
                    gpu_tex.as_range(TextureLayout::Undefined, .., ..),
                    tex.pixels.as_slice(),
                    TextureState::Sampled,
                )?;
            }
        }

        let immediate = ImmediateCommandEncoder::new(device, device.graphics_queue())?;
        upload_pool.flush(&immediate, device)?;
        self.meshes.extend(pending.into_iter().map(|p| p.gpu));

        Ok(())
    }
    
    pub fn render(
        &mut self,
        builder: &mut RenderGraphBuilder,
        camera: &Camera,
        output: &mut RenderGraphResource<Texture>,
    ) {
        let view = builder.create(
            BufferDesc::uniform("view", size_of::<GpuViewData>() as _)
        );

        let scene_textures = self.render_meshes(builder, camera, &view);
        self.direct_lighting_renderer.render(builder, scene_textures, &view, output);
    }

    fn render_meshes(
        &mut self,
        builder: &mut RenderGraphBuilder,
        camera: &Camera,
        view: &RenderGraphResource<Buffer>,
    ) -> SceneTextures {
        let mut scene_textures = SceneTextures::new(builder, self.width, self.height);

        let pipeline_desc = GraphicPipelineDesc::new(
            "gbuffer",
            GraphicShaderInputBuilder::default()
                .vertex_shader(self.vertex_shader.clone())
                .fragment_shader(self.fragment_shader.clone())
                .vertex_layout::<Vertex>()
                .build().unwrap(),
            GraphicPipelineState::default(),
            GraphicPipelineAttachmentsBuilder::default()
                .color_no_blending(scene_textures.base_color.desc(builder).format)
                .color_no_blending(scene_textures.normal_mra.desc(builder).format)
                .depth_stencil(scene_textures.depth.desc(builder).format, DepthStencilStateBuilder::default()
                    .depth_test_enable(true).depth_write_enable(true)
                    .build().unwrap())
                .build().unwrap()
        );

        // Import buffers first (RenderGraphBuilder is mutably borrowed when a node builder exists).
        let mut imported = Vec::with_capacity(self.meshes.len());
        for m in &self.meshes {
            let vb = builder.import(m.vertex_buffer.clone(), BufferState::Vertex);
            let ib = builder.import(m.index_buffer.clone(), BufferState::Index);
            imported.push((vb, ib, m.index_count, m.model, m.material.clone()));
        }

        let mut node = builder.add_graphic_node("gbuffer");

        // Register pipeline and get handle
        let pipeline_handle = node.register_pipeline(pipeline_desc.clone());

        let view = node.read(&view, BufferState::Uniform);
        let gbuffer_base_rt = node.write(&mut scene_textures.base_color, TextureState::Color);
        let gbuffer_nmr_rt = node.write(&mut scene_textures.normal_mra, TextureState::Color);
        let depth_rt = node.write(&mut scene_textures.depth, TextureState::DepthStencil);

        // Import buffers and capture per-mesh draw info.
        let mut draws = Vec::with_capacity(self.meshes.len());

        for (vb_res, ib_res, index_count, model, material) in imported {
            let vb = node.read(&vb_res, BufferState::Vertex);
            let ib = node.read(&ib_res, BufferState::Index);
            draws.push((vb, ib, index_count, model, material));
        }

        let view_data = GpuViewData::from_view_data(&camera.view_data());
        let width = self.width;
        let height = self.height;
        node.execute(move |ctx| {
            if let Some(pipe) = ctx.bind_pipeline(pipeline_handle) {
                ctx.get(&view)
                    .as_range(..)
                    .write(bytemuck::bytes_of(&view_data))
                    .map_err(|e| anyhow!("failed to write view buffer: {e:?}"))?;

                pipe.bind_raw(BindlessPool::SET_INDEX, &[ctx.device().bindless_pool().lock().set()], &[])
                    .bind("view", view)?
                    .finish();

                ctx.begin_rendering(
                    (width, height),
                    &[
                        ColorAttachment::new(gbuffer_base_rt).clear_input().clear_value([0.05, 0.05, 0.05, 1.0]),
                        ColorAttachment::new(gbuffer_nmr_rt).clear_input().clear_value([0.5, 0.5, 1.0, 1.0]),
                    ],
                    Some(DepthStencilAttachment::new(depth_rt).clear_depth_input()),
                );

                for (vb, ib, index_count, model, material) in draws.into_iter() {
                    let pc = PushConstants {
                        model: model.to_cols_array(),
                        base_color_factor: material.base_color_factor,
                        base_color_tex: material.base_color_tex_handle.raw(),
                        mra_tex: material.mra_tex_handle.raw(),
                        normal_tex: material.normal_tex_handle.raw(),
                    };
                    ctx.push_constants(pipeline_handle, 0, &pc);

                    ctx.bind_vertex_buffers(vb, 0, &[0]);
                    ctx.bind_index_buffer(ib, 0, vk::IndexType::UINT32);
                    ctx.encoder().draw_indexed(0..index_count, 0..1, 0);
                }

                ctx.end_rendering();
            }

            Ok(())
        });

        scene_textures
    }
}
