use std::path::PathBuf;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use zenith_asset::AssetHandle;
use zenith_asset::render::{Material, Mesh};
use zenith_build::{ShaderEntry};
use zenith_core::collections::SmallVec;
use zenith_render::{define_shader, GraphicShader, RenderDevice};
use zenith_rendergraph::{Buffer, DepthStencilInfo, RenderGraphBuilder, RenderGraphResource, RenderResource, Texture, TextureDesc};

pub struct SimpleMeshRenderer {
    mesh_buffers: MeshBuffers,
    material: MaterialResources,
    default_texture: RenderResource<Texture>,
    default_sampler: Arc<wgpu::Sampler>,
    shader: Arc<GraphicShader>,
    base_color: [f32; 3],
}

struct MeshBuffers {
    vertex_buffer: RenderResource<Buffer>,
    index_buffer: RenderResource<Buffer>,
    index_count: u32,
    // material_index: Option<usize>,
    // _name: Option<String>,
}

struct MaterialResources {
    base_color_texture: Option<RenderResource<Texture>>,
    base_color_sampler: Arc<wgpu::Sampler>,
    _material: Material,
}

pub struct MeshRenderData {
    mesh: AssetHandle<Mesh>,
    material: AssetHandle<Material>,
}

// "/mesh/cerberus/scene.mesh"
impl MeshRenderData {
    pub fn invalid() -> Self {
        Self {
            mesh: AssetHandle::null(),
            material: AssetHandle::null(),
        }
    }

    pub fn new(name: &str) -> Self {
        let mut mesh_path = PathBuf::from(name);
        mesh_path.set_extension("mesh");
        let mut material_path = PathBuf::from(name);
        material_path.set_extension("mat");

        Self {
            mesh: AssetHandle::new(mesh_path.into()),
            material: AssetHandle::new(material_path.into()),
        }
    }
}

impl SimpleMeshRenderer {
    #[profiling::function]
    pub fn from_model(device: &RenderDevice, data: MeshRenderData) -> Self {
        let mat = data.material.get().unwrap();
        let material = Self::create_material_resources(device, &mat);

        let mesh = data.mesh.get().unwrap();
        let mesh_buffers = Self::create_mesh_buffers(device, &mesh);

        let (default_texture, default_sampler) = Self::create_default_texture(device);

        let shader = Self::create_shader();

        Self {
            mesh_buffers,
            material,
            default_texture,
            default_sampler,
            shader: Arc::new(shader),
            base_color: [0.8, 0.8, 0.8],
        }
    }

    pub fn set_base_color(&mut self, color: [f32; 3]) {
        self.base_color = color;
    }

    #[profiling::function]
    fn create_mesh_buffers(device: &RenderDevice, mesh: &Mesh) -> MeshBuffers {
        let device = device.device();

        let vertex_buffer = RenderResource::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex_buffer"),
            contents: mesh.vertices_bytes(),
            usage: wgpu::BufferUsages::VERTEX,
        }));

        let index_buffer = RenderResource::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index_buffer"),
            contents: mesh.indices_bytes(),
            usage: wgpu::BufferUsages::INDEX,
        }));

        MeshBuffers {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as u32,
            // _name: mesh.name.clone(),
        }
    }

    #[profiling::function]
    fn create_material_resources(render_device: &RenderDevice, material: &Material) -> MaterialResources {
        let device = render_device.device();
        
        let base_color_texture = if let Some(texture_data) = &material.base_color_tex {
            let format = texture_data.format.to_wgpu_format();
            let pixels = &texture_data.pixels;

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("base_color"),
                size: wgpu::Extent3d {
                    width: texture_data.width,
                    height: texture_data.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            render_device.queue().write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(texture_data.width * texture_data.format.bytes_per_pixel()),
                    rows_per_image: Some(texture_data.height),
                },
                wgpu::Extent3d {
                    width: texture_data.width,
                    height: texture_data.height,
                    depth_or_array_layers: 1,
                },
            );
            
            Some(RenderResource::new(texture))
        } else {
            None
        };
        
        let base_color_sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lll_r_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));
        
        MaterialResources {
            base_color_texture,
            base_color_sampler,
            _material: material.clone(),
        }
    }

    #[profiling::function]
    fn create_default_texture(render_device: &RenderDevice) -> (RenderResource<wgpu::Texture>, Arc<wgpu::Sampler>) {
        let device = render_device.device();
        
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default White Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let white_pixel = [255u8; 4]; // White RGBA
        render_device.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &white_pixel,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Default Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        
        (RenderResource::new(texture), Arc::new(sampler))
    }

    #[profiling::function]
    fn create_shader() -> GraphicShader {
        define_shader! {
            let shader = Graphic(mesh, "mesh.wgsl", ShaderEntry::Mesh, wgpu::VertexStepMode::Vertex, 1, 1)
        }
        shader.unwrap()
    }

    #[profiling::function]
    pub fn build_render_graph(
        &self, 
        builder: &mut RenderGraphBuilder, 
        view_matrix: glam::Mat4,
        proj_matrix: glam::Mat4,
        model_matrix: glam::Mat4,
        width: u32,
        height: u32,
    ) -> RenderGraphResource<Texture>  {
        let mut output = builder.create("triangle.output", TextureDesc {
            label: Some("mesh output render target"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[wgpu::TextureFormat::Bgra8UnormSrgb],
        });

        let mut depth_buffer = builder.create("mesh.depth", TextureDesc {
            label: Some("mesh depth buffer"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        
        let view_uniform = builder.create("mesh.camera_uniform", wgpu::BufferDescriptor {
            label: Some("Camera Uniform Buffer"),
            size: size_of::<[[f32; 4]; 4]>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let model_uniform = builder.create("mesh.model_uniform", wgpu::BufferDescriptor {
            label: Some("Model Uniform Buffer"),
            size: (size_of::<[[f32; 4]; 4]>() + size_of::<[f32; 3]>() + 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vb = builder.import(
            "mesh.vertex",
            self.mesh_buffers.vertex_buffer.clone(),
            wgpu::BufferUses::empty()
        );
        let ib = builder.import(
            "mesh.index",
            self.mesh_buffers.index_buffer.clone(),
            wgpu::BufferUses::empty()
        );

        // Import default texture
        let default_texture = builder.import(
            "default_texture",
            self.default_texture.clone(),
            wgpu::TextureUses::empty()
        );

        let base_color = if let Some(texture) = &self.material.base_color_texture {
            Some(builder.import(
                "base_color",
                texture.clone(),
                wgpu::TextureUses::empty()
            ))
        } else {
            None
        };

        {
            let mut node = builder.add_graphic_node("mesh_render");

            let view_uniform = node.read(&view_uniform, wgpu::BufferUses::UNIFORM);
            let model_uniform = node.read(&model_uniform, wgpu::BufferUses::UNIFORM);
            let output = node.write(&mut output, wgpu::TextureUses::COLOR_TARGET);
            let depth_buffer = node.write(&mut depth_buffer, wgpu::TextureUses::DEPTH_STENCIL_WRITE);

            let vb_read = node.read(&vb, wgpu::BufferUses::VERTEX);
            let ib_read = node.read(&ib, wgpu::BufferUses::INDEX);

            let default_texture_read = node.read(&default_texture, wgpu::TextureUses::RESOURCE);

            let tex_read = if let Some(texture) = &base_color {
                Some(node.read(texture, wgpu::TextureUses::RESOURCE))
            } else {
                None
            };

            node.setup_pipeline()
                .with_shader(self.shader.clone())
                .with_color(output, Default::default())
                .with_depth_stencil(depth_buffer, DepthStencilInfo {
                    depth_write: true,
                    compare: wgpu::CompareFunction::Greater,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                    depth_load_op: wgpu::LoadOp::Clear(0.0),
                    depth_store_op: wgpu::StoreOp::Store,
                    stencil_load_op: wgpu::LoadOp::Clear(0),
                    stencil_store_op: wgpu::StoreOp::Discard,
                });

            let view_proj = proj_matrix * view_matrix;
            let base_color = self.base_color.into();
            let default_sampler_clone = self.default_sampler.clone();
            let index_count = self.mesh_buffers.index_count;
            let base_color_sampler = self.material.base_color_sampler.clone();

            node.execute(move |ctx, encoder| {
                let view_uniform_data = zenith_build::mesh::ViewUniforms::new(view_proj);
                ctx.write_buffer(&view_uniform, 0, view_uniform_data);
                let model_uniform_data = zenith_build::mesh::ModelUniforms::new(model_matrix, base_color);
                ctx.write_buffer(&model_uniform, 0, model_uniform_data);

                let view_buffer = ctx.get_buffer(&view_uniform);
                let model_buffer = ctx.get_buffer(&model_uniform);

                let mut render_pass = ctx.begin_render_pass(encoder);

                let vertex_buffer = ctx.get_buffer(&vb_read);
                let index_buffer = ctx.get_buffer(&ib_read);

                let (tex, sampler) = if let Some(tex) = tex_read {
                    (ctx.get_texture(&tex), base_color_sampler.clone())
                } else {
                    (ctx.get_texture(&default_texture_read), default_sampler_clone.clone())
                };

                let texture_view = tex.create_view(&wgpu::TextureViewDescriptor::default());

                // Bind all resources for this mesh
                ctx.bind_pipeline(&mut render_pass)
                    .with_binding(0, 0, view_buffer.as_entire_binding())
                    .with_binding(0, 1, model_buffer.as_entire_binding())
                    .with_binding(0, 2, wgpu::BindingResource::TextureView(&texture_view))
                    .with_binding(0, 3, wgpu::BindingResource::Sampler(&sampler))
                    .bind();

                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..index_count, 0, 0..1);
            });
        }

        output
    }
} 