use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use gltf::{buffer::Data as BufferData, image::Data as ImageData, Document, Primitive};
use zenith_core::file::load_with_memory_mapping;
use zenith_core::log::info;
use crate::render::{Material, MaterialBuilder, Mesh, MeshBuilder, MeshCollection, TextureBuilder, TextureFormat, Vertex};
use crate::{Asset, RawResourceBaker, AssetRegistry, RawResource, RawResourceLoader, AssetUrl, serialize_asset};

#[derive(Debug, Clone)]
pub struct GltfLoader;

impl GltfLoader {
    pub fn new() -> Self {
        Self
    }
}

pub struct RawGltf {
    path: PathBuf,
    gltf: gltf::Gltf,
    buffers: Vec<BufferData>,
    images: Vec<ImageData>,
}

impl RawResource for RawGltf {
    fn load_path(&self) -> &Path {
        self.path.as_path()
    }
}

impl RawResourceLoader for GltfLoader {
    type Raw = RawGltf;

    #[profiling::function]
    fn load(path: &Path) -> Result<Self::Raw> {
        let mmap = load_with_memory_mapping(path)?;

        let gltf = gltf::Gltf::from_slice(&mmap)
            .map_err(|e| anyhow!("Failed to parse GLTF: {}", e))?;

        let mut raw = RawGltf {
            path: path.to_owned(),
            gltf,
            buffers: vec![],
            images: vec![],
        };
        
        Self::load_gltf(path, &mut raw)?;

        Ok(raw)
    }
}

pub struct RawGltfProcessor;

impl RawGltfProcessor {
    pub fn new() -> Self {
        Self
    }
}

impl RawGltfProcessor {
    #[profiling::function]
    fn process_node(
        base_directory: &PathBuf,
        node: &gltf::Node,
        buffers: &[BufferData],
        registry: &AssetRegistry,
        meshes_url: &mut Vec<AssetUrl>,
        main_url: &str,
    ) -> Result<()> {
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                // TODO: abstract asset serialize and register logic
                let mesh_asset = Self::bake_mesh(&primitive, buffers)?;
                let url = mesh_asset.url(&main_url);

                let asset_serialize_path = base_directory.join(&url);
                serialize_asset(&mesh_asset, &asset_serialize_path)?;

                meshes_url.push(url.clone());
                registry.register(url, mesh_asset);
            }
        }

        for child in node.children() {
            Self::process_node(base_directory, &child, buffers, registry, meshes_url, main_url)?;
        }

        Ok(())
    }

    #[profiling::function]
    fn bake_mesh(
        primitive: &Primitive,
        buffers: &[BufferData],
    ) -> Result<Mesh> {
        let reader = primitive.reader(|buffer| Some(&*buffers[buffer.index()]));

        let positions = reader
            .read_positions()
            .ok_or(anyhow!("Missing positions"))?
            .collect::<Vec<_>>();

        let normals = if let Some(normals) = reader.read_normals() {
            normals.collect::<Vec<_>>()
        } else {
            // Generate flat normals if missing
            Self::generate_flat_normals(&positions)?
        };

        let tex_coords = if let Some(tex_coords) = reader.read_tex_coords(0) {
            tex_coords.into_f32().collect::<Vec<_>>()
        } else {
            // Generate default UV coordinates
            vec![[0.0, 0.0]; positions.len()]
        };

        let indices = reader
            .read_indices()
            .ok_or(anyhow!("Missing indices"))?
            .into_u32()
            .collect::<Vec<_>>();

        if positions.len() != normals.len() || positions.len() != tex_coords.len() {
            return Err(anyhow!("Vertex attribute count mismatch"));
        }

        let vertices: Vec<Vertex> = positions
            .into_iter()
            .zip(normals.into_iter())
            .zip(tex_coords.into_iter())
            .map(|((pos, norm), uv)| {
                Vertex::new(
                    glam::Vec3::from_array(pos),
                    glam::Vec3::from_array(norm),
                    glam::Vec2::from_array(uv),
                )
            })
            .collect();

        let mesh = MeshBuilder::default()
            .vertices(vertices)
            .indices(indices)
            .build()?;

        Ok(mesh)
    }

    #[profiling::function]
    fn generate_flat_normals(positions: &Vec<[f32; 3]>) -> Result<Vec<[f32; 3]>> {
        if positions.len() % 3 != 0 {
            return Err(anyhow!("Position count must be divisible by 3 for flat normals"));
        }

        let mut normals = vec![[0.0, 0.0, 0.0]; positions.len()];

        for i in (0..positions.len()).step_by(3) {
            let v0 = glam::Vec3::from_array(positions[i]);
            let v1 = glam::Vec3::from_array(positions[i + 1]);
            let v2 = glam::Vec3::from_array(positions[i + 2]);

            let normal = (v1 - v0).cross(v2 - v0).normalize();

            normals[i] = normal.to_array();
            normals[i + 1] = normal.to_array();
            normals[i + 2] = normal.to_array();
        }

        Ok(normals)
    }

    #[profiling::function]
    fn bake_materials(gltf: &Document, images: &[ImageData]) -> Result<Vec<Material>> {
        let mut materials = Vec::new();

        for material in gltf.materials() {
            let pbr = material.pbr_metallic_roughness();

            let mut builder = MaterialBuilder::default();
            builder.base_color(pbr.base_color_factor())
                .metallic(pbr.metallic_factor())
                .roughness(pbr.roughness_factor())
                .emissive(material.emissive_factor());

            if let Some(texture) = pbr.base_color_texture() {
                let image_index = texture.texture().source().index();
                if let Some(image_data) = images.get(image_index) {
                    let tex = Self::create_texture_from_gltf_image(image_data)?;
                    builder.base_color_tex(tex);
                }
            }

            if let Some(texture) = pbr.metallic_roughness_texture() {
                let image_index = texture.texture().source().index();
                if let Some(image_data) = images.get(image_index) {
                    let tex = Self::create_texture_from_gltf_image(image_data)?;
                    builder.mra_tex(tex);
                }
            }

            if let Some(texture) = material.normal_texture() {
                let image_index = texture.texture().source().index();
                if let Some(image_data) = images.get(image_index) {
                    let tex = Self::create_texture_from_gltf_image(image_data)?;
                    builder.normal_tex(tex);
                }
            }

            // if let Some(texture) = material.occlusion_texture() {
            //     let image_index = texture.texture().source().index();
            //     if let Some(image_data) = images.get(image_index) {
            //         pbr_material.textures.occlusion = Some(TextureData {
            //             pixels: image_data.pixels.clone(),
            //             width: image_data.width,
            //             height: image_data.height,
            //             format: image_data.format,
            //         });
            //     }
            // }

            if let Some(texture) = material.emissive_texture() {
                let image_index = texture.texture().source().index();
                if let Some(image_data) = images.get(image_index) {
                    let tex = Self::create_texture_from_gltf_image(image_data)?;
                    builder.emissive_tex(tex);
                }
            }

            materials.push(builder.build()?);
        }

        if materials.is_empty() {
            materials.push(MaterialBuilder::default().build()?);
        }

        Ok(materials)
    }

    #[profiling::function]
    fn create_texture_from_gltf_image(image_data: &ImageData) -> Result<crate::render::Texture> {
        // Convert GLTF format to wgpu-compatible format and pixels
        let (wgpu_pixels, texture_format) = Self::convert_gltf_pixels_to_wgpu(image_data);

        TextureBuilder::default()
            .width(image_data.width)
            .height(image_data.height)
            .format(texture_format)
            .pixels(wgpu_pixels)
            .build()
            .map_err(|e| anyhow!("Failed to build texture: {}", e))
    }

    #[profiling::function]
    fn convert_gltf_pixels_to_wgpu(data: &ImageData) -> (Vec<u8>, TextureFormat) {
        match data.format {
            gltf::image::Format::R8G8B8 => {
                // Convert RGB to RGBA
                let mut rgba_data = Vec::with_capacity(data.pixels.len() * 4 / 3);
                for chunk in data.pixels.chunks(3) {
                    rgba_data.extend_from_slice(chunk);
                    rgba_data.push(255); // Add alpha = 1.0
                }
                (rgba_data, TextureFormat::R8G8B8A8)
            }
            gltf::image::Format::R16G16B16 => {
                // Convert RGB16 to RGBA16
                let mut rgba_data = Vec::with_capacity(data.pixels.len() * 8 / 6);
                for chunk in data.pixels.chunks(6) { // 6 bytes = 3 * 16-bit values
                    rgba_data.extend_from_slice(chunk);
                    rgba_data.extend_from_slice(&[255, 255]); // Add alpha = 1.0 (16-bit)
                }
                (rgba_data, TextureFormat::R16G16B16A16)
            }
            gltf::image::Format::R32G32B32FLOAT => {
                // Convert RGB32F to RGBA32F
                let mut rgba_data = Vec::with_capacity(data.pixels.len() * 16 / 12);
                for chunk in data.pixels.chunks(12) { // 12 bytes = 3 * 32-bit floats
                    rgba_data.extend_from_slice(chunk);
                    rgba_data.extend_from_slice(&1.0f32.to_le_bytes()); // Add alpha = 1.0 (32-bit float)
                }
                (rgba_data, TextureFormat::R32G32B32A32Float)
            }
            gltf::image::Format::R8 => {
                (data.pixels.clone(), TextureFormat::R8)
            }
            gltf::image::Format::R8G8 => {
                (data.pixels.clone(), TextureFormat::R8G8)
            }
            gltf::image::Format::R8G8B8A8 => {
                (data.pixels.clone(), TextureFormat::R8G8B8A8)
            }
            gltf::image::Format::R16 => {
                (data.pixels.clone(), TextureFormat::R16)
            }
            gltf::image::Format::R16G16 => {
                (data.pixels.clone(), TextureFormat::R16G16)
            }
            gltf::image::Format::R16G16B16A16 => {
                (data.pixels.clone(), TextureFormat::R16G16B16A16)
            }
            gltf::image::Format::R32G32B32A32FLOAT => {
                (data.pixels.clone(), TextureFormat::R32G32B32A32Float)
            }
        }
    }
}

impl RawResourceBaker for RawGltfProcessor {
    type Raw = RawGltf;

    #[profiling::function]
    fn bake(raw: Self::Raw, registry: &AssetRegistry, base_directory: &PathBuf, url: &AssetUrl) -> Result<()> {
        let RawGltf {
            gltf,
            buffers,
            images,
            ..
        } = raw;

        let asset_url = url.path.to_str().ok_or(anyhow!(format!("Invalid asset url: {:?}", url)))?;

        let materials = Self::bake_materials(&gltf, &images)?;
        let mut material_urls = Vec::with_capacity(materials.len());
        for material in materials {
            // TODO: abstract asset serialize and register logic
            let url = material.url(asset_url);

            let asset_serialize_path = base_directory.join(&url);
            serialize_asset(&material, &asset_serialize_path)?;

            material_urls.push(url.clone());
            registry.register(url, material);
        }

        let mut meshes_urls = Vec::with_capacity(material_urls.len());
        for scene in gltf.scenes() {
            for node in scene.nodes() {
                Self::process_node(&base_directory, &node, &buffers, registry, &mut meshes_urls, asset_url)?;
            }
        }

        assert_eq!(meshes_urls.len(), material_urls.len());

        let mut mesh_collection = MeshCollection::new(&url);
        for (mat, mesh) in material_urls.into_iter().zip(meshes_urls.into_iter()) {
            mesh_collection.add_mesh(mesh, mat);
        }

        let mesh_collection_url = mesh_collection.url(asset_url);
        let asset_serialize_path = base_directory.join(&mesh_collection_url);
        serialize_asset(&mesh_collection, &asset_serialize_path)?;

        info!("[{}] is loaded and serialized.", asset_url);
        info!("{:?}", mesh_collection);

        Ok(())
    }
}

impl GltfLoader {
    #[profiling::function]
    fn load_gltf<P: AsRef<Path>>(path: P, raw: &mut RawGltf) -> Result<()> {
        let base_dir = path.as_ref().parent().ok_or(anyhow!("Invalid gltf load path."))?;

        let buffer_count = raw.gltf.buffers().len();
        let image_count = raw.gltf.images().len();

        raw.buffers.clear();
        raw.buffers.reserve(buffer_count);

        for buffer in raw.gltf.buffers() {
            match buffer.source() {
                gltf::buffer::Source::Uri(uri) => {
                    if uri.starts_with("data:") {
                        info!("inspecting gltf buffer uri: {:?}", uri);

                        let mut blob = None;
                        let data = BufferData::from_source_and_blob(buffer.source(), None, &mut blob)
                            .map_err(|e| anyhow!("Failed to decode data URI: {}", e))?;
                        
                        raw.buffers.push(data);
                    } else {
                        info!("inspecting gltf buffer uri: {:?}", uri);

                        let buffer_path = base_dir.join(uri);
                        let mmap = load_with_memory_mapping(&buffer_path)?;

                        raw.buffers.push(BufferData(mmap[..].to_vec()));
                    }
                }
                gltf::buffer::Source::Bin => {
                    return Err(anyhow!("Unexpected binary chunk in .gltf file"));
                }
            }
        }

        raw.images.clear();
        raw.images.reserve(image_count);

        for image in raw.gltf.images() {
            match image.source() {
                gltf::image::Source::Uri { uri, .. } => {
                    if uri.starts_with("data:") {
                        info!("inspecting gltf image uri: {:?}", uri);

                        let data = ImageData::from_source(image.source(), None, &raw.buffers)
                            .map_err(|e| anyhow!("Failed to decode image data uri: {}", e))?;
                        
                        raw.images.push(data);
                    } else {
                        info!("inspecting gltf image uri: {:?}", uri);

                        let image_path = base_dir.join(uri);
                        let uri = uri.to_owned();
                        let mmap = load_with_memory_mapping(&image_path)?;

                        raw.images.push(Self::decode_image(&mmap, &uri).expect("Failed to decode gltf image"));
                    }
                }
                gltf::image::Source::View { .. } => {
                    let data = ImageData::from_source(image.source(), None, &raw.buffers)
                        .map_err(|e| anyhow!("Failed to decode embedded image: {}", e))?;
                    
                    raw.images.push(data);
                }
            }
        }

        Ok(())
    }

    #[profiling::function]
    fn decode_image(data: &[u8], filename: &str) -> Result<ImageData> {
        // Fast path: try to guess format from magic bytes first (no file extension parsing)
        let format = image::guess_format(data).unwrap_or_else(|_| {
            // Fallback: use file extension
            let extension = Path::new(filename)
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("")
                .to_lowercase();
                
            match extension.as_str() {
                "png" => image::ImageFormat::Png,
                "jpg" | "jpeg" => image::ImageFormat::Jpeg,
                "bmp" => image::ImageFormat::Bmp,
                "tga" => image::ImageFormat::Tga,
                "tiff" | "tif" => image::ImageFormat::Tiff,
                "webp" => image::ImageFormat::WebP,
                _ => image::ImageFormat::Png, // Default fallback
            }
        });
        
        // Use optimized image loading with format hint
        let img = image::load_from_memory_with_format(data, format)
            .map_err(|e| anyhow!("Failed to decode image {}: {}", filename, e))?;
            
        // Convert to RGBA8 efficiently
        let rgba_img = match img {
            image::DynamicImage::ImageRgba8(rgba) => rgba,
            other => other.into_rgba8(),
        };
        
        let (width, height) = rgba_img.dimensions();
        
        Ok(ImageData {
            pixels: rgba_img.into_raw(),
            format: gltf::image::Format::R8G8B8A8,
            width,
            height,
        })
    }
}