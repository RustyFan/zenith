use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use gltf::{buffer::Data as BufferData, image::Data as ImageData, Document, Primitive};
use ispc_texcomp::{RgbaSurface, bc5, bc7};
use zenith_core::file::load_with_memory_mapping;
use zenith_core::log;
use zenith_core::log::info;
use crate::render::{Material, MaterialBuilder, Mesh, MeshBuilder, MeshCollection, TextureBuilder, TextureFormat, Texture, Vertex};
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

#[derive(Clone, Copy, Default)]
struct TextureUsageMask(u8);

impl TextureUsageMask {
    const BASE_COLOR: u8 = 1 << 0;
    const MRA: u8 = 1 << 1;
    const NORMAL: u8 = 1 << 2;
    const EMISSIVE: u8 = 1 << 3;

    fn add_base_color(&mut self) {
        self.0 |= Self::BASE_COLOR;
    }

    fn add_mra(&mut self) {
        self.0 |= Self::MRA;
    }

    fn add_normal(&mut self) {
        self.0 |= Self::NORMAL;
    }

    fn add_emissive(&mut self) {
        self.0 |= Self::EMISSIVE;
    }

    fn has(&self, flag: u8) -> bool {
        (self.0 & flag) != 0
    }
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
        meshes_url: &mut Vec<(AssetUrl, Option<usize>)>,
        mesh_counter: &mut u32,
        main_url: &str,
    ) -> Result<()> {
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                // TODO: abstract asset serialize and register logic
                let mesh_asset = Self::bake_mesh(&primitive, buffers)?;
                let url = {
                    let main = PathBuf::from(main_url);
                    let parent = main.parent().unwrap_or_else(|| Path::new(""));
                    let stem = main
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("mesh");
                    let idx = *mesh_counter;
                    *mesh_counter += 1;
                    let mut p = parent.join(format!("{stem}_mesh{idx}"));
                    p.set_extension(Mesh::<Vertex>::extension());
                    AssetUrl::from(p)
                };

                let asset_serialize_path = base_directory.join(&url);
                serialize_asset(&mesh_asset, &asset_serialize_path)?;

                meshes_url.push((url.clone(), mesh_asset.material));
                registry.register(url, mesh_asset);
            }
        }

        for child in node.children() {
            Self::process_node(base_directory, &child, buffers, registry, meshes_url, mesh_counter, main_url)?;
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

        let material = primitive.material().index();

        let mesh = MeshBuilder::default()
            .vertices(vertices)
            .indices(indices)
            .material(material)
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
    fn bake_materials(gltf: &Document, texture_urls: &[AssetUrl]) -> Result<Vec<Material>> {
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
                if let Some(url) = texture_urls.get(image_index) {
                    builder.base_color_tex(url.clone());
                }
            }

            if let Some(texture) = pbr.metallic_roughness_texture() {
                let image_index = texture.texture().source().index();
                if let Some(url) = texture_urls.get(image_index) {
                    builder.mra_tex(url.clone());
                }
            }

            if let Some(texture) = material.normal_texture() {
                let image_index = texture.texture().source().index();
                if let Some(url) = texture_urls.get(image_index) {
                    builder.normal_tex(url.clone());
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
                if let Some(url) = texture_urls.get(image_index) {
                    builder.emissive_tex(url.clone());
                }
            }

            materials.push(builder.build()?);
        }

        if materials.is_empty() {
            materials.push(MaterialBuilder::default().build()?);
        }

        Ok(materials)
    }

    fn collect_texture_usages(gltf: &Document, image_count: usize) -> Vec<TextureUsageMask> {
        let mut usages = vec![TextureUsageMask::default(); image_count];

        for material in gltf.materials() {
            let pbr = material.pbr_metallic_roughness();
            if let Some(texture) = pbr.base_color_texture() {
                let image_index = texture.texture().source().index();
                if let Some(slot) = usages.get_mut(image_index) {
                    slot.add_base_color();
                }
            }
            if let Some(texture) = pbr.metallic_roughness_texture() {
                let image_index = texture.texture().source().index();
                if let Some(slot) = usages.get_mut(image_index) {
                    slot.add_mra();
                }
            }
            if let Some(texture) = material.normal_texture() {
                let image_index = texture.texture().source().index();
                if let Some(slot) = usages.get_mut(image_index) {
                    slot.add_normal();
                }
            }
            if let Some(texture) = material.emissive_texture() {
                let image_index = texture.texture().source().index();
                if let Some(slot) = usages.get_mut(image_index) {
                    slot.add_emissive();
                }
            }
        }

        usages
    }

    #[profiling::function]
    fn create_texture_from_gltf_image(image_data: &ImageData, usage: TextureUsageMask) -> Result<crate::render::Texture> {
        if let Some((pixels, format)) = Self::compress_texture_if_possible(image_data, usage) {
            return TextureBuilder::default()
                .width(image_data.width)
                .height(image_data.height)
                .format(format)
                .pixels(pixels)
                .build()
                .map_err(|e| anyhow!("Failed to build texture: {}", e));
        }

        // Fallback: store uncompressed pixels.
        let (pixels, texture_format) = Self::convert_gltf_pixels(image_data);
        log::warn!("Uncompressed glTF image: format[{:?}]", texture_format);

        TextureBuilder::default()
            .width(image_data.width)
            .height(image_data.height)
            .format(texture_format)
            .pixels(pixels)
            .build()
            .map_err(|e| anyhow!("Failed to build texture: {}", e))
    }

    fn compress_texture_if_possible(image_data: &ImageData, usage: TextureUsageMask) -> Option<(Vec<u8>, TextureFormat)> {
        if !Self::is_8bit_format(image_data.format) {
            return None;
        }

        if usage.has(TextureUsageMask::NORMAL) {
            let rg_pixels = Self::rg8_from_gltf(image_data)?;
            let rgba_pixels = Self::rg8_to_rgba(&rg_pixels);
            let compressed = Self::compress_bc5(&rgba_pixels, image_data.width, image_data.height);
            return Some((compressed, TextureFormat::Bc5Unorm));
        }

        if usage.has(TextureUsageMask::BASE_COLOR) || usage.has(TextureUsageMask::EMISSIVE) {
            let rgba_pixels = Self::rgba8_from_gltf(image_data)?;
            let compressed = Self::compress_bc7(&rgba_pixels, image_data.width, image_data.height, true);
            return Some((compressed, TextureFormat::Bc7Srgb));
        }

        if usage.has(TextureUsageMask::MRA) {
            let rgba_pixels = Self::rgba8_from_gltf(image_data)?;
            let compressed = Self::compress_bc7(&rgba_pixels, image_data.width, image_data.height, false);
            return Some((compressed, TextureFormat::Bc7Unorm));
        }

        None
    }

    fn is_8bit_format(format: gltf::image::Format) -> bool {
        matches!(
            format,
            gltf::image::Format::R8
                | gltf::image::Format::R8G8
                | gltf::image::Format::R8G8B8
                | gltf::image::Format::R8G8B8A8
        )
    }

    fn rgba8_from_gltf(data: &ImageData) -> Option<Vec<u8>> {
        match data.format {
            gltf::image::Format::R8G8B8A8 => Some(data.pixels.clone()),
            gltf::image::Format::R8G8B8 => {
                let mut rgba = Vec::with_capacity(data.pixels.len() * 4 / 3);
                for chunk in data.pixels.chunks(3) {
                    rgba.extend_from_slice(chunk);
                    rgba.push(255);
                }
                Some(rgba)
            }
            gltf::image::Format::R8G8 => {
                let mut rgba = Vec::with_capacity(data.pixels.len() * 2);
                for chunk in data.pixels.chunks(2) {
                    rgba.push(chunk[0]);
                    rgba.push(chunk[1]);
                    rgba.push(0);
                    rgba.push(255);
                }
                Some(rgba)
            }
            gltf::image::Format::R8 => {
                let mut rgba = Vec::with_capacity(data.pixels.len() * 4);
                for value in data.pixels.iter().copied() {
                    rgba.push(value);
                    rgba.push(0);
                    rgba.push(0);
                    rgba.push(255);
                }
                Some(rgba)
            }
            _ => None,
        }
    }

    fn rg8_from_gltf(data: &ImageData) -> Option<Vec<u8>> {
        match data.format {
            gltf::image::Format::R8G8 => Some(data.pixels.clone()),
            gltf::image::Format::R8G8B8 => {
                let mut rg = Vec::with_capacity(data.pixels.len() * 2 / 3);
                for chunk in data.pixels.chunks(3) {
                    rg.push(chunk[0]);
                    rg.push(chunk[1]);
                }
                Some(rg)
            }
            gltf::image::Format::R8G8B8A8 => {
                let mut rg = Vec::with_capacity(data.pixels.len() / 2);
                for chunk in data.pixels.chunks(4) {
                    rg.push(chunk[0]);
                    rg.push(chunk[1]);
                }
                Some(rg)
            }
            gltf::image::Format::R8 => {
                let mut rg = Vec::with_capacity(data.pixels.len() * 2);
                for value in data.pixels.iter().copied() {
                    rg.push(value);
                    rg.push(0);
                }
                Some(rg)
            }
            _ => None,
        }
    }

    fn rg8_to_rgba(pixels: &[u8]) -> Vec<u8> {
        let mut rgba = Vec::with_capacity(pixels.len() * 2);
        for chunk in pixels.chunks(2) {
            rgba.push(chunk[0]);
            rgba.push(chunk[1]);
            rgba.push(0);
            rgba.push(255);
        }
        rgba
    }

    fn compress_bc7(pixels: &[u8], width: u32, height: u32, use_alpha: bool) -> Vec<u8> {
        let output_size = bc7::calc_output_size(width, height);
        let mut output = vec![0u8; output_size];
        let surface = RgbaSurface {
            data: pixels,
            width,
            height,
            stride: width * 4,
        };
        let settings = if use_alpha {
            bc7::alpha_fast_settings()
        } else {
            bc7::opaque_fast_settings()
        };
        bc7::compress_blocks_into(&settings, &surface, &mut output);
        output
    }

    fn compress_bc5(pixels: &[u8], width: u32, height: u32) -> Vec<u8> {
        let output_size = bc5::calc_output_size(width, height);
        let mut output = vec![0u8; output_size];
        let surface = RgbaSurface {
            data: pixels,
            width,
            height,
            stride: width * 4,
        };
        bc5::compress_blocks_into(&surface, &mut output);
        output
    }

    #[profiling::function]
    fn convert_gltf_pixels(data: &ImageData) -> (Vec<u8>, TextureFormat) {
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

        let texture_usage_by_image = Self::collect_texture_usages(&gltf, images.len());

        // Bake textures once (shared by materials via AssetUrl references).
        let texture_urls: Vec<AssetUrl> = images
            .iter()
            .enumerate()
            .map(|(idx, img)| {
                let usage = texture_usage_by_image.get(idx).copied().unwrap_or_default();
                let tex = Self::create_texture_from_gltf_image(img, usage)?;

                let main = PathBuf::from(asset_url);
                let parent = main.parent().unwrap_or_else(|| Path::new(""));
                let stem = main
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("asset");

                let mut p = parent.join(format!("{stem}_tex{idx}_{}x{}", tex.width, tex.height));
                p.set_extension(Texture::extension());
                let url: AssetUrl = p.into();

                let asset_serialize_path = base_directory.join(&url);
                serialize_asset(&tex, &asset_serialize_path)?;
                registry.register(url.clone(), tex);

                Ok(url)
            })
            .collect::<Result<Vec<_>>>()?;

        let materials = Self::bake_materials(&gltf, &texture_urls)?;
        let mut material_urls = Vec::with_capacity(materials.len());
        for (mat_idx, material) in materials.into_iter().enumerate() {
            // TODO: abstract asset serialize and register logic
            let url = {
                let main = PathBuf::from(asset_url);
                let parent = main.parent().unwrap_or_else(|| Path::new(""));
                let stem = main
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("material");
                let mut p = parent.join(format!("{stem}_mat{mat_idx}"));
                p.set_extension(Material::extension());
                AssetUrl::from(p)
            };

            let asset_serialize_path = base_directory.join(&url);
            serialize_asset(&material, &asset_serialize_path)?;

            material_urls.push(url.clone());
            registry.register(url, material);
        }

        let mut meshes_urls: Vec<(AssetUrl, Option<usize>)> = Vec::new();
        let mut mesh_counter: u32 = 0;
        for scene in gltf.scenes() {
            for node in scene.nodes() {
                Self::process_node(&base_directory, &node, &buffers, registry, &mut meshes_urls, &mut mesh_counter, asset_url)?;
            }
        }

        let mut mesh_collection = MeshCollection::new(&url);
        for (mesh_url, mat_index) in meshes_urls.into_iter() {
            let idx = mat_index.unwrap_or(0);
            let mat_url = material_urls
                .get(idx)
                .cloned()
                .unwrap_or_else(|| material_urls.first().cloned().unwrap());
            mesh_collection.add_mesh(mesh_url, mat_url);
        }

        let mesh_collection_url = mesh_collection.url(asset_url);
        let asset_serialize_path = base_directory.join(&mesh_collection_url);
        serialize_asset(&mesh_collection, &asset_serialize_path)?;

        // Also register the collection so users can immediately get it via AssetHandle<MeshCollection>.
        registry.register(mesh_collection_url.clone(), mesh_collection.clone());

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