use std::any::Any;
use std::path::{Path, PathBuf};
use bincode::{Decode, Encode};
use bytemuck::{NoUninit, Pod, Zeroable};
use derive_builder::Builder;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use super::{Asset, AssetUrl};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Serialize, Deserialize, Encode, Decode)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coord: [f32; 2],
}

impl Vertex {
    pub fn new(position: Vec3, normal: Vec3, tex_coord: Vec2) -> Self {
        Self {
            position: position.to_array(),
            normal: normal.to_array(),
            tex_coord: tex_coord.to_array(),
        }
    }
}

#[derive(Debug, Clone, Builder, Serialize, Deserialize, Encode, Decode)]
#[builder(setter(into))]
pub struct Mesh<V = Vertex> {
    pub vertices: Vec<V>,
    pub indices: Vec<u32>,
    #[builder(default)]
    #[bincode(with_serde)]
    pub material: Option<usize>,
}

impl<V: NoUninit> Mesh<V> {
    pub fn new(vertices: Vec<V>, indices: Vec<u32>, material: Option<usize>) -> Self {
        Self {
            vertices,
            indices,
            material,
        }
    }
    
    pub fn vertices_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.vertices)
    }

    pub fn indices_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.indices)
    }
}

impl<V: 'static + Send + Sync> Asset for Mesh<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn url(&self, name: &str) -> AssetUrl {
        let mut url = PathBuf::from(name);
        url.set_extension(Self::extension());
        url.into()
    }

    fn extension() -> &'static str {
        "mesh"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub enum TextureFormat {
    R8,
    R8G8,
    R8G8B8A8,
    R16,
    R16G16,
    R16G16B16A16,
    R32G32B32A32Float,
}

impl TextureFormat {
    pub fn bytes_per_pixel(&self) -> u32 {
        match self {
            TextureFormat::R8 => 1,
            TextureFormat::R8G8 => 2,
            TextureFormat::R8G8B8A8 => 4,
            TextureFormat::R16 => 2,
            TextureFormat::R16G16 => 4,
            TextureFormat::R16G16B16A16 => 8,
            TextureFormat::R32G32B32A32Float => 16,
        }
    }

    pub fn to_vk_format(&self) -> ash::vk::Format {
        match self {
            TextureFormat::R8 => ash::vk::Format::R8_UNORM,
            TextureFormat::R8G8 => ash::vk::Format::R8G8_UNORM,
            TextureFormat::R8G8B8A8 => ash::vk::Format::R8G8B8A8_SRGB,
            TextureFormat::R16 => ash::vk::Format::R16_UNORM,
            TextureFormat::R16G16 => ash::vk::Format::R16G16_UNORM,
            TextureFormat::R16G16B16A16 => ash::vk::Format::R16G16B16A16_UNORM,
            TextureFormat::R32G32B32A32Float => ash::vk::Format::R32G32B32A32_SFLOAT,
        }
    }
    
}

#[derive(Debug, Clone, Builder, Serialize, Deserialize, Encode, Decode)]
#[builder(setter(into))]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub pixels: Vec<u8>,
}

impl Asset for Texture {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn url(&self, name: &str) -> AssetUrl {
        let mut url = PathBuf::from(format!("{}_{}_{}", name, self.width, self.height));
        url.set_extension(Self::extension());
        url.into()
    }

    fn extension() -> &'static str {
        "tex"
    }
}

#[derive(Debug, Clone, Builder, Serialize, Deserialize, Encode, Decode)]
#[builder(setter(into))]
pub struct Material {
    #[builder(default = [1., 0., 1., 1.])]
    pub base_color: [f32; 4],
    #[builder(default = 1.0)]
    pub metallic: f32,
    #[builder(default = 0.5)]
    pub roughness: f32,
    #[builder(default = [0., 0., 0.])]
    pub emissive: [f32; 3],

    // TODO: replace with asset path reference
    #[builder(default)]
    #[bincode(with_serde)]
    pub base_color_tex: Option<Texture>,
    #[builder(default)]
    #[bincode(with_serde)]
    pub mra_tex: Option<Texture>,
    #[builder(default)]
    #[bincode(with_serde)]
    pub normal_tex: Option<Texture>,
    #[builder(default)]
    #[bincode(with_serde)]
    pub emissive_tex: Option<Texture>,
}

impl Asset for Material {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn url(&self, name: &str) -> AssetUrl {
        let mut url = PathBuf::from(name);
        url.set_extension(Self::extension());
        url.into()
    }

    fn extension() -> &'static str {
        "mat"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct MeshCollection {
    pub raw_asset_path: PathBuf,
    #[bincode(with_serde)]
    pub meshes: Vec<AssetUrl>,
    #[bincode(with_serde)]
    pub materials: Vec<AssetUrl>,
}

impl Asset for MeshCollection {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn url(&self, name: &str) -> AssetUrl {
        let mut url = PathBuf::from(name);
        url.set_extension(Self::extension());
        url.into()
    }

    fn extension() -> &'static str {
        "mscl"
    }
}

impl MeshCollection {
    pub fn new(raw_asset_path: impl AsRef<Path>) -> Self {
        Self {
            raw_asset_path: raw_asset_path.as_ref().into(),
            meshes: vec![],
            materials: vec![],
        }
    }

    pub fn add_mesh(&mut self, mesh_url: AssetUrl, mat_url: AssetUrl) {
        self.meshes.push(mesh_url);
        self.materials.push(mat_url);
    }

    // "mesh/cerberus/scene.gltf" -> "mesh/cerberus/scene.mscl"
    pub fn asset_url(&self) -> AssetUrl {
        let mut baked_asset_path = self.raw_asset_path.clone();
        baked_asset_path.set_extension(Self::extension());
        baked_asset_path.into()
    }
}