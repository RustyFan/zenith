use std::any::Any;
use std::path::{Path, PathBuf};
use anyhow::Result;
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

impl<V: 'static + Send + Sync + NoUninit> Asset for Mesh<V> {
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

    fn gpu_size_in_bytes(&self) -> usize {
        self.vertices_bytes().len() + self.indices_bytes().len()
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
    Bc5Unorm,
    Bc7Unorm,
    Bc7Srgb,
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
            TextureFormat::Bc5Unorm | TextureFormat::Bc7Unorm | TextureFormat::Bc7Srgb => 0,
        }
    }

    pub fn is_block_compressed(&self) -> bool {
        matches!(self, TextureFormat::Bc5Unorm | TextureFormat::Bc7Unorm | TextureFormat::Bc7Srgb)
    }

    pub fn block_dimensions(&self) -> (u32, u32) {
        match self {
            TextureFormat::Bc5Unorm | TextureFormat::Bc7Unorm | TextureFormat::Bc7Srgb => (4, 4),
            _ => (1, 1),
        }
    }

    pub fn bytes_per_block(&self) -> u32 {
        match self {
            TextureFormat::Bc5Unorm | TextureFormat::Bc7Unorm | TextureFormat::Bc7Srgb => 16,
            _ => self.bytes_per_pixel(),
        }
    }

    pub fn data_size_in_bytes(&self, width: u32, height: u32) -> usize {
        if self.is_block_compressed() {
            let (bw, bh) = self.block_dimensions();
            let blocks_x = (width + bw - 1) / bw;
            let blocks_y = (height + bh - 1) / bh;
            (blocks_x * blocks_y * self.bytes_per_block()) as usize
        } else {
            (width * height * self.bytes_per_pixel()) as usize
        }
    }

    pub fn to_vk(&self) -> ash::vk::Format {
        match self {
            TextureFormat::R8 => ash::vk::Format::R8_UNORM,
            TextureFormat::R8G8 => ash::vk::Format::R8G8_UNORM,
            TextureFormat::R8G8B8A8 => ash::vk::Format::R8G8B8A8_SRGB,
            TextureFormat::R16 => ash::vk::Format::R16_UNORM,
            TextureFormat::R16G16 => ash::vk::Format::R16G16_UNORM,
            TextureFormat::R16G16B16A16 => ash::vk::Format::R16G16B16A16_UNORM,
            TextureFormat::R32G32B32A32Float => ash::vk::Format::R32G32B32A32_SFLOAT,
            TextureFormat::Bc5Unorm => ash::vk::Format::BC5_UNORM_BLOCK,
            TextureFormat::Bc7Unorm => ash::vk::Format::BC7_UNORM_BLOCK,
            TextureFormat::Bc7Srgb => ash::vk::Format::BC7_SRGB_BLOCK,
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

    fn gpu_size_in_bytes(&self) -> usize {
        self.pixels.len()
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

    // Texture assets referenced by URL (baked separately as `.tex`).
    #[builder(default)]
    #[bincode(with_serde)]
    pub base_color_tex: Option<AssetUrl>,
    #[builder(default)]
    #[bincode(with_serde)]
    pub mra_tex: Option<AssetUrl>,
    #[builder(default)]
    #[bincode(with_serde)]
    pub normal_tex: Option<AssetUrl>,
    #[builder(default)]
    #[bincode(with_serde)]
    pub emissive_tex: Option<AssetUrl>,
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

    fn gpu_size_in_bytes(&self) -> usize {
        size_of_val(&self.base_color) +
            size_of_val(&self.metallic) +
            size_of_val(&self.roughness) +
            size_of_val(&self.emissive)
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

    fn gpu_size_in_bytes(&self) -> usize {
        0
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

    /// Iterate mesh/material pairs. This is fallible because the two lists may be mismatched.
    pub fn iter(&self) -> Result<impl Iterator<Item = (&AssetUrl, &AssetUrl)>> {
        if self.meshes.len() != self.materials.len() {
            anyhow::bail!(
                "MeshCollection meshes/materials length mismatch ({} vs {})",
                self.meshes.len(),
                self.materials.len()
            );
        }

        Ok(self.meshes.iter().zip(self.materials.iter()))
    }

    // "mesh/cerberus/scene.gltf" -> "mesh/cerberus/scene.mscl"
    pub fn asset_url(&self) -> AssetUrl {
        let mut baked_asset_path = self.raw_asset_path.clone();
        baked_asset_path.set_extension(Self::extension());
        baked_asset_path.into()
    }
}