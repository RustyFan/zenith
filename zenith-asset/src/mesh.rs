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