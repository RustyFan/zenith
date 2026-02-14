use std::any::Any;
use std::path::{Path, PathBuf};
use anyhow::Result;
use derive_more::Error;
use zenith_core::log::info;
use zenith_core::{log, workspace_root};
use crate::gltf::{GltfLoader, GltfBaker};
use crate::hdr::{HdrLoader, RawHdrProcessor};
use crate::{AssetBaker, AssetLoadRequest, AssetType, AssetLoader, ASSET_REGISTRY, AssetLoadRequestBuilder, Asset, AssetUrl, deserialize_asset, RawAssetType, serialize_asset};
use crate::material::Material;
use crate::mesh::{Mesh, Scene};
use crate::texture::Texture;

/// Managing the loading, registering of assets.
pub struct AssetRequestor {
    cache_dir: PathBuf,
    content_dir: PathBuf,
}

const CACHE_VERSION: &str = "9def1275-2dc5-47c7-b4c4-daed2c7d3951";
const CACHE_VERSION_FILE: &str = ".cache_version";

#[derive(Debug, Clone, Copy)]
pub enum AssetDirectory {
    Content,
    Asset,
}

impl AssetDirectory {
    pub fn folder_name(&self) -> &str {
        match self {
            AssetDirectory::Content => "content",
            AssetDirectory::Asset => "asset",
        }
    }
}

impl AssetRequestor {
    pub fn new() -> Self {
        let root = workspace_root();
        Self {
            cache_dir: root.to_owned().join("cache/"),
            content_dir: root.join("content/"),
        }
    }

    /// Send a load request to the asset manager.
    /// Loading will complete synchronously before returning.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use zenith_asset::manager::AssetRequestor;
    /// let manager = AssetRequestor::new();
    /// let gltf_path = "mesh/cerberus/scene.gltf";
    /// manager.request_load(gltf_path).expect("Failed to load asset");
    /// ```
    #[profiling::function]
    pub fn request_load(&self, request: AssetLoadRequest) -> Result<()> {
        // let url = request.url.into();

        if self.should_bake_asset(&request) {
            self.request_load_raw(request)?;
        } else {
            self.request_load_asset(request)?;
            // let extension = request.url.as_ref().extension()
            //     .and_then(|e| e.to_str())
            //     .unwrap_or("");
            //
            // if extension == "hdr" {
            //     // Load the cached cubemap texture directly
            //     if let Some(cached_path) = self.find_hdr_cache_file(&request) {
            //         let cache_url: AssetUrl = cached_path.into();
            //         return self.request_load_asset(AssetLoadRequestBuilder::default()
            //             .url(cache_url)
            //             .build()?);
            //     }
            //     anyhow::bail!("HDR cache file not found for {:?}", request.url);
            // }
            //
            // // Default: assume MeshCollection
            // let mut url = url;
            // url.set_extension(Scene::extension());
            //
            // self.request_load_asset(AssetLoadRequestBuilder::default()
            //     .url(url)
            //     .build()?)
        }

        Ok(())
    }

    #[profiling::function]
    fn should_bake_asset(&self, request: &AssetLoadRequest) -> bool {
        let raw_path = request.absolute_raw_asset_path();
        let extension = raw_path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        if !self.cache_version_matches() {
            return true;
        }

        if extension == "gltf" {
            let cached_file_path = request.absolute_asset_path();

            // if no cache had been found, rebake
            if !cached_file_path.exists() {
                return true;
            }

            let asset_metadata = match std::fs::metadata(cached_file_path) {
                Ok(metadata) => metadata,
                Err(_) => return false,
            };

            let source_metadata = match std::fs::metadata(raw_path) {
                Ok(metadata) => metadata,
                Err(_) => return false,
            };

            let asset_last_modified_time = match asset_metadata.modified() {
                Ok(time) => time,
                Err(_) => return false,
            };

            let raw_last_modified_time = match source_metadata.modified() {
                Ok(time) => time,
                Err(_) => return false,
            };

            // if the raw asset had been modified, rebake
            return raw_last_modified_time > asset_last_modified_time;
        }

        // For HDR files, check if a cached cubemap texture exists and is up-to-date
        if extension == "hdr" {
            if !raw_path.exists() {
                return false;
            }

            if let Some(cached_path) = self.find_hdr_cache_file(&request) {
                let cached_abs_path = self.cache_dir.join(&cached_path);
                let cache_modified = std::fs::metadata(&cached_abs_path)
                    .and_then(|m| m.modified())
                    .ok();
                let source_modified = std::fs::metadata(&raw_path)
                    .and_then(|m| m.modified())
                    .ok();

                match (cache_modified, source_modified) {
                    (Some(cache_time), Some(source_time)) => return source_time > cache_time,
                    _ => return true,
                }
            }

            return true;
        }
        
        true
    }

    fn cache_version_matches(&self) -> bool {
        let version_path = self.cache_dir.join(CACHE_VERSION_FILE);
        let contents = match std::fs::read_to_string(version_path) {
            Ok(contents) => contents,
            Err(_) => return false,
        };
        contents.trim() == CACHE_VERSION
    }

    fn write_cache_version(&self) -> Result<()> {
        std::fs::create_dir_all(&self.cache_dir)?;
        let version_path = self.cache_dir.join(CACHE_VERSION_FILE);
        std::fs::write(version_path, CACHE_VERSION)?;
        Ok(())
    }

    #[profiling::function]
    fn request_load_raw(&self, request: AssetLoadRequest) -> Result<()> {
        // TODO: make it close for modification
        let assets = match request.raw_asset_type() {
            RawAssetType::Gltf => {
                GltfLoader::load(&PathBuf::default())?;

                let raw = GltfLoader::load(&request.absolute_raw_asset_path())?;
                GltfBaker::bake(raw, request.url.clone())?
            }
            RawAssetType::Hdr => {
                let raw = HdrLoader::load(&request.absolute_raw_asset_path())?;
                RawHdrProcessor::bake(raw, request.url.clone())?
            }
            _ => anyhow::bail!("Unsupported asset format: {:?}", request.url),
        };

        for asset in assets {
            let url = asset.url().to_owned();
            let ty = url.asset_type();

            let asset = asset as Box<dyn Any>;

            // TODO: make it close for modification
            match ty {
                AssetType::Mesh => {
                    let asset = *asset.downcast::<Mesh>().unwrap();
                    serialize_asset(&asset, &workspace_root()
                        .join(AssetDirectory::Asset.folder_name())
                        .join(url))?;
                    Self::register_asset(asset);
                },
                AssetType::Texture => {
                    let asset = *asset.downcast::<Texture>().unwrap();
                    serialize_asset(&asset, &workspace_root()
                        .join(AssetDirectory::Asset.folder_name())
                        .join(url))?;
                    Self::register_asset(asset);
                },
                AssetType::Material => {
                    let asset = *asset.downcast::<Material>().unwrap();
                    serialize_asset(&asset, &workspace_root()
                        .join(AssetDirectory::Asset.folder_name())
                        .join(url))?;
                    Self::register_asset(asset);
                },
                AssetType::Scene => {
                    let asset = *asset.downcast::<Scene>().unwrap();
                    serialize_asset(&asset, &workspace_root()
                        .join(AssetDirectory::Asset.folder_name())
                        .join(url))?;
                    Self::register_asset(asset);
                },
            }

        }

        self.write_cache_version()?;
        info!("Asset {:?} baked successfully.", request.raw_asset_path);
        Ok(())
    }

    #[profiling::function]
    fn request_load_asset(&self, request: AssetLoadRequest) -> Result<()> {
        let asset_type = request.url.asset_type();
        let asset_path = request.absolute_asset_path();
        info!("Loading baked asset: {:?}", asset_path);

        // TODO: load dependencies
        // TODO: notice a 1-to-1 mapping between AssetType and static asset type, further abstract the deserialize logic
        if asset_type == AssetType::Scene {
            let mut asset: Scene = deserialize_asset(&asset_path)?;
            asset.url = request.url.clone();

            for mesh_url in &asset.meshes {
                self.request_load_asset(Self::build_asset_request(mesh_url.clone())?)?;
            }

            Self::register_asset(asset);

            return Ok(());
        }

        match asset_type {
            AssetType::Mesh => {
                let mut asset: Mesh = deserialize_asset(&asset_path)?;
                asset.url = request.url.clone();
                if let Some(mat) = &asset.material {
                    self.request_load_asset(Self::build_asset_request(mat.clone())?)?;
                }
                Self::register_asset(asset);
            }
            AssetType::Texture => {
                let mut asset: Texture = deserialize_asset(&asset_path)?;
                asset.url = request.url.clone();
                Self::register_asset(asset);
            }
            AssetType::Material => {
                let mut asset: Material = deserialize_asset(&asset_path)?;
                asset.url = request.url.clone();

                let tex_urls = [
                    asset.base_color_tex.clone(),
                    asset.mra_tex.clone(),
                    asset.normal_tex.clone(),
                    asset.emissive_tex.clone(),
                ];
                for url in tex_urls.into_iter().flatten() {
                    self.request_load_asset(Self::build_asset_request(url)?)?;
                }

                Self::register_asset(asset);
            }
            _ => unreachable!()
        }

        Ok(())
    }

    fn build_asset_request(url: AssetUrl) -> Result<AssetLoadRequest> {
        Ok(AssetLoadRequestBuilder::default()
            .url(url)
            .build()?)
    }

    fn register_asset<A: Asset>(asset: A) {
        ASSET_REGISTRY
            .get()
            .unwrap()
            .register(asset);
    }

    /// Find a cached cubemap texture file for a given HDR source path.
    /// The cache file is named `{stem}_cubemap.tex`.
    fn find_hdr_cache_file(&self, request: &AssetLoadRequest) -> Option<PathBuf> {
        let parent = request.url.as_ref().parent().unwrap_or(Path::new(""));
        let stem = request.url.as_ref().file_stem()?.to_str()?;
        // let cache_parent = self.cache_dir.join(parent);
        let asset_path = request.absolute_asset_path();
        let prefix = format!("{}_cubemap", stem);

        let entries = std::fs::read_dir(&asset_path).ok()?;
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(&prefix) && name_str.ends_with(".tex") {
                // Return relative path (without cache_dir prefix) for use as AssetUrl
                return Some(parent.join(name_str.into_owned()));
            }
        }
        None
    }
}