use std::path::{Path, PathBuf};
use anyhow::Result;
use zenith_core::log::info;
use zenith_core::workspace_root;
use crate::gltf::{GltfLoader, RawGltfProcessor};
use crate::hdr::{HdrLoader, RawHdrProcessor};
use crate::{RawResourceBaker, AssetLoadRequest, AssetType, RawResourceLoadRequest, RawResourceLoader, ASSET_REGISTRY, RawResourceLoadRequestBuilder, AssetLoadRequestBuilder, Asset, AssetUrl, deserialize_asset};
use crate::material::Material;
use crate::mesh::{Mesh, MeshCollection};
use crate::texture::Texture;

/// Managing the loading, registering of assets and maintaining assets' cache.
/// Asset lifetime:
///     Load -> Register -> Unregister -> Unload
pub struct AssetManager {
    cache_dir: PathBuf,
    content_dir: PathBuf,
}

const CACHE_VERSION: &str = "9def1275-2dc5-47c7-b4c4-daed2c7d3951";
const CACHE_VERSION_FILE: &str = ".cache_version";

impl AssetManager {
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
    /// # use zenith_asset::manager::AssetManager;
    /// let manager = AssetManager::new();
    /// let gltf_path = "mesh/cerberus/scene.gltf";
    /// manager.request_load(gltf_path).expect("Failed to load asset");
    /// ```
    #[profiling::function]
    pub fn request_load(&self, url: impl Into<PathBuf>) -> Result<()> {
        let url = url.into();
        info!("Loading raw asset {:?}", url);

        if self.should_bake_asset(&url) {
            self.request_load_raw(RawResourceLoadRequestBuilder::default()
                .relative_path(url)
                .build()?)
        } else {
            let extension = url.extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");

            if extension == "hdr" {
                // Load the cached cubemap texture directly
                if let Some(cached_path) = self.find_hdr_cache_file(&url) {
                    let cache_url: AssetUrl = cached_path.into();
                    return self.request_load_asset(AssetLoadRequestBuilder::default()
                        .url(cache_url)
                        .build()?);
                }
                anyhow::bail!("HDR cache file not found for {:?}", url);
            }

            // Default: assume MeshCollection
            let mut url = url;
            url.set_extension(MeshCollection::extension());

            self.request_load_asset(AssetLoadRequestBuilder::default()
                .url(url)
                .build()?)
        }
    }

    #[profiling::function]
    fn should_bake_asset(&self, path: &impl AsRef<Path>) -> bool {
        let raw_path = self.content_dir.join(path.as_ref());
        let extension = raw_path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        if !self.cache_version_matches() {
            return true;
        }

        // For GLTF files, check the mesh collection cache
        if extension == "gltf" {
            let mesh_collection = MeshCollection::new(path);
            let asset_url = mesh_collection.asset_url();
            let cached_file_path = self.cache_dir.join(asset_url.path);

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

            if let Some(cached_path) = self.find_hdr_cache_file(path) {
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
    fn request_load_raw(&self, load_request: RawResourceLoadRequest) -> Result<()> {
        let extension = load_request.relative_path.extension()
            .and_then(|e| e.to_str());

        let raw_content_path = self.content_dir.join(&load_request.relative_path);
        let asset_url = AssetUrl::from(load_request.relative_path.clone());

        match extension {
            Some("gltf") => {
                let raw = GltfLoader::load(&raw_content_path)?;
                RawGltfProcessor::bake(raw, ASSET_REGISTRY.get().unwrap(), &self.cache_dir, &asset_url)?;
            }
            Some("hdr") => {
                let raw = HdrLoader::load(&raw_content_path)?;
                RawHdrProcessor::bake(raw, ASSET_REGISTRY.get().unwrap(), &self.cache_dir, &asset_url)?;
            }
            _ => anyhow::bail!("Unsupported asset format: {:?}", extension),
        }

        self.write_cache_version()?;
        info!("Successfully baked asset {:?}", raw_content_path);
        Ok(())
    }

    #[profiling::function]
    fn request_load_asset(&self, load_request: AssetLoadRequest) -> Result<()> {
        let asset_type = load_request.url.ty();

        let cache_asset_path = self.cache_dir.join(&load_request.url);
        info!("Loading baked asset: {:?}", cache_asset_path);

        // TODO: load dependencies
        // TODO: notice a 1-to-1 mapping between AssetType and static asset type, further abstract the deserialize logic
        if asset_type == AssetType::MeshCollection {
            let asset: MeshCollection = deserialize_asset(&cache_asset_path)?;

            // Register MeshCollection itself so AssetHandle<MeshCollection>::get() works.
            ASSET_REGISTRY
                .get()
                .unwrap()
                .register(load_request.url.clone(), asset.clone());

            for mesh_url in &asset.meshes {
                self.request_load_asset(Self::build_asset_request(mesh_url.clone())?)?;
            }

            for mat_url in &asset.materials {
                self.request_load_asset(Self::build_asset_request(mat_url.clone())?)?;
            }

            return Ok(());
        }

        match asset_type {
            AssetType::Mesh => {
                let asset: Mesh = deserialize_asset(&cache_asset_path)?;
                Self::register_asset(load_request.url, asset);
            }
            AssetType::Texture => {
                let asset: Texture = deserialize_asset(&cache_asset_path)?;
                Self::register_asset(load_request.url, asset);
            }
            AssetType::Material => {
                let asset: Material = deserialize_asset(&cache_asset_path)?;

                let tex_urls = [
                    asset.base_color_tex.clone(),
                    asset.mra_tex.clone(),
                    asset.normal_tex.clone(),
                    asset.emissive_tex.clone(),
                ];

                Self::register_asset(load_request.url.clone(), asset);

                for url in tex_urls.into_iter().flatten() {
                    self.request_load_asset(Self::build_asset_request(url)?)?;
                }
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

    fn register_asset<A: Asset>(url: AssetUrl, asset: A) {
        ASSET_REGISTRY
            .get()
            .unwrap()
            .register(url, asset);
    }

    /// Find a cached cubemap texture file for a given HDR source path.
    /// The cache file is named `{stem}_cubemap_{size}x{size}.tex`.
    fn find_hdr_cache_file(&self, hdr_path: &impl AsRef<Path>) -> Option<PathBuf> {
        let parent = hdr_path.as_ref().parent().unwrap_or(Path::new(""));
        let stem = hdr_path.as_ref().file_stem()?.to_str()?;
        let cache_parent = self.cache_dir.join(parent);
        let prefix = format!("{}_cubemap_", stem);

        let entries = std::fs::read_dir(&cache_parent).ok()?;
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