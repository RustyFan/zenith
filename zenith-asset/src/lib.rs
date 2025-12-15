use std::any::{Any, TypeId};
use std::fs::File;
use std::io::Write;
use std::marker::PhantomData;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use anyhow::{anyhow, Result};
use bincode::Encode;
use derive_builder::Builder;
use derive_more::From;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use zenith_core::collections::hashmap::HashMap;
use zenith_core::file::load_with_memory_mapping;

pub mod render;
pub mod manager;
pub mod gltf_loader;

static ASSET_REGISTRY: OnceLock<AssetRegistry> = OnceLock::new();

pub fn initialize() -> Result<()> {
    ASSET_REGISTRY.set(AssetRegistry::new()).map_err(|_| anyhow!("Failed to initialize asset registry!"))
}

type AssetId = (AssetUrl, TypeId);
type AssetMap = HashMap<AssetId, Arc<dyn Asset>>;

#[derive(Default)]
pub struct AssetRegistry {
    assets_map: RwLock<AssetMap>,
}

unsafe impl Send for AssetRegistry {}
unsafe impl Sync for AssetRegistry {}

impl AssetRegistry {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    /// Register an asset.
    pub fn register<A: Asset>(&self, url: impl Into<AssetUrl>, asset: A) {
        let key = (url.into(), TypeId::of::<A>());
        self.assets_map.write().insert(key, Arc::new(asset));
    }

    /// Unregister an asset, return true if this asset was exists.
    pub fn unregister<A: Asset>(&self, url: impl Into<AssetUrl>) -> bool {
        let key = (url.into(), TypeId::of::<A>());
        self.assets_map.write().remove(&key).is_some()
    }

    /// Get an asset by url. Return None is this asset had NOT been loaded.
    fn get<A: Asset>(&self, url: AssetUrl) -> Option<AssetRef<'_, A>> {
        let assets = self.assets_map.read();
        let key = (url, TypeId::of::<A>());

        assets.get(&key)
            .map(Arc::clone)
            .and_then(AssetRef::new)
    }
}

/// Engine asset type.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum AssetType {
    Mesh,
    Texture,
    Material,
    MeshCollection,
}

fn asset_type_extension(ty: AssetType) -> &'static str {
    match ty {
        AssetType::Mesh => "mesh",
        AssetType::Texture => "tex",
        AssetType::Material => "mat",
        AssetType::MeshCollection => "mscl",
    }
}

fn extension_asset_type(extension: &str) -> AssetType {
    match extension {
        "mesh" => AssetType::Mesh,
        "tex" => AssetType::Texture,
        "mat" => AssetType::Material,
        "mscl" => AssetType::MeshCollection,
        _ => unreachable!()
    }
}

impl AssetType {
    pub fn extension(&self) -> &str {
        asset_type_extension(*self)
    }
}

/// Url to unique identify an asset.
/// This is a relative path start with words, points to a file located inside content/ folder.
/// TODO: Validation. AssetUrl should always have a valid extension.
///
/// # Example
///
/// ```
/// use zenith_asset::AssetUrl;
/// use std::path::PathBuf;
/// let asset_url: AssetUrl = PathBuf::from("mesh/cerberus/scene.mesh").into();
/// ```
#[derive(Clone, Debug, From, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetUrl {
    path: PathBuf,
}

impl From<String> for AssetUrl {
    fn from(path: String) -> Self {
        AssetUrl { path: path.into() }
    }
}

impl AssetUrl {
    /// Return an invalid url represents nothing.
    pub fn invalid() -> Self {
        Self {
            path: Default::default(),
        }
    }

    /// Return the asset type this AssetUrl points to.
    pub fn ty(&self) -> AssetType {
        let extension = self
            .path
            .extension()
            .and_then(|os_str| os_str.to_str())
            .map(|str| str.to_lowercase())
            .unwrap_or("unknown".to_owned());
        extension_asset_type(&extension)
    }
}

impl AsRef<Path> for AssetUrl {
    fn as_ref(&self) -> &Path {
        self.path.as_path()
    }
}

/// Asset handle represents a loaded and registered asset.
pub struct AssetHandle<A> {
    url: AssetUrl,
    _marker: PhantomData<A>,
}

impl<A: Asset> AssetHandle<A> {
    /// Return a null asset handle which points to nothing.
    pub fn null() -> Self {
        Self {
            url: AssetUrl::invalid(),
            _marker: PhantomData,
        }
    }

    /// Create a new asset handle using AssetUrl.
    pub fn new(url: AssetUrl) -> Self {
        Self {
            url,
            _marker: PhantomData,
        }
    }

    /// Get the underlying asset data if this asset is successfully loaded and registered.
    pub fn get(&self) -> Option<AssetRef<'_, A>> {
        ASSET_REGISTRY.get().unwrap().get(self.url.clone())
    }
}

pub struct AssetRef<'a, A> {
    asset: Arc<dyn Asset>,
    _marker: PhantomData<&'a A>,
}

impl<'a, A: Asset> AssetRef<'a, A> {
    fn new(asset: Arc<dyn Asset>) -> Option<Self> {
        Some(Self {
            asset,
            _marker: PhantomData,
        })
    }
}

impl<'a, A: Asset> Deref for AssetRef<'a, A> {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        unsafe {
            // Safety: asset type is checked by TypeId in AssetRegistry when calling get()
            self.asset.as_ref().as_any().downcast_ref::<A>().unwrap_unchecked()
        }
    }
}

impl<'a, A: Asset> AsRef<A> for AssetRef<'a, A> {
    fn as_ref(&self) -> &A {
        &self
    }
}

/// Asset is any type of data which can be serialized and deserialized.
/// Asset should be read-only which is thread-safe.
///
/// Raw data is stored at content/ folder.
/// The baked asset which had been turned into engine representation is stored at cache/ folder.
pub trait Asset: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn url(&self, name: &str) -> AssetUrl;
    fn extension() -> &'static str where Self: Sized;
}

/// Data needed to send a raw resource load request.
#[derive(Clone, Debug, Builder)]
#[builder(setter(into))]
pub struct RawResourceLoadRequest {
    /// Relative path starts at content/ folder.
    relative_path: PathBuf,
}

/// Type represents a raw resource.
pub trait RawResource: Sized {
    fn load_path(&self) -> &Path;
}

/// Raw resource loader interface.
pub trait RawResourceLoader {
    type Raw: RawResource;

    fn load(path: &Path) -> Result<Self::Raw>;
}

/// Raw resource baker interface.
pub trait RawResourceBaker {
    type Raw: RawResource;

    fn bake(raw: Self::Raw, registry: &AssetRegistry, directory: &PathBuf, url: &AssetUrl) -> Result<()>;
}

/// Data needed to send an asset load request.
#[derive(Clone, Debug, Builder)]
#[builder(setter(into))]
pub struct AssetLoadRequest {
    url: AssetUrl,
}

fn serialize_asset<A: Asset + Encode>(asset: &A, absolute_path: &PathBuf) -> Result<()> {
    if let Some(parent) = absolute_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let config = bincode::config::standard();
    let encoded_data = bincode::encode_to_vec(asset, config)?;

    let mut file = File::create(absolute_path)?;
    file.write_all(&encoded_data)?;
    file.flush()?;

    Ok(())
}

fn deserialize_asset<A: Asset + Encode + DeserializeOwned>(absolute_path: &PathBuf) -> Result<A> {
    let absolute_path = absolute_path.canonicalize()?;
    let mmap = load_with_memory_mapping(&absolute_path)?;

    let (asset, _): (A, usize) = bincode::serde::decode_from_slice(&mmap, bincode::config::standard())
        .expect(&format!("Failed to deserialize asset {:?}", absolute_path));

    Ok(asset)
}