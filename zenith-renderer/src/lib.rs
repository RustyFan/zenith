mod defer_shading;
mod triangle;
mod world;
mod lighting;
mod helpers;

use std::sync::{Arc, OnceLock};
use parking_lot::Mutex;
use zenith_rhi::RenderDevice;
use crate::helpers::DefaultRenderResources;

pub use triangle::TriangleRenderer;
pub use world::WorldRenderer;

static DEFAULT_RENDER_RESOURCES: OnceLock<Mutex<Option<DefaultRenderResources>>> = OnceLock::new();

pub fn initialize(device: &Arc<RenderDevice>) -> anyhow::Result<()> {
    let res = Mutex::new(Some(DefaultRenderResources::new(device)?));
    DEFAULT_RENDER_RESOURCES.set(res).map_err(|_| anyhow::anyhow!("Failed to initialize render resources"))?;
    Ok(())
}

pub fn deinitialize() {
    DEFAULT_RENDER_RESOURCES.get().unwrap().lock().take().unwrap();
}
