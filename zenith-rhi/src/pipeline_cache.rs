//! Pipeline cache for caching graphics pipelines with robust hashing.

use crate::pipeline::{GraphicPipeline, GraphicPipelineDesc};
use ash::{vk};
use std::sync::Arc;
use zenith_core::collections::hashmap::HashMap;
use zenith_rhi_derive::DeviceObject;
use crate::RenderDevice;
use crate::device::DebuggableObject;
use crate::device::set_debug_name_handle;

#[derive(Debug, Clone, Copy, Default)]
pub struct PipelineCacheStats {
    pub graphic_pipeline_count: usize,
}

/// Pipeline cache for storing and reusing graphics pipelines.
#[DeviceObject]
pub struct PipelineCache {
    name: String,
    cache: vk::PipelineCache,
    pipelines: HashMap<GraphicPipelineDesc, Arc<GraphicPipeline>>,
}

impl PipelineCache {
    /// Create a new pipeline cache.
    pub fn new(name: &str, device: &RenderDevice) -> Result<Self, vk::Result> {
        let cache_info = vk::PipelineCacheCreateInfo::default();
        let vk_cache = unsafe { device.handle().create_pipeline_cache(&cache_info, None)? };

        let pc = Self {
            name: name.to_owned(),
            cache: vk_cache,
            pipelines: HashMap::new(),
            device: device.handle().clone(),
        };
        device.set_debug_name(&pc);
        Ok(pc)
    }

    /// Create a pipeline cache with initial data.
    pub fn with_data(name: &str, device: &RenderDevice, data: &[u8]) -> Result<Self, vk::Result> {
        let cache_info = vk::PipelineCacheCreateInfo::default().initial_data(data);
        let vk_cache = unsafe { device.handle().create_pipeline_cache(&cache_info, None)? };

        let pc = Self {
            name: name.to_owned(),
            cache: vk_cache,
            pipelines: HashMap::new(),
            device: device.handle().clone(),
        };
        device.set_debug_name(&pc);
        Ok(pc)
    }

    #[inline]
    pub fn name(&self) -> &str { &self.name }

    #[inline]
    pub fn handle(&self) -> vk::PipelineCache { self.cache }

    /// Get or create a graphics pipeline.
    pub fn get_or_create(&mut self, name: &str, device: &RenderDevice, desc: &GraphicPipelineDesc) -> Result<Arc<GraphicPipeline>, vk::Result> {
        if let Some(cached) = self.pipelines.get(desc) {
            return Ok(cached.clone());
        }

        let pipeline = Arc::new(GraphicPipeline::with_cache(name, device, desc, self.cache)?);
        self.pipelines.insert(desc.clone(), pipeline.clone());
        Ok(pipeline)
    }

    /// Get cached pipeline data for serialization.
    pub fn get_cache_data(&self) -> Result<Vec<u8>, vk::Result> {
        unsafe { self.device.get_pipeline_cache_data(self.cache) }
    }

    /// Get the number of cached pipelines.
    pub fn len(&self) -> usize {
        self.pipelines.len()
    }

    pub fn stats(&self) -> PipelineCacheStats {
        PipelineCacheStats {
            graphic_pipeline_count: self.pipelines.len(),
        }
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }

    /// Clear all cached pipelines.
    pub fn clear(&mut self) {
        self.pipelines.clear();
    }
}

impl DebuggableObject for PipelineCache {
    fn set_debug_name(&self, device: &RenderDevice) {
        set_debug_name_handle(device, self.cache, vk::ObjectType::PIPELINE_CACHE, self.name());
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        self.clear();
        unsafe {
            self.device.destroy_pipeline_cache(self.cache, None);
        }
    }
}
