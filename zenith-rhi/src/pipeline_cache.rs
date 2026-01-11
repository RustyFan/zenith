//! Pipeline cache for caching graphics pipelines with robust hashing.

use crate::pipeline::{GraphicPipeline, GraphicPipelineDesc};
use ash::{vk, Device};
use std::sync::Arc;
use zenith_core::collections::hashmap::HashMap;
use zenith_rhi_derive::DeviceObject;

#[derive(Debug, Clone, Copy, Default)]
pub struct PipelineCacheStats {
    pub graphic_pipeline_count: usize,
}

/// Pipeline cache for storing and reusing graphics pipelines.
#[DeviceObject]
pub struct PipelineCache {
    cache: vk::PipelineCache,
    pipelines: HashMap<GraphicPipelineDesc, Arc<GraphicPipeline>>,
}

impl PipelineCache {
    /// Create a new pipeline cache.
    pub fn new(device: &Device) -> Result<Self, vk::Result> {
        let cache_info = vk::PipelineCacheCreateInfo::default();
        let vk_cache = unsafe { device.create_pipeline_cache(&cache_info, None)? };

        Ok(Self {
            cache: vk_cache,
            pipelines: HashMap::new(),
            device: device.clone(),
        })
    }

    /// Create a pipeline cache with initial data.
    pub fn with_data(device: &Device, data: &[u8]) -> Result<Self, vk::Result> {
        let cache_info = vk::PipelineCacheCreateInfo::default().initial_data(data);
        let vk_cache = unsafe { device.create_pipeline_cache(&cache_info, None)? };

        Ok(Self {
            cache: vk_cache,
            pipelines: HashMap::new(),
            device: device.clone(),
        })
    }

    /// Get or create a graphics pipeline.
    pub fn get_or_create(&mut self, desc: &GraphicPipelineDesc) -> Result<Arc<GraphicPipeline>, vk::Result> {
        if let Some(cached) = self.pipelines.get(desc) {
            return Ok(cached.clone());
        }

        let pipeline = Arc::new(GraphicPipeline::with_cache(&self.device, desc, self.cache)?);
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

impl Drop for PipelineCache {
    fn drop(&mut self) {
        self.clear();
        unsafe {
            self.device.destroy_pipeline_cache(self.cache, None);
        }
    }
}
