//! Pipeline cache for caching graphics and compute pipelines with robust hashing.

use crate::pipeline::{ComputePipeline, ComputePipelineDesc, GraphicPipeline, GraphicPipelineDesc};
use crate::shader::ShaderReflection;
use ash::{vk};
use std::sync::Arc;
use zenith_core::collections::hashmap::HashMap;
use zenith_core::collections::SmallVec;
use crate::{DescriptorSetLayout, RenderDevice};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineHandle(pub u32);

pub enum PipelineEntry {
    Graphic(Arc<GraphicPipeline>),
    Compute(Arc<ComputePipeline>),
}

impl PipelineEntry {
    pub fn layout(&self) -> vk::PipelineLayout {
        match self {
            PipelineEntry::Graphic(p) => p.layout(),
            PipelineEntry::Compute(p) => p.layout(),
        }
    }

    pub fn handle(&self) -> vk::Pipeline {
        match self {
            PipelineEntry::Graphic(p) => p.handle(),
            PipelineEntry::Compute(p) => p.handle(),
        }
    }

    pub fn bind_point(&self) -> vk::PipelineBindPoint {
        match self {
            PipelineEntry::Graphic(_) => vk::PipelineBindPoint::GRAPHICS,
            PipelineEntry::Compute(_) => vk::PipelineBindPoint::COMPUTE,
        }
    }

    pub fn shader_reflection(&self) -> &ShaderReflection {
        match self {
            PipelineEntry::Graphic(p) => &p.desc().shader.merged_reflection,
            PipelineEntry::Compute(p) => p.desc().shader.reflection(),
        }
    }

    pub fn descriptor_layouts(&self) -> &[Arc<DescriptorSetLayout>] {
        match self {
            PipelineEntry::Graphic(p) => p.descriptor_layouts(),
            PipelineEntry::Compute(p) => p.descriptor_layouts(),
        }
    }

    pub fn as_graphic(&self) -> Option<&Arc<GraphicPipeline>> {
        match self {
            PipelineEntry::Graphic(p) => Some(p),
            _ => None,
        }
    }

    pub fn as_compute(&self) -> Option<&Arc<ComputePipeline>> {
        match self {
            PipelineEntry::Compute(p) => Some(p),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PipelineCacheStats {
    pub graphic_pipeline_count: usize,
    pub compute_pipeline_count: usize,
}

pub struct PipelineRegistry {
    graphic_pipeline_handles: HashMap<GraphicPipelineDesc, PipelineHandle>,
    compute_pipeline_handles: HashMap<ComputePipelineDesc, PipelineHandle>,
    pipeline_storages: Vec<PipelineEntry>,
}

impl PipelineRegistry {
    pub fn new() -> Self {
        Self {
            graphic_pipeline_handles: HashMap::new(),
            compute_pipeline_handles: HashMap::new(),
            pipeline_storages: Vec::new(),
        }
    }

    pub fn register_graph_pipeline(
        &mut self,
        device: &RenderDevice,
        desc: &GraphicPipelineDesc,
    ) -> Result<PipelineHandle, vk::Result> {
        debug_assert!(self.pipeline_storages.len() < u32::MAX as _);

        if let Some(&handle) = self.graphic_pipeline_handles.get(desc) {
            return Ok(handle);
        }

        let mut descriptor_layouts: SmallVec<[Arc<DescriptorSetLayout>; 4]> = SmallVec::new();

        let max_set = desc.shader.merged_reflection.max_set().unwrap_or(0);
        for idx in 0..=max_set {
            let layout_name = format!("descriptor_layout.graphic.s{idx}");
            let layout = DescriptorSetLayout::from_reflection(&layout_name, device, &desc.shader.merged_reflection.bindings, idx)?;
            descriptor_layouts.push(Arc::new(layout));
        }

        let pipeline = Arc::new(GraphicPipeline::new(
            device,
            desc,
            &descriptor_layouts,
        )?);

        let handle = PipelineHandle(self.pipeline_storages.len() as u32);
        self.graphic_pipeline_handles.insert(desc.clone(), handle);
        self.pipeline_storages.push(PipelineEntry::Graphic(pipeline));

        Ok(handle)
    }

    pub fn register_compute_pipeline(
        &mut self,
        device: &RenderDevice,
        desc: &ComputePipelineDesc,
    ) -> Result<PipelineHandle, vk::Result> {
        debug_assert!(self.pipeline_storages.len() < u32::MAX as _);
        
        if let Some(&handle) = self.compute_pipeline_handles.get(desc) {
            return Ok(handle);
        }

        let reflection = desc.shader.reflection();
        let max_set = reflection.max_set().unwrap_or(0);
        let mut descriptor_layouts: SmallVec<[Arc<DescriptorSetLayout>; 4]> = SmallVec::new();
        for idx in 0..=max_set {
            let layout_name = format!("descriptor_layout.compute.s{idx}");
            let layout = DescriptorSetLayout::from_reflection(&layout_name, device, &reflection.bindings, idx)?;
            descriptor_layouts.push(Arc::new(layout));
        }

        let pipeline = Arc::new(ComputePipeline::new(
            device,
            desc,
            &descriptor_layouts,
        )?);
        let handle = PipelineHandle(self.pipeline_storages.len() as u32);
        self.compute_pipeline_handles.insert(desc.clone(), handle);
        self.pipeline_storages.push(PipelineEntry::Compute(pipeline));

        Ok(handle)
    }

    #[inline]
    pub fn try_get_pipeline(&self, handle: PipelineHandle) -> Option<&PipelineEntry> {
        self.pipeline_storages.get(handle.0 as usize)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pipeline_storages.len()
    }

    pub fn stats(&self) -> PipelineCacheStats {
        PipelineCacheStats {
            graphic_pipeline_count: self.graphic_pipeline_handles.len(),
            compute_pipeline_count: self.compute_pipeline_handles.len(),
        }
    }

    pub fn clear(&mut self) {
        self.graphic_pipeline_handles.clear();
        self.compute_pipeline_handles.clear();
        self.pipeline_storages.clear();
    }
}
