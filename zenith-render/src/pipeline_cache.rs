use std::hash::{Hash, Hasher};
use zenith_core::collections::{DefaultHasher};
use zenith_core::collections::hashmap::{Entry, HashMap};
use crate::shader::{GraphicShader};

/// Cache all types of pipelines created during rendering.
pub struct PipelineCache {
    raster_pipelines: HashMap<u64, wgpu::RenderPipeline>,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {
            raster_pipelines: HashMap::new(),
        }
    }

    /// If this pipeline is exist, return the cached pipeline.
    /// If this pipeline is NOT exists, create one and return it.
    #[profiling::function]
    pub fn get_or_create_graphic_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader: &GraphicShader,
        color_states: &[Option<wgpu::ColorTargetState>],
        depth_stencil_state: Option<wgpu::DepthStencilState>,
    ) -> anyhow::Result<wgpu::RenderPipeline> {
        let mut hasher = DefaultHasher::new();
        shader.hash(&mut hasher);
        let hash = hasher.finish();

        match self.raster_pipelines.entry(hash) {
            Entry::Occupied(pipeline) => {
                Ok(pipeline.get().clone())
            }
            Entry::Vacant(entry) => {
                let module = shader.create_shader_module(
                    device,
                    Default::default(),
                )?;

                let layout = shader.create_pipeline_layout(device);

                let vertex = shader.create_vertex_state(&module);
                let fragment = shader.create_fragment_state(&module, color_states);

                let pipeline = device.create_render_pipeline(
                    &wgpu::RenderPipelineDescriptor {
                        label: Some(&shader.name()),
                        layout: Some(&layout),
                        vertex,
                        primitive: Default::default(),
                        depth_stencil: depth_stencil_state,
                        multisample: Default::default(),
                        fragment,
                        multiview: None,
                        cache: None,
                    }
                );

                entry.insert(pipeline.clone());
                Ok(pipeline)
            }
        }
    }
}
