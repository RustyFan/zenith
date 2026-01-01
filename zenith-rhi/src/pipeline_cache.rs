//! Pipeline cache for caching graphics pipelines with robust hashing.

use crate::pipeline::GraphicPipeline;
use crate::shader::Shader;
use ash::vk::Handle;
use ash::{vk, Device};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use zenith_core::collections::hashmap::HashMap;

/// Hashable key for graphics pipeline configuration.
#[derive(Clone)]
pub struct GraphicPipelineKey {
    // Shaders (Arc ensures shader modules stay alive while in cache)
    pub vertex_shader: Arc<Shader>,
    pub fragment_shader: Option<Arc<Shader>>,

    // Vertex input
    pub vertex_bindings: Vec<VertexBinding>,
    pub vertex_attributes: Vec<VertexAttribute>,

    // Input assembly
    pub topology: vk::PrimitiveTopology,
    pub primitive_restart: bool,

    // Rasterization
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub depth_clamp: bool,
    pub depth_bias_enable: bool,
    pub depth_bias_constant: i32,
    pub depth_bias_slope: i32,
    pub line_width: i32,

    // Multisample
    pub samples: vk::SampleCountFlags,
    pub sample_shading: bool,
    pub min_sample_shading: i32,

    // Depth stencil
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_compare_op: vk::CompareOp,
    pub depth_bounds_test: bool,
    pub stencil_test: bool,
    pub stencil_front: StencilOpKey,
    pub stencil_back: StencilOpKey,

    // Color blend
    pub color_attachments: Vec<ColorBlendAttachment>,
    pub blend_constants: [i32; 4],

    // Dynamic states
    pub dynamic_states: Vec<vk::DynamicState>,

    // Render targets
    pub color_formats: Vec<vk::Format>,
    pub depth_format: Option<vk::Format>,
    pub stencil_format: Option<vk::Format>,

    // Descriptor layouts (raw handles for hashing)
    pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub push_constant_size: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct VertexBinding {
    pub binding: u32,
    pub stride: u32,
    pub input_rate: vk::VertexInputRate,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct VertexAttribute {
    pub location: u32,
    pub binding: u32,
    pub format: vk::Format,
    pub offset: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub struct StencilOpKey {
    pub fail_op: vk::StencilOp,
    pub pass_op: vk::StencilOp,
    pub depth_fail_op: vk::StencilOp,
    pub compare_op: vk::CompareOp,
    pub compare_mask: u32,
    pub write_mask: u32,
    pub reference: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ColorBlendAttachment {
    pub blend_enable: bool,
    pub src_color: vk::BlendFactor,
    pub dst_color: vk::BlendFactor,
    pub color_op: vk::BlendOp,
    pub src_alpha: vk::BlendFactor,
    pub dst_alpha: vk::BlendFactor,
    pub alpha_op: vk::BlendOp,
    pub write_mask: vk::ColorComponentFlags,
}

impl Default for ColorBlendAttachment {
    fn default() -> Self {
        Self {
            blend_enable: false,
            src_color: vk::BlendFactor::ONE,
            dst_color: vk::BlendFactor::ZERO,
            color_op: vk::BlendOp::ADD,
            src_alpha: vk::BlendFactor::ONE,
            dst_alpha: vk::BlendFactor::ZERO,
            alpha_op: vk::BlendOp::ADD,
            write_mask: vk::ColorComponentFlags::RGBA,
        }
    }
}

impl GraphicPipelineKey {
    /// Create a new pipeline key with required shaders and default settings.
    pub fn new(vertex_shader: Arc<Shader>, fragment_shader: Option<Arc<Shader>>) -> Self {
        Self {
            vertex_shader,
            fragment_shader,
            vertex_bindings: Vec::new(),
            vertex_attributes: Vec::new(),
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart: false,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_clamp: false,
            depth_bias_enable: false,
            depth_bias_constant: 0,
            depth_bias_slope: 0,
            line_width: f32::to_bits(1.0) as i32,
            samples: vk::SampleCountFlags::TYPE_1,
            sample_shading: false,
            min_sample_shading: 0,
            depth_test: false,
            depth_write: false,
            depth_compare_op: vk::CompareOp::LESS,
            depth_bounds_test: false,
            stencil_test: false,
            stencil_front: StencilOpKey::default(),
            stencil_back: StencilOpKey::default(),
            color_attachments: Vec::new(),
            blend_constants: [0; 4],
            dynamic_states: Vec::new(),
            color_formats: Vec::new(),
            depth_format: None,
            stencil_format: None,
            descriptor_set_layouts: Vec::new(),
            push_constant_size: 0,
        }
    }
}

impl Hash for GraphicPipelineKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash by shader module handle, not Arc pointer
        self.vertex_shader.module().as_raw().hash(state);
        self.fragment_shader.as_ref().map(|s| s.module().as_raw()).hash(state);
        self.vertex_bindings.hash(state);
        self.vertex_attributes.hash(state);
        (self.topology.as_raw() as i32).hash(state);
        self.primitive_restart.hash(state);
        (self.polygon_mode.as_raw() as i32).hash(state);
        self.cull_mode.as_raw().hash(state);
        (self.front_face.as_raw() as i32).hash(state);
        self.depth_clamp.hash(state);
        self.depth_bias_enable.hash(state);
        self.depth_bias_constant.hash(state);
        self.depth_bias_slope.hash(state);
        self.line_width.hash(state);
        self.samples.as_raw().hash(state);
        self.sample_shading.hash(state);
        self.min_sample_shading.hash(state);
        self.depth_test.hash(state);
        self.depth_write.hash(state);
        (self.depth_compare_op.as_raw() as i32).hash(state);
        self.depth_bounds_test.hash(state);
        self.stencil_test.hash(state);
        self.stencil_front.hash(state);
        self.stencil_back.hash(state);
        self.color_attachments.hash(state);
        self.blend_constants.hash(state);
        for ds in &self.dynamic_states {
            (ds.as_raw() as i32).hash(state);
        }
        for fmt in &self.color_formats {
            (fmt.as_raw() as i32).hash(state);
        }
        self.depth_format.map(|f| f.as_raw() as i32).hash(state);
        self.stencil_format.map(|f| f.as_raw() as i32).hash(state);
        for layout in &self.descriptor_set_layouts {
            layout.as_raw().hash(state);
        }
        self.push_constant_size.hash(state);
    }
}

impl PartialEq for GraphicPipelineKey {
    fn eq(&self, other: &Self) -> bool {
        // Compare by shader module handle, not Arc pointer
        self.vertex_shader.module() == other.vertex_shader.module()
            && self.fragment_shader.as_ref().map(|s| s.module()) == other.fragment_shader.as_ref().map(|s| s.module())
            && self.vertex_bindings == other.vertex_bindings
            && self.vertex_attributes == other.vertex_attributes
            && self.topology == other.topology
            && self.primitive_restart == other.primitive_restart
            && self.polygon_mode == other.polygon_mode
            && self.cull_mode == other.cull_mode
            && self.front_face == other.front_face
            && self.depth_clamp == other.depth_clamp
            && self.depth_bias_enable == other.depth_bias_enable
            && self.depth_bias_constant == other.depth_bias_constant
            && self.depth_bias_slope == other.depth_bias_slope
            && self.line_width == other.line_width
            && self.samples == other.samples
            && self.sample_shading == other.sample_shading
            && self.min_sample_shading == other.min_sample_shading
            && self.depth_test == other.depth_test
            && self.depth_write == other.depth_write
            && self.depth_compare_op == other.depth_compare_op
            && self.depth_bounds_test == other.depth_bounds_test
            && self.stencil_test == other.stencil_test
            && self.stencil_front == other.stencil_front
            && self.stencil_back == other.stencil_back
            && self.color_attachments == other.color_attachments
            && self.blend_constants == other.blend_constants
            && self.dynamic_states == other.dynamic_states
            && self.color_formats == other.color_formats
            && self.depth_format == other.depth_format
            && self.stencil_format == other.stencil_format
            && self.descriptor_set_layouts == other.descriptor_set_layouts
            && self.push_constant_size == other.push_constant_size
    }
}

impl Eq for GraphicPipelineKey {}

/// Pipeline cache for storing and reusing graphics pipelines.
pub struct PipelineCache {
    device: Device,
    cache: vk::PipelineCache,
    pipelines: HashMap<GraphicPipelineKey, Arc<GraphicPipeline>>,
}

impl PipelineCache {
    /// Create a new pipeline cache.
    pub fn new(device: &Device) -> Result<Self, vk::Result> {
        let cache_info = vk::PipelineCacheCreateInfo::default();
        let vk_cache = unsafe { device.create_pipeline_cache(&cache_info, None)? };

        Ok(Self {
            device: device.clone(),
            cache: vk_cache,
            pipelines: HashMap::new(),
        })
    }

    /// Create a pipeline cache with initial data.
    pub fn with_data(device: &Device, data: &[u8]) -> Result<Self, vk::Result> {
        let cache_info = vk::PipelineCacheCreateInfo::default().initial_data(data);
        let vk_cache = unsafe { device.create_pipeline_cache(&cache_info, None)? };

        Ok(Self {
            device: device.clone(),
            cache: vk_cache,
            pipelines: HashMap::new(),
        })
    }

    /// Get or create a graphics pipeline.
    pub fn get_or_create(&mut self, key: &GraphicPipelineKey) -> Result<Arc<GraphicPipeline>, vk::Result> {
        if let Some(cached) = self.pipelines.get(key) {
            return Ok(cached.clone());
        }

        let pipeline = Arc::new(self.create_graphic_pipeline(key)?);
        self.pipelines.insert(key.clone(), pipeline.clone());
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

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }

    /// Clear all cached pipelines.
    pub fn clear(&mut self) {
        self.pipelines.clear();
    }

    fn create_graphic_pipeline(&self, key: &GraphicPipelineKey) -> Result<GraphicPipeline, vk::Result> {
        // Build vertex input state
        let bindings: Vec<vk::VertexInputBindingDescription> = key
            .vertex_bindings
            .iter()
            .map(|b| {
                vk::VertexInputBindingDescription::default()
                    .binding(b.binding)
                    .stride(b.stride)
                    .input_rate(b.input_rate)
            })
            .collect();

        let attributes: Vec<vk::VertexInputAttributeDescription> = key
            .vertex_attributes
            .iter()
            .map(|a| {
                vk::VertexInputAttributeDescription::default()
                    .location(a.location)
                    .binding(a.binding)
                    .format(a.format)
                    .offset(a.offset)
            })
            .collect();

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);

        // Input assembly
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(key.topology)
            .primitive_restart_enable(key.primitive_restart);

        // Rasterization
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(key.depth_clamp)
            .rasterizer_discard_enable(false)
            .polygon_mode(key.polygon_mode)
            .cull_mode(key.cull_mode)
            .front_face(key.front_face)
            .depth_bias_enable(key.depth_bias_enable)
            .depth_bias_constant_factor(f32::from_bits(key.depth_bias_constant as u32))
            .depth_bias_slope_factor(f32::from_bits(key.depth_bias_slope as u32))
            .line_width(f32::from_bits(key.line_width as u32));

        // Multisample
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(key.samples)
            .sample_shading_enable(key.sample_shading)
            .min_sample_shading(f32::from_bits(key.min_sample_shading as u32));

        // Depth stencil
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(key.depth_test)
            .depth_write_enable(key.depth_write)
            .depth_compare_op(key.depth_compare_op)
            .depth_bounds_test_enable(key.depth_bounds_test)
            .stencil_test_enable(key.stencil_test)
            .front(vk::StencilOpState {
                fail_op: key.stencil_front.fail_op,
                pass_op: key.stencil_front.pass_op,
                depth_fail_op: key.stencil_front.depth_fail_op,
                compare_op: key.stencil_front.compare_op,
                compare_mask: key.stencil_front.compare_mask,
                write_mask: key.stencil_front.write_mask,
                reference: key.stencil_front.reference,
            })
            .back(vk::StencilOpState {
                fail_op: key.stencil_back.fail_op,
                pass_op: key.stencil_back.pass_op,
                depth_fail_op: key.stencil_back.depth_fail_op,
                compare_op: key.stencil_back.compare_op,
                compare_mask: key.stencil_back.compare_mask,
                write_mask: key.stencil_back.write_mask,
                reference: key.stencil_back.reference,
            });

        // Color blend
        let blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = key
            .color_attachments
            .iter()
            .map(|a| {
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(a.blend_enable)
                    .src_color_blend_factor(a.src_color)
                    .dst_color_blend_factor(a.dst_color)
                    .color_blend_op(a.color_op)
                    .src_alpha_blend_factor(a.src_alpha)
                    .dst_alpha_blend_factor(a.dst_alpha)
                    .alpha_blend_op(a.alpha_op)
                    .color_write_mask(a.write_mask)
            })
            .collect();

        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&blend_attachments)
            .blend_constants([
                f32::from_bits(key.blend_constants[0] as u32),
                f32::from_bits(key.blend_constants[1] as u32),
                f32::from_bits(key.blend_constants[2] as u32),
                f32::from_bits(key.blend_constants[3] as u32),
            ]);

        // Extract shader modules from Arc<Shader>
        let vertex_module = key.vertex_shader.module();
        let fragment_module = key.fragment_shader.as_ref().map(|s| s.module()).unwrap_or(vk::ShaderModule::null());

        // Build info and use GraphicPipeline's creation logic
        let info = crate::pipeline::GraphicPipelineInfo {
            vertex_shader: vertex_module,
            fragment_shader: fragment_module,
            vertex_input_state: vertex_input,
            input_assembly_state: input_assembly,
            rasterization_state: rasterization,
            multisample_state: multisample,
            depth_stencil_state: if key.depth_test || key.stencil_test {
                Some(depth_stencil)
            } else {
                None
            },
            color_blend_state: color_blend,
            dynamic_states: &key.dynamic_states,
            color_attachment_formats: &key.color_formats,
            depth_attachment_format: key.depth_format,
        };

        // Build push constant range if needed
        let push_constant_ranges = if key.push_constant_size > 0 {
            vec![vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::ALL_GRAPHICS,
                offset: 0,
                size: key.push_constant_size,
            }]
        } else {
            vec![]
        };

        GraphicPipeline::with_cache(&self.device, &key.descriptor_set_layouts, &push_constant_ranges, &info, self.cache)
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
