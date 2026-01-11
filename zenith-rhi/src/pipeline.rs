//! Vulkan Pipeline - pipeline layout and graphics pipeline management.

use zenith_core::log;
use crate::descriptor::DescriptorSetLayout;
use crate::shader::{Shader, ShaderReflection};
use derive_builder::Builder;
use ash::{vk};
use ash::vk::Handle;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use zenith_core::collections::SmallVec;
use zenith_rhi_derive::DeviceObject;
use crate::{RenderDevice};
use crate::device::DebuggableObject;
use crate::device::set_debug_name_handle;

/// Dynamic rendering attachment formats for a graphics pipeline.
///
/// The binding order is:
/// - `color_formats[0..]` (in-order)
/// - `depth_format` (optional)
/// - `stencil_format` (optional)
#[derive(Clone, Debug, Default)]
pub struct GraphicPipelineAttachments {
    pub color_formats: Vec<vk::Format>,
    pub depth_format: Option<vk::Format>,
    pub stencil_format: Option<vk::Format>,
}

impl Hash for GraphicPipelineAttachments {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for fmt in &self.color_formats {
            (fmt.as_raw() as i32).hash(state);
        }
        self.depth_format.map(|f| f.as_raw() as i32).hash(state);
        self.stencil_format.map(|f| f.as_raw() as i32).hash(state);
    }
}

impl PartialEq for GraphicPipelineAttachments {
    fn eq(&self, other: &Self) -> bool {
        self.color_formats == other.color_formats
            && self.depth_format == other.depth_format
            && self.stencil_format == other.stencil_format
    }
}

impl Eq for GraphicPipelineAttachments {}

impl GraphicPipelineAttachments {
    #[inline]
    pub fn to_vk_rendering_info(&self) -> vk::PipelineRenderingCreateInfo<'_> {
        let mut info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&self.color_formats);

        if let Some(depth) = self.depth_format {
            info = info.depth_attachment_format(depth);
        }
        if let Some(stencil) = self.stencil_format {
            info = info.stencil_attachment_format(stencil);
        }

        info
    }
}

#[derive(Clone)]
pub struct GraphicShaderInput {
    pub vertex_shader: Arc<Shader>,
    pub fragment_shader: Option<Arc<Shader>>,

    pub vertex_bindings: Vec<VertexBinding>,
    pub vertex_attributes: Vec<VertexAttribute>,

    pub merged_reflection: ShaderReflection,
    // pub descriptor_set_layouts: Vec<Arc<DescriptorSetLayout>>,
    pub push_constant_size: u32,
}

impl GraphicShaderInput {
    pub fn new(
        vertex_shader: Arc<Shader>,
        fragment_shader: Option<Arc<Shader>>,
        vertex_bindings: Vec<VertexBinding>,
        vertex_attributes: Vec<VertexAttribute>,
    ) -> Result<Self, GraphicShaderInputBuildError> {
        validate_vertex_inputs(vertex_shader.reflection(), &vertex_attributes)?;

        let mut reflections: Vec<&ShaderReflection> = Vec::new();
        reflections.push(vertex_shader.reflection());
        if let Some(fs) = &fragment_shader {
            reflections.push(fs.reflection());
        }

        let merged_reflection = ShaderReflection::merge(&reflections);
        // let descriptor_set_layouts = crate::shader::create_layouts_from_reflection(vertex_shader.device(), &merged_reflection)
        //     .map_err(GraphicShaderInputBuildError::DescriptorLayoutCreationFailed)?;

        Ok(Self {
            vertex_shader,
            fragment_shader,
            vertex_bindings,
            vertex_attributes,
            push_constant_size: merged_reflection.push_constant_size,
            merged_reflection,
            // descriptor_set_layouts,
        })
    }

    pub fn create_pipeline_layout(&self, device: &RenderDevice, layouts: &[DescriptorSetLayout]) -> Result<vk::PipelineLayout, vk::Result> {
        let push_constant_ranges = if self.push_constant_size > 0 {
            vec![vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::ALL_GRAPHICS,
                offset: 0,
                size: self.push_constant_size,
            }]
        } else {
            vec![]
        };

        let layouts = layouts.iter()
            .map(|layout| layout.handle())
            .collect::<SmallVec<[_; 3]>>();

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&push_constant_ranges);

        unsafe { device.handle().create_pipeline_layout(&layout_info, None) }
    }
}

#[derive(Debug)]
pub enum GraphicShaderInputBuildError {
    MissingVertexShader,
    VertexInputReflectionMissing,
    DuplicateVertexAttributeLocation { location: u32 },
    MissingVertexAttribute { location: u32, expected: vk::Format },
    VertexAttributeFormatMismatch { location: u32, expected: vk::Format, provided: vk::Format },
    UnexpectedVertexAttribute { location: u32, provided: vk::Format },
    DescriptorLayoutCreationFailed(vk::Result),
}

impl std::fmt::Display for GraphicShaderInputBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphicShaderInputBuildError::MissingVertexShader => write!(f, "missing vertex shader"),
            GraphicShaderInputBuildError::VertexInputReflectionMissing => {
                write!(f, "vertex shader reflection contains no vertex_inputs, but vertex_attributes were provided")
            }
            GraphicShaderInputBuildError::DuplicateVertexAttributeLocation { location } => {
                write!(f, "duplicate vertex attribute location: {}", location)
            }
            GraphicShaderInputBuildError::MissingVertexAttribute { location, expected } => write!(
                f,
                "missing vertex attribute for location {} (expected {:?})",
                location, expected
            ),
            GraphicShaderInputBuildError::VertexAttributeFormatMismatch { location, expected, provided } => write!(
                f,
                "vertex attribute format mismatch at location {} (expected {:?}, provided {:?})",
                location, expected, provided
            ),
            GraphicShaderInputBuildError::UnexpectedVertexAttribute { location, provided } => write!(
                f,
                "unexpected vertex attribute at location {} (provided {:?}, but shader has no input at this location)",
                location, provided
            ),
            GraphicShaderInputBuildError::DescriptorLayoutCreationFailed(e) => {
                write!(f, "failed to create merged descriptor set layouts: {:?}", e)
            }
        }
    }
}

impl std::error::Error for GraphicShaderInputBuildError {}

#[derive(Default)]
pub struct GraphicShaderInputBuilder {
    vertex_shader: Option<Arc<Shader>>,
    fragment_shader: Option<Arc<Shader>>,
    vertex_bindings: Vec<VertexBinding>,
    vertex_attributes: Vec<VertexAttribute>,
}

impl GraphicShaderInputBuilder {
    pub fn vertex_shader(mut self, shader: Arc<Shader>) -> Self {
        self.vertex_shader = Some(shader);
        self
    }

    pub fn fragment_shader(mut self, shader: Arc<Shader>) -> Self {
        self.fragment_shader = Some(shader);
        self
    }

    pub fn push_vertex_binding(mut self, binding: VertexBinding) -> Self {
        self.vertex_bindings.push(binding);
        self
    }

    pub fn push_vertex_attribute(mut self, attribute: VertexAttribute) -> Self {
        self.vertex_attributes.push(attribute);
        self
    }

    pub fn vertex_layout<T: VertexLayout>(mut self) -> Self {
        let (binding, attributes) = T::vertex_layout();
        self.vertex_bindings.push(binding);
        self.vertex_attributes.extend(attributes);
        self
    }

    pub fn build(self) -> Result<GraphicShaderInput, GraphicShaderInputBuildError> {
        let Some(vs) = self.vertex_shader else {
            return Err(GraphicShaderInputBuildError::MissingVertexShader);
        };
        GraphicShaderInput::new(
            vs,
            self.fragment_shader,
            self.vertex_bindings,
            self.vertex_attributes,
        )
    }
}

fn validate_vertex_inputs(
    vs_reflection: &ShaderReflection,
    vertex_attributes: &[VertexAttribute],
) -> Result<(), GraphicShaderInputBuildError> {
    // If shader reflection doesn't provide inputs, accept only empty attributes.
    if vs_reflection.vertex_inputs.is_empty() {
        if vertex_attributes.is_empty() {
            return Ok(());
        }
        return Err(GraphicShaderInputBuildError::VertexInputReflectionMissing);
    }

    let mut expected: HashMap<u32, vk::Format> = HashMap::new();
    for vi in &vs_reflection.vertex_inputs {
        expected.insert(vi.location, vi.format);
    }

    let mut provided: HashMap<u32, vk::Format> = HashMap::new();
    for a in vertex_attributes {
        if let Some(prev) = provided.insert(a.location, a.format) {
            if prev != a.format {
                return Err(GraphicShaderInputBuildError::DuplicateVertexAttributeLocation {
                    location: a.location,
                });
            }
        }
    }

    // Ensure every shader input has a matching attribute.
    for (loc, exp_fmt) in &expected {
        match provided.get(loc) {
            None => {
                return Err(GraphicShaderInputBuildError::MissingVertexAttribute {
                    location: *loc,
                    expected: *exp_fmt,
                })
            }
            Some(got_fmt) if got_fmt != exp_fmt => {
                return Err(GraphicShaderInputBuildError::VertexAttributeFormatMismatch {
                    location: *loc,
                    expected: *exp_fmt,
                    provided: *got_fmt,
                })
            }
            _ => {}
        }
    }

    // Disallow unexpected attributes (strict match).
    for (loc, got_fmt) in &provided {
        if !expected.contains_key(loc) {
            return Err(GraphicShaderInputBuildError::UnexpectedVertexAttribute {
                location: *loc,
                provided: *got_fmt,
            });
        }
    }

    Ok(())
}

impl Hash for GraphicShaderInput {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash by shader module handle + stage + entry point, not Arc pointer
        self.vertex_shader.module().as_raw().hash(state);
        (self.vertex_shader.vk_stage().as_raw() as i32).hash(state);
        self.vertex_shader.entry_point().as_bytes().hash(state);

        self.fragment_shader.as_ref().map(|s| s.module().as_raw()).hash(state);
        self.fragment_shader
            .as_ref()
            .map(|s| s.vk_stage().as_raw() as i32)
            .hash(state);
        self.fragment_shader
            .as_ref()
            .map(|s| s.entry_point().as_bytes())
            .hash(state);

        self.vertex_bindings.hash(state);
        self.vertex_attributes.hash(state);
    }
}

impl PartialEq for GraphicShaderInput {
    fn eq(&self, other: &Self) -> bool {
        self.vertex_shader.module() == other.vertex_shader.module()
            && self.vertex_shader.vk_stage() == other.vertex_shader.vk_stage()
            && self.vertex_shader.entry_point() == other.vertex_shader.entry_point()
            && self.fragment_shader.as_ref().map(|s| s.module())
                == other.fragment_shader.as_ref().map(|s| s.module())
            && self.fragment_shader.as_ref().map(|s| s.vk_stage())
                == other.fragment_shader.as_ref().map(|s| s.vk_stage())
            && self.fragment_shader.as_ref().map(|s| s.entry_point())
                == other.fragment_shader.as_ref().map(|s| s.entry_point())
            && self.vertex_bindings == other.vertex_bindings
            && self.vertex_attributes == other.vertex_attributes
    }
}

impl Eq for GraphicShaderInput {}

/// Trait implemented by vertex types that can describe their Vulkan vertex input layout.
///
/// Prefer `#[derive(VertexLayout)]` (re-exported by `zenith-rhi`) to generate this automatically.
pub trait VertexLayout {
    fn vertex_layout() -> (VertexBinding, Vec<VertexAttribute>);
}

/// Color attachment configuration used by dynamic rendering and (partially) by pipeline blend state.
#[derive(Clone, Debug, Builder)]
#[builder(setter(into), default)]
pub struct ColorAttachmentDesc {
    pub blend_enable: bool,
    pub src_color_blend: vk::BlendFactor,
    pub dst_color_blend: vk::BlendFactor,
    pub color_blend_op: vk::BlendOp,
    pub src_alpha_blend: vk::BlendFactor,
    pub dst_alpha_blend: vk::BlendFactor,
    pub alpha_blend_op: vk::BlendOp,
    pub write_mask: vk::ColorComponentFlags,

    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub clear_value: [f32; 4],
}

impl Default for ColorAttachmentDesc {
    fn default() -> Self {
        Self {
            blend_enable: false,
            src_color_blend: vk::BlendFactor::ONE,
            dst_color_blend: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend: vk::BlendFactor::ONE,
            dst_alpha_blend: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            write_mask: vk::ColorComponentFlags::RGBA,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

impl ColorAttachmentDesc {
    #[inline]
    pub fn to_vk_blend_attachment(&self) -> vk::PipelineColorBlendAttachmentState {
        vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(self.blend_enable)
            .src_color_blend_factor(self.src_color_blend)
            .dst_color_blend_factor(self.dst_color_blend)
            .color_blend_op(self.color_blend_op)
            .src_alpha_blend_factor(self.src_alpha_blend)
            .dst_alpha_blend_factor(self.dst_alpha_blend)
            .alpha_blend_op(self.alpha_blend_op)
            .color_write_mask(self.write_mask)
    }
}

impl ColorAttachmentDescBuilder {
    pub fn discard_input(&mut self) -> &mut Self {
        self.load_op.replace(vk::AttachmentLoadOp::DONT_CARE);
        self
    }

    pub fn clear_input(&mut self) -> &mut Self {
        self.load_op.replace(vk::AttachmentLoadOp::CLEAR);
        self
    }

    pub fn discard_output(&mut self) -> &mut Self {
        self.store_op.replace(vk::AttachmentStoreOp::DONT_CARE);
        self
    }

    pub fn translucent(&mut self) -> &mut Self {
        self.blend_enable.replace(true);
        self.src_color_blend.replace(vk::BlendFactor::SRC_ALPHA);
        self.dst_color_blend.replace(vk::BlendFactor::DST_ALPHA);
        self.color_blend_op.replace(vk::BlendOp::ADD);
        self.src_alpha_blend.replace(vk::BlendFactor::ZERO);
        self.dst_alpha_blend.replace(vk::BlendFactor::SRC_ALPHA);
        self
    }
}

#[derive(Clone, Debug, Builder)]
#[builder(setter(into), default)]
pub struct DepthStencilDesc {
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_compare_op: vk::CompareOp,
    pub depth_bounds_test_enable: bool,

    pub depth_load_op: vk::AttachmentLoadOp,
    pub depth_store_op: vk::AttachmentStoreOp,
    pub depth_clear_value: f32,

    pub stencil_test_enable: bool,
    pub stencil_front: vk::StencilOpState,
    pub stencil_back: vk::StencilOpState,

    pub stencil_load_op: vk::AttachmentLoadOp,
    pub stencil_store_op: vk::AttachmentStoreOp,
    pub stencil_clear_value: u32,
}

impl Default for DepthStencilDesc {
    fn default() -> Self {
        Self {
            depth_test_enable: false,
            depth_write_enable: false,
            depth_compare_op: vk::CompareOp::LESS,
            depth_bounds_test_enable: false,
            depth_load_op: vk::AttachmentLoadOp::CLEAR,
            depth_store_op: vk::AttachmentStoreOp::STORE,
            depth_clear_value: 1.0,
            stencil_test_enable: false,
            stencil_front: vk::StencilOpState::default(),
            stencil_back: vk::StencilOpState::default(),
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_clear_value: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Builder)]
#[builder(setter(into), default)]
pub struct InputAssemblyState {
    pub topology: vk::PrimitiveTopology,
    pub primitive_restart: bool,
}

impl DepthStencilDesc {
    #[inline]
    pub fn test_enabled(&self) -> bool {
        self.depth_test_enable || self.stencil_test_enable
    }

    #[inline]
    pub fn to_vk(&self) -> vk::PipelineDepthStencilStateCreateInfo<'static> {
        // Note: this create-info does not borrow anything; the lifetime is only a phantom.
        vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(self.depth_test_enable)
            .depth_write_enable(self.depth_write_enable)
            .depth_compare_op(self.depth_compare_op)
            .depth_bounds_test_enable(self.depth_bounds_test_enable)
            .stencil_test_enable(self.stencil_test_enable)
            .front(self.stencil_front)
            .back(self.stencil_back)
    }
}

impl Default for InputAssemblyState {
    fn default() -> Self {
        Self {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart: false,
        }
    }
}

impl Hash for InputAssemblyState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.topology.as_raw() as i32).hash(state);
        self.primitive_restart.hash(state);
    }
}

impl InputAssemblyState {
    #[inline]
    pub fn to_vk(&self) -> vk::PipelineInputAssemblyStateCreateInfo<'static> {
        vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(self.topology)
            .primitive_restart_enable(self.primitive_restart)
    }
}

#[derive(Clone, Debug, Builder)]
#[builder(setter(into), default)]
pub struct RasterizationState {
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub depth_clamp: bool,
    pub depth_bias_enable: bool,
    pub depth_bias_constant: f32,
    pub depth_bias_slope: f32,
    pub line_width: f32,
}

impl Default for RasterizationState {
    fn default() -> Self {
        Self {
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_clamp: false,
            depth_bias_enable: false,
            depth_bias_constant: 0.0,
            depth_bias_slope: 0.0,
            line_width: 1.0,
        }
    }
}

impl PartialEq for RasterizationState {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.polygon_mode.as_raw() == other.polygon_mode.as_raw()
            && self.cull_mode.as_raw() == other.cull_mode.as_raw()
            && self.front_face.as_raw() == other.front_face.as_raw()
            && self.depth_clamp == other.depth_clamp
            && self.depth_bias_enable == other.depth_bias_enable
            && self.depth_bias_constant.to_bits() == other.depth_bias_constant.to_bits()
            && self.depth_bias_slope.to_bits() == other.depth_bias_slope.to_bits()
            && self.line_width.to_bits() == other.line_width.to_bits()
    }
}

impl Eq for RasterizationState {}

impl Hash for RasterizationState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.polygon_mode.as_raw() as i32).hash(state);
        self.cull_mode.as_raw().hash(state);
        (self.front_face.as_raw() as i32).hash(state);
        self.depth_clamp.hash(state);
        self.depth_bias_enable.hash(state);
        self.depth_bias_constant.to_bits().hash(state);
        self.depth_bias_slope.to_bits().hash(state);
        self.line_width.to_bits().hash(state);
    }
}

impl RasterizationState {
    #[inline]
    pub fn to_vk(&self) -> vk::PipelineRasterizationStateCreateInfo<'static> {
        vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(self.depth_clamp)
            .rasterizer_discard_enable(false)
            .polygon_mode(self.polygon_mode)
            .cull_mode(self.cull_mode)
            .front_face(self.front_face)
            .depth_bias_enable(self.depth_bias_enable)
            .depth_bias_constant_factor(f32::from_bits(self.depth_bias_constant as u32))
            .depth_bias_slope_factor(f32::from_bits(self.depth_bias_slope as u32))
            .line_width(self.line_width)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Builder)]
#[builder(setter(into))]
pub struct MultisampleState {
    pub samples: vk::SampleCountFlags,
    pub sample_shading: bool,
    pub min_sample_shading: i32,
}

impl Default for MultisampleState {
    fn default() -> Self {
        Self {
            samples: vk::SampleCountFlags::TYPE_1,
            sample_shading: false,
            min_sample_shading: 0,
        }
    }
}

impl Hash for MultisampleState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.samples.as_raw().hash(state);
        self.sample_shading.hash(state);
        self.min_sample_shading.hash(state);
    }
}

impl MultisampleState {
    #[inline]
    pub fn to_vk(&self) -> vk::PipelineMultisampleStateCreateInfo<'static> {
        vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(self.samples)
            .sample_shading_enable(self.sample_shading)
            .min_sample_shading(f32::from_bits(self.min_sample_shading as u32))
    }
}


#[derive(Clone, Debug, Builder)]
#[builder(setter(into), default)]
pub struct ColorBlendState {
    pub attachments: Vec<ColorAttachmentDesc>,
    pub blend_constants: [i32; 4],
}

impl Default for ColorBlendState {
    fn default() -> Self {
        Self {
            attachments: Vec::new(),
            blend_constants: [0; 4],
        }
    }
}

impl Hash for ColorBlendState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash full attachment behavior + clear value for cache key
        for a in &self.attachments {
            a.blend_enable.hash(state);
            (a.src_color_blend.as_raw() as i32).hash(state);
            (a.dst_color_blend.as_raw() as i32).hash(state);
            (a.color_blend_op.as_raw() as i32).hash(state);
            (a.src_alpha_blend.as_raw() as i32).hash(state);
            (a.dst_alpha_blend.as_raw() as i32).hash(state);
            (a.alpha_blend_op.as_raw() as i32).hash(state);
            a.write_mask.as_raw().hash(state);
            (a.load_op.as_raw() as i32).hash(state);
            (a.store_op.as_raw() as i32).hash(state);
            a.clear_value[0].to_bits().hash(state);
            a.clear_value[1].to_bits().hash(state);
            a.clear_value[2].to_bits().hash(state);
            a.clear_value[3].to_bits().hash(state);
        }
        self.blend_constants.hash(state);
    }
}

impl PartialEq for ColorBlendState {
    fn eq(&self, other: &Self) -> bool {
        if self.blend_constants != other.blend_constants {
            return false;
        }
        if self.attachments.len() != other.attachments.len() {
            return false;
        }

        self.attachments
            .iter()
            .zip(other.attachments.iter())
            .all(|(a, b)| {
                a.blend_enable == b.blend_enable
                    && a.src_color_blend == b.src_color_blend
                    && a.dst_color_blend == b.dst_color_blend
                    && a.color_blend_op == b.color_blend_op
                    && a.src_alpha_blend == b.src_alpha_blend
                    && a.dst_alpha_blend == b.dst_alpha_blend
                    && a.alpha_blend_op == b.alpha_blend_op
                    && a.write_mask == b.write_mask
                    && a.load_op == b.load_op
                    && a.store_op == b.store_op
                    && a.clear_value[0].to_bits() == b.clear_value[0].to_bits()
                    && a.clear_value[1].to_bits() == b.clear_value[1].to_bits()
                    && a.clear_value[2].to_bits() == b.clear_value[2].to_bits()
                    && a.clear_value[3].to_bits() == b.clear_value[3].to_bits()
            })
    }
}

impl Eq for ColorBlendState {}

impl ColorBlendState {
    #[inline]
    pub fn to_vk_attachments(&self) -> Vec<vk::PipelineColorBlendAttachmentState> {
        self.attachments.iter().map(|a| a.to_vk_blend_attachment()).collect()
    }

    #[inline]
    pub fn to_vk<'a>(
        &self,
        attachments: &'a [vk::PipelineColorBlendAttachmentState],
    ) -> vk::PipelineColorBlendStateCreateInfo<'a> {
        vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(attachments)
            .blend_constants([
                f32::from_bits(self.blend_constants[0] as u32),
                f32::from_bits(self.blend_constants[1] as u32),
                f32::from_bits(self.blend_constants[2] as u32),
                f32::from_bits(self.blend_constants[3] as u32),
            ])
    }
}

/// Graphics pipeline state collection (rasterization/blend/depth-stencil/etc...).
///
/// Conservative defaults are provided via `Default`.
#[derive(Clone)]
pub struct GraphicPipelineState {
    pub input_assembly: InputAssemblyState,
    pub rasterization: RasterizationState,
    pub multisample: MultisampleState,

    pub depth_stencil: Option<DepthStencilDesc>,
    pub color_blend: ColorBlendState,

    pub dynamic_states: Vec<vk::DynamicState>,
}

impl Default for GraphicPipelineState {
    fn default() -> Self {
        Self {
            input_assembly: InputAssemblyState::default(),
            rasterization: RasterizationState::default(),
            multisample: MultisampleState::default(),
            depth_stencil: None,
            color_blend: ColorBlendState::default(),
            dynamic_states: vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR],
        }
    }
}

impl Hash for GraphicPipelineState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input_assembly.hash(state);
        self.rasterization.hash(state);
        self.multisample.hash(state);
        // Depth/stencil (now includes attachment behavior + clear values for cache key)
        if let Some(ds) = &self.depth_stencil {
            ds.depth_test_enable.hash(state);
            ds.depth_write_enable.hash(state);
            (ds.depth_compare_op.as_raw() as i32).hash(state);
            ds.depth_bounds_test_enable.hash(state);
            (ds.depth_load_op.as_raw() as i32).hash(state);
            (ds.depth_store_op.as_raw() as i32).hash(state);
            ds.depth_clear_value.to_bits().hash(state);
            ds.stencil_test_enable.hash(state);
            (ds.stencil_load_op.as_raw() as i32).hash(state);
            (ds.stencil_store_op.as_raw() as i32).hash(state);
            ds.stencil_clear_value.hash(state);

            // front/back stencil op state
            hash_vk_stencil_op_state(&ds.stencil_front, state);
            hash_vk_stencil_op_state(&ds.stencil_back, state);
        }
        self.color_blend.hash(state);
        for ds in &self.dynamic_states {
            (ds.as_raw() as i32).hash(state);
        }
    }
}

impl PartialEq for GraphicPipelineState {
    fn eq(&self, other: &Self) -> bool {
        self.input_assembly == other.input_assembly
            && self.rasterization == other.rasterization
            && self.multisample == other.multisample
            && eq_depth_stencil_opt(&self.depth_stencil, &other.depth_stencil)
            && self.color_blend == other.color_blend
            && self.dynamic_states == other.dynamic_states
    }
}

impl Eq for GraphicPipelineState {}

impl GraphicPipelineState {
    #[inline]
    pub fn to_vk_dynamic_state(&self) -> vk::PipelineDynamicStateCreateInfo<'_> {
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&self.dynamic_states)
    }
}

#[derive(Default)]
pub struct GraphicPipelineStateBuilder {
    state: GraphicPipelineState,
}

impl GraphicPipelineStateBuilder {
    pub fn input_assembly(mut self, s: InputAssemblyState) -> Self {
        self.state.input_assembly = s;
        self
    }

    pub fn rasterization(mut self, s: RasterizationState) -> Self {
        self.state.rasterization = s;
        self
    }

    pub fn multisample(mut self, s: MultisampleState) -> Self {
        self.state.multisample = s;
        self
    }

    pub fn push_color_attachment(mut self, a: ColorAttachmentDesc) -> Self {
        self.state.color_blend.attachments.push(a);
        self
    }

    pub fn blend_constants(mut self, c: [i32; 4]) -> Self {
        self.state.color_blend.blend_constants = c;
        self
    }

    pub fn depth_stencil(mut self, ds: DepthStencilDesc) -> Self {
        self.state.depth_stencil = Some(ds);
        self
    }

    pub fn dynamic_states(mut self, states: Vec<vk::DynamicState>) -> Self {
        self.state.dynamic_states = states;
        self
    }

    pub fn build(self) -> GraphicPipelineState {
        self.state
    }
}

fn hash_vk_stencil_op_state<H: Hasher>(s: &vk::StencilOpState, state: &mut H) {
    (s.fail_op.as_raw() as i32).hash(state);
    (s.pass_op.as_raw() as i32).hash(state);
    (s.depth_fail_op.as_raw() as i32).hash(state);
    (s.compare_op.as_raw() as i32).hash(state);
    s.compare_mask.hash(state);
    s.write_mask.hash(state);
    s.reference.hash(state);
}

fn eq_vk_stencil_op_state(a: &vk::StencilOpState, b: &vk::StencilOpState) -> bool {
    a.fail_op == b.fail_op
        && a.pass_op == b.pass_op
        && a.depth_fail_op == b.depth_fail_op
        && a.compare_op == b.compare_op
        && a.compare_mask == b.compare_mask
        && a.write_mask == b.write_mask
        && a.reference == b.reference
}

fn eq_depth_stencil_opt(a: &Option<DepthStencilDesc>, b: &Option<DepthStencilDesc>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.depth_test_enable == b.depth_test_enable
                && a.depth_write_enable == b.depth_write_enable
                && a.depth_compare_op == b.depth_compare_op
                && a.depth_bounds_test_enable == b.depth_bounds_test_enable
                && a.depth_load_op == b.depth_load_op
                && a.depth_store_op == b.depth_store_op
                && a.depth_clear_value.to_bits() == b.depth_clear_value.to_bits()
                && a.stencil_test_enable == b.stencil_test_enable
                && a.stencil_load_op == b.stencil_load_op
                && a.stencil_store_op == b.stencil_store_op
                && a.stencil_clear_value == b.stencil_clear_value
                && eq_vk_stencil_op_state(&a.stencil_front, &b.stencil_front)
                && eq_vk_stencil_op_state(&a.stencil_back, &b.stencil_back)
        }
        _ => false,
    }
}

/// Hashable graphics pipeline description.
#[derive(Clone)]
pub struct GraphicPipelineDesc {
    pub shader: GraphicShaderInput,
    pub state: GraphicPipelineState,
    pub attachments: GraphicPipelineAttachments,
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

impl GraphicPipelineDesc {
    pub fn new(shader: GraphicShaderInput, state: GraphicPipelineState, attachments: GraphicPipelineAttachments) -> Self {
        Self { shader, state, attachments }
    }
}

impl Hash for GraphicPipelineDesc {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.shader.hash(state);
        self.state.hash(state);
        self.attachments.hash(state);
    }
}

impl PartialEq for GraphicPipelineDesc {
    fn eq(&self, other: &Self) -> bool {
        self.shader == other.shader && self.state == other.state && self.attachments == other.attachments
    }
}

impl Eq for GraphicPipelineDesc {}

#[DeviceObject]
pub struct CommonPipeline {
    name: String,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl CommonPipeline {
    /// Create a new pipeline layout.
    pub fn new_graphic(
        name: &str,
        device: &RenderDevice,
        desc: &GraphicPipelineDesc,
        cache: vk::PipelineCache,
    ) -> Result<(Vec<DescriptorSetLayout>, Self), vk::Result> {
        let max_set = desc.shader.merged_reflection.max_set().unwrap_or(0);

        let layouts = (0..=max_set).into_iter()
            .map(|idx| {
                let name = format!("descriptor_layout.s{idx}");
                DescriptorSetLayout::from_reflection(&name, device, &desc.shader.merged_reflection.bindings, idx)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let layout = desc.shader.create_pipeline_layout(device, &layouts)?;

        // Build shader stages (fragment shader is optional)
        let mut shader_stages = Vec::with_capacity(if desc.shader.fragment_shader.is_some() { 2 } else { 1 });
        shader_stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(desc.shader.vertex_shader.vk_stage())
                .module(desc.shader.vertex_shader.module())
                .name(desc.shader.vertex_shader.entry_point()),
        );
        if let Some(fs) = &desc.shader.fragment_shader {
            shader_stages.push(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(fs.vk_stage())
                    .module(fs.module())
                    .name(fs.entry_point()),
            );
        }

        // Vertex input
        let bindings: Vec<vk::VertexInputBindingDescription> = desc
            .shader
            .vertex_bindings
            .iter()
            .map(|b| {
                vk::VertexInputBindingDescription::default()
                    .binding(b.binding)
                    .stride(b.stride)
                    .input_rate(b.input_rate)
            })
            .collect();

        let attributes: Vec<vk::VertexInputAttributeDescription> = desc
            .shader
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

        // Input assembly / rasterization / multisample
        let input_assembly = desc.state.input_assembly.to_vk();
        let rasterization = desc.state.rasterization.to_vk();
        let multisample = desc.state.multisample.to_vk();

        // Depth stencil (if provided)
        let depth_stencil_state = desc
            .state
            .depth_stencil
            .as_ref()
            .map(|ds| ds.to_vk());

        // Color blend
        let blend_attachments = desc.state.color_blend.to_vk_attachments();
        let color_blend_state = desc.state.color_blend.to_vk(&blend_attachments);

        // Viewport state (dynamic)
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        // Dynamic state
        let dynamic_state = desc.state.to_vk_dynamic_state();

        // Dynamic rendering info
        let mut rendering_info = desc.attachments.to_vk_rendering_info();

        // Build pipeline create info
        let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .push_next(&mut rendering_info);

        if let Some(ref depth_stencil) = depth_stencil_state {
            pipeline_info = pipeline_info.depth_stencil_state(depth_stencil);
        }

        let pipelines = unsafe { device.handle().create_graphics_pipelines(cache, &[pipeline_info], None) }
            .map_err(|e| e.1)?;

        log::trace!("create graphic pipeline.");

        let pipeline = Self {
            name: name.to_owned(),
            layout,
            pipeline: pipelines[0],
            device: device.handle().clone(),
        };
        device.set_debug_name(&pipeline);
        Ok((layouts, pipeline))
    }

    #[inline]
    pub fn name(&self) -> &str { &self.name }

    #[inline]
    pub fn handle(&self) -> vk::Pipeline { self.pipeline }
}

impl Drop for CommonPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
        }

        log::trace!("destroy graphic pipeline.");
    }
}

impl DebuggableObject for CommonPipeline {
    fn set_debug_name(&self, device: &RenderDevice) {
        // Name both pipeline and layout for better debugging.
        set_debug_name_handle(device, self.pipeline, vk::ObjectType::PIPELINE, self.name());
        set_debug_name_handle(
            device,
            self.layout,
            vk::ObjectType::PIPELINE_LAYOUT,
            &format!("pipeline_layout.{}", self.name()),
        );
    }
}

/// Graphics pipeline using dynamic rendering (Vulkan 1.3+).
pub struct GraphicPipeline {
    pipeline: CommonPipeline,
    pub(crate) descriptor_layouts: Vec<DescriptorSetLayout>,
}

impl GraphicPipeline {
    /// Create a new graphics pipeline with dynamic rendering.
    pub fn new(
        name: &str,
        device: &RenderDevice,
        desc: &GraphicPipelineDesc,
    ) -> Result<Self, vk::Result> {
        Self::with_cache(name, device, desc, vk::PipelineCache::null())
    }

    /// Create a new graphics pipeline with a pipeline cache.
    pub fn with_cache(
        name: &str,
        device: &RenderDevice,
        desc: &GraphicPipelineDesc,
        cache: vk::PipelineCache,
    ) -> Result<Self, vk::Result> {
        let (layouts, pipeline) = CommonPipeline::new_graphic(name, device, desc, cache)?;
        Ok(Self {
            descriptor_layouts: layouts,
            pipeline
        })
    }

    /// Get the raw Vulkan pipeline handle.
    pub fn handle(&self) -> vk::Pipeline {
        self.pipeline.pipeline
    }

    pub fn layout(&self) -> vk::PipelineLayout { self.pipeline.layout }
}
