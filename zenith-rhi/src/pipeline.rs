//! Vulkan Pipeline - pipeline layout and graphics pipeline management.

use zenith_core::log;
use ash::{vk, Device};

pub struct CommonPipeline {
    device: Device,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl CommonPipeline {
    /// Create a new pipeline layout.
    pub fn new_graphic(
        device: &Device,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
        info: &GraphicPipelineInfo,
        cache: vk::PipelineCache,
    ) -> Result<Self, vk::Result> {
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);

        let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        let entry_point = c"main";

        // Shader stages
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(info.vertex_shader)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(info.fragment_shader)
                .name(entry_point),
        ];

        // Viewport state (dynamic)
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        // Dynamic state
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(info.dynamic_states);

        // Dynamic rendering info
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(info.color_attachment_formats);

        if let Some(depth_format) = info.depth_attachment_format {
            rendering_info = rendering_info.depth_attachment_format(depth_format);
        }

        // Build pipeline create info
        let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&info.vertex_input_state)
            .input_assembly_state(&info.input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&info.rasterization_state)
            .multisample_state(&info.multisample_state)
            .color_blend_state(&info.color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .push_next(&mut rendering_info);

        if let Some(ref depth_stencil) = info.depth_stencil_state {
            pipeline_info = pipeline_info.depth_stencil_state(depth_stencil);
        }

        let pipelines =
            unsafe { device.create_graphics_pipelines(cache, &[pipeline_info], None) }
                .map_err(|e| e.1)?;

        log::info!("create common graphic pipeline!");

        Ok(Self {
            device: device.clone(),
            layout,
            pipeline: pipelines[0],
        })
    }

    /// Get the raw Vulkan pipeline layout handle.
    pub fn handle(&self) -> vk::PipelineLayout {
        self.layout
    }
}

impl Drop for CommonPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

/// Graphics pipeline configuration.
pub struct GraphicPipelineInfo<'a> {
    pub vertex_shader: vk::ShaderModule,
    pub fragment_shader: vk::ShaderModule,
    pub vertex_input_state: vk::PipelineVertexInputStateCreateInfo<'a>,
    pub input_assembly_state: vk::PipelineInputAssemblyStateCreateInfo<'a>,
    pub rasterization_state: vk::PipelineRasterizationStateCreateInfo<'a>,
    pub multisample_state: vk::PipelineMultisampleStateCreateInfo<'a>,
    pub depth_stencil_state: Option<vk::PipelineDepthStencilStateCreateInfo<'a>>,
    pub color_blend_state: vk::PipelineColorBlendStateCreateInfo<'a>,
    pub dynamic_states: &'a [vk::DynamicState],
    pub color_attachment_formats: &'a [vk::Format],
    pub depth_attachment_format: Option<vk::Format>,
}

/// Graphics pipeline using dynamic rendering (Vulkan 1.3+).
pub struct GraphicPipeline {
    pipeline: CommonPipeline,
}

impl GraphicPipeline {
    /// Create a new graphics pipeline with dynamic rendering.
    pub fn new(
        device: &Device,
        info: &GraphicPipelineInfo,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange]
    ) -> Result<Self, vk::Result> {
        Self::with_cache(device, descriptor_set_layouts, push_constant_ranges, info, vk::PipelineCache::null())
    }

    /// Create a new graphics pipeline with a pipeline cache.
    pub fn with_cache(
        device: &Device,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
        info: &GraphicPipelineInfo,
        cache: vk::PipelineCache,
    ) -> Result<Self, vk::Result> {
        let pipeline = CommonPipeline::new_graphic(device, descriptor_set_layouts, push_constant_ranges, info, cache)?;
        Ok(Self {
            pipeline
        })
    }

    /// Get the raw Vulkan pipeline handle.
    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline.pipeline
    }

    pub fn layout(&self) -> vk::PipelineLayout { self.pipeline.layout }
}
