//! Vulkan Shader - HLSL compilation and SPIR-V reflection.

use ash::{vk, Device};
use hassle_rs::{Dxc, HassleError};
use rspirv_reflect::{Reflection, DescriptorType, BindingCount};
use std::ffi::CString;
use std::sync::Arc;
use crate::descriptor::DescriptorSetLayout;

/// Shader compilation and reflection errors.
#[derive(Debug)]
pub enum ShaderError {
    CompilationFailed(String),
    ReflectionFailed(String),
    VulkanError(vk::Result),
    IoError(std::io::Error),
}

impl From<vk::Result> for ShaderError {
    fn from(e: vk::Result) -> Self {
        ShaderError::VulkanError(e)
    }
}

impl From<std::io::Error> for ShaderError {
    fn from(e: std::io::Error) -> Self {
        ShaderError::IoError(e)
    }
}

impl From<HassleError> for ShaderError {
    fn from(e: HassleError) -> Self {
        ShaderError::CompilationFailed(format!("{:?}", e))
    }
}

impl std::fmt::Display for ShaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShaderError::CompilationFailed(msg) => write!(f, "Shader compilation failed: {}", msg),
            ShaderError::ReflectionFailed(msg) => write!(f, "Shader reflection failed: {}", msg),
            ShaderError::VulkanError(e) => write!(f, "Vulkan error: {:?}", e),
            ShaderError::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for ShaderError {}

/// Shader stage type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

impl ShaderStage {
    /// Get the DXC target profile string for this stage.
    fn to_profile(&self, shader_model: &str) -> String {
        let prefix = match self {
            ShaderStage::Vertex => "vs",
            ShaderStage::Fragment => "ps",
            ShaderStage::Compute => "cs",
        };
        format!("{}_{}", prefix, shader_model)
    }

    /// Convert to Vulkan shader stage flags.
    pub fn to_vk_stage(&self) -> vk::ShaderStageFlags {
        match self {
            ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
            ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
            ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
        }
    }
}

/// A single shader resource binding.
#[derive(Debug, Clone)]
pub struct ShaderBinding {
    pub name: String,
    pub set: u32,
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub stage_flags: vk::ShaderStageFlags,
    pub count: u32,
}

/// Shader reflection data.
#[derive(Debug, Clone, Default)]
pub struct ShaderReflection {
    pub bindings: Vec<ShaderBinding>,
    pub push_constant_size: u32,
}

impl ShaderReflection {
    /// Merge multiple shader reflections into one.
    /// Combines stage_flags for bindings at the same (set, binding).
    pub fn merge(reflections: &[&ShaderReflection]) -> Self {
        use std::collections::HashMap;

        let mut binding_map: HashMap<(u32, u32), ShaderBinding> = HashMap::new();
        let mut push_constant_size = 0u32;

        for reflection in reflections {
            push_constant_size = push_constant_size.max(reflection.push_constant_size);

            for binding in &reflection.bindings {
                let key = (binding.set, binding.binding);
                if let Some(existing) = binding_map.get_mut(&key) {
                    existing.stage_flags |= binding.stage_flags;
                } else {
                    binding_map.insert(key, binding.clone());
                }
            }
        }

        let mut bindings: Vec<ShaderBinding> = binding_map.into_values().collect();
        bindings.sort_by_key(|b| (b.set, b.binding));

        Self {
            bindings,
            push_constant_size,
        }
    }

    /// Find a binding by name.
    pub fn find_binding(&self, name: &str) -> Option<&ShaderBinding> {
        self.bindings.iter().find(|b| b.name == name)
    }

    /// Get the maximum set index used.
    pub fn max_set(&self) -> Option<u32> {
        self.bindings.iter().map(|b| b.set).max()
    }
}

/// Compile HLSL source to SPIR-V bytecode using DXC.
pub fn compile_hlsl(
    source: &str,
    entry_point: &str,
    stage: ShaderStage,
    shader_model: &str,
) -> Result<Vec<u8>, ShaderError> {
    let dxc = Dxc::new(None)?;
    let compiler = dxc.create_compiler()?;
    let library = dxc.create_library()?;

    let blob = library.create_blob_with_encoding_from_str(source)?;

    let profile = stage.to_profile(shader_model);

    let args = [
        "-spirv",
        "-fspv-target-env=vulkan1.2",
        "-fvk-use-scalar-layout",
        "-fvk-use-dx-position-w",
        "-Zpc", // Pack matrices in column-major order
    ];

    let result = compiler.compile(
        &blob,
        "shader.hlsl",
        entry_point,
        &profile,
        &args,
        None,
        &[],
    );

    match result {
        Ok(compiled) => {
            let result_blob = compiled.get_result()?;
            Ok(result_blob.to_vec())
        }
        Err(e) => {
            let error_msg = e
                .0
                .get_error_buffer()
                .ok()
                .and_then(|buf| library.get_blob_as_string(&buf.into()).ok())
                .unwrap_or_else(|| "Unknown hlsl compilation error".to_string());
            Err(ShaderError::CompilationFailed(error_msg))
        }
    }
}

/// Reflect SPIR-V bytecode to extract resource bindings using rspirv_reflect.
pub fn reflect_spirv(spirv: &[u8], stage: ShaderStage) -> Result<ShaderReflection, ShaderError> {
    // rspirv_reflect takes &[u8] directly
    if spirv.len() % 4 != 0 {
        return Err(ShaderError::ReflectionFailed("SPIR-V must be 4-byte aligned".to_string()));
    }

    let reflection = match Reflection::new_from_spirv(spirv) {
        Ok(r) => r,
        Err(e) => {
            return Err(ShaderError::ReflectionFailed(format!("{:?}", e)));
        }
    };

    let mut bindings = Vec::new();
    let stage_flags = stage.to_vk_stage();

    // Get descriptor bindings
    let descriptor_sets = reflection.get_descriptor_sets();
    if let Ok(sets) = descriptor_sets {
        for (set_index, set_bindings) in sets.iter() {
            for (binding_index, binding_info) in set_bindings.iter() {
                let descriptor_type = convert_descriptor_type(binding_info.ty);
                let count = match &binding_info.binding_count {
                    BindingCount::One => 1,
                    BindingCount::StaticSized(n) => *n as u32,
                    BindingCount::Unbounded => u32::MAX,
                };

                bindings.push(ShaderBinding {
                    name: binding_info.name.clone(),
                    set: *set_index,
                    binding: *binding_index,
                    descriptor_type,
                    stage_flags,
                    count,
                });
            }
        }
    }

    // Get push constants
    let push_constant_size = reflection
        .get_push_constant_range()
        .ok()
        .flatten()
        .map(|info| info.size)
        .unwrap_or(0);

    Ok(ShaderReflection {
        bindings,
        push_constant_size,
    })
}

/// Convert rspirv_reflect descriptor type to Vulkan descriptor type.
fn convert_descriptor_type(reflect_type: DescriptorType) -> vk::DescriptorType {
    // DescriptorType is a transparent wrapper around u32, matching Vulkan values
    vk::DescriptorType::from_raw(reflect_type.0 as i32)
}

/// Compiled shader with Vulkan shader module and reflection data.
pub struct Shader {
    device: Device,
    module: vk::ShaderModule,
    stage: ShaderStage,
    entry_point: CString,
    reflection: ShaderReflection,
    /// Cached descriptor set layouts derived from this shader's reflection.
    descriptor_set_layouts: Vec<Arc<DescriptorSetLayout>>,
}

impl Shader {
    /// Create a shader from HLSL source code.
    pub fn from_hlsl(
        device: &Device,
        source: &str,
        entry_point: &str,
        stage: ShaderStage,
        shader_model: &str,
    ) -> Result<Self, ShaderError> {
        let spirv = compile_hlsl(source, entry_point, stage, shader_model)?;
        Self::from_spirv(device, &spirv, entry_point, stage)
    }

    /// Create a shader from an HLSL file.
    pub fn from_hlsl_file(
        device: &Device,
        path: impl AsRef<std::path::Path>,
        entry_point: &str,
        stage: ShaderStage,
        shader_model: &str,
    ) -> Result<Self, ShaderError> {
        let source = std::fs::read_to_string(path)?;
        Self::from_hlsl(device, &source, entry_point, stage, shader_model)
    }

    /// Create a shader from pre-compiled SPIR-V bytecode.
    pub fn from_spirv(
        device: &Device,
        spirv: &[u8],
        entry_point: &str,
        stage: ShaderStage,
    ) -> Result<Self, ShaderError> {
        // Reflect the shader
        let reflection = reflect_spirv(spirv, stage)?;

        // Create shader module
        let module = create_shader_module(device, spirv)?;

        // Create and cache descriptor set layouts
        let descriptor_set_layouts = crate::descriptor::create_layouts_from_reflection(device, &reflection)
            .map_err(ShaderError::VulkanError)?;

        Ok(Self {
            device: device.clone(),
            module,
            stage,
            entry_point: CString::new(entry_point).unwrap(),
            reflection,
            descriptor_set_layouts,
        })
    }

    /// Get the Vulkan shader module handle.
    pub fn module(&self) -> vk::ShaderModule {
        self.module
    }

    /// Get the shader stage.
    pub fn stage(&self) -> ShaderStage {
        self.stage
    }

    /// Get the entry point name.
    pub fn entry_point(&self) -> &CString {
        &self.entry_point
    }

    /// Get the shader reflection data.
    pub fn reflection(&self) -> &ShaderReflection {
        &self.reflection
    }

    /// Get Vulkan shader stage flags.
    pub fn vk_stage(&self) -> vk::ShaderStageFlags {
        self.stage.to_vk_stage()
    }

    /// Get the cached descriptor set layouts for this shader.
    pub fn descriptor_set_layouts(&self) -> &[Arc<DescriptorSetLayout>] {
        &self.descriptor_set_layouts
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.module, None);
        }
    }
}

/// Create a Vulkan shader module from SPIR-V bytecode.
fn create_shader_module(device: &Device, spirv: &[u8]) -> Result<vk::ShaderModule, ShaderError> {
    assert_eq!(spirv.len() % 4, 0, "SPIR-V bytecode must be 4-byte aligned");

    let code: &[u32] = unsafe { std::slice::from_raw_parts(spirv.as_ptr() as *const u32, spirv.len() / 4) };

    let create_info = vk::ShaderModuleCreateInfo::default().code(code);
    let module = unsafe { device.create_shader_module(&create_info, None)? };

    Ok(module)
}
