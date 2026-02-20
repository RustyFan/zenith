//! Vulkan Shader - Slang compilation and reflection.
//!
//! All shaders are compiled via the Slang C API ([`slang_api::compile_slang_and_reflect`]).
//! Reflection (bindings, push constants, vertex inputs) is extracted from Slang's layout.

use ash::{vk, Device};
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Arc;
use zenith_rhi_derive::DeviceObject;
use crate::RenderDevice;
use crate::device::DebuggableObject;
use crate::device::set_debug_name_handle;
use std::path::Path;

#[DeviceObject]
pub struct Shader {
    name: String,
    module: vk::ShaderModule,
    stage: ShaderStage,
    entry_point: CString,
    reflection: ShaderReflection,
}

impl Shader {
    pub fn from_file(
        name: &str,
        device: &Arc<RenderDevice>,
        path: impl AsRef<Path>,
        stage: ShaderStage,
    ) -> Result<Self, ShaderError> {
        let path = path.as_ref();

        #[cfg(debug_assertions)]
        let debug = true;
        #[cfg(not(debug_assertions))]
        let debug = false;

        let (spirv, reflection) = crate::slang_compiler::compile_slang_and_reflect(name, path, stage, device.bindless_caps(), debug)?;
        let module = create_shader_module(device.handle(), &spirv)?;

        let shader = Self {
            name: name.to_owned(),
            module,
            stage,
            entry_point: CString::new("main").unwrap(),
            reflection,
            device: device.clone(),
        };
        device.set_debug_name(&shader, name);

        Ok(shader)
    }

    #[inline]
    pub fn name(&self) -> &str { &self.name }

    #[inline]
    pub fn handle(&self) -> vk::ShaderModule { self.module }

    #[inline]
    pub fn module(&self) -> vk::ShaderModule {
        self.module
    }

    #[inline]
    pub fn stage(&self) -> ShaderStage {
        self.stage
    }

    #[inline]
    pub fn entry_point(&self) -> &CString {
        &self.entry_point
    }

    #[inline]
    pub fn reflection(&self) -> &ShaderReflection {
        &self.reflection
    }

    #[inline]
    pub fn vk_stage(&self) -> vk::ShaderStageFlags {
        self.stage.to_vk()
    }
}

impl DebuggableObject for Shader {
    fn set_debug_name(&self, device: &ash::ext::debug_utils::Device, name: &str) {
        set_debug_name_handle(device, self.module, vk::ObjectType::SHADER_MODULE, name);
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            self.device.handle().destroy_shader_module(self.module, None);
        }
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

impl ShaderStage {
    pub fn to_vk(&self) -> vk::ShaderStageFlags {
        match self {
            ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
            ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
            ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShaderBinding {
    pub name: String,
    pub set: u32,
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub stage_flags: vk::ShaderStageFlags,
    pub count: u32,
    pub binding_flags: vk::DescriptorBindingFlags,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexInputAttribute {
    pub location: u32,
    pub format: vk::Format,
}

#[derive(Debug, Clone, Default)]
pub struct ShaderReflection {
    pub bindings: Vec<ShaderBinding>,
    pub push_constant_size: u32,
    pub vertex_inputs: Vec<VertexInputAttribute>,
}

impl ShaderReflection {
    pub fn merge(reflections: &[&ShaderReflection]) -> Self {
        if reflections.len() == 1 {
            return reflections[0].clone();
        }
        
        let mut binding_map: HashMap<(u32, u32), ShaderBinding> = HashMap::new();
        let mut push_constant_size = 0u32;
        let mut vertex_inputs_map: HashMap<u32, vk::Format> = HashMap::new();

        for reflection in reflections {
            push_constant_size = push_constant_size.max(reflection.push_constant_size);

            for binding in &reflection.bindings {
                let key = (binding.set, binding.binding);
                if let Some(existing) = binding_map.get_mut(&key) {
                    existing.stage_flags |= binding.stage_flags;
                    existing.binding_flags |= binding.binding_flags;
                } else {
                    binding_map.insert(key, binding.clone());
                }
            }

            // Merge vertex inputs by location (first wins on conflicts).
            for vi in &reflection.vertex_inputs {
                vertex_inputs_map.entry(vi.location).or_insert(vi.format);
            }
        }

        let mut bindings: Vec<ShaderBinding> = binding_map.into_values().collect();
        bindings.sort_by_key(|b| (b.set, b.binding));

        let mut vertex_inputs: Vec<VertexInputAttribute> = vertex_inputs_map
            .into_iter()
            .map(|(location, format)| VertexInputAttribute { location, format })
            .collect();
        vertex_inputs.sort_by_key(|v| v.location);

        Self {
            bindings,
            push_constant_size,
            vertex_inputs,
        }
    }

    pub fn find_binding(&self, name: &str) -> Option<&ShaderBinding> {
        self.bindings.iter().find(|b| b.name == name)
    }

    pub fn max_set(&self) -> Option<u32> {
        self.bindings.iter().map(|b| b.set).max()
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
