//! Slang Compilation API integration.
//!
//! Uses Slang's C API (via `shader-slang` crate) as the sole shader compiler.
//! A `SlangCompiler` holds a reusable `GlobalSession` and is stored in a thread-local
//! to avoid recreating it per shader. Each compilation creates a lightweight `Session`
//! with search paths configured for `#include` resolution.

use crate::shader::{ShaderBinding, ShaderError, ShaderReflection, ShaderStage, VertexInputAttribute};
use ash::vk;
use shader_slang::{
    BindingType, CompilerOptions, ComponentType, GlobalSession, ParameterCategory, ScalarType,
    SessionDesc, Stage, TargetDesc, TypeKind,
    reflection::{self, TypeLayout},
};
use std::cell::RefCell;
use std::ffi::CString;
use std::path::Path;
use crate::BindlessCaps;

/// Persistent Slang compiler that reuses a `GlobalSession` across compilations.
struct SlangCompiler {
    global_session: GlobalSession,
}

impl SlangCompiler {
    fn new() -> Result<Self, ShaderError> {
        let global_session = GlobalSession::new().ok_or_else(|| {
            ShaderError::CompilationFailed("Slang GlobalSession::new() failed".into())
        })?;
        Ok(Self { global_session })
    }

    fn compile(
        &self,
        _name: &str,
        path: &Path,
        stage: ShaderStage,
        caps: &BindlessCaps,
        debug: bool,
    ) -> Result<(Vec<u8>, ShaderReflection), ShaderError> {
        let source = std::fs::read_to_string(path).map_err(|e| {
            ShaderError::CompilationFailed(format!("failed to read shader file: {}", e))
        })?;
        let path_str = path
            .to_str()
            .ok_or_else(|| ShaderError::CompilationFailed("shader path not UTF-8".into()))?;
        let module_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("module");

        // Build compiler options.
        let target_stage = match stage {
            ShaderStage::Vertex => Stage::Vertex,
            ShaderStage::Fragment => Stage::Fragment,
            ShaderStage::Compute => Stage::Compute,
        };

        let profile = self.global_session.find_profile("spirv_1_5");
        let mut options = CompilerOptions::default()
            .target(shader_slang::CompileTarget::Spirv)
            .stage(target_stage)
            .matrix_layout_column(true);

        if debug {
            options = options
                .debug_information(shader_slang::DebugInfoLevel::Maximal)
                .optimization(shader_slang::OptimizationLevel::None);
        } else {
            options = options
                .debug_information(shader_slang::DebugInfoLevel::None)
                .optimization(shader_slang::OptimizationLevel::Maximal);
        }

        let target = TargetDesc::default()
            .format(shader_slang::CompileTarget::Spirv)
            .profile(profile)
            .options(&options);

        // Configure search paths so #include "foo.slang" resolves from the shader's directory.
        let search_dir = path
            .parent()
            .and_then(|p| p.canonicalize().ok())
            .ok_or_else(|| {
                ShaderError::CompilationFailed("shader path has no parent directory".into())
            })?;
        let search_dir_str = search_dir
            .to_str()
            .ok_or_else(|| ShaderError::CompilationFailed("shader dir not UTF-8".into()))?;
        let search_dir_cstr = CString::new(search_dir_str)
            .map_err(|_| ShaderError::CompilationFailed("shader dir contains null byte".into()))?;
        let search_paths: Vec<*const i8> = vec![search_dir_cstr.as_ptr()];

        let session_desc = SessionDesc::default()
            .targets(std::slice::from_ref(&target))
            .search_paths(&search_paths);
        let session = self
            .global_session
            .create_session(&session_desc)
            .ok_or_else(|| ShaderError::CompilationFailed("Slang create_session failed".into()))?;

        // Load the module from source.
        let module = session
            .load_module_from_source_string(module_name, path_str, &source)
            .map_err(|e| {
                ShaderError::CompilationFailed(format!("Slang load_module failed: {:?}", e))
            })?;

        // Collect all entry points from the module, then link them together.
        // After linking, we use the layout to find which entry point index matches
        // the requested stage, and extract SPIR-V for that one only.
        let entry_points: Vec<shader_slang::EntryPoint> = module.entry_points().collect();
        if entry_points.is_empty() {
            return Err(ShaderError::CompilationFailed(
                "no entry points found in shader module".into(),
            ));
        }

        let mut components: Vec<ComponentType> = Vec::with_capacity(1 + entry_points.len());
        components.push(ComponentType::from(module.clone()));
        for ep in &entry_points {
            components.push(ComponentType::from(ep.clone()));
        }

        let composite = session.create_composite_component_type(&components).map_err(|e| {
            ShaderError::CompilationFailed(format!(
                "Slang create_composite_component_type failed: {:?}",
                e
            ))
        })?;

        let linked = composite.link().map_err(|e| {
            ShaderError::CompilationFailed(format!("Slang link failed: {:?}", e))
        })?;

        let layout = linked.layout(0).map_err(|e| {
            ShaderError::ReflectionFailed(format!("Slang getLayout failed: {:?}", e))
        })?;

        // Find the entry point matching the requested stage.
        let ep_index = layout
            .entry_points()
            .enumerate()
            .find(|(_, ep)| ep.stage() == target_stage)
            .map(|(i, _)| i)
            .ok_or_else(|| {
                ShaderError::CompilationFailed(format!(
                    "no entry point found for stage {:?} in {:?}",
                    stage, path
                ))
            })?;

        // Extract SPIR-V for the matched entry point.
        let code = linked.entry_point_code(ep_index as i64, 0).map_err(|e| {
            ShaderError::CompilationFailed(format!("Slang getEntryPointCode failed: {:?}", e))
        })?;
        let spirv = code.as_slice().to_vec();

        // Build reflection from the layout.
        let reflection = build_reflection(stage, layout, ep_index, caps)?;
        Ok((spirv, reflection))
    }
}

thread_local! {
    static SLANG_COMPILER: RefCell<Option<SlangCompiler>> = const { RefCell::new(None) };
}

/// Compile a Slang file and produce SPIR-V plus reflection.
/// Uses the Slang C API in-process; the `GlobalSession` is reused across calls.
pub fn compile_slang_and_reflect(
    name: &str,
    path: &Path,
    stage: ShaderStage,
    caps: &BindlessCaps,
    debug: bool,
) -> Result<(Vec<u8>, ShaderReflection), ShaderError> {
    SLANG_COMPILER.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            *opt = Some(SlangCompiler::new()?);
        }
        opt.as_ref().unwrap().compile(name, path, stage, caps, debug)
    })
}

// ---------------------------------------------------------------------------
// Reflection
// ---------------------------------------------------------------------------

/// Build `ShaderReflection` from Slang's program layout.
///
/// - Descriptor bindings and push constants come from `layout.parameters()`.
/// - Vertex inputs come from the entry point's parameters (handles both flat
///   and struct-typed inputs).
/// - Bindless set 0 is normalized via `BindlessPool::canonical_shader_bindings()`.
fn build_reflection(
    stage: ShaderStage,
    layout: &reflection::Shader,
    entry_point_index: usize,
    caps: &BindlessCaps,
) -> Result<ShaderReflection, ShaderError> {
    let stage_flags = stage.to_vk();
    let mut bindings = Vec::new();
    let mut push_constant_size = 0u32;

    // Global parameters -> descriptor bindings / push constants.
    for param in layout.parameters() {
        let type_layout = match param.type_layout() {
            Some(t) => t,
            None => continue,
        };

        let set = param.binding_space();
        let binding = param.binding_index();
        let name = param.name().unwrap_or("").to_string();

        // Check if this parameter is a push constant.
        let is_push_constant = param.category() == Some(ParameterCategory::PushConstantBuffer)
            || type_layout.categories().any(|c| c == ParameterCategory::PushConstantBuffer);

        if is_push_constant {
            // Push constant: extract the size of the inner data.
            // For ConstantBuffer<T>, the element type layout gives the size of T.
            let size = type_layout
                .element_type_layout()
                .map(|el| el.size(ParameterCategory::Uniform))
                .unwrap_or_else(|| type_layout.size(ParameterCategory::Uniform));
            push_constant_size = push_constant_size.max(size as u32);
        } else if let Some(dty) = slang_type_layout_to_vk_descriptor_type(type_layout) {
            let count = type_layout
                .ty()
                .map(|t| t.total_array_element_count())
                .filter(|&c| c > 0)
                .unwrap_or(1) as u32;
            let count = if count > 1 { count } else { 1 };

            let binding_flags = if name.starts_with(crate::BindlessPool::BINDING_PREFIX) {
                vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            } else {
                vk::DescriptorBindingFlags::empty()
            };

            bindings.push(ShaderBinding {
                name,
                set,
                binding,
                descriptor_type: dty,
                stage_flags,
                count: if count == 0 || count > 0x7FFF { u32::MAX } else { count },
                binding_flags,
            });
        } else {
            // Non-descriptor, non-push-constant parameter (e.g. specialization constants).
            // Try to detect any remaining uniform data as push constants.
            for cat in type_layout.categories() {
                if cat == ParameterCategory::Uniform {
                    let size = type_layout.size(cat);
                    push_constant_size = push_constant_size.max(size as u32);
                    break;
                }
            }
        }
    }

    // Bindless normalization for set 0 - ensure bindless bindings use canonical names and types
    let has_bindless_set0 = bindings.iter().any(|b| {
        b.set == crate::BindlessPool::SET_INDEX && b.name.starts_with(crate::BindlessPool::BINDING_PREFIX)
    });
    if has_bindless_set0 {
        for canonical in crate::BindlessPool::shader_bindings(caps) {
            if let Some(existing) = bindings
                .iter_mut()
                .find(|b| b.set == canonical.set && b.binding == canonical.binding)
            {
                existing.name = canonical.name.clone();
                existing.descriptor_type = canonical.descriptor_type;
                existing.stage_flags = canonical.stage_flags;
                existing.count = canonical.count;
                existing.binding_flags = canonical.binding_flags;
            } else {
                bindings.push(canonical);
            }
        }
    }

    // Vertex inputs from entry point parameters (vertex stage only).
    let vertex_inputs = if stage == ShaderStage::Vertex {
        let entry = layout.entry_point_by_index(entry_point_index as u32).ok_or_else(|| {
            ShaderError::ReflectionFailed("entry point not found in layout".into())
        })?;
        reflect_vertex_inputs(entry)?
    } else {
        Vec::new()
    };

    Ok(ShaderReflection {
        bindings,
        push_constant_size,
        vertex_inputs,
    })
}

/// Extract vertex input attributes from a Slang entry point's parameters.
///
/// Handles two patterns:
/// - **Struct parameter**: `main(VSInput input)` -- iterates struct fields for per-member locations.
/// - **Flat parameters**: `vsmain(float3 position, float3 color)` -- each parameter is one attribute.
fn reflect_vertex_inputs(
    entry: &reflection::EntryPoint,
) -> Result<Vec<VertexInputAttribute>, ShaderError> {
    let mut out = Vec::new();

    for param in entry.parameters() {
        let type_layout = match param.type_layout() {
            Some(t) => t,
            None => continue,
        };

        let base_location = param.offset(ParameterCategory::VaryingInput) as u32;

        match type_layout.kind() {
            TypeKind::Struct => {
                // Struct parameter: iterate fields for individual vertex attributes.
                for field in type_layout.fields() {
                    let field_tl = match field.type_layout() {
                        Some(t) => t,
                        None => continue,
                    };
                    let location =
                        base_location + field.offset(ParameterCategory::VaryingInput) as u32;
                    let format = slang_type_layout_to_vertex_format(field_tl)?;
                    out.push(VertexInputAttribute { location, format });
                }
            }
            _ => {
                // Scalar / vector parameter: single vertex attribute.
                let format = slang_type_layout_to_vertex_format(type_layout)?;
                out.push(VertexInputAttribute {
                    location: base_location,
                    format,
                });
            }
        }
    }

    out.sort_by_key(|a| a.location);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Type mapping helpers
// ---------------------------------------------------------------------------

fn slang_type_layout_to_vk_descriptor_type(type_layout: &TypeLayout) -> Option<vk::DescriptorType> {
    // For array types, unwrap to the element type.
    let type_layout = type_layout.unwrap_array();

    match type_layout.kind() {
        TypeKind::Resource => {
            if type_layout.binding_range_count() > 0 {
                let binding_type = type_layout.binding_range_type(0);
                let dty = match binding_type {
                    BindingType::Sampler => vk::DescriptorType::SAMPLER,
                    BindingType::Texture => vk::DescriptorType::SAMPLED_IMAGE,
                    BindingType::MutableTeture => vk::DescriptorType::STORAGE_IMAGE,
                    BindingType::ConstantBuffer => vk::DescriptorType::UNIFORM_BUFFER,
                    BindingType::RawBuffer | BindingType::MutableRawBuffer => {
                        vk::DescriptorType::STORAGE_BUFFER
                    }
                    BindingType::TypedBuffer | BindingType::MutableTypedBuffer => {
                        vk::DescriptorType::STORAGE_BUFFER
                    }
                    _ => return None,
                };
                return Some(dty);
            }
            None
        }
        TypeKind::SamplerState => Some(vk::DescriptorType::SAMPLER),
        TypeKind::ConstantBuffer | TypeKind::ParameterBlock => {
            Some(vk::DescriptorType::UNIFORM_BUFFER)
        }
        TypeKind::ShaderStorageBuffer => Some(vk::DescriptorType::STORAGE_BUFFER),
        TypeKind::TextureBuffer => Some(vk::DescriptorType::UNIFORM_TEXEL_BUFFER),
        _ => None,
    }
}

fn slang_type_layout_to_vertex_format(type_layout: &TypeLayout) -> Result<vk::Format, ShaderError> {
    let ty = type_layout.ty().ok_or_else(|| {
        ShaderError::ReflectionFailed("vertex input type_layout has no type".into())
    })?;

    match type_layout.kind() {
        TypeKind::Scalar => {
            let scalar = ty.scalar_type();
            match scalar {
                ScalarType::Float32 => Ok(vk::Format::R32_SFLOAT),
                ScalarType::Int32 => Ok(vk::Format::R32_SINT),
                ScalarType::Uint32 => Ok(vk::Format::R32_UINT),
                _ => Err(ShaderError::ReflectionFailed(
                    "unsupported vertex scalar type".into(),
                )),
            }
        }
        TypeKind::Vector => {
            let elem = ty.element_type().ok_or_else(|| {
                ShaderError::ReflectionFailed("vector has no element type".into())
            })?;
            let count = ty.element_count();
            let scalar = elem.scalar_type();
            match (scalar, count) {
                (ScalarType::Float32, 2) => Ok(vk::Format::R32G32_SFLOAT),
                (ScalarType::Float32, 3) => Ok(vk::Format::R32G32B32_SFLOAT),
                (ScalarType::Float32, 4) => Ok(vk::Format::R32G32B32A32_SFLOAT),
                (ScalarType::Int32, 2) => Ok(vk::Format::R32G32_SINT),
                (ScalarType::Int32, 3) => Ok(vk::Format::R32G32B32_SINT),
                (ScalarType::Int32, 4) => Ok(vk::Format::R32G32B32A32_SINT),
                (ScalarType::Uint32, 2) => Ok(vk::Format::R32G32_UINT),
                (ScalarType::Uint32, 3) => Ok(vk::Format::R32G32B32_UINT),
                (ScalarType::Uint32, 4) => Ok(vk::Format::R32G32B32A32_UINT),
                _ => Err(ShaderError::ReflectionFailed(
                    "unsupported vertex vector type".into(),
                )),
            }
        }
        TypeKind::Matrix => {
            // Matrices occupy one location per column vector.
            // For vertex inputs this is uncommon; report an error for now.
            Err(ShaderError::ReflectionFailed(
                "matrix vertex input not supported; use separate vectors".into(),
            ))
        }
        _ => Err(ShaderError::ReflectionFailed(format!(
            "unsupported vertex input type kind: {:?}",
            type_layout.kind()
        ))),
    }
}
