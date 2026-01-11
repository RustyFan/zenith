//! Vulkan Shader - HLSL compilation and SPIR-V reflection.

use ash::{vk, Device};
// use hassle_rs::HassleError;
use rspirv_reflect::{Reflection, DescriptorType, BindingCount};
use std::ffi::CString;
use std::collections::HashMap;
use zenith_rhi_derive::DeviceObject;
use crate::RenderDevice;
use crate::device::DebuggableObject;
use std::path::{Path, PathBuf};
use std::process::Command;
use crate::device::set_debug_name_handle;

pub enum ShaderModel {
    SM6,
}

impl ShaderModel {
    pub fn as_str(&self) -> &str {
        match self {
            ShaderModel::SM6 => { "6_0" }
        }
    }
}

/// Compiled shader with Vulkan shader module and reflection data.
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
        device: &RenderDevice,
        path: &Path,
        entry_point: &str,
        stage: ShaderStage,
    ) -> Result<Self, ShaderError> {
        // Compile two variants:
        // - runtime SPIR-V: has embedded debug info for RenderDoc
        // - reflection SPIR-V: no debug info to keep reflection robust
        let runtime_spirv = compile_slang_file_to_spirv(name, path, entry_point, stage, true)?;
        let reflection_spirv = compile_slang_file_to_spirv(name, path, entry_point, stage, false)?;

        let reflection = reflect_spirv(&reflection_spirv, stage)?;
        let module = create_shader_module(device.handle(), &runtime_spirv)?;

        let shader = Self {
            name: name.to_owned(),
            module,
            stage,
            entry_point: CString::new(entry_point).unwrap(),
            reflection,
            device: device.handle().clone(),
        };
        device.set_debug_name(&shader);
        Ok(shader)
    }

    /// Create a shader from pre-compiled SPIR-V bytecode.
    pub fn from_spirv(
        name: &str,
        device: &RenderDevice,
        spirv: &[u8],
        entry_point: &str,
        stage: ShaderStage,
    ) -> Result<Self, ShaderError> {
        // Reflect the shader.
        let reflection = reflect_spirv(spirv, stage)?;

        // Create shader module
        let module = create_shader_module(device.handle(), spirv)?;

        let shader = Self {
            name: name.to_owned(),
            module,
            stage,
            entry_point: CString::new(entry_point).unwrap(),
            reflection,
            // descriptor_set_layouts,
            device: device.handle().clone(),
        };
        device.set_debug_name(&shader);
        Ok(shader)
    }

    #[inline]
    pub fn name(&self) -> &str { &self.name }

    #[inline]
    pub fn handle(&self) -> vk::ShaderModule { self.module }

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
}

impl DebuggableObject for Shader {
    fn set_debug_name(&self, device: &RenderDevice) {
        set_debug_name_handle(device, self.module, vk::ObjectType::SHADER_MODULE, self.name());
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.module, None);
        }
    }
}

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

/// Vertex shader input attribute reflected from SPIR-V.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexInputAttr {
    pub location: u32,
    pub format: vk::Format,
}

/// Shader reflection data.
#[derive(Debug, Clone, Default)]
pub struct ShaderReflection {
    pub bindings: Vec<ShaderBinding>,
    pub push_constant_size: u32,
    /// Vertex inputs (only populated for vertex stage).
    pub vertex_inputs: Vec<VertexInputAttr>,
}

impl ShaderReflection {
    /// Merge multiple shader reflections into one.
    /// Combines stage_flags for bindings at the same (set, binding).
    pub fn merge(reflections: &[&ShaderReflection]) -> Self {
        let mut binding_map: HashMap<(u32, u32), ShaderBinding> = HashMap::new();
        let mut push_constant_size = 0u32;
        let mut vertex_inputs_map: HashMap<u32, vk::Format> = HashMap::new();

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

            // Merge vertex inputs by location (first wins on conflicts).
            for vi in &reflection.vertex_inputs {
                vertex_inputs_map.entry(vi.location).or_insert(vi.format);
            }
        }

        let mut bindings: Vec<ShaderBinding> = binding_map.into_values().collect();
        bindings.sort_by_key(|b| (b.set, b.binding));

        let mut vertex_inputs: Vec<VertexInputAttr> = vertex_inputs_map
            .into_iter()
            .map(|(location, format)| VertexInputAttr { location, format })
            .collect();
        vertex_inputs.sort_by_key(|v| v.location);

        Self {
            bindings,
            push_constant_size,
            vertex_inputs,
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

fn slangc_path() -> Result<PathBuf, ShaderError> {
    if let Ok(p) = std::env::var("SLANGC") {
        return Ok(PathBuf::from(p));
    }
    if let Ok(vk) = std::env::var("VULKAN_SDK") {
        return Ok(PathBuf::from(vk).join("Bin").join("slangc.exe"));
    }
    Ok(PathBuf::from("slangc"))
}

fn stage_arg(stage: ShaderStage) -> &'static str {
    match stage {
        ShaderStage::Vertex => "vertex",
        ShaderStage::Fragment => "fragment",
        ShaderStage::Compute => "compute",
    }
}

fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' { c } else { '_' })
        .collect()
}

/// Compile Slang source file to SPIR-V using VulkanSDK `slangc.exe`.
///
/// We enable SPIR-V debug info (DWARF) for RenderDoc.
pub fn compile_slang_file_to_spirv(
    shader_name: &str,
    path: &Path,
    entry_point: &str,
    stage: ShaderStage,
    debug: bool,
) -> Result<Vec<u8>, ShaderError> {
    compile_slang_file_to_spirv_cli(shader_name, path, entry_point, stage, debug)
}

fn compile_slang_file_to_spirv_cli(
    shader_name: &str,
    path: &Path,
    entry_point: &str,
    stage: ShaderStage,
    debug: bool,
) -> Result<Vec<u8>, ShaderError> {
    let slangc = slangc_path()?;

    let out_dir = PathBuf::from("target").join("shader_pdb");
    std::fs::create_dir_all(&out_dir)?;

    let out_spv = out_dir.join(format!(
        "{}.{}.{}.{}.spv",
        sanitize_filename(shader_name),
        stage_arg(stage),
        sanitize_filename(entry_point),
        if debug { "debug" } else { "nodebug" },
    ));

    let include_dir = path
        .parent()
        .ok_or_else(|| ShaderError::CompilationFailed("Shader path has no parent dir".into()))?;

    let mut cmd = Command::new(slangc);
    cmd.arg(path)
        .arg("-target")
        .arg("spirv")
        .arg("-profile")
        .arg("spirv_1_6")
        .arg("-fvk-use-entrypoint-name")
        .arg("-entry")
        .arg(entry_point)
        .arg("-stage")
        .arg(stage_arg(stage))
        .arg("-I")
        .arg(include_dir)
        .arg("-o")
        .arg(&out_spv);

    if debug {
        // Debug: include debug info (level 3) in DWARF format for SPIR-V.
        cmd.arg("-g3").arg("-gdwarf");
    }

    let output = cmd.output()?;
    if !output.status.success() {
        let mut msg = String::new();
        msg.push_str("slangc failed\n");
        msg.push_str(&String::from_utf8_lossy(&output.stdout));
        msg.push_str(&String::from_utf8_lossy(&output.stderr));
        return Err(ShaderError::CompilationFailed(msg));
    }

    Ok(std::fs::read(out_spv)?)
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

    // Vertex inputs (VS only)
    let vertex_inputs = if stage == ShaderStage::Vertex {
        reflect_vertex_inputs_from_spirv(spirv)?
    } else {
        Vec::new()
    };

    Ok(ShaderReflection {
        bindings,
        push_constant_size,
        vertex_inputs,
    })
}

#[derive(Debug, Clone)]
enum SpirvType {
    Int { width: u32, signed: bool },
    Float { width: u32 },
    Vector { component_type: u32, count: u32 },
    Matrix { column_type: u32, count: u32 },
    Array { element_type: u32, length_id: u32 },
    Struct { members: Vec<u32> },
    Pointer { storage_class: u32, pointee_type: u32 },
}

#[derive(Default)]
struct MemberDecos {
    location: Option<u32>,
    builtin: Option<u32>,
}

fn reflect_vertex_inputs_from_spirv(spirv: &[u8]) -> Result<Vec<VertexInputAttr>, ShaderError> {
    // Minimal SPIR-V parser for stage inputs:
    // - OpVariable (Input)
    // - OpDecorate / OpMemberDecorate (Location/BuiltIn)
    // - Type graph enough to map to vk::Format

    let words: &[u32] = unsafe {
        std::slice::from_raw_parts(spirv.as_ptr() as *const u32, spirv.len() / 4)
    };
    if words.len() < 5 {
        return Err(ShaderError::ReflectionFailed("SPIR-V header too small".into()));
    }

    // Opcode values from SPIR-V spec.
    const OP_NAME: u16 = 5;
    const OP_DECORATE: u16 = 71;
    const OP_MEMBER_DECORATE: u16 = 72;
    const OP_VARIABLE: u16 = 59;
    const OP_TYPE_INT: u16 = 21;
    const OP_TYPE_FLOAT: u16 = 22;
    const OP_TYPE_VECTOR: u16 = 23;
    const OP_TYPE_MATRIX: u16 = 24;
    const OP_TYPE_ARRAY: u16 = 28;
    const OP_TYPE_STRUCT: u16 = 30;
    const OP_TYPE_POINTER: u16 = 32;
    const OP_CONSTANT: u16 = 43;

    // Decorations.
    const DECORATION_BUILTIN: u32 = 11;
    const DECORATION_LOCATION: u32 = 30;

    // StorageClass.
    const STORAGE_CLASS_INPUT: u32 = 1;

    let mut types: HashMap<u32, SpirvType> = HashMap::new();
    let mut const_u32: HashMap<u32, u32> = HashMap::new();
    let mut var_ptr_type: HashMap<u32, u32> = HashMap::new();
    let mut var_storage_class: HashMap<u32, u32> = HashMap::new();
    let mut var_location: HashMap<u32, u32> = HashMap::new();
    let mut var_builtin: HashMap<u32, u32> = HashMap::new();
    let mut member_decos: HashMap<(u32, u32), MemberDecos> = HashMap::new();

    // Skip header (5 words).
    let mut i = 5usize;
    while i < words.len() {
        let first = words[i];
        let wc = (first >> 16) as usize;
        let op = (first & 0xFFFF) as u16;
        if wc == 0 || i + wc > words.len() {
            return Err(ShaderError::ReflectionFailed("invalid SPIR-V instruction word count".into()));
        }

        let inst = &words[i..i + wc];
        match op {
            OP_TYPE_INT => {
                // OpTypeInt %result width signedness
                if wc >= 4 {
                    let result_id = inst[1];
                    let width = inst[2];
                    let signed = inst[3] != 0;
                    types.insert(result_id, SpirvType::Int { width, signed });
                }
            }
            OP_TYPE_FLOAT => {
                // OpTypeFloat %result width
                if wc >= 3 {
                    let result_id = inst[1];
                    let width = inst[2];
                    types.insert(result_id, SpirvType::Float { width });
                }
            }
            OP_TYPE_VECTOR => {
                // OpTypeVector %result %component count
                if wc >= 4 {
                    let result_id = inst[1];
                    let component_type = inst[2];
                    let count = inst[3];
                    types.insert(result_id, SpirvType::Vector { component_type, count });
                }
            }
            OP_TYPE_MATRIX => {
                // OpTypeMatrix %result %column_type count
                if wc >= 4 {
                    let result_id = inst[1];
                    let column_type = inst[2];
                    let count = inst[3];
                    types.insert(result_id, SpirvType::Matrix { column_type, count });
                }
            }
            OP_TYPE_ARRAY => {
                // OpTypeArray %result %element_type %length_id
                if wc >= 4 {
                    let result_id = inst[1];
                    let element_type = inst[2];
                    let length_id = inst[3];
                    types.insert(result_id, SpirvType::Array { element_type, length_id });
                }
            }
            OP_TYPE_STRUCT => {
                // OpTypeStruct %result %member0 %member1 ...
                if wc >= 2 {
                    let result_id = inst[1];
                    let members = inst[2..].to_vec();
                    types.insert(result_id, SpirvType::Struct { members });
                }
            }
            OP_TYPE_POINTER => {
                // OpTypePointer %result StorageClass %type
                if wc >= 4 {
                    let result_id = inst[1];
                    let storage_class = inst[2];
                    let pointee_type = inst[3];
                    types.insert(result_id, SpirvType::Pointer { storage_class, pointee_type });
                }
            }
            OP_CONSTANT => {
                // OpConstant %type %result value...
                if wc >= 4 {
                    let result_type = inst[1];
                    let result_id = inst[2];
                    // Only handle 32-bit scalar ints for array lengths.
                    match types.get(&result_type) {
                        Some(SpirvType::Int { width: 32, .. }) => {
                            const_u32.insert(result_id, inst[3]);
                        }
                        _ => {}
                    }
                }
            }
            OP_VARIABLE => {
                // OpVariable %result_type %result StorageClass [initializer]
                if wc >= 4 {
                    let result_type = inst[1];
                    let result_id = inst[2];
                    let storage_class = inst[3];
                    var_ptr_type.insert(result_id, result_type);
                    var_storage_class.insert(result_id, storage_class);
                }
            }
            OP_DECORATE => {
                // OpDecorate %target Decoration [literals...]
                if wc >= 3 {
                    let target_id = inst[1];
                    let deco = inst[2];
                    match deco {
                        DECORATION_LOCATION => {
                            if wc >= 4 {
                                var_location.insert(target_id, inst[3]);
                            }
                        }
                        DECORATION_BUILTIN => {
                            if wc >= 4 {
                                var_builtin.insert(target_id, inst[3]);
                            }
                        }
                        _ => {}
                    }
                }
            }
            OP_MEMBER_DECORATE => {
                // OpMemberDecorate %struct member Decoration [literals...]
                if wc >= 4 {
                    let struct_id = inst[1];
                    let member = inst[2];
                    let deco = inst[3];
                    let entry = member_decos.entry((struct_id, member)).or_insert_with(MemberDecos::default);
                    match deco {
                        DECORATION_LOCATION => {
                            if wc >= 5 {
                                entry.location = Some(inst[4]);
                            }
                        }
                        DECORATION_BUILTIN => {
                            if wc >= 5 {
                                entry.builtin = Some(inst[4]);
                            }
                        }
                        _ => {}
                    }
                }
            }
            OP_NAME => {
                // not needed for minimal schema
            }
            _ => {}
        }

        i += wc;
    }

    let mut out: Vec<VertexInputAttr> = Vec::new();

    for (&var_id, &storage_class) in &var_storage_class {
        if storage_class != STORAGE_CLASS_INPUT {
            continue;
        }
        if var_builtin.contains_key(&var_id) {
            continue;
        }

        let Some(&ptr_type_id) = var_ptr_type.get(&var_id) else { continue };
        let pointee_type_id = match types.get(&ptr_type_id) {
            Some(SpirvType::Pointer { storage_class: sc, pointee_type }) if *sc == STORAGE_CLASS_INPUT => *pointee_type,
            Some(SpirvType::Pointer { pointee_type, .. }) => *pointee_type,
            _ => continue,
        };

        match types.get(&pointee_type_id) {
            Some(SpirvType::Struct { members }) => {
                for (member_index, &member_ty) in members.iter().enumerate() {
                    let key = (pointee_type_id, member_index as u32);
                    let Some(decos) = member_decos.get(&key) else { continue };
                    if decos.builtin.is_some() {
                        continue;
                    }
                    let Some(loc) = decos.location else { continue };
                    expand_type_to_vertex_attrs(&types, &const_u32, member_ty, loc, &mut out)?;
                }
            }
            _ => {
                let Some(&loc) = var_location.get(&var_id) else { continue };
                expand_type_to_vertex_attrs(&types, &const_u32, pointee_type_id, loc, &mut out)?;
            }
        }
    }

    out.sort_by_key(|v| v.location);

    // Dedup by location: keep first if format matches, otherwise keep first.
    out.dedup_by(|a, b| a.location == b.location && a.format == b.format);

    Ok(out)
}

fn expand_type_to_vertex_attrs(
    types: &HashMap<u32, SpirvType>,
    const_u32: &HashMap<u32, u32>,
    ty_id: u32,
    base_location: u32,
    out: &mut Vec<VertexInputAttr>,
) -> Result<(), ShaderError> {
    match types.get(&ty_id) {
        Some(SpirvType::Float { width: 32 }) => {
            out.push(VertexInputAttr { location: base_location, format: vk::Format::R32_SFLOAT });
            Ok(())
        }
        Some(SpirvType::Int { width: 32, signed }) => {
            out.push(VertexInputAttr {
                location: base_location,
                format: if *signed { vk::Format::R32_SINT } else { vk::Format::R32_UINT },
            });
            Ok(())
        }
        Some(SpirvType::Vector { component_type, count }) => {
            let comp = types.get(component_type).ok_or_else(|| ShaderError::ReflectionFailed("unknown vector component type".into()))?;
            let (is_float, is_signed_int) = match comp {
                SpirvType::Float { width: 32 } => (true, false),
                SpirvType::Int { width: 32, signed } => (false, *signed),
                _ => return Err(ShaderError::ReflectionFailed("unsupported vertex input component type".into())),
            };

            let fmt = match (is_float, is_signed_int, *count) {
                (true, _, 2) => vk::Format::R32G32_SFLOAT,
                (true, _, 3) => vk::Format::R32G32B32_SFLOAT,
                (true, _, 4) => vk::Format::R32G32B32A32_SFLOAT,

                (false, true, 2) => vk::Format::R32G32_SINT,
                (false, true, 3) => vk::Format::R32G32B32_SINT,
                (false, true, 4) => vk::Format::R32G32B32A32_SINT,

                (false, false, 2) => vk::Format::R32G32_UINT,
                (false, false, 3) => vk::Format::R32G32B32_UINT,
                (false, false, 4) => vk::Format::R32G32B32A32_UINT,

                _ => return Err(ShaderError::ReflectionFailed("unsupported vertex vector width/count".into())),
            };

            out.push(VertexInputAttr { location: base_location, format: fmt });
            Ok(())
        }
        Some(SpirvType::Matrix { column_type, count }) => {
            // Matrices occupy multiple locations: one per column vector.
            let cols = *count;
            for c in 0..cols {
                expand_type_to_vertex_attrs(types, const_u32, *column_type, base_location + c, out)?;
            }
            Ok(())
        }
        Some(SpirvType::Array { element_type, length_id }) => {
            let Some(&len) = const_u32.get(length_id) else {
                return Err(ShaderError::ReflectionFailed("unsupported array length (non-constant)".into()));
            };
            for idx in 0..len {
                expand_type_to_vertex_attrs(types, const_u32, *element_type, base_location + idx, out)?;
            }
            Ok(())
        }
        Some(SpirvType::Struct { .. }) => Err(ShaderError::ReflectionFailed(
            "unexpected struct vertex input without member locations".into(),
        )),
        _ => Err(ShaderError::ReflectionFailed("unsupported vertex input type".into())),
    }
}

/// Convert rspirv_reflect descriptor type to Vulkan descriptor type.
fn convert_descriptor_type(reflect_type: DescriptorType) -> vk::DescriptorType {
    // DescriptorType is a transparent wrapper around u32, matching Vulkan values
    vk::DescriptorType::from_raw(reflect_type.0 as i32)
}

/// Create a Vulkan shader module from SPIR-V bytecode.
fn create_shader_module(device: &Device, spirv: &[u8]) -> Result<vk::ShaderModule, ShaderError> {
    assert_eq!(spirv.len() % 4, 0, "SPIR-V bytecode must be 4-byte aligned");

    let code: &[u32] = unsafe { std::slice::from_raw_parts(spirv.as_ptr() as *const u32, spirv.len() / 4) };

    let create_info = vk::ShaderModuleCreateInfo::default().code(code);
    let module = unsafe { device.create_shader_module(&create_info, None)? };

    Ok(module)
}

// /// Create all descriptor set layouts from shader reflection.
// pub(crate) fn create_layouts_from_reflection(
//     device: &Device,
//     reflection: &ShaderReflection,
// ) -> Result<Vec<Arc<DescriptorSetLayout>>, vk::Result> {
//     let max_set = reflection.max_set().unwrap_or(0);
//     let mut layouts = Vec::with_capacity((max_set + 1) as usize);
//
//     for set_index in 0..=max_set {
//         let name = format!("descriptor_set_layout[{set_index}]");
//         let layout = DescriptorSetLayout::from_reflection(&name, device, &reflection.bindings, set_index)?;
//         layouts.push(Arc::new(layout));
//     }
//
//     Ok(layouts)
// }