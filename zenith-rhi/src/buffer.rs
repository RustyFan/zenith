//! Vulkan Buffer - GPU buffer resource management.

use ash::{vk};
use zenith_core::log;
use std::hash::{Hash, Hasher};
use std::ops::RangeBounds;
use zenith_rhi_derive::DeviceObject;
use crate::RenderDevice;
use crate::device::{DebuggableObject};
use crate::utility::{find_memory_type, normalize_range_u64};
use crate::device::set_debug_name_handle;

/// Buffer descriptor for creating GPU buffers.
#[derive(Debug, Clone)]
pub struct BufferDesc {
    pub name: String,
    /// Size of the buffer in bytes.
    pub size: vk::DeviceSize,
    /// Buffer usage flags (e.g., VERTEX_BUFFER, INDEX_BUFFER, UNIFORM_BUFFER).
    pub usage: vk::BufferUsageFlags,
    /// Memory property flags for allocation.
    pub memory_flags: vk::MemoryPropertyFlags,
}

impl Default for BufferDesc {
    fn default() -> Self {
        Self {
            name: "Unnamed buffer".to_string(),
            size: 0,
            usage: vk::BufferUsageFlags::empty(),
            memory_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        }
    }
}

impl BufferDesc {
    /// Create a new buffer descriptor with the specified size.
    pub fn new(name: &str, size: vk::DeviceSize) -> Self {
        Self {
            name: name.to_owned(),
            size,
            ..Default::default()
        }
    }

    /// Create a vertex buffer descriptor.
    pub fn vertex(name: &str, size: vk::DeviceSize) -> Self {
        Self {
            name: name.to_owned(),
            size,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        }
    }

    /// Create an index buffer descriptor.
    pub fn index(name: &str, size: vk::DeviceSize) -> Self {
        Self {
            name: name.to_owned(),
            size,
            usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        }
    }

    /// Create a uniform buffer descriptor.
    pub fn uniform(name: &str, size: vk::DeviceSize) -> Self {
        Self {
            name: name.to_owned(),
            size,
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            memory_flags: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        }
    }

    /// Create a storage buffer descriptor.
    pub fn storage(name: &str, size: vk::DeviceSize) -> Self {
        Self {
            name: name.to_owned(),
            size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        }
    }

    /// Create a staging buffer descriptor (CPU-visible for transfers).
    pub fn staging(name: &str, size: vk::DeviceSize) -> Self {
        Self {
            name: name.to_owned(),
            size,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            memory_flags: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the buffer usage flags.
    pub fn with_usage(mut self, usage: vk::BufferUsageFlags) -> Self {
        self.usage = usage;
        self
    }

    /// Add additional usage flags.
    pub fn with_additional_usage(mut self, usage: vk::BufferUsageFlags) -> Self {
        self.usage |= usage;
        self
    }

    /// Set memory property flags.
    pub fn with_memory_flags(mut self, flags: vk::MemoryPropertyFlags) -> Self {
        self.memory_flags = flags;
        self
    }

    /// Make the buffer host-visible (CPU accessible).
    pub fn host_visible(mut self) -> Self {
        self.memory_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        self
    }

    /// Make the buffer device-local (GPU-only).
    pub fn device_local(mut self) -> Self {
        self.memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        self
    }

    /// Enable buffer device address.
    pub fn with_device_address(mut self) -> Self {
        self.usage |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        self
    }
}

impl PartialEq for BufferDesc {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.size == other.size
            && self.usage.as_raw() == other.usage.as_raw()
            && self.memory_flags.as_raw() == other.memory_flags.as_raw()
    }
}

impl Eq for BufferDesc {}

impl Hash for BufferDesc {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.size.hash(state);
        self.usage.as_raw().hash(state);
        self.memory_flags.as_raw().hash(state);
    }
}

/// GPU buffer with memory allocation.
#[DeviceObject]
pub struct Buffer {
    buffer: vk::Buffer,
    desc: BufferDesc,
    memory: vk::DeviceMemory,
}

impl Buffer {
    /// Create a new buffer from a descriptor.
    pub fn new(
        device: &RenderDevice,
        desc: &BufferDesc,
    ) -> Result<Self, vk::Result> {
        let memory_properties = device.memory_properties();
        // Create buffer
        let buffer_info = vk::BufferCreateInfo::default()
            .size(desc.size)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.handle().create_buffer(&buffer_info, None)? };

        // Get memory requirements
        let mem_requirements = unsafe { device.handle().get_buffer_memory_requirements(buffer) };

        // Find suitable memory type
        let memory_type_index = find_memory_type(memory_properties, mem_requirements.memory_type_bits, desc.memory_flags)
            .ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;

        // Allocate memory
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { device.handle().allocate_memory(&alloc_info, None)? };

        // Bind memory to buffer
        unsafe { device.handle().bind_buffer_memory(buffer, memory, 0)? };

        log::trace!("new buffer created.");

        let buf = Self {
            buffer,
            desc: desc.clone(),
            memory,
            device: device.handle().clone(),
        };
        device.set_debug_name(&buf);
        Ok(buf)
    }

    pub fn as_range<R: RangeBounds<u64>>(&self, range: R) -> Result<BufferRange<'_>, vk::Result> {
        let (offset, size) = normalize_range_u64(range, self.desc.size as u64)?;
        Ok(BufferRange {
            buffer: self,
            offset,
            size,
        })
    }

    /// Get buffer device address (requires BUFFER_DEVICE_ADDRESS usage flag).
    pub fn device_address(&self) -> vk::DeviceAddress {
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        unsafe { self.device.get_buffer_device_address(&info) }
    }

    /// Get the raw Vulkan buffer handle.
    pub fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    #[inline]
    pub fn name(&self) -> &str {
        &self.desc.name
    }
    
    #[inline]
    pub fn desc(&self) -> &BufferDesc {
        &self.desc
    }

    #[inline]
    pub fn size(&self) -> vk::DeviceSize {
        self.desc.size
    }

    #[inline]
    pub fn usage(&self) -> vk::BufferUsageFlags {
        self.desc.usage
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }

        log::trace!("buffer destroyed.");
    }
}

impl DebuggableObject for Buffer {
    fn set_debug_name(&self, device: &RenderDevice) {
        set_debug_name_handle(device, self.buffer, vk::ObjectType::BUFFER, self.name());
        set_debug_name_handle(
            device,
            self.memory,
            vk::ObjectType::DEVICE_MEMORY,
            &format!("{}.memory", self.name()),
        );
    }
}

#[derive(Clone, Copy)]
pub struct BufferRange<'a> {
    buffer: &'a Buffer,
    offset: u64,
    size: u64,
}

impl<'a> BufferRange<'a> {
    #[inline]
    pub fn buffer(&self) -> &'a Buffer { self.buffer }

    #[inline]
    pub fn offset(&self) -> u64 { self.offset }

    #[inline]
    pub fn size(&self) -> u64 { self.size }

    pub fn to_binding(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffer.handle())
            .offset(self.offset)
            .range(self.size)
    }

    pub fn write(&self, data: &[u8]) -> Result<(), vk::Result>  {
        let len = data.len() as u64;
        if len == 0 {
            return Ok(());
        }
        if len > self.size {
            return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        }

        // SAFETY: range is checked before constructing, and mapping is limited to `len`.
        unsafe {
            let ptr = self.buffer.device.map_memory(
                self.buffer.memory,
                self.offset as vk::DeviceSize,
                len as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            self.buffer.device.unmap_memory(self.buffer.memory);
        }

        Ok(())
    }
}
