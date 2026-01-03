//! Vulkan Buffer - GPU buffer resource management.

use ash::{vk};
use zenith_core::log;

/// Buffer descriptor for creating GPU buffers.
#[derive(Debug, Clone)]
pub struct BufferDesc {
    /// Size of the buffer in bytes.
    pub size: vk::DeviceSize,
    /// Buffer usage flags (e.g., VERTEX_BUFFER, INDEX_BUFFER, UNIFORM_BUFFER).
    pub usage: vk::BufferUsageFlags,
    /// Memory property flags for allocation.
    pub memory_flags: vk::MemoryPropertyFlags,
    /// Sharing mode between queue families.
    pub sharing_mode: vk::SharingMode,
}

impl Default for BufferDesc {
    fn default() -> Self {
        Self {
            size: 0,
            usage: vk::BufferUsageFlags::empty(),
            memory_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
        }
    }
}

impl BufferDesc {
    /// Create a new buffer descriptor with the specified size.
    pub fn new(size: vk::DeviceSize) -> Self {
        Self {
            size,
            ..Default::default()
        }
    }

    /// Create a vertex buffer descriptor.
    pub fn vertex(size: vk::DeviceSize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
        }
    }

    /// Create an index buffer descriptor.
    pub fn index(size: vk::DeviceSize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
        }
    }

    /// Create a uniform buffer descriptor.
    pub fn uniform(size: vk::DeviceSize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            memory_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
        }
    }

    /// Create a storage buffer descriptor.
    pub fn storage(size: vk::DeviceSize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
        }
    }

    /// Create a staging buffer descriptor (CPU-visible for transfers).
    pub fn staging(size: vk::DeviceSize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            memory_flags: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
        }
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

    /// Set sharing mode.
    pub fn with_sharing_mode(mut self, mode: vk::SharingMode) -> Self {
        self.sharing_mode = mode;
        self
    }
}

/// Find a suitable memory type index.
pub fn find_memory_type(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..memory_properties.memory_type_count {
        let memory_type = memory_properties.memory_types[i as usize];
        if (type_filter & (1 << i)) != 0 && memory_type.property_flags.contains(properties) {
            return Some(i);
        }
    }
    None
}

/// GPU buffer with memory allocation.
pub struct Buffer {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    mapped_ptr: Option<*mut std::ffi::c_void>,
}

impl Buffer {
    /// Create a new buffer from a descriptor.
    pub fn new(
        device: &crate::RenderDevice,
        desc: &BufferDesc,
    ) -> Result<Self, vk::Result> {
        let memory_properties = device.memory_properties();
        let device = device.handle();
        // Create buffer
        let buffer_info = vk::BufferCreateInfo::default()
            .size(desc.size)
            .usage(desc.usage)
            .sharing_mode(desc.sharing_mode);

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

        // Get memory requirements
        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        // Find suitable memory type
        let memory_type_index =
            find_memory_type(memory_properties, mem_requirements.memory_type_bits, desc.memory_flags)
                .ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;

        // Allocate memory
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };

        // Bind memory to buffer
        unsafe { device.bind_buffer_memory(buffer, memory, 0)? };

        log::trace!("new buffer created.");

        Ok(Self {
            device: device.clone(),
            buffer,
            memory,
            size: desc.size,
            usage: desc.usage,
            mapped_ptr: None,
        })
    }

    /// Map buffer memory for CPU access. Returns a pointer to the mapped memory.
    pub fn map(&mut self) -> Result<*mut std::ffi::c_void, vk::Result> {
        if let Some(ptr) = self.mapped_ptr {
            return Ok(ptr);
        }

        let ptr = unsafe {
            self.device
                .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())?
        };

        self.mapped_ptr = Some(ptr);
        Ok(ptr)
    }

    /// Unmap buffer memory.
    pub fn unmap(&mut self) {
        if self.mapped_ptr.is_some() {
            unsafe { self.device.unmap_memory(self.memory) };
            self.mapped_ptr = None;
        }
    }

    /// Map buffer memory, write data, and unmap.
    pub fn map_and_write(&self, data: &[u8]) {
        unsafe {
            let ptr = self.device
                .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
                .expect("Failed to map buffer memory");
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            self.device.unmap_memory(self.memory);
        }
    }

    /// Write bytes into the buffer at `offset` (in bytes). This is intended for staging buffers.
    pub fn write_at(&self, offset: vk::DeviceSize, data: &[u8]) -> Result<(), vk::Result> {
        let len = data.len() as vk::DeviceSize;
        if len == 0 {
            return Ok(());
        }
        if offset + len > self.size {
            return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        }
        unsafe {
            let ptr = self.device.map_memory(self.memory, offset, len, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            self.device.unmap_memory(self.memory);
        }
        Ok(())
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

    /// Get the buffer size.
    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    /// Get the buffer usage flags.
    pub fn usage(&self) -> vk::BufferUsageFlags {
        self.usage
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.unmap();
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }

        log::trace!("buffer destroyed.");
    }
}
