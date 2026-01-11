use std::collections::Bound;
use std::ops::RangeBounds;
use ash::vk;

macro_rules! normalize_range_function {
    ($unsigned:ty) => {
        $crate::paste! {
            pub(crate) fn [<normalize_range_ $unsigned>]<R: RangeBounds<$unsigned>>(
                bounds: R,
                size: $unsigned,
            ) -> Result<($unsigned, $unsigned), vk::Result> {
                let start = match bounds.start_bound() {
                    Bound::Included(&v) => v,
                    Bound::Excluded(&v) => v.checked_add(1).ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?,
                    Bound::Unbounded => 0,
                };
                let end_exclusive = match bounds.end_bound() {
                    Bound::Included(&v) => v.checked_add(1).ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?,
                    Bound::Excluded(&v) => v,
                    Bound::Unbounded => size,
                };

                if start > end_exclusive {
                    return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
                }
                if end_exclusive > size {
                    return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
                }

                Ok((start, end_exclusive - start))
            }
        }
    };
}

normalize_range_function!(u64);
normalize_range_function!(u32);

/// Find a suitable memory type index.
pub(crate) fn find_memory_type(
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