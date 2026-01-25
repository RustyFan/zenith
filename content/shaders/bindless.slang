static const uint BINDLESS_TYPE_SHIFT = 29;
static const uint BINDLESS_INDEX_MASK = 0x1FFFFFFF;

static const uint BINDLESS_BINDING_TEXTURE = 0;
static const uint BINDLESS_BINDING_BUFFER = 1;
static const uint BINDLESS_BINDING_SAMPLER = 2;

uint bindless_index(uint handle) {
    return handle & BINDLESS_INDEX_MASK;
}

uint bindless_type(uint handle) {
    return handle >> BINDLESS_TYPE_SHIFT;
}

[[vk::binding(0, 0)]]
Texture2D texture_heap[];

[[vk::binding(1, 0)]]
ByteAddressBuffer buffer_heap[];

[[vk::binding(3, 0)]]
SamplerState sampler_heap[];

T get_bindless_buffer<T>(uint handle) {
    uint index = bindless_index(handle);
    return buffer_heap[NonUniformResourceIndex(index)].Load<T>(0);
}

T get_bindless_buffer_with_offset<T>(uint handle) {
    uint index = bindless_index(handle);
    return buffer_heap[NonUniformResourceIndex(index)].Load<T>(0);
}