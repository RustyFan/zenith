static const uint BINDLESS_TYPE_SHIFT = 29;
static const uint BINDLESS_INDEX_MASK = 0x1FFFFFFF;

static const uint BINDLESS_BINDING_TEXTURE = 0;
static const uint BINDLESS_BINDING_BUFFER = 1;
static const uint BINDLESS_BINDING_SAMPLER = 2;

func bindless_index(uint handle) -> uint {
    return handle & BINDLESS_INDEX_MASK;
}

func bindless_type(uint handle) -> uint {
    return handle >> BINDLESS_TYPE_SHIFT;
}

func is_valid_bindless_handle(uint handle) -> bool {
    return handle != (~0);
}

[[vk::binding(0, 0)]]
Texture2D bindless_texture_heap[];

[[vk::binding(1, 0)]]
StructuredBuffer<float4> bindless_buffer_heap[];

[[vk::binding(2, 0)]]
SamplerState bindless_sampler_heap[];

func load_bindless_float(uint handle, uint element_index) -> float {
    const uint index = bindless_index(handle);
    return bindless_buffer_heap[NonUniformResourceIndex(index)][element_index].x;
}

func load_bindless_float4(uint handle, uint element_index) -> float4 {
    const uint index = bindless_index(handle);
    return bindless_buffer_heap[NonUniformResourceIndex(index)][element_index];
}

func load_bindless_float4x4(uint handle, uint element_index) -> float4x4 {
    const uint index = bindless_index(handle);
    float4 c0 = bindless_buffer_heap[NonUniformResourceIndex(index)][element_index];
    float4 c1 = bindless_buffer_heap[NonUniformResourceIndex(index)][element_index + 1];
    float4 c2 = bindless_buffer_heap[NonUniformResourceIndex(index)][element_index + 2];
    float4 c3 = bindless_buffer_heap[NonUniformResourceIndex(index)][element_index + 3];
    // float4x4(a,b,c,d) in Slang/HLSL treats arguments as rows.
    return float4x4(c0, c1, c2, c3);
}

enum SamplerKind : uint {
    LinearRepeat = 0,
    LinearClamp  = 1,
    NearestRepeat = 2,
    NearestClamp  = 3,
};

func get_sampler(kind: SamplerKind) -> SamplerState {
    return bindless_sampler_heap[NonUniformResourceIndex((uint)kind)];
}

func sample_bindless_texture(uint handle, SamplerKind sampler_kind, float2 uv) -> float4 {
    const uint index = bindless_index(handle);
    const Texture2D tex = bindless_texture_heap[NonUniformResourceIndex(index)];
    const SamplerState sampler = bindless_sampler_heap[NonUniformResourceIndex((uint)sampler_kind)];
    return tex.Sample(sampler, uv);
}