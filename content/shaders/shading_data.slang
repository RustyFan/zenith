func encode_normal_rg(float3 normal) -> float2 {
    return normal.xy * 0.5 + 0.5;
}

func decode_normal_rg(float2 packed) -> float3 {
    float2 xy = packed * 2.0 - 1.0;
    float z = sqrt(saturate(1.0 - dot(xy, xy)));
    return normalize(float3(xy, z));
}

func encode_normal_mra(float3 normal, float metallic, float roughness) -> float4 {
    float2 packed_normal = encode_normal_rg(normalize(normal));
    return float4(packed_normal.x, packed_normal.y, metallic, roughness);
}

func decode_normal_mra(float4 packed, out float3 normal, out float metallic, out float roughness) {
    normal = decode_normal_rg(packed.xy);
    metallic = packed.z;
    roughness = packed.w;
}

struct ShadingData {
    float3 basecolor;
    float3 normal;
    float metallic;
    float roughness;

    static func new(float3 basecolor, float3 normal, float metallic, float roughness) -> ShadingData {
        ShadingData data;
        data.basecolor = basecolor;
        data.normal = normal;
        data.metallic = metallic;
        data.roughness = roughness;
        return data;
    }

    void export_to_gbuffer(out float4 gbuffer_basecolor, out float4 gbuffer_normal_mra) {
        gbuffer_basecolor = float4(basecolor, 1.0);
        gbuffer_normal_mra = encode_normal_mra(normal, metallic, roughness);
    }
};
