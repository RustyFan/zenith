import bindless;
import view;
import shading_data;

struct VSInput {
    [[vk::location(0)]]
    float3 position : POSITION;
    [[vk::location(1)]]
    float3 normal : NORMAL;
    [[vk::location(2)]]
    float2 tex_coord : TEXCOORD0;
};

struct PSInput {
    float2 uv : TEXCOORD0;
    float3 normal : NORMAL;
};

struct VSOutput {
    float4 position : SV_Position;
    PSInput psout;
};

static const uint FLAG_HAS_BASE_COLOR_TEX = 1u << 0;
static const uint FLAG_HAS_MRA_TEX = 1u << 1;
static const uint FLAG_HAS_NORMAL_TEX = 1u << 2;

struct PushConstants {
    float4x4 model;
    float4 base_color_factor;
    uint base_color_tex;
    uint mra_tex;
    uint normal_tex;
    uint sampler_kind;
    uint flags;
    float keep_normal_scale;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> constant;

[[vk::binding(0, 1)]]
ConstantBuffer<ViewData> view;

[shader("vertex")]
VSOutput main(VSInput input) {
    VSOutput output;
    const float4 world_pos = mul(constant.model, float4(input.position, 1.0));
    output.position = mul(view.view_proj, world_pos);
    // Keep `normal` as a live vertex input for reflection/layout, without affecting output.
    output.psout.uv = input.tex_coord;
    output.psout.normal = input.normal;
    return output;
}

struct PSOutput {
    float4 base_color : SV_Target0;
    float4 nmr : SV_Target1; // Normal, metallic and roughness
};

[shader("fragment")]
PSOutput main(PSInput input) {
    var base = constant.base_color_factor;

    if ((constant.flags & FLAG_HAS_BASE_COLOR_TEX) != 0u) {
        base *= sample_bindless_texture(constant.base_color_tex, SamplerKind::LinearRepeat, input.uv);
    }

    var normal_ws = normalize(input.normal);
    if ((constant.flags & FLAG_HAS_NORMAL_TEX) != 0u) {
        float2 packed_normal = sample_bindless_texture(constant.normal_tex, SamplerKind::LinearRepeat, input.uv).xy;
        normal_ws = decode_normal_rg(packed_normal);
    }

    var metallic = 0.0;
    var roughness = 1.0;
    if ((constant.flags & FLAG_HAS_MRA_TEX) != 0u) {
        float3 mra = sample_bindless_texture(constant.mra_tex, SamplerKind::LinearRepeat, input.uv).rgb;
        metallic = mra.b;
        roughness = mra.g;
    }

    PSOutput output;
    let data = ShadingData::new(base.xyz, normal_ws, metallic, roughness);
    data.export_to_gbuffer(output.base_color, output.nmr);
    return output;
}

