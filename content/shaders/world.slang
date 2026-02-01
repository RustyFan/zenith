#include "bindless.slang"

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
};

struct VSOutput {
    float4 position : SV_Position;
    PSInput psout;
};

static const uint FLAG_HAS_BASE_COLOR_TEX = 1u << 0;

struct PushConstants {
    float4x4 mvp;
    float4 base_color_factor;
    uint base_color_tex;
    uint sampler_kind;
    uint flags;
    float keep_normal_scale;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> constant;

[shader("vertex")]
VSOutput vsmain(VSInput input) {
    VSOutput output;
    output.position = mul(constant.mvp, float4(input.position, 1.0));
    // Keep `normal` as a live vertex input for reflection/layout, without affecting output.
    output.psout.uv = input.tex_coord + input.normal.xy * constant.keep_normal_scale;
    return output;
}

struct PSOutput {
    float4 output : SV_Target;
};

[shader("fragment")]
PSOutput psmain(PSInput input) {
    float4 base = constant.base_color_factor;

    if ((constant.flags & FLAG_HAS_BASE_COLOR_TEX) != 0u) {
        base *= sample_bindless_texture(constant.base_color_tex, (SamplerKind)constant.sampler_kind, input.uv);
    }

    PSOutput o;
    o.output = base;
    return o;
}

