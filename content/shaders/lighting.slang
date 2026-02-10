import bindless;
import view;
import shading_data;
import brdf;

struct VSInput {
    [[vk::location(0)]]
    float2 position : POSITION;
    [[vk::location(1)]]
    float2 uv : TEXCOORD0;
};

struct PSInput {
    float2 uv : TEXCOORD0;
};

struct VSOutput {
    float4 position : SV_Position;
    PSInput psout;
};

[shader("vertex")]
VSOutput main(VSInput input) {
    VSOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.psout.uv = input.uv;
    return output;
}

struct PushConstants {
    uint gbuffer_base;
    uint gbuffer_nmr;
    uint scene_depth;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> constant;

[[vk::binding(0, 1)]]
ConstantBuffer<ViewData> view;

struct PSOutput {
    float4 output : SV_Target;
};

[shader("fragment")]
PSOutput main(PSInput input) {
    let depth = sample_bindless_texture(constant.scene_depth, SamplerKind::NearestClamp, input.uv).r;
    if (depth == 0.0) {
        discard;
    }

    let base = sample_bindless_texture(constant.gbuffer_base, SamplerKind::LinearClamp, input.uv);
    let nmr = sample_bindless_texture(constant.gbuffer_nmr, SamplerKind::LinearClamp, input.uv);

    float3 n;
    float metallic;
    float roughness;
    decode_normal_mra(nmr, n, metallic, roughness);

    let clip_pos = float4((input.uv - 0.5) * float2(2.0, -2.0), 1.0, 1.0);
    var world_point = mul(view.inv_view_proj, clip_pos);
    world_point.xyz /= world_point.w;

    let l = float3(0.0, 0.0, 1.0);
    let light_color = float3(1.0, 1.0, 1.0);
    let v = normalize(view.position.xyz - world_point.xyz);

    let specular_color = lerp(float3(0.04), base.xyz, metallic);
    let diffuse_color = max(1.0 - metallic, 0.0) * base.xyz;

    let alpha = roughness * roughness;
    let h = normalize(l + v);
    let f = fresnel_schlick_appro(specular_color, l ,h);
    let d = ndf_ggx(n, h, alpha * alpha);
    let vis = vis_smith_joint(v, l, n, alpha);
    let specular = f * d * vis;

    let diffuse = diffuse_lambert(diffuse_color);

    let radiance = (diffuse + specular) * light_color * max(dot(n, l), 0.0);

    PSOutput output;
    output.output = float4(radiance, 1.0);
    return output;
}
