import bindless;
import view;
import shading_data;
import brdf;
import screen_quad;

[[vk::binding(0, 1)]]
ConstantBuffer<ViewData> view;

[[vk::binding(1, 1)]]
Texture2D<float4> base_color_tex;

[[vk::binding(2, 1)]]
Texture2D<float4> normal_mra_tex;

[[vk::binding(3, 1)]]
Texture2D<float> depth_tex;

[[vk::binding(4, 1)]]
TextureCube skybox_cubemap;

struct PSOutput {
    float4 output : SV_Target;
};

[shader("fragment")]
PSOutput main(PSInput input) {
    let ndc_pos = float4((input.uv - 0.5) * float2(2.0, -2.0), 1.0, 1.0);
    var world_pos = mul(view.inv_view_proj, ndc_pos);
    world_pos.xyz /= world_pos.w;

    let view_dir = normalize(world_pos.xyz - view.position.xyz);

    let depth = depth_tex.Sample(get_sampler(SamplerKind::NearestClamp), input.uv).r;
    if (depth == 0.0) {
        let skybox_color = skybox_cubemap.Sample(get_sampler(SamplerKind::LinearClamp), view_dir);

        PSOutput output;
        output.output = float4(skybox_color.rgb, 1.0);
        return output;
    }

    let sampler = get_sampler(SamplerKind::LinearClamp);
    let base = base_color_tex.Sample(sampler, input.uv);
    let nmr = normal_mra_tex.Sample(sampler, input.uv);

    float3 n;
    float metallic;
    float roughness;
    decode_normal_mra(nmr, n, metallic, roughness);

    let l = float3(0.0, 0.0, 1.0);
    let light_color = float3(1.0, 1.0, 1.0);
    let v = -view_dir;

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
