import bindless;
import view;
import shading_data;
import brdf;
import screen_quad;
import sh;
import debug;
import math_constants;
import tonemapping;

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

[[vk::binding(5, 1)]]
StructuredBuffer<float4> sh_diffuse;

struct PushConstant {
    uint debug_mode;
};

[[vk::push_constant]]
PushConstant pc;

struct PSOutput {
    float4 output : SV_Target;
};

[shader("fragment")]
PSOutput main(PSInput input) {
    let ndc_pos = float4((input.uv - 0.5) * float2(2.0, -2.0), 1.0, 1.0);
    var world_pos = mul(view.inv_view_proj, ndc_pos);
    world_pos.xyz /= world_pos.w;

    let sampler = get_sampler(SamplerKind::LinearClamp);
    let base = base_color_tex.Sample(sampler, input.uv);
    let nmr = normal_mra_tex.Sample(sampler, input.uv);

    float3 n;
    float metallic;
    float roughness;
    decode_normal_mra(nmr, n, metallic, roughness);

    let view_dir = normalize(world_pos.xyz - view.position.xyz);

    ThreeBandSH sh3[3] = { ThreeBandSH(), ThreeBandSH(), ThreeBandSH() };
    sh3[0].coeff0 = sh_diffuse.Load(0);
    sh3[0].coeff1 = sh_diffuse.Load(1);
    sh3[0].coeff2 = sh_diffuse.Load(2).x;

    sh3[1].coeff0 = sh_diffuse.Load(3);
    sh3[1].coeff1 = sh_diffuse.Load(4);
    sh3[1].coeff2 = sh_diffuse.Load(5).x;

    sh3[2].coeff0 = sh_diffuse.Load(6);
    sh3[2].coeff1 = sh_diffuse.Load(7);
    sh3[2].coeff2 = sh_diffuse.Load(8).x;

    let depth = depth_tex.Sample(get_sampler(SamplerKind::NearestClamp), input.uv).r;
    if (depth == 0.0) {
        let skybox_color = skybox_cubemap.Sample(get_sampler(SamplerKind::LinearClamp), view_dir).rgb;

        PSOutput output;
        if ((pc.debug_mode & DebugMode.DiffuseSH) != 0) {
            output.output = float4(float3(sh3[0].evaluate(view_dir), sh3[1].evaluate(view_dir), sh3[2].evaluate(view_dir)), 1.0);
        } else {
            output.output = float4(skybox_color, 1.0);
        }
        return output;
    }

    let l = float3(0.0, 0.0, 1.0);
    let light_color = float3(1.0, 1.0, 1.0);
    let v = -view_dir;

    metallic = 0.0;

    let specular_color = lerp(float3(0.04), base.xyz, metallic);
    let diffuse_color = max(1.0 - metallic, 0.0) * base.xyz;

    let alpha = roughness * roughness;
    let h = normalize(l + v);
    let f = fresnel_schlick_appro(specular_color, l, h);
    let d = ndf_ggx(n, h, alpha * alpha);
    let vis = vis_smith_joint(v, l, n, alpha);
    let specular = f * d * vis;

    //let f_roughness = fresnel_schlick_roughness_appro(specular_color, l, h, roughness);
    //let kd = (float3(1.0) - f_roughness) * (1.0 - metallic);

    let diffuse = diffuse_lambert(diffuse_color);
    //let irradiance = diffuse_lambert(float3(1.0)) * ibl_diffuse_lighting * 100.0;
    let ibl_diffuse_lighting = float3(sh3[0].evaluate(n), sh3[1].evaluate(n), sh3[2].evaluate(n));
    let irradiance = diffuse * ibl_diffuse_lighting;
    let radiance = (diffuse + specular) * light_color * max(dot(n, l), 0.0);

    let final_output = gamma(aces_film(irradiance));

    PSOutput output;
    output.output = float4(final_output, 1.0);
    return output;
}
