import sh;
import bindless;
import basis;
import math_constants;

[[vk::binding(0, 1)]]
TextureCube skybox_cubemap;

[[vk::binding(1, 1)]]
RWStructuredBuffer<float4> outSHCoefficients;

static const uint SAMPLES = 1024;

// TODO: sample a direction per thread
[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    if (threadId.x > 0 || threadId.y > 0) {
        return;
    }

    ThreeBandSH sh3[3] = { ThreeBandSH(), ThreeBandSH(), ThreeBandSH() };

    let phi = PI * (sqrt(5.0) - 1.0);
    for (uint i = 0; i < SAMPLES; i++)
    {
        let y = 1.0 - ((float)i / (float)(SAMPLES)) * 2.0;
        let radius = sqrt(1.0 - y * y);

        let theta = phi * i;
        let x = cos(theta) * radius;
        let z = sin(theta) * radius;
        let dir = normalize(float3(x, y, z));

        let skybox_color = skybox_cubemap.SampleLevel(get_sampler(SamplerKind::LinearClamp), dir, 0).rgb;

        sh3[0].project(dir, skybox_color.r);
        sh3[1].project(dir, skybox_color.g);
        sh3[2].project(dir, skybox_color.b);
    }

    let normalize_factor = 4 * PI / SAMPLES;

    sh3[0].multiply(normalize_factor);
    sh3[1].multiply(normalize_factor);
    sh3[2].multiply(normalize_factor);

    outSHCoefficients[0] = sh3[0].coeff0;
    outSHCoefficients[1] = sh3[0].coeff1;
    outSHCoefficients[2] = float4(sh3[0].coeff2, 0.0, 0.0, 0.0);

    outSHCoefficients[3] = sh3[1].coeff0;
    outSHCoefficients[4] = sh3[1].coeff1;
    outSHCoefficients[5] = float4(sh3[1].coeff2, 0.0, 0.0, 0.0);

    outSHCoefficients[6] = sh3[2].coeff0;
    outSHCoefficients[7] = sh3[2].coeff1;
    outSHCoefficients[8] = float4(sh3[2].coeff2, 0.0, 0.0, 0.0);
}