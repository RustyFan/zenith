cbuffer TimeData : register(b0, space0) {
    float time;
};

struct PSInput {
    float4 position : SV_Position;
    float3 color : COLOR;
};

float4 main(PSInput input) : SV_Target {
    float3 animated = input.color * (0.5 + 0.5 * sin(time * 3.0 + input.color * 6.28));
    return float4(animated, 1.0);
}
