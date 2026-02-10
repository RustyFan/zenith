struct VSInput {
    [[vk::location(0)]]
    float3 position : POSITION;
    [[vk::location(1)]]
    float3 color : COLOR;
};

struct VSOutput {
    float4 position : SV_Position;
    PSInput psout;
};

[shader("vertex")]
VSOutput vsmain(VSInput input) {
    VSOutput output;
    output.position = float4(input.position, 1.0);
    output.psout.color = input.color;
    return output;
}

struct PSInput {
    float3 color : COLOR;
};

struct PSOutput {
    float4 output : SV_Target;
};

struct TimeData {
    float time;
};

[[vk::binding(0, 0)]]
ConstantBuffer<TimeData> time;

[shader("fragment")]
PSOutput psmain(PSInput input) {
    let animated = input.color * (0.5 + 0.5 * sin(time.time * 3.0 + input.color * 6.28));
    PSOutput output;
    output.output = float4(animated, 1.0);
    return output;
}