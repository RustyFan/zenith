module triangle;

struct VSInput {
    [[vk::location(0)]]
    float3 position : POSITION;
    [[vk::location(1)]]
    float3 color : COLOR;
};

struct PSInput {
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

struct TimeData {
    float time;
};

[[vk::binding(0, 0)]]
ConstantBuffer<TimeData> Time;

[shader("fragment")]
float4 psmain(PSInput input) : SV_Target {
    float3 animated = input.color * (0.5 + 0.5 * sin(Time.time * 3.0 + input.color * 6.28));
    return float4(animated, 1.0);
}
