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
