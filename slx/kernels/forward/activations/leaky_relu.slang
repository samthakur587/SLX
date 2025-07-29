[numthreads(256, 1, 1)]
void leaky_relu_kernel(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    StructuredBuffer<float> input,
    RWStructuredBuffer<float> output,
    float negative_slope,
    uint size
) {
    uint index = dispatchThreadID.x;
    if (index >= size) return;
    
    float x = input[index];
    output[index] = x > 0.0 ? x : negative_slope * x;
}