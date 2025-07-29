[numthreads(256, 1, 1)]
void sigmoid_kernel(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    StructuredBuffer<float> input,
    RWStructuredBuffer<float> output,
    uint size
) {
    uint index = dispatchThreadID.x;
    if (index >= size) return;
    
    float x = input[index];
    output[index] = 1.0 / (1.0 + exp(-x));
}