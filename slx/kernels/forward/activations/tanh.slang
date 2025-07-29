[numthreads(256, 1, 1)]
void tanh_kernel(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    StructuredBuffer<float> input,
    RWStructuredBuffer<float> output,
    uint size
) {
    uint index = dispatchThreadID.x;
    if (index >= size) return;
    
    float x = input[index];
    float exp_2x = exp(2.0 * x);
    output[index] = (exp_2x - 1.0) / (exp_2x + 1.0);
}