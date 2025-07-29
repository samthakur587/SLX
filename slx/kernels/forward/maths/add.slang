[numthreads(256, 1, 1)]
void add_kernel(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    StructuredBuffer<float> input_a,
    StructuredBuffer<float> input_b,
    RWStructuredBuffer<float> output,
    uint size
) {
    uint index = dispatchThreadID.x;
    if (index >= size) return;
    
    output[index] = input_a[index] + input_b[index];
}