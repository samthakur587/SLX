[numthreads(256, 1, 1)]
void relu_backward_kernel(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    StructuredBuffer<float> input,
    StructuredBuffer<float> grad_output,
    RWStructuredBuffer<float> grad_input,
    uint size
) {
    uint index = dispatchThreadID.x;
    if (index >= size) return;
    
    grad_input[index] = input[index] > 0.0 ? grad_output[index] : 0.0;
}
