[numthreads(256, 1, 1)]
void tanh_backward_kernel(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    StructuredBuffer<float> tanh_output,
    StructuredBuffer<float> grad_output,
    RWStructuredBuffer<float> grad_input,
    uint size
) {
    uint index = dispatchThreadID.x;
    if (index >= size) return;
    
    float t = tanh_output[index];
    grad_input[index] = grad_output[index] * (1.0 - t * t);
}
