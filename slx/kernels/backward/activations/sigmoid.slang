[numthreads(256, 1, 1)]
void sigmoid_backward_kernel(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    StructuredBuffer<float> sigmoid_output,
    StructuredBuffer<float> grad_output,
    RWStructuredBuffer<float> grad_input,
    uint size
) {
    uint index = dispatchThreadID.x;
    if (index >= size) return;
    
    float s = sigmoid_output[index];
    grad_input[index] = grad_output[index] * s * (1.0 - s);
}
