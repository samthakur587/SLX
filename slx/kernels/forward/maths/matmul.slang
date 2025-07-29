[numthreads(16, 16, 1)]
void matmul_kernel(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    StructuredBuffer<float> input_a,
    StructuredBuffer<float> input_b,
    RWStructuredBuffer<float> output,
    uint M, uint N, uint K
) {
    uint row = dispatchThreadID.x;
    uint col = dispatchThreadID.y;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += input_a[row * K + k] * input_b[k * N + col];
    }
    
    output[row * N + col] = sum;
}