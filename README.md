# SLX - Scalable Learning eXperiments

SLX is a high-performance deep learning framework built on top of PyTorch, designed for efficient tensor operations with support for custom shader-based kernels through Slang. It provides a clean, intuitive interface for building and training neural networks while enabling low-level optimization through custom GPU shaders.

## Features

- **High-Performance Tensor Operations**: Optimized tensor operations with PyTorch backend
- **Custom Shader Support**: Write and compile custom shaders for performance-critical operations
- **Easy Model Building**: Intuitive API for building neural networks
- **Seamless PyTorch Integration**: Compatible with existing PyTorch models and workflows
- **Multi-Device Support**: Run on CPU or CUDA-enabled GPUs

## Installation

### Prerequisites

- Python 3.8+
- PyTorch (will be installed automatically if not present)
- CUDA Toolkit (for GPU acceleration, optional but recommended)
- slangtorch (for custom shader support)

### Using pip

```bash
pip install slx
```

### From Source

```bash
git clone https://github.com/samthakur587/SLX.git
cd SLX
pip install -e .
```

## Quick Start

### Basic Tensor Operations

```python
import slx
import torch

# Create tensors
x = slx.tensor([1.0, 2.0, 3.0], device='cuda')
y = slx.tensor([4.0, 5.0, 6.0], device='cuda')

# Basic operations
z = x + y  # Element-wise addition
z = x * y  # Element-wise multiplication
z = slx.matmul(x, y.T)  # Matrix multiplication

# Activation functions
z = slx.relu(x)  # ReLU activation
z = slx.sigmoid(x)  # Sigmoid activation
z = slx.tanh(x)  # Hyperbolic tangent
```

### Building a Neural Network

```python
import slx
import slx.nn as nn
import slx.optim as optim

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten input
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model, loss function, and optimizer
model = Net().to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (example)
for epoch in range(10):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Using Custom Shaders

SLX allows you to write custom shaders for performance-critical operations. Here's an example of a custom ReLU shader:

1. Create a shader file (e.g., `custom_relu.slang`):
```hlsl
[shader("compute")]
[numthreads(256, 1, 1)]
void csMain(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    StructuredBuffer<float> input : register(t0),
    RWStructuredBuffer<float> output : register(u0)
) {
    uint index = dispatchThreadID.x;
    uint size;
    uint stride;
    input.GetDimensions(size, stride);
    
    if (index >= size) return;
    output[index] = max(0.0, input[index]);
}
```

2. Use the custom shader in Python:
```python
import slx
from slx.kernels import KernelManager

# Initialize kernel manager
km = KernelManager()

# Load and compile the shader
shader = km.load_shader('custom_relu.slang')

# Create input tensor
x = slx.tensor([-1.0, 0.5, 2.0, -0.5], device='cuda')
output = slx.zeros_like(x)

# Execute the shader
shader(input=x, output=output)

print(output)  # Should be [0.0, 0.5, 2.0, 0.0]
```

## Examples

Check out the `examples/` directory for more comprehensive examples:

- `basic_operations.py`: Basic tensor operations
- `custom_kernel.py`: Using custom shaders
- `neural_network.py`: Building and training a simple neural network

## Documentation

For detailed documentation, please refer to the [documentation](docs/).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch for the amazing deep learning framework
- NVIDIA for CUDA and Slang shader language
- The open-source community for inspiration and support
