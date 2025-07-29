# examples/basic_operations.py

"""
SLX Framework - Basic Operations Example
This example demonstrates basic tensor operations and automatic differentiation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import slx
from slx.core.tensor import tensor
from slx.kernels.kernel_manager import KernelManager

def test_basic_operations():
    """Test basic tensor operations"""
    print("=== SLX Framework - Basic Operations ===\n")
    
    # Create tensors
    print("1. Creating tensors:")
    a = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = tensor([[2.0, 1.0], [1.0, 2.0]], requires_grad=True)
    
    print(f"Tensor a: {a}")
    print(f"Tensor b: {b}")
    print(f"Shape a: {a.shape}")
    print(f"Shape b: {b.shape}\n")
    
    # Test addition
    print("2. Testing addition (a + b):")
    c = a + b
    print(f"Result: {c}")
    print(f"Requires grad: {c.requires_grad}\n")
    
    # Test multiplication
    print("3. Testing element-wise multiplication (a * b):")
    d = a * b
    print(f"Result: {d}")
    print(f"Requires grad: {d.requires_grad}\n")
    
    # Test matrix multiplication
    print("4. Testing matrix multiplication (a @ b):")
    e = a @ b
    print(f"Result: {e}")
    print(f"Requires grad: {e.requires_grad}\n")
    
    # Test activation functions
    print("5. Testing activation functions:")
    print("ReLU:")
    f = a.relu()
    print(f"ReLU(a): {f}")
    
    print("Sigmoid:")
    g = a.sigmoid()
    print(f"Sigmoid(a): {g}")
    
    print("Tanh:")
    h = a.tanh()
    print(f"Tanh(a): {h}\n")
    
    # Test reduction operations
    print("6. Testing reduction operations:")
    sum_all = a.sum()
    print(f"Sum all elements: {sum_all}")
    
    sum_dim0 = a.sum(dim=0)
    print(f"Sum along dim 0: {sum_dim0}")
    
    sum_dim1 = a.sum(dim=1)
    print(f"Sum along dim 1: {sum_dim1}\n")
    
    return a, b, c, d, e

def test_autograd():
    """Test automatic differentiation"""
    print("=== Testing Automatic Differentiation ===\n")
    
    # Create input tensors
    x = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = tensor([[2.0, 1.0], [1.0, 2.0]], requires_grad=True)
    
    print(f"Input x: {x}")
    print(f"Input y: {y}")
    
    # Forward pass
    z = x * y  # Element-wise multiplication
    w = z.sum()  # Sum all elements
    
    print(f"z = x * y: {z}")
    print(f"w = z.sum(): {w}")
    
    # Backward pass
    print("\nPerforming backward pass...")
    w.backward()
    
    print(f"Gradient of x: {x.grad}")
    print(f"Gradient of y: {y.grad}")
    
    # Verify gradients manually
    print("\nManual verification:")
    print("dw/dx should equal y (since w = sum(x * y), dw/dx = y)")
    print("dw/dy should equal x (since w = sum(x * y), dw/dy = x)")
    
    return x, y, z, w

def test_neural_network_components():
    """Test basic neural network components"""
    print("\n=== Testing Neural Network Components ===\n")
    
    from slx.nn.modules import Linear, ReLU, Sequential
    from slx.optim.sgd import SGD
    
    # Create a simple network
    print("1. Creating a simple neural network:")
    net = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1)
    )
    
    print("Network architecture:")
    for i, layer in enumerate(net._modules.values()):
        print(f"  Layer {i}: {type(layer).__name__}")
    
    # Create input data
    x = tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=False)
    target = tensor([[1.0], [0.0], [1.0]], requires_grad=False)
    
    print(f"\nInput: {x}")
    print(f"Target: {target}")
    
    # Forward pass
    output = net(x)
    print(f"Output: {output}")
    
    # Compute loss (simple MSE)
    loss = ((output.data - target.data) ** 2).mean()
    loss_tensor = tensor(loss, requires_grad=True)
    print(f"Loss: {loss_tensor}")
    
    # Create optimizer
    optimizer = SGD(net.parameters(), lr=0.01)
    
    print(f"\nOptimizer: {type(optimizer).__name__}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    
    return net, x, target, output, loss_tensor, optimizer

def main():
    """Main function to run all tests"""
    print("SLX Framework Test Suite")
    print("=" * 50)
    
    try:
        # Test basic operations
        a, b, c, d, e = test_basic_operations()
        
        # Test autograd
        x, y, z, w = test_autograd()
        
        # Test neural network components
        #net, input_data, target, output, loss, optimizer = test_neural_network_components()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
        # Show kernel information
        print("\nCompiled kernels:")
        kernels = KernelManager.list_kernels()
        for kernel in kernels:
            info = KernelManager.get_kernel_info(kernel)
            print(f"  - {kernel}: {'✓' if info['compiled'] else '✗'}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()