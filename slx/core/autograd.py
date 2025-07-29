# slx/core/autograd.py

import torch
from typing import List, Optional, Tuple, Any

class Function:
    """Base class for differentiable functions"""
    
    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError("Forward method must be implemented")
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Backward method must be implemented")
    
    @classmethod
    def apply(cls, *args):
        """Apply the function with automatic differentiation"""
        ctx = Context()
        
        # Extract tensors and check if any require grad
        tensors = []
        requires_grad = False
        for arg in args:
            if hasattr(arg, 'requires_grad'):
                tensors.append(arg)
                if arg.requires_grad:
                    requires_grad = True
        
        # Forward pass
        result = cls.forward(ctx, *args)
        
        # Set up backward pass if needed
        if requires_grad:
            result.grad_fn = BackwardFunction(cls, ctx, tensors)
            result.requires_grad = True
        
        return result

class Context:
    """Context object to save information for backward pass"""
    
    def __init__(self):
        self.saved_tensors = []
        self.saved_variables = {}
    
    def save_for_backward(self, *tensors):
        """Save tensors for use in backward pass"""
        self.saved_tensors = tensors
    
    def save_variable(self, name, value):
        """Save a variable for use in backward pass"""
        self.saved_variables[name] = value

class BackwardFunction:
    """Handles backward pass for a function"""
    
    def __init__(self, function_class, ctx, input_tensors):
        self.function_class = function_class
        self.ctx = ctx
        self.input_tensors = input_tensors
    
    def backward(self, grad_output):
        """Execute backward pass"""
        grads = self.function_class.backward(self.ctx, grad_output)
        
        if not isinstance(grads, tuple):
            grads = (grads,)
        
        # Accumulate gradients for input tensors
        for i, (tensor, grad) in enumerate(zip(self.input_tensors, grads)):
            if tensor.requires_grad and grad is not None:
                if tensor.grad is None:
                    tensor.grad = torch.zeros_like(tensor.data)
                tensor.grad += grad
                
                # Continue backpropagation
                if tensor.grad_fn is not None:
                    tensor.grad_fn.backward(grad)

class Variable:
    """Wrapper for tensors with gradient tracking (legacy support)"""
    
    def __init__(self, tensor, requires_grad=True):
        self.data = tensor
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

# Automatic differentiation functions
class AddFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        from ..core.tensor import SlxTensor
        result_data = a.data + (b.data if hasattr(b, 'data') else b)
        return SlxTensor(result_data, requires_grad=False, device=a.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

class MulFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        from ..core.tensor import SlxTensor
        ctx.save_for_backward(a, b)
        
        if hasattr(b, 'data'):
            result_data = a.data * b.data
        else:
            result_data = a.data * b
            
        return SlxTensor(result_data, requires_grad=False, device=a.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        
        if hasattr(b, 'data'):
            grad_a = grad_output * b.data
            grad_b = grad_output * a.data
            return grad_a, grad_b
        else:
            grad_a = grad_output * b
            return grad_a, None

class MatMulFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        from ..core.tensor import SlxTensor
        ctx.save_for_backward(a, b)
        result_data = torch.matmul(a.data, b.data)
        return SlxTensor(result_data, requires_grad=False, device=a.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = torch.matmul(grad_output, b.data.transpose(-2, -1))
        grad_b = torch.matmul(a.data.transpose(-2, -1), grad_output)
        return grad_a, grad_b

class ReluFunction(Function):
    @staticmethod
    def forward(ctx, input):
        from ..core.tensor import SlxTensor
        ctx.save_for_backward(input)
        result_data = torch.relu(input.data)
        return SlxTensor(result_data, requires_grad=False, device=input.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * (input.data > 0).float()
        return grad_input