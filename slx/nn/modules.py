# slx/nn/modules.py

from ..core.tensor import SlxTensor, tensor
from typing import Optional, List
import torch
import math

class Module:
    """Base class for all neural network modules"""
    
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method must be implemented")
    
    def parameters(self):
        """Returns an iterator over module parameters"""
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            for param in module.parameters():
                yield param
    
    def named_parameters(self):
        """Returns an iterator over module parameters with names"""
        for name, param in self._parameters.items():
            yield name, param
        for module_name, module in self._modules.items():
            for name, param in module.named_parameters():
                yield f"{module_name}.{name}", param
    
    def zero_grad(self):
        """Zero out gradients of all parameters"""
        for param in self.parameters():
            param.zero_grad()
    
    def train(self, mode=True):
        """Set training mode"""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        return self.train(False)
    
    def to(self, device):
        """Move module to device"""
        for param in self._parameters.values():
            param.data = param.to(device)
        for module in self._modules.values():
            module.to(device)
        return self

class Linear(Module):
    """Linear transformation layer"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using Xavier uniform
        bound = 1 / math.sqrt(in_features)
        weight_data = torch.empty(out_features, in_features).uniform_(-bound, bound)
        self.weight = tensor(weight_data, requires_grad=True)
        self._parameters['weight'] = self.weight
        
        if bias:
            bias_data = torch.empty(out_features).uniform_(-bound, bound)
            self.bias = tensor(bias_data, requires_grad=True)
            self._parameters['bias'] = self.bias
        else:
            self.bias = None
    
    def forward(self, input):
        """Forward pass through linear layer"""
        # input @ weight.T + bias
        output = input @ self.weight.data.T
        if self.bias is not None:
            output = output + self.bias
        return tensor(output, requires_grad=input.requires_grad)

class ReLU(Module):
    """ReLU activation function"""
    
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, input):
        return input.relu()

class Sigmoid(Module):
    """Sigmoid activation function"""
    
    def forward(self, input):
        return input.sigmoid()

class Tanh(Module):
    """Tanh activation function"""
    
    def forward(self, input):
        return input.tanh()

class Sequential(Module):
    """Sequential container for modules"""
    
    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            self._modules[str(i)] = module
    
    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input
    
    def __len__(self):
        return len(self._modules)
    
    def __getitem__(self, idx):
        return self._modules[str(idx)]

class Conv2d(Module):
    """2D Convolution layer"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Initialize weights
        n = in_channels * self.kernel_size[0] * self.kernel_size[1]
        bound = 1 / math.sqrt(n)
        weight_shape = (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
        weight_data = torch.empty(weight_shape).uniform_(-bound, bound)
        self.weight = tensor(weight_data, requires_grad=True)
        self._parameters['weight'] = self.weight
        
        if bias:
            bias_data = torch.empty(out_channels).uniform_(-bound, bound)
            self.bias = tensor(bias_data, requires_grad=True)
            self._parameters['bias'] = self.bias
        else:
            self.bias = None
    
    def forward(self, input):
        # Simple convolution using PyTorch backend for now
        # In a real implementation, this would use Slang kernels
        output = torch.nn.functional.conv2d(
            input.data, self.weight.data, 
            bias=self.bias.data if self.bias else None,
            stride=self.stride, padding=self.padding
        )
        return tensor(output, requires_grad=input.requires_grad)

class BatchNorm1d(Module):
    """1D Batch Normalization"""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = tensor(torch.ones(num_features), requires_grad=True)
        self.bias = tensor(torch.zeros(num_features), requires_grad=True)
        self._parameters['weight'] = self.weight
        self._parameters['bias'] = self.bias
        
        # Running statistics
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
    
    def forward(self, input):
        if self.training:
            # Calculate batch statistics
            mean = input.data.mean(dim=0, keepdim=True)
            var = input.data.var(dim=0, keepdim=True, unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            mean = self.running_mean.unsqueeze(0)
            var = self.running_var.unsqueeze(0)
        
        # Normalize
        normalized = (input.data - mean) / torch.sqrt(var + self.eps)
        output = normalized * self.weight.data + self.bias.data
        
        return tensor(output, requires_grad=input.requires_grad)

class Dropout(Module):
    """Dropout layer"""
    
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace
    
    def forward(self, input):
        if self.training:
            # Apply dropout
            mask = torch.rand_like(input.data) > self.p
            output = input.data * mask / (1 - self.p)
        else:
            output = input.data
        
        return tensor(output, requires_grad=input.requires_grad)