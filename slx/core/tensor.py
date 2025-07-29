# slx/core/tensor.py

import torch
import numpy as np
from typing import Optional, Tuple, Union, List
from .autograd import Function, Variable

class SlxTensor:
    """
    Custom tensor class that integrates Slang kernels with PyTorch backend
    """
    
    def __init__(self, data, requires_grad=False, device='cpu'):
        # Import here to avoid circular imports
        from .device import to_torch_device
        
        try:
            # Convert input to torch.Tensor first
            if isinstance(data, (list, tuple)):
                data = torch.tensor(data, dtype=torch.float32)
            elif isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            elif not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            
            # Convert device string to torch.device
            torch_device = to_torch_device(device)
            
            # Move tensor to the specified device
            self.data = data.to(torch_device)
            self.requires_grad = requires_grad
            self.grad = None
            self.grad_fn = None
            self.device = str(device)  # Store as string for consistency
            self._version = 0
            
            if requires_grad:
                self.grad = torch.zeros_like(self.data)
                
        except Exception as e:
            raise RuntimeError(f"Failed to create tensor: {str(e)}")
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __repr__(self):
        return f"SlxTensor({self.data}, requires_grad={self.requires_grad})"
    
    def __add__(self, other):
        from ..kernels.kernel_manager import KernelManager
        return KernelManager.execute_add(self, other)
    
    def __mul__(self, other):
        from ..kernels.kernel_manager import KernelManager
        return KernelManager.execute_mul(self, other)
    
    def __matmul__(self, other):
        from ..kernels.kernel_manager import KernelManager
        return KernelManager.execute_matmul(self, other)
    
    def relu(self):
        from ..kernels.kernel_manager import KernelManager
        return KernelManager.execute_relu(self)
    
    def sigmoid(self):
        from ..kernels.kernel_manager import KernelManager
        return KernelManager.execute_sigmoid(self)
    
    def tanh(self):
        from ..kernels.kernel_manager import KernelManager
        return KernelManager.execute_tanh(self)
    
    def sum(self, dim=None, keepdim=False):
        from ..kernels.kernel_manager import KernelManager
        return KernelManager.execute_sum(self, dim, keepdim)
    
    def backward(self, gradient=None):
        """Compute gradients using automatic differentiation"""
        if not self.requires_grad:
            return
        
        if gradient is None:
            if self.data.numel() != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            gradient = torch.ones_like(self.data)
        
        if self.grad_fn is not None:
            self.grad_fn.backward(gradient)
        else:
            if self.grad is None:
                self.grad = torch.zeros_like(self.data)
            self.grad += gradient
    
    def zero_grad(self):
        """Zero out the gradients"""
        if self.grad is not None:
            self.grad.zero_()
    
    def detach(self):
        """Return a new tensor detached from the computation graph"""
        new_tensor = SlxTensor(self.data.clone(), requires_grad=False, device=self.device)
        return new_tensor
    
    def to(self, device):
        """
        Move tensor to specified device.
        
        Args:
            device: Target device (string, torch.device, or Device object)
            
        Returns:
            SlxTensor: A new tensor on the specified device
        """
        from .device import to_torch_device, get_device
        
        try:
            # Convert device to string for comparison
            device_str = str(device)
            
            # If already on target device, return self
            if device_str == self.device:
                return self
                
            # Convert device to torch device
            torch_device = to_torch_device(device)
            
            # Create new tensor on target device
            new_tensor = SlxTensor(
                self.data.to(torch_device), 
                requires_grad=self.requires_grad, 
                device=device_str
            )
            
            # Move gradient if it exists
            if self.grad is not None:
                new_tensor.grad = self.grad.to(torch_device)
                
            return new_tensor
            
        except Exception as e:
            raise RuntimeError(f"Failed to move tensor to device {device}: {str(e)}")
    
    def numpy(self):
        """Convert to numpy array"""
        return self.data.detach().cpu().numpy()
    
    def item(self):
        """Get scalar value"""
        return self.data.item()

def tensor(data, requires_grad=False, device='cpu'):
    """
    Factory function to create SlxTensor
    
    Args:
        data: Input data (list, np.ndarray, torch.Tensor, or SlxTensor)
        requires_grad: Whether the tensor requires gradient computation
        device: Target device for the tensor
        
    Returns:
        SlxTensor: A new tensor with the specified properties
    """
    # If data is already an SlxTensor, handle specially
    if isinstance(data, SlxTensor):
        # If no changes needed, return as is
        if (data.requires_grad == requires_grad and 
            str(device) == str(data.device)):
            return data
        # Otherwise create a new tensor with the same data but new properties
        return SlxTensor(
            data.data.detach().clone(),
            requires_grad=requires_grad,
            device=device
        )
    
    # For all other types, create a new tensor
    return SlxTensor(data, requires_grad, device)