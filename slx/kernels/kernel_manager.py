# slx/kernels/kernel_manager.py

import torch
import os
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path

from ..core.autograd import AddFunction, MulFunction, MatMulFunction, ReluFunction
from ..core.tensor import SlxTensor
from .slang_compiler import compile_shader, execute_shader

# Map kernel names to their corresponding shader files and entry points
KERNEL_MAPPING = {
    'add': {'shader': 'basic_ops', 'entry_point': 'add_kernel'},
    'mul': {'shader': 'basic_ops', 'entry_point': 'mul_kernel'},
    'matmul': {'shader': 'basic_ops', 'entry_point': 'matmul_kernel'},
    'relu': {'shader': 'activation', 'entry_point': 'relu_kernel'},
    'sigmoid': {'shader': 'activation', 'entry_point': 'sigmoid_kernel'},
    'tanh': {'shader': 'activation', 'entry_point': 'tanh_kernel'},
    'sum': {'shader': 'reduction', 'entry_point': 'sum_kernel'}
}

class KernelManager:
    """Manages Slang kernel compilation and execution"""
    
    _compiled_kernels = {}
    _kernel_cache = {}
    
    @classmethod
    def _get_shader_path(cls, shader_name: str) -> Path:
        """Get the full path to a shader file."""
        kernel_dir = Path(__file__).parent / 'slang'
        return kernel_dir / f"{shader_name}.slang"
    
    @classmethod
    def _get_kernel_info(cls, kernel_name: str) -> dict:
        """Get shader and entry point information for a kernel."""
        return KERNEL_MAPPING.get(kernel_name, {})
    
    @classmethod
    def compile_slang_kernel(cls, kernel_name: str, slang_file: str = None) -> bool:
        """
        Compile a Slang shader for a specific kernel.
        
        Args:
            kernel_name: Name of the kernel to compile
            slang_file: Optional override for the shader file name
            
        Returns:
            bool: True if compilation was successful, False otherwise
        """
        try:
            # Get kernel info
            kernel_info = cls._get_kernel_info(kernel_name)
            if not kernel_info:
                print(f"No mapping found for kernel: {kernel_name}")
                return False
            
            # Use provided shader file or get from mapping
            shader_name = slang_file.replace('.slang', '') if slang_file else kernel_info['shader']
            entry_point = kernel_info['entry_point']
            
            # Compile the shader
            success = compile_shader(
                shader_name=shader_name,
                entry_point=entry_point
            )
            
            if success:
                cls._compiled_kernels[kernel_name] = {
                    'shader': shader_name,
                    'entry_point': entry_point,
                    'compiled': True
                }
                print(f"Successfully compiled kernel: {kernel_name}")
                return True
            else:
                print(f"Failed to compile kernel: {kernel_name}")
                return False
            
        except Exception as e:
            print(f"Failed to compile kernel {kernel_name}: {e}")
            return False
    
    @classmethod
    def execute_add(cls, a, b):
        """Execute element-wise addition using Slang shader or PyTorch fallback."""
        try:
            # Try to use Slang shader first
            if 'add' not in cls._compiled_kernels:
                if not cls.compile_slang_kernel('add'):
                    raise RuntimeError("Failed to compile add kernel")
            
            # Ensure tensors are on the same device
            device = a.device if hasattr(a, 'device') else 'cpu'
            b_on_device = b.to(device) if hasattr(b, 'to') else b
            
            # Execute the shader
            result = cls._execute_shader('add', a, b_on_device, output_shape=a.shape)
            return result
            
        except Exception as e:
            print(f"Shader execution failed, falling back to PyTorch: {e}")
            return AddFunction.apply(a, b)
    
    @classmethod
    def execute_mul(cls, a, b):
        """Execute element-wise multiplication using Slang shader or PyTorch fallback."""
        try:
            # Try to use Slang shader first
            if 'mul' not in cls._compiled_kernels:
                if not cls.compile_slang_kernel('mul'):
                    raise RuntimeError("Failed to compile mul kernel")
            
            # Ensure tensors are on the same device
            device = a.device if hasattr(a, 'device') else 'cpu'
            b_on_device = b.to(device) if hasattr(b, 'to') else b
            
            # Execute the shader
            result = cls._execute_shader('mul', a, b_on_device, output_shape=a.shape)
            return result
            
        except Exception as e:
            print(f"Shader execution failed, falling back to PyTorch: {e}")
            return MulFunction.apply(a, b)
    
    @classmethod
    def execute_matmul(cls, a, b):
        """Execute matrix multiplication kernel"""
        # Calculate output shape for matrix multiplication
        output_shape = (a.shape[0], b.shape[1])
        
        # Try to use the shader first
        if 'matmul' not in cls._compiled_kernels:
            if not cls.compile_slang_kernel('matmul', 'basic_ops.slang'):
                # Fallback to PyTorch if compilation fails
                return MatMulFunction.apply(a, b)
        
        try:
            # Execute the shader
            output = cls._execute_shader('matmul', a, b, output_shape=output_shape)
            
            # Create a new tensor with the same properties as input
            from ..core.tensor import SlxTensor
            return SlxTensor(
                output,
                requires_grad=a.requires_grad or b.requires_grad,
                device=a.device
            )
        except Exception as e:
            print(f"Shader execution failed, falling back to PyTorch: {e}")
            return MatMulFunction.apply(a, b)
    
    @classmethod
    def _prepare_tensor_for_shader(cls, tensor) -> torch.Tensor:
        """Convert a tensor to a format suitable for shader input."""
        if isinstance(tensor, SlxTensor):
            return tensor.data
        return tensor
    
    @classmethod
    def _execute_shader(cls, kernel_name: str, *inputs, **kwargs):
        """
        Execute a shader with the given inputs.
        
        Args:
            kernel_name: Name of the kernel to execute
            *inputs: Input tensors (can be torch.Tensor or SlxTensor)
            **kwargs: Additional arguments for the shader
            
        Returns:
            The result of the shader execution as a torch.Tensor
        """
        if kernel_name not in cls._compiled_kernels:
            # Try to compile the kernel if not already compiled
            if not cls.compile_slang_kernel(kernel_name):
                raise RuntimeError(f"Failed to compile kernel: {kernel_name}")
        
        try:
            # Get the entry point for this kernel
            entry_point = cls._compiled_kernels[kernel_name]['entry_point']
            
            # Prepare inputs - convert SlxTensors to torch.Tensor if needed
            prepared_inputs = [cls._prepare_tensor_for_shader(x) for x in inputs]
            
            # Get output shape from kwargs or infer from first input
            output_shape = kwargs.pop('output_shape', None)
            if output_shape is None and prepared_inputs:
                output_shape = prepared_inputs[0].shape
            
            # Execute the shader
            result = execute_shader(
                entry_point=entry_point,
                *prepared_inputs,
                output_shape=output_shape,
                **kwargs
            )
            
            return result
            
        except Exception as e:
            print(f"Error executing shader {kernel_name}: {e}")
            raise
    
    @classmethod
    def execute_relu(cls, input_tensor):
        """Execute ReLU activation using Slang shader or PyTorch fallback."""
        try:
            # Try to use Slang shader first
            if 'relu' not in cls._compiled_kernels:
                if not cls.compile_slang_kernel('relu'):
                    raise RuntimeError("Failed to compile relu kernel")
            
            # Execute the shader
            result = cls._execute_shader('relu', input_tensor, output_shape=input_tensor.shape)
            return result
            
        except Exception as e:
            print(f"Shader execution failed, falling back to PyTorch: {e}")
            # Convert SlxTensor to PyTorch tensor if needed
            if hasattr(input_tensor, 'data'):
                return ReluFunction.apply(input_tensor.data)
            return ReluFunction.apply(input_tensor)
    
    @classmethod
    def execute_sigmoid(cls, input_tensor):
        """Execute sigmoid activation using Slang shader or PyTorch fallback."""
        try:
            # Try to use Slang shader first
            if 'sigmoid' not in cls._compiled_kernels:
                if not cls.compile_slang_kernel('sigmoid'):
                    raise RuntimeError("Failed to compile sigmoid kernel")
            
            # Execute the shader
            result = cls._execute_shader('sigmoid', input_tensor, output_shape=input_tensor.shape)
            return result
            
        except Exception as e:
            print(f"Shader execution failed, falling back to PyTorch: {e}")
            # Convert SlxTensor to PyTorch tensor if needed
            if hasattr(input_tensor, 'data'):
                return torch.sigmoid(input_tensor.data)
            return torch.sigmoid(input_tensor)
    
    @classmethod
    def execute_tanh(cls, input_tensor):
        """Execute tanh activation using Slang shader or PyTorch fallback."""
        try:
            # Try to use Slang shader first
            if 'tanh' not in cls._compiled_kernels:
                if not cls.compile_slang_kernel('tanh'):
                    raise RuntimeError("Failed to compile tanh kernel")
            
            # Execute the shader
            result = cls._execute_shader('tanh', input_tensor, output_shape=input_tensor.shape)
            return result
            
        except Exception as e:
            print(f"Shader execution failed, falling back to PyTorch: {e}")
            # Convert SlxTensor to PyTorch tensor if needed
            if hasattr(input_tensor, 'data'):
                return torch.tanh(input_tensor.data)
            return torch.tanh(input_tensor)
    
    @classmethod
    def execute_sum(cls, input_tensor, dim=None, keepdim=False):
        """
        Execute sum reduction using Slang shader or PyTorch fallback.
        
        Args:
            input_tensor: Input tensor (can be SlxTensor or torch.Tensor)
            dim: Dimension to reduce. If None, all dimensions are reduced.
            keepdim: Whether to keep the reduced dimension.
            
        Returns:
            The result of the sum operation as a tensor
        """
        # Get the underlying PyTorch tensor if input is SlxTensor
        tensor_data = input_tensor.data if hasattr(input_tensor, 'data') else input_tensor
        
        try:
            # For now, we'll use PyTorch's implementation for sum with dim/keepdim
            # as it's more complex to implement in shaders
            if dim is not None or not keepdim:
                result = tensor_data.sum(dim=dim, keepdim=keepdim)
                # Return SlxTensor if input was SlxTensor, otherwise return PyTorch tensor
                if hasattr(input_tensor, 'data'):
                    from ..core.tensor import SlxTensor
                    return SlxTensor(result, requires_grad=input_tensor.requires_grad, device=input_tensor.device)
                return result
                
            # Try to use Slang shader for full reduction
            if 'sum' not in cls._compiled_kernels:
                if not cls.compile_slang_kernel('sum'):
                    raise RuntimeError("Failed to compile sum kernel")
            
            # Execute the shader for full reduction
            result = cls._execute_shader('sum', input_tensor, output_shape=(1,))
            return result
            
        except Exception as e:
            print(f"Shader execution failed, falling back to PyTorch: {e}")
            result = tensor_data.sum(dim=dim, keepdim=keepdim)
            # Return SlxTensor if input was SlxTensor, otherwise return PyTorch tensor
            if hasattr(input_tensor, 'data'):
                from ..core.tensor import SlxTensor
                return SlxTensor(result, requires_grad=input_tensor.requires_grad, device=input_tensor.device)
            return result
    
    @classmethod
    def list_kernels(cls):
        """List all compiled kernels"""
        return list(cls._compiled_kernels.keys())
    
    @classmethod
    def get_kernel_info(cls, kernel_name: str):
        """Get information about a compiled kernel"""
        return cls._compiled_kernels.get(kernel_name, None)