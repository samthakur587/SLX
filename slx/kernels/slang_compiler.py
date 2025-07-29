"""
Slang shader compiler for SLX framework.

This module handles the compilation and execution of Slang shaders using slangtorch.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import torch
import numpy as np
import slangtorch
from ..core.tensor import SlxTensor

class SlangCompiler:
    """
    Compiles and manages Slang shaders for execution using slangtorch.
    
    This class handles the loading and execution of Slang shaders with PyTorch tensor support.
    """
    
    def __init__(self):
        """Initialize the Slang compiler."""
        self.compiled_shaders = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.shader_dir = Path(__file__).parent / 'slang'
    
    def _get_shader_path(self, shader_name: str) -> Path:
        """Get the full path to a shader file."""
        return self.shader_dir / f"{shader_name}.slang"
    
    def compile_shader(
        self,
        shader_name: str,
        entry_point: str,
        **kwargs
    ) -> bool:
        """
        Load and compile a Slang shader using slangtorch.
        
        Args:
            shader_name: Name of the shader file (without .slang extension)
            entry_point: Name of the entry point function
            
        Returns:
            bool: True if compilation was successful, False otherwise
        """
        try:
            shader_path = self._get_shader_path(shader_name)
            if not shader_path.exists():
                print(f"Shader file not found: {shader_path}")
                return False
            
            # Load the shader module using slangtorch
            module = slangtorch.loadModule(
                str(shader_path),
                verbose=True
            )
            
            # Store the compiled shader
            self.compiled_shaders[entry_point] = {
                'module': module,
                'entry_point': entry_point,
                'shader_name': shader_name
            }
            
            return True
            
        except Exception as e:
            print(f"Error loading shader {shader_name} with entry point {entry_point}: {e}")
            return False
    

    

    
    def execute_shader(
        self,
        entry_point: str,
        *inputs: Union[torch.Tensor, SlxTensor],
        output_shape: Optional[Tuple[int, ...]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Execute a compiled shader with the given inputs.
        
        Args:
            entry_point: Name of the shader entry point to execute
            *inputs: Input tensors (can be torch.Tensor or SlxTensor)
            output_shape: Optional shape of the output tensor (inferred from inputs if not provided)
            **kwargs: Additional arguments to pass to the shader
            
        Returns:
            torch.Tensor: Output tensor
            
        Raises:
            RuntimeError: If shader execution fails
        """
        if entry_point not in self.compiled_shaders:
            raise RuntimeError(f"Shader {entry_point} not found in compiled shaders")
            
        shader_info = self.compiled_shaders[entry_point]
        module = shader_info['module']
        
        try:
            # Convert SlxTensors to torch.Tensor if needed
            torch_inputs = []
            for x in inputs:
                if isinstance(x, SlxTensor):
                    torch_inputs.append(x.data)
                else:
                    torch_inputs.append(x)
            
            # Get the kernel function
            kernel_func = getattr(module, entry_point, None)
            if kernel_func is None:
                raise RuntimeError(f"Entry point {entry_point} not found in shader module")
            
            # Execute the shader
            if output_shape is None:
                # For element-wise operations, output shape is same as input
                output_shape = torch_inputs[0].shape
            
            # Create output tensor on the same device as first input
            device = torch_inputs[0].device
            output = torch.empty(output_shape, device=device)
            
            # Call the shader
            kernel_func(*torch_inputs, output=output, **kwargs)
            
            return output
            
        except Exception as e:
            raise RuntimeError(f"Failed to execute shader {entry_point}: {str(e)}")

# Global compiler instance
compiler = SlangCompiler()

def compile_shader(*args, **kwargs):
    """
    Compile a shader using the global compiler instance.
    
    Args:
        shader_name: Name of the shader file (without .slang extension)
        entry_point: Name of the entry point function
        **kwargs: Additional arguments for shader compilation
        
    Returns:
        bool: True if compilation was successful, False otherwise
    """
    return compiler.compile_shader(*args, **kwargs)

def execute_shader(entry_point: str, *inputs, **kwargs):
    """
    Execute a shader using the global compiler instance.
    
    Args:
        entry_point: Name of the shader entry point to execute
        *inputs: Input tensors (can be torch.Tensor or SlxTensor)
        **kwargs: Additional arguments to pass to the shader
        
    Returns:
        torch.Tensor: Output tensor
    """
    return compiler.execute_shader(entry_point, *inputs, **kwargs)