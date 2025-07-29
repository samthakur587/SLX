"""
SLX Core Module

This is the core module of the SLX (Slang Extension) library, providing
fundamental tensor operations, automatic differentiation, and device management.
"""

# Import and expose core functionality
from .tensor import tensor, SlxTensor
from .autograd import Variable, Function
from .device import (
    Device,
    get_device,
    is_cuda_available,
    get_cuda_device_count,
    set_default_device,
    get_default_device,
    to_torch_device,
    synchronize
)

# Set up package-level exports
__all__ = [
    # Tensor and autograd
    'tensor',
    'SlxTensor',
    'Variable',
    'Function',
    
    # Device management
    'Device',
    'get_device',
    'is_cuda_available',
    'get_cuda_device_count',
    'set_default_device',
    'get_default_device',
    'to_torch_device',
    'synchronize',
]