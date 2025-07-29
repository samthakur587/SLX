"""
Device management for SLX (Slang Extension) library.

This module provides functionality for managing devices (CPU, GPU) and handling
data movement between devices.
"""

import torch
from typing import Union, Optional, Any, Dict, List, Tuple

class Device:
    """
    A class representing a compute device.
    
    This class provides an abstraction over different compute devices (CPU, GPU)
    and handles device-related operations.
    
    Attributes:
        type (str): The type of the device ('cpu' or 'cuda').
        index (int, optional): The device index (for CUDA devices).
    """
    
    def __init__(self, device: Union[str, 'Device', torch.device] = 'cpu'):
        """
        Initialize a device.
        
        Args:
            device: A string ('cpu' or 'cuda'), a torch.device object, or another Device object.
                   Can include device index for CUDA (e.g., 'cuda:0' or 'cuda:1').
        """
        if isinstance(device, Device):
            self.type = device.type
            self.index = device.index
            return
            
        if isinstance(device, torch.device):
            device = str(device)
            
        if isinstance(device, str):
            if ':' in device:
                self.type, idx = device.split(':')
                self.index = int(idx)
            else:
                self.type = device
                self.index = None
        else:
            raise ValueError(f"Unsupported device type: {type(device)}")
            
        # Validate device type
        if self.type not in ['cpu', 'cuda']:
            raise ValueError(f"Unsupported device type: {self.type}. Must be 'cpu' or 'cuda'.")
            
        # Validate CUDA availability
        if self.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot create CUDA device.")
    
    def __str__(self) -> str:
        """Return string representation of the device."""
        if self.index is not None:
            return f"{self.type}:{self.index}"
        return self.type
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the device."""
        return f"Device(type='{self.type}', index={self.index})"
    
    def __eq__(self, other: Any) -> bool:
        """Check if two devices are equal."""
        if not isinstance(other, (Device, str, torch.device)):
            return False
        
        other = Device(other)
        return self.type == other.type and self.index == other.index
    
    def to_torch(self) -> torch.device:
        """Convert to PyTorch device."""
        if self.index is not None:
            return torch.device(f"{self.type}:{self.index}")
        return torch.device(self.type)


def get_device(device: Union[str, Device, torch.device] = 'cpu') -> str:
    """
    Get the device string for the given device.
    
    Args:
        device: Device specification (string, Device, or torch.device).
        
    Returns:
        str: Device string (e.g., 'cpu', 'cuda:0').
    """
    return str(Device(device))


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def get_cuda_device_count() -> int:
    """Get the number of available CUDA devices."""
    return torch.cuda.device_count() if is_cuda_available() else 0


def set_default_device(device: Union[str, Device, torch.device]) -> None:
    """
    Set the default device for tensor creation.
    
    Args:
        device: Device to set as default.
    """
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = str(Device(device))


# Initialize default device
_DEFAULT_DEVICE = 'cpu'


def get_default_device() -> str:
    """Get the default device for tensor creation."""
    return _DEFAULT_DEVICE


def to_torch_device(device: Union[str, Device, torch.device]) -> torch.device:
    """
    Convert a device specification to a PyTorch device.
    
    Args:
        device: Device specification (string, Device, or torch.device).
        
    Returns:
        torch.device: The corresponding PyTorch device.
    """
    if isinstance(device, torch.device):
        return device
    return Device(device).to_torch()


def synchronize(device: Union[str, Device, torch.device] = None) -> None:
    """
    Wait for all kernels in all streams on the given device to complete.
    
    Args:
        device: Device to synchronize. If None, synchronizes all devices.
    """
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    else:
        device = str(Device(device))
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)
