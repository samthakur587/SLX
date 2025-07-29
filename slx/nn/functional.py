# slx/nn/functional.py

import torch
from ..core.tensor import tensor, SlxTensor
from typing import Optional

def relu(input):
    """Apply ReLU activation function"""
    return input.relu()

def sigmoid(input):
    """Apply Sigmoid activation function"""
    return input.sigmoid()

def tanh(input):
    """Apply Tanh activation function"""
    return input.tanh()

def softmax(input, dim=-1):
    """Apply Softmax activation function"""
    # Subtract max for numerical stability
    x_max = torch.max(input.data, dim=dim, keepdim=True)[0]
    x_shifted = input.data - x_max
    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    result = exp_x / sum_exp
    return tensor(result, requires_grad=input.requires_grad)

def log_softmax(input, dim=-1):
    """Apply Log Softmax activation function"""
    x_max = torch.max(input.data, dim=dim, keepdim=True)[0]
    x_shifted = input.data - x_max
    log_sum_exp = torch.log(torch.sum(torch.exp(x_shifted), dim=dim, keepdim=True))
    result = x_shifted - log_sum_exp
    return tensor(result, requires_grad=input.requires_grad)

def linear(input, weight, bias=None):
    """Apply linear transformation"""
    output = input @ weight.T
    if bias is not None:
        output = output + bias
    return output

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Apply 2D convolution"""
    # Use PyTorch backend for now
    output = torch.nn.functional.conv2d(
        input.data, weight.data, 
        bias=bias.data if bias else None,
        stride=stride, padding=padding, dilation=dilation, groups=groups
    )
    return tensor(output, requires_grad=input.requires_grad)

def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    """Apply 2D max pooling"""
    output = torch.nn.functional.max_pool2d(
        input.data, kernel_size, stride, padding, dilation, ceil_mode
    )
    return tensor(output, requires_grad=input.requires_grad)

def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    """Apply 2D average pooling"""
    output = torch.nn.functional.avg_pool2d(
        input.data, kernel_size, stride, padding, ceil_mode, count_include_pad
    )
    return tensor(output, requires_grad=input.requires_grad)

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=True, momentum=0.1, eps=1e-5):
    """Apply batch normalization"""
    output = torch.nn.functional.batch_norm(
        input.data, running_mean, running_var, weight.data if weight else None, 
        bias.data if bias else None, training, momentum, eps
    )
    return tensor(output, requires_grad=input.requires_grad)

def dropout(input, p=0.5, training=True, inplace=False):
    """Apply dropout"""
    if training:
        output = torch.nn.functional.dropout(input.data, p, training, inplace)
    else:
        output = input.data
    return tensor(output, requires_grad=input.requires_grad)

def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
    """Compute cross entropy loss"""
    loss = torch.nn.functional.cross_entropy(
        input.data, target, weight, size_average, ignore_index, reduce, reduction
    )
    return tensor(loss, requires_grad=input.requires_grad)

def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    """Compute mean squared error loss"""
    loss = torch.nn.functional.mse_loss(input.data, target.data if hasattr(target, 'data') else target, 
                                       size_average, reduce, reduction)
    return tensor(loss, requires_grad=input.requires_grad)

def binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction='mean'):
    """Compute binary cross entropy loss"""
    loss = torch.nn.functional.binary_cross_entropy(
        input.data, target.data if hasattr(target, 'data') else target,
        weight, size_average, reduce, reduction
    )
    return tensor(loss, requires_grad=input.requires_grad)

def l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    """Compute L1 loss"""
    loss = torch.nn.functional.l1_loss(
        input.data, target.data if hasattr(target, 'data') else target,
        size_average, reduce, reduction
    )
    return tensor(loss, requires_grad=input.requires_grad)

def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    """Interpolate tensor"""
    output = torch.nn.functional.interpolate(
        input.data, size, scale_factor, mode, align_corners
    )
    return tensor(output, requires_grad=input.requires_grad)

def pad(input, pad, mode='constant', value=0):
    """Pad tensor"""
    output = torch.nn.functional.pad(input.data, pad, mode, value)
    return tensor(output, requires_grad=input.requires_grad)