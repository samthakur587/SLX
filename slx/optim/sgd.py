# slx/optim/sgd.py

import torch
from .optimizer import Optimizer

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, 
                       weight_decay=weight_decay, nesterov=nesterov)
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad.data
                if weight_decay != 0:
                    grad = grad + weight_decay * param.data
                
                if momentum != 0:
                    param_state = self.state.setdefault(id(param), {})
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(param.data)
                        buf.mul_(momentum).add_(grad)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                    
                    if nesterov:
                        grad = grad + momentum * buf
                    else:
                        grad = buf
                
                param.data.add_(grad, alpha=-group['lr'])
        
        return loss
