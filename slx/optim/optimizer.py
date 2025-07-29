# slx/optim/optimizer.py

from typing import List, Optional, Dict, Any
from ..core.tensor import SlxTensor

class Optimizer:
    """Base class for all optimizers"""
    
    def __init__(self, params, defaults):
        self.defaults = defaults
        self.state = {}
        self.param_groups = []
        
        if isinstance(params, SlxTensor):
            params = [params]
        elif not isinstance(params, list):
            params = list(params)
        
        if len(params) == 0:
            raise ValueError("optimizer got an empty parameter list")
        
        param_groups = [{'params': params}]
        
        for param_group in param_groups:
            self.add_param_group(param_group)
    
    def add_param_group(self, param_group):
        """Add a param group to the optimizer"""
        params = param_group['params']
        if isinstance(params, SlxTensor):
            param_group['params'] = [params]
        elif isinstance(params, list):
            param_group['params'] = params
        else:
            param_group['params'] = list(params)
        
        for default_key, default_value in self.defaults.items():
            param_group.setdefault(default_key, default_value)
        
        self.param_groups.append(param_group)
    
    def zero_grad(self):
        """Clear gradients of all parameters"""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.zero_()
    
    def step(self, closure=None):
        """Perform a single optimization step"""
        raise NotImplementedError("Subclasses must implement step method")
    
    def state_dict(self):
        """Return the state of the optimizer as a dict"""
        return {
            'state': self.state,
            'param_groups': self.param_groups
        }
    
    def load_state_dict(self, state_dict):
        """Load the optimizer state"""
        self.state = state_dict['state']
        self.param_groups = state_dict['param_groups']



