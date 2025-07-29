# examples/neural_network.py

"""
SLX Framework - Neural Network Training Example
This example demonstrates training a simple neural network for binary classification.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import slx
from slx.core.tensor import tensor
from slx.nn.modules import Linear, ReLU, Sigmoid, Sequential
from slx.nn import functional as F
from slx.optim.sgd import SGD, Adam
import torch
import numpy as np

def generate_data(n_samples=100):
    """Generate synthetic binary classification data"""
    np.random.seed(42)
    
    # Generate two classes of data
    class_0 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    class_1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    
    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]
    
    return X, y

class BinaryClassifier(Sequential):
    """Simple binary classifier"""
    
    def __init__(self, input_size, hidden_size=10):
        super().__init__(
            Linear(input_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, 1),
            Sigmoid()
        )

def binary_cross_entropy_loss(predictions, targets, epsilon=1e-15):
    """Compute binary cross entropy loss"""
    # Clip predictions to prevent log(0)
    predictions_clipped = torch.clamp(predictions.data, epsilon, 1 - epsilon)
    targets_data = targets.data if hasattr(targets, 'data') else targets
    
    loss = -(targets_data * torch.log(predictions_clipped) + 
             (1 - targets_data) * torch.log(1 - predictions_clipped))
    
    return tensor(loss.mean(), requires_grad=predictions.requires_grad)

def accuracy(predictions, targets):
    """Compute accuracy"""
    pred_labels = (predictions.data > 0.5).float()
    targets_data = targets.data if hasattr(targets, 'data') else targets
    correct = (pred_labels == targets_data).float()
    return correct.mean().item()

