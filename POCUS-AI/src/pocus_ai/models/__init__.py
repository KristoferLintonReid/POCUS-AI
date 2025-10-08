"""
Base models and model utilities for POCUS-AI.
"""

import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """Base class for all models in POCUS-AI."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        raise NotImplementedError
        
    def save(self, path):
        """Save model weights."""
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        """Load model weights."""
        self.load_state_dict(torch.load(path))