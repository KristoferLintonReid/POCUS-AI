"""
Self-supervised learning methods for ultrasound image analysis.
"""

import torch
import torch.nn as nn

class ContrastiveLearning(nn.Module):
    """Base class for contrastive learning methods."""
    
    def __init__(self, encoder: nn.Module, projection_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder.output_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z