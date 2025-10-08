"""
Base Model Module for POCUS-AI

This module defines the base model classes and interfaces for all POCUS-AI models.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Union, List, Tuple, Optional
import numpy as np
import os
import json

class BaseModel(nn.Module, ABC):
    """Base class for all POCUS-AI models"""
    
    def __init__(self, name: str = "base_model", **kwargs):
        """
        Initialize the base model
        
        Args:
            name: Unique name for the model
            **kwargs: Additional model parameters
        """
        super(BaseModel, self).__init__()
        self.name = name
        self.config = kwargs
        self.model_type = "base"
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
        
    def save(self, path: str) -> None:
        """
        Save the model weights and configuration
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), f"{path}.pt")
        
        # Save model configuration
        config = {
            "name": self.name,
            "model_type": self.model_type,
            "config": self.config
        }
        
        with open(f"{path}.json", "w") as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def load(cls, path: str, **kwargs) -> "BaseModel":
        """
        Load a model from saved weights and configuration
        
        Args:
            path: Path to the saved model
            **kwargs: Additional parameters to override saved configuration
            
        Returns:
            Loaded model
        """
        # Load configuration
        with open(f"{path}.json", "r") as f:
            config = json.load(f)
            
        # Update configuration with any provided overrides
        if kwargs:
            config["config"].update(kwargs)
            
        # Create model instance
        model = cls(**config["config"])
        model.name = config["name"]
        model.model_type = config["model_type"]
        
        # Load weights
        state_dict = torch.load(f"{path}.pt")
        model.load_state_dict(state_dict)
        
        return model
    
    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Make a prediction on input data
        
        Args:
            x: Input data (torch tensor or numpy array)
            
        Returns:
            Prediction as numpy array
        """
        # Convert to torch tensor if numpy array
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
            
        # Add batch dimension if needed
        if len(x.shape) == 3:  # C, H, W format for images
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Set model to evaluation mode
        self.eval()
        
        # Make prediction
        with torch.no_grad():
            output = self.forward(x)
            
        # Convert to numpy and return
        return output.cpu().numpy()
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters and configuration
        
        Returns:
            Dictionary of model parameters
        """
        return {
            "name": self.name,
            "model_type": self.model_type,
            "config": self.config,
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

# Import submodules
from . import segmentation
from . import classification
from . import detection