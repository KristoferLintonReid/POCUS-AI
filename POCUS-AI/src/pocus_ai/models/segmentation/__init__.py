"""
UNet segmentation model for ultrasound image segmentation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from ..import BaseModel

class DoubleConv(nn.Module):
    """Double convolution block for UNet"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetSegmentation(BaseModel):
    """
    UNet model for medical image segmentation.
    
    Paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    Olaf Ronneberger, Philipp Fischer, Thomas Brox
    """
    
    def __init__(self, 
                 in_channels: int = 1, 
                 out_channels: int = 1, 
                 features: List[int] = [64, 128, 256, 512],
                 pretrained: bool = False,
                 name: str = "unet_segmentation",
                 **kwargs):
        """
        Initialize UNet segmentation model
        
        Args:
            in_channels: Number of input channels (default: 1 for grayscale)
            out_channels: Number of output channels (default: 1 for binary segmentation)
            features: List of feature dimensions for each level
            pretrained: Whether to load pretrained weights
            name: Model name
            **kwargs: Additional model parameters
        """
        super(UNetSegmentation, self).__init__(name=name, in_channels=in_channels, 
                                               out_channels=out_channels, features=features, **kwargs)
        self.model_type = "segmentation"
        
        # Encoder (downsampling) path
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Initial double conv
        self.initial = DoubleConv(in_channels, features[0])
        
        # Encoder blocks
        for i in range(len(features)-1):
            self.downs.append(DoubleConv(features[i], features[i+1]))
            
        # Decoder (upsampling) path
        self.ups = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        # Decoder blocks
        for i in range(len(features)-1, 0, -1):
            self.upconvs.append(
                nn.ConvTranspose2d(features[i], features[i-1], kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(features[i], features[i-1]))
            
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Load pretrained weights if specified
        if pretrained:
            self._load_pretrained_weights()
            
    def _load_pretrained_weights(self):
        """Load pretrained weights if available"""
        try:
            weights_path = os.path.join(
                os.path.dirname(__file__), 
                "pretrained", 
                f"{self.name}_pretrained.pt"
            )
            self.load_state_dict(torch.load(weights_path))
            print(f"Loaded pretrained weights from {weights_path}")
        except:
            print("Pretrained weights not found. Using random initialization.")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Segmentation mask of shape [B, out_channels, H, W]
        """
        # Store skip connections
        skip_connections = []
        
        # Initial double conv
        x = self.initial(x)
        skip_connections.append(x)
        
        # Encoder path
        for down in self.downs:
            x = self.pool(x)
            x = down(x)
            skip_connections.append(x)
            
        # Remove last skip connection (bottleneck)
        skip_connections.pop()
        
        # Decoder path (with skip connections)
        for idx, (up, upconv) in enumerate(zip(self.ups, self.upconvs)):
            x = upconv(x)
            skip = skip_connections.pop()
            
            # Handle size mismatch (if needed)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
                
            # Concatenate skip connection
            x = torch.cat((skip, x), dim=1)
            x = up(x)
            
        # Final convolution
        x = self.final_conv(x)
        
        return x