import math
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from . import layers

class MappingNetwork(pl.LightningModule):
    def __init__(self,
                  latent_size: int = 512,
                  style_size: int = 512,
                  state_size: int = 512,
                  num_layers: int = 8,
                  label_size: int = 0,
                  lr_mul: float = 1
                ):
        super().__init__()

        self.label_size = label_size

        in_channels = latent_size

        if label_size > 0:
            self.embed_label = layers.EqualizedLinear(label_size, latent_size, lr_mul=lr_mul, bias=False)
            in_channels *= 2

        linear_layers = []
        
        for i in range(0, num_layers):
            if i == num_layers - 1:
                linear_layer = layers.EqualizedLinear(in_channels, style_size, lr_mul=lr_mul)
            else:
                linear_layer = layers.EqualizedLinear(in_channels, state_size, lr_mul=lr_mul)
            linear_layers.append(linear_layer)
            linear_layers.append(nn.LeakyReLU(negative_slope=0.2))
    
            in_channels = state_size

        self.linear_layers = nn.Sequential(linear_layers)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        if labels is not None and self.label_size > 0:
            label_embedding = self.embed_label(labels)
            x = torch.cat([x, label_embedding], dim=-1)
        
        return self.linear_layers.forward(x)