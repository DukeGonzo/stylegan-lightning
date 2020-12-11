import math
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from . import layers

class MappingNetwork(pl.LightningModule):
    def __init__(self,
                  input_size: int = 512,
                  latent_size: int = 512,
                  state_size: int = 512,
                  num_layers: int = 8,
                  label_size: int = 0,
                  lr_mul: float = 1,
                  normalize_z: bool = True
                ):
        super().__init__()

        self.label_size = label_size
        self.normalize_z = normalize_z
        self.input_size = input_size

        in_channels = input_size

        if label_size > 0:
            # self.embed_label = layers.EqualizedLinear(label_size, latent_size, lr_mul=lr_mul, bias=False)
            self.embed_label = torch.nn.Linear(label_size, input_size, bias=False)
            torch.nn.init.normal_(self.embed_label.weight)
            in_channels *= 2

        linear_layers = []
        
        for i in range(0, num_layers):
            if i == num_layers - 1:
                linear_layer = layers.EqualizedLinear(in_channels, latent_size, lr_mul=lr_mul)
            else:
                linear_layer = layers.EqualizedLinear(in_channels, state_size, lr_mul=lr_mul)
            linear_layers.append(linear_layer)
            linear_layers.append(layers.ScaledLeakyReLU(negative_slope=0.2, inplace=True))
    
            in_channels = state_size

        self.linear_layers = nn.Sequential(*linear_layers)
    
    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

    def forward(self, z: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:

        if self.label_size > 0:
            if labels is None:
                raise ValueError('Where is label, Lebowski? ') # TODO: Make [0.5, 0.5]
            label_embedding = self.embed_label(labels)
            z = torch.cat([z, label_embedding], dim=-1)
            
        if self.normalize_z:
            z = self.normalize(z)

        return self.linear_layers.forward(z)