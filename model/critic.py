
import math
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from . import layers

class CriticNetwork(pl.LightningModule):
    def __init__(self, 
                resolution: int = 512,
                label_size: int = 0,
                channel_multiplier: int = 2): # TODO: TO CONFIG! 
        super().__init__()
        self._leaky_relu_slope = 0.2
        
        log_res = np.log2(resolution)
        assert log_res > 1 and (log_res * 10) % 10 == 0, f'resolution must be a power of 2.'
        log_res = int(log_res)
        self.resolution = resolution

        assert label_size >= 0
        self.label_size = label_size

        ############ TO CONFIG ############
        self.channels = {  #TODO: channel_multiplier sounds meaningless and the whole channel stuff must be moved to config explicitly 
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        in_channels = self.channels[self.resolution]

        self.conv = nn.Sequential(
                    layers.EqualizedConv(in_channels=3, 
                                      out_channels=in_channels,
                                      kernel_size=1,
                                      padding=1,
                                      bias = True),
                    nn.LeakyReLU(negative_slope=self._leaky_relu_slope)
                    )
        
        block_sequence = [] 

        for i in range(log_res, 2, -1):
            out_channels = self.channels[2 ** (i - 1)]
            block = CriticBlock(in_channels, out_channels)
            block_sequence.append(block)
            in_channels = out_channels
        
        self.blocks = nn.Sequential(block_sequence)

        self.conv_final = nn.Sequential(
                 layers.EqualizedConv(in_channels=in_channels+1, 
                                      out_channels=self.channels[4],
                                      kernel_size=3,
                                      padding=1,
                                      bias = True),
                 nn.LeakyReLU(negative_slope=self._leaky_relu_slope)
                 )

        self.stddev_group = 4
        self.stddev_feat = 1

        self.head = nn.Sequential(
            layers.EqualizedLinear(self.channels[4] * 4 * 4, self.channels[4]),
            nn.LeakyReLU(negative_slope=self._leaky_relu_slope),
            layers.EqualizedLinear(self.channels[4], max(1, self.label_size))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv.forward(x)
        x = self.blocks.forward(x)

        # TODO: extract to separate layer ########################
        out = x
        batch_size, channel, height, width = out.shape
        group = min(batch_size, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        #############################################

        x = self.conv_final.forward(out)
        x = x.view(batch_size, -1)
        x = self.head.forward(x)

        return x

class CriticBlock(pl.LightningModule):
    def __init__(self, 
                in_channels: int, 
                out_channels: int):
        super().__init__()

        self._leaky_relu_slope = 0.2

        self.conv = layers.EqualizedConv(in_channels, in_channels, kernel_size= 3, bias=True, padding=1)

        self.bilinear = layers.BilinearFilter(in_channels, padding=2)
        self.conv_down = layers.EqualizedConv(in_channels, out_channels, 3, stride=2, bias=True, padding=0)

        self.shortcut_filter = layers.BilinearFilter(in_channels, padding=1)                                    
        self.shortcut_down = layers.EqualizedConv(in_channels, out_channels, 1, stride=2, bias=False, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut_filter.forward(x)
        shortcut = self.shortcut_down.forward(shortcut)

        x = self.conv.forward(x)
        x = F.leaky_relu(x, self._leaky_relu_slope)

        x = self.bilinear.forward(x)
        x = self.conv_down.forward(x)
        x = F.leaky_relu(x, self._leaky_relu_slope)

        return (x + shortcut) / math.sqrt(2)