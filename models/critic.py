import math
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint_sequential
import pytorch_lightning as pl
from . import layers


class CriticNetwork(pl.LightningModule):
    def __init__(self,
                 resolution: int = 512,
                 label_size: int = 0,
                 channel_multiplier: int = 2):
        """[summary]

        Args:
            resolution (int, optional): [description]. Defaults to 512.
            label_size (int, optional): [description]. Defaults to 0.
            channel_multiplier (int, optional): [description]. Defaults to 2.
            multiresolution (bool, optional): https://arxiv.org/abs/2103.03243. Defaults to False.
        """
        super().__init__()
        self._leaky_relu_slope = 0.2

        log_res = np.log2(resolution)
        assert log_res > 1 and (log_res * 10) % 10 == 0, f'resolution must be a power of 2.'
        log_res = int(log_res)
        self.resolution = resolution

        assert label_size >= 0
        self.label_size = label_size

        ############ TO CONFIG ############
        self.channels = {  # TODO: channel_multiplier sounds meaningless and the whole channel stuff must be moved to config explicitly
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

        self.activation = layers.ScaledLeakyReLU(self._leaky_relu_slope, inplace=True)

        self.conv = layers.EqualizedConv(in_channels=3,
                                         out_channels=in_channels,
                                         kernel_size=1,
                                         padding=0,
                                         bias=True)

        block_sequence = []

        for i in range(log_res, 2, -1):
            out_channels = self.channels[2 ** (i - 1)]
            block = CriticBlock(in_channels, out_channels)
            block_sequence.append(block)
            in_channels = out_channels

        self.blocks = nn.Sequential(*block_sequence)

        self.add_std_channel = layers.AddStdChannel()

        self.conv_final = layers.EqualizedConv(in_channels=in_channels+1,
                                               out_channels=self.channels[4],
                                               kernel_size=3,
                                               padding=1,
                                               bias=True)

        self.head = nn.Sequential(
            layers.EqualizedLinear(self.channels[4] * 4 * 4, self.channels[4]),
            self.activation,
            layers.EqualizedLinear(self.channels[4], max(1, self.label_size))
        )

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor], return_activations: bool = False) -> torch.Tensor:
        x = self.conv.forward(x)
        x = self.activation(x)

        x = self.blocks(x)

        x = self.add_std_channel.forward(x)
        x = self.conv_final.forward(x)
        x = self.activation(x)

        x = torch.flatten(x, start_dim=1)
        x = self.head(x)

        if label != None:
            x = torch.mean(x * label, dim=-1, keepdim=True)

        if return_activations:
            raise NotImplementedError()
            return x, None

        return x


class MultiResCriticNetwork(CriticNetwork):
    def __init__(self,
                 resolution: int = 512,
                 label_size: int = 0,
                 channel_multiplier: int = 2):
        super().__init__(resolution=resolution,
                         label_size=label_size,
                         channel_multiplier=channel_multiplier)

        self.multi_inputs = nn.ModuleList([self.conv])

        current_resolution = self.resolution

        while current_resolution > 4:
            current_resolution //= 2
            out_channels = self.channels[current_resolution]
            conv = layers.EqualizedConv(in_channels=3,
                                        out_channels=out_channels,
                                        kernel_size=1,
                                        padding=0,
                                        bias=True)
            self.multi_inputs.append(conv)

    def forward(self, x: torch.Tensor,
                label: Optional[torch.Tensor],
                target_resolution: Optional[int] = None,
                return_activations: bool = False) -> torch.Tensor:

        if target_resolution is None:
            target_resolution = self.resolution

        cutoff_index = int(np.log2(self.resolution // target_resolution))

        x = self.multi_inputs[cutoff_index].forward(x)
        x = self.activation(x)

        x = self.blocks[cutoff_index:](x)

        x = self.add_std_channel.forward(x)
        x = self.conv_final.forward(x)
        x = self.activation(x)

        x = torch.flatten(x, start_dim=1)
        x = self.head(x)

        if label != None:
            x = torch.mean(x * label, dim=-1, keepdim=True)

        if return_activations:
            raise NotImplementedError()
            return x, None

        return x


class CriticBlock(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()

        self._leaky_relu_slope = 0.2

        self.conv = layers.EqualizedConv(in_channels, in_channels, kernel_size=3, bias=True, padding=1)
        self.activation = layers.ScaledLeakyReLU(self._leaky_relu_slope, inplace=True)

        self.bilinear = layers.BilinearFilter(in_channels, padding=2)
        self.conv_down = layers.EqualizedConv(in_channels, out_channels, 3, stride=2, bias=True, padding=0)

        self.shortcut_filter = layers.BilinearFilter(in_channels, padding=1)
        self.shortcut_down = layers.EqualizedConv(in_channels, out_channels, 1, stride=2, bias=False, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shortcut = self.shortcut_filter.forward(x)
        shortcut = x
        shortcut = self.shortcut_down.forward(shortcut)

        x = self.conv.forward(x)
        x = self.activation(x)

        x = self.bilinear.forward(x)
        x = self.conv_down.forward(x)
        x = self.activation(x)

        # return fused_shortcut(x, shortcut)
        return (x + shortcut) / math.sqrt(2)
