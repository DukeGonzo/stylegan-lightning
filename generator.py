import math
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import layers

class ModulatedBlock(pl.LightningModule):
    def __init__(self, 
                in_channels: int, 
                out_channels: int,
                latent_size: int):
        super().__init__()

        assert in_channels >= 1
        assert out_channels >= 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.style_conv_up = layers.StyleConvUp(latent_size, in_channels, out_channels, filter_size = 3)
        self.style_conv = layers.StyleConv(latent_size, out_channels, out_channels, filter_size = 3)
        self.to_rgb = layers.ToRgb(latent_size, out_channels)
        self.upscale_shortcut = nn.Sequential(layers.UpsampleZeros(), layers.BilinearFilter(3, scaling_factor=2))



    def forward(self, x: torch.Tensor, shortcut: torch.Tensor, latents: torch.Tensor, noise: torch.Tensor):
        # assert len(noise) == 2
        # assert noise.size() == 2
        
        x = self.style_conv_up.forward(x, latent = latents[:, 0], noise = noise[0])
        x = self.style_conv.forward(x, latent = latents[:, 1], noise = noise[1])
        
        shortcut = self.upscale_shortcut.forward(shortcut)
        rgb = self.to_rgb(x, latents[:, 2])
        rgb += shortcut

        return x, rgb

class SynthesisNetwork(pl.LightningModule):
    def __init__(self, 
                resolution: int = 512,
                latent_size: int = 512,
                channel_multiplier: int = 2): #TODO: TO Config?):
                
        super().__init__()
        
        log_res = np.log2(resolution)
        assert log_res > 1 and (log_res * 10) % 10 == 0, f'resolution must be a power of 2.'
        log_res = int(log_res)
        self.resolution = resolution

        assert latent_size > 0, f'Latent size must be 1 or greater'
        self.latent_size = latent_size

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

        self.blocks = nn.ModuleList()
        in_channels = self.channels[4] # TODO: remove hardcode

        self.constant = layers.ConstantLayer(in_channels, 4) # TODO: remove hardcode

        self.style_conv = layers.StyleConv(latent_size, in_channels, in_channels, filter_size = 3)
        self.to_rgb = layers.ToRgb(latent_size, in_channels)

        for i in range(3, log_res + 1):
            out_channels = self.channels[2 ** i]
            self.blocks.append(ModulatedBlock(in_channels, out_channels, latent_size))
            in_channels = out_channels  

    def make_noise(self,  batch_size: int) -> List[torch.Tensor]:
        """ Make spatial noise for each layer

        Returns:
            List of spatial noises for each activation map
        """
        assert batch_size > 0

        noises = []

        for i, map_size in enumerate(self.channels.keys()):
            
            if map_size > self.resolution:
                break

            for _ in range(2):
                noise = torch.randn(batch_size, 1, map_size, map_size, device=self.device)
                # noise.type_as(latent_vectors, self.device) # TODO: not sure about this stuff 
                noises.append(noise)

                if i == 0: # the first block has only one injection
                    break

        return noises

    def forward(self, latent_vectors: torch.Tensor, noise: Optional[List[torch.Tensor]] = None):
        batch_size = latent_vectors.shape[0]

        if noise is None:
            noise = self.make_noise(batch_size)

        x = self.constant.forward(batch_size)
        x = self.style_conv.forward(x, latent_vectors[:, 0], noise=noise[0])
        rgb = self.to_rgb.forward(x, latent_vectors[:, 1])
        i = 1
        for block in self.blocks:
            x, rgb = block(x, rgb, latent_vectors[:, i:i+3], noise[i:i+2])
            i += 2

        return rgb

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
            