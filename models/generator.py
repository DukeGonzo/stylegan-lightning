from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from . import layers

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
        # self.upscale_shortcut = nn.Sequential(layers.UpsampleZeros(), layers.BilinearFilter(3, scaling_factor=2))
        self.upscale_shortcut = nn.UpsamplingBilinear2d(scale_factor=2)



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
        self.num_layers = log_res * 2 - 2


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