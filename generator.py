import math
from typing import Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
# from layers import *

import pytorch_lightning as pl

class ConstantLayer(pl.LightningModule):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, batch_size: int) -> torch.Tensor:
        assert batch_size > 0
        out = self.input.repeat(batch_size, 1, 1, 1)

        return out

class EqualizedLrLayer(pl.LightningModule):
    def __init__(self, 
                weight_shape: tuple, 
                equalize_lr: bool = True, 
                lr_mul: float = 1,
                nonlinearity = 'leaky_relu') -> None:
        """ Base class for layer with equalized LR technique 
            Description in section 4.1 of  https://arxiv.org/abs/1710.10196
            If you want to use eqlr just inherite from this class and use 
            self.get_weight() method to get weights in forward

        Args:
            weight_shape (tuple): shape of a weight tensor
            equalize_lr (bool, optional): use equalize lr. Defaults to True.
            lr_mul (float, optional): Lerning rate multiplier. Defaults to 1.
            nonlinearity (str, optional): Activation function, it affects He coefficient. Defaults to 'leaky_relu'.
        """
        super().__init__()

        self.scale = 1
        self.lr_mul = lr_mul
        self.eps = 1e-8

        # Equalized learning rate from https://arxiv.org/abs/1710.10196
        self.weight = nn.Parameter(torch.randn(weight_shape))

        if not equalize_lr:
            nn.init.kaiming_normal_(self.weight, a=0.2, mode='fan_in', nonlinearity=nonlinearity)
            self.scale = 1
        else:
            fan = nn.init._calculate_correct_fan(self.weight, 'fan_in') 
            # gain = nn.init.calculate_gain(nonlinearity='leaky_relu', a = 0.2) # TODO: 1 in original paper! 
            gain = 1.
            he_coefficient = gain / math.sqrt(fan)
            self.scale = he_coefficient * lr_mul

    def get_weight(self) -> torch.Tensor:
         # To ensure equalized learning rate from https://arxiv.org/abs/1710.10196
        return self.weight * self.scale 
    
class Conv2DMod(EqualizedLrLayer):
    def __init__(self, in_channels: int, 
                       out_channels: int, 
                       filter_size: int, 
                       demodulate: bool = True, 
                       stride: int=1, 
                       dilation: int=1,
                       equalize_lr: bool = True,
                       lr_mul: float = 1):
        super().__init__(
            weight_shape=(out_channels, in_channels, filter_size, filter_size),
            equalize_lr = equalize_lr,
            lr_mul=lr_mul, nonlinearity='leaky_relu')

        self.out_channels = out_channels
        self.demodulate = demodulate
        self.filter_size = filter_size
        self.stride = stride
        self.dilation = dilation

    def _get_same_padding(self, input_size):
        return ((input_size - 1) * (self.stride - 1) + self.dilation * (self.filter_size - 1)) // 2

    def forward(self, x, style: torch.Tensor):
        batch_size, in_channels, h, w = x.shape
        
        kernel = self.get_weight()

        x = x.type_as(kernel)
        style = style.type_as(kernel)

        # expand dimentions of style vectors and conv kernel to ensure broadcasting
        expanded_style = style.view(batch_size, 1, in_channels, 1, 1)
        expanded_kernel = self.weight.view(1, self.out_channels, in_channels, self.filter_size, self.filter_size)
        
        # modulation
        expanded_kernel *= expanded_style 

        if self.demodulate:
            demodulation_coefficient = torch.rsqrt((expanded_kernel ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps) #TODO: rewrite in einops?
            expanded_kernel = expanded_kernel * demodulation_coefficient

        # reshape to represent batch dimention as channel groups
        # unique kernel for each object in batch 
        x = x.view(1, -1, h, w) 
        _, _, *ws = expanded_kernel.shape
        kernel = expanded_kernel.reshape(batch_size * self.out_channels, *ws)

        padding = self._get_same_padding(h)
        x = F.conv2d(x, kernel, padding=padding, groups=batch_size)

        x = x.view(batch_size, self.out_channels, h, w)
        return x

class ModulatedBlock(pl.LightningModule):
    def __init__(self, 
                in_channels: int, 
                out_channels: int,
                latent_size: int,
                blur_kernel: list = [1, 3, 3, 1],
                **kwargs):
        super().__init__()

        assert in_channels >= 1
        assert out_channels >= 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1 = StyledConv(in_channels, out_channels, 3, latent_size, upsample=True, blur_kernel=blur_kernel)
        self.conv_2 = StyledConv(out_channels, out_channels, 3, latent_size, upsample=False, blur_kernel=blur_kernel)
        self.to_rgb = ToRGB(out_channels, latent_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, styles: torch.Tensor, noise: torch.Tensor, **kwargs): #TODO: styles should be a tensor?
        # assert len(noise) == 2
        # assert noise.size() == 2
        
        x = self.conv_1(x, styles[:, 0], noise=noise[:, 0])
        x = self.conv_2(x, styles[:, 1], noise=noise[:, 1])
        skip = self.to_rgb(x, styles[:, 2], skip)

        return x, skip


class SynthesisNetwork(pl.LightningModule):
    def __init__(self, 
                resolution: int = 512,
                latent_size: int = 512,
                channel_multiplier: int = 2, #TODO: TO Config?
                blur_kernel: list =[1, 3, 3, 1], #TODO: to config? Or hardcode this shait? 
                **kwargs):
                
        super().__init__()
        
        log_res = np.log2(resolution)
        assert log_res > 1 and log_res * (log_res -1) == 0, f'resolution must be a power of 2.'
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

        self.blur_kernel = blur_kernel

        self.blocks = nn.ModuleList()
        in_channels = self.channels[4] # TODO: remove hardcode

        self.constant = ConstantLayer(in_channels, 4) # TODO: remove hardcode

        self.conv1 = StyledConv(in_channels, in_channels, 3, latent_size, blur_kernel=self.blur_kernel)
        self.to_rgb1 = ToRGB(in_channels, latent_size, upsample=False)

        for i in range(3, log_res + 1):
            out_channels = self.channels[2 ** i]
            self.blocks.append(ModulatedBlock(in_channels, out_channels, latent_size, self.blur_kernel))

    def make_noise(self,  batch_size: int) -> List[torch.Tensor]:
        """ Make spatial noise for each layer

        Returns:
            List of spatial noises for each activation map
        """
        assert batch_size > 0

        noises = []

        for i, map_size in enumerate(self.channels.keys()):
            for _ in range(2):
                noise = torch.randn(batch_size, 1, map_size, map_size)
                # noise.type_as(latent_vectors, self.device) # TODO: not sure about this stuff 
                noises.append(noise)

                if i == 0: # the first block has only one injection
                    break

        return noises

    def forward(self, latent_vectors: torch.Tensor, noise: Optional[List[torch.Tensor]] = None):
        batch_size = latent_vectors.shape[0]

        if noise is None:
            noise = self.make_noise(batch_size)

        x = self.constant(latent_vectors)
        x = self.conv1(x, latent_vectors[:, 0], noise=noise[0])
        skip = self.to_rgb1(x, latent_vectors[:, 1])

        i = 1
        for block in self.blocks:
            x, skip = block(x, skip, latent_vectors[: i:i+3], noise[i:i+2])
            i += 2

        return skip
