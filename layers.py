import math
from typing import Any, Optional, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
                nonlinearity = 'leaky_relu') -> None:  # TODO: remove nonlinearity arg?
        """ Base class for layer with equalized LR technique 
            Description in section 4.1 of  https://arxiv.org/abs/1710.10196
            If you want to use eqlr just inherit from this class and use 
            self.get_weight() method to get weights in forward

        Args:
            weight_shape (tuple): shape of a weight tensor
            equalize_lr (bool, optional): use equalize lr. Defaults to True.
            lr_mul (float, optional): Learning rate multiplier. Defaults to 1.
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

class ModConvBase(EqualizedLrLayer):
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.demodulate = demodulate
        self.filter_size = filter_size
        self.stride = stride
        self.dilation = dilation

    def _get_same_padding(self, input_size):
        return ((input_size - 1) * (self.stride - 1) + self.dilation * (self.filter_size - 1)) // 2

    def modulate(self, kernel: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        batch_size = style.shape[0]

        # expand dimentions of style vectors and conv kernel to ensure broadcasting
        expanded_style = style.view(batch_size, 1, self.in_channels, 1, 1)
        expanded_kernel = kernel.view(1, self.out_channels, self.in_channels, self.filter_size, self.filter_size)
        
        # modulation
        expanded_kernel = expanded_kernel * expanded_style 

        if self.demodulate:
            demodulation_coefficient = torch.rsqrt((expanded_kernel ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps) #TODO: rewrite in einops?
            expanded_kernel = expanded_kernel * demodulation_coefficient

        return expanded_kernel
    
class ModConv2d(ModConvBase):
    def __init__(self, in_channels: int, 
                       out_channels: int, 
                       filter_size: int, 
                       demodulate: bool = True, 
                       stride: int=1, 
                       dilation: int=1,
                       equalize_lr: bool = True,
                       lr_mul: float = 1):
        super().__init__(
            in_channels, 
            out_channels, 
            filter_size, 
            demodulate, 
            stride, 
            dilation,
            equalize_lr,
            lr_mul)


    def forward(self, x, style: torch.Tensor):
        batch_size, c , h, w = x.shape

        assert c == self.in_channels
        
        kernel = self.get_weight()

        x = x.type_as(kernel)
        style = style.type_as(kernel)

        expanded_kernel = self.modulate(kernel, style)

        # reshape to represent batch dimention as channel groups
        # unique kernel for each object in batch 
        x = x.view(1, -1, h, w) 
        _, _, *ws = expanded_kernel.shape
        kernel = expanded_kernel.view(batch_size * self.out_channels, *ws)

        padding = self._get_same_padding(h)
        x = F.conv2d(x, kernel, padding=padding, groups=batch_size)

        return x.view(batch_size, self.out_channels, h, w)

class ModTransposedConv2d(ModConvBase):
    def __init__(self, in_channels: int, 
                       out_channels: int, 
                       filter_size: int, 
                       demodulate: bool = True, 
                       stride: int=1, 
                       dilation: int=1,
                       equalize_lr: bool = True,
                       lr_mul: float = 1):
        super().__init__(
            in_channels, 
            out_channels, 
            filter_size, 
            demodulate, 
            stride, 
            dilation,
            equalize_lr,
            lr_mul)


    def forward(self, x, style: torch.Tensor):
        batch_size, c , h, w = x.shape

        assert c == self.in_channels
        
        kernel = self.get_weight()

        x = x.type_as(kernel)
        style = style.type_as(kernel)

        expanded_kernel = self.modulate(kernel, style)

        expanded_kernel = expanded_kernel.transpose(1, 2) 

        x = x.view(1, -1, h, w) 
        _, _, *ws = expanded_kernel.shape
        kernel = expanded_kernel.reshape(batch_size * self.in_channels, *ws) #SUSPICIOUS 

        x = F.conv_transpose2d(x, kernel, padding=0, stride=2, groups=batch_size) # TODO: remove hardcode, calculate padding properly
        _,_, h,w = x.shape
        return x.view(batch_size, self.out_channels, h, w)

class BilinearFilter(pl.LightningModule):
    def __init__(self, channels: int, kernel: Union[List[float], np.ndarray] = [1.,3.,3.,1.], upsample_factor: Optional[int] = None):
        super().__init__()

        self.channels = channels
        self.upsample_factor = upsample_factor

        kernel = self._make_kernel(kernel)

        if self.upsample_factor is not None:
            kernel *= (self.upsample_factor ** 2)

        self.register_buffer('kernel', kernel[None, None, :, :].repeat((1, self.channels, 1, 1))) # TODO: maybe it's a not good idea

    @staticmethod
    def _make_kernel(kernel):
        kernel = torch.tensor(kernel)

        if kernel.ndim == 1:
            kernel = kernel[None, :] * kernel[:, None]

        kernel /= kernel.sum()

        return kernel

    def _calculate_padding(self) -> Tuple[int]: # TODO: CHECK! It could easily be wrong! 
        factor = self.upsample_factor if self.upsample_factor is not None else 1
        padding = self.kernel.shape[2] - factor

        pad0 = (padding + 1) // 2 + factor - 1
        pad1 = padding // 2

        return pad0, pad1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel
        _, in_c, in_h, in_w = x.shape

        if self.upsample_factor is not None:
            x = x.view(-1, in_c, in_h, 1, in_w, 1)
            x = F.pad(x, [0, self.upsample_factor - 1, 0, 0, 0, self.upsample_factor - 1])
            x = x.view(-1, in_c, in_h * self.upsample_factor, in_w * self.upsample_factor)

        pad0, pad1 = self._calculate_padding()
        x = F.pad(x, [pad0, pad1, pad0, pad1])
        x = F.conv2d(x, kernel)

        return x

class StyleConvBase(pl.LightningModule):
    def __init__(self, 
                 latent_size: int,
                 in_channels: int,
                 out_channels: int,
                 equalize_lr: bool = True,
                 lr_mul: float = 1,
                 ) -> None:
        super().__init__()

        self.latent2style = EqualizedLinear(latent_size, 
                                            in_channels, 
                                            bias_init=1.0, 
                                            equalize_lr=equalize_lr, 
                                            lr_mul=lr_mul)
        self.inject_noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def bias_and_activation(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias
        scale = 2 * 0.5 # TODO: check it

        return F.leaky_relu(x + bias, negative_slope=0.2) * scale


class StyleConv(StyleConvBase):
    def __init__(self, 
                 latent_size: int,
                 in_channels: int,
                 out_channels: int,
                 filter_size: int, 
                 equalize_lr: bool = True,
                 lr_mul: float = 1,
                 ) -> None:
        super().__init__(latent_size, 
                        in_channels, 
                        out_channels, 
                        equalize_lr, 
                        lr_mul)

        self.conv = ModConv2d(in_channels, out_channels, filter_size, stride=1)
        self.inject_noise = NoiseInjection()


    def forward(self, x: torch.Tensor, latent: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        
        style = self.latent2style.forward(latent)
        x = self.conv.forward(x, style)
        x = self.inject_noise.forward(x, noise)
        return self.bias_and_activation(x)

class StyleConvUp(StyleConvBase):
    def __init__(self, 
                 latent_size: int,
                 in_channels: int,
                 out_channels: int,
                 filter_size: int, 
                 equalize_lr: bool = True,
                 lr_mul: float = 1,
                 ) -> None:
        super().__init__(latent_size, 
                        in_channels, 
                        out_channels, 
                        equalize_lr, 
                        lr_mul)

        self.conv = ModTransposedConv2d(in_channels, out_channels, filter_size, stride=2)
        self.filter_bilinear = BilinearFilter(out_channels)
        self.inject_noise = NoiseInjection()


    def forward(self, x: torch.Tensor, latent: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:       
        style = self.latent2style.forward(latent)
        x = self.conv.forward(x, style)
        x = self.filter_bilinear.forward(x)
        x = self.inject_noise.forward(x, noise)
        return self.bias_and_activation(x)


class ToRgb(StyleConvBase):
    def __init__(self, 
                 latent_size: int,
                 in_channels: int,
                 equalize_lr: bool = True,
                 lr_mul: float = 1,
                 ) -> None:
        super().__init__(latent_size, 
                        in_channels, 
                        3, 
                        equalize_lr, 
                        lr_mul)

        self.conv = ModConv2d(in_channels, out_channels = 3, demodulate = False, filter_size = 1, stride=1)


    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:        
        style = self.latent2style.forward(latent)
        x = self.conv.forward(x, style)
        x += self.bias
        return 

class EqualizedLinear(EqualizedLrLayer):
    def __init__(self, 
                in_features: int, 
                out_features: int, 
                bias: bool = True, 
                bias_init: float = 0.0, 
                equalize_lr: bool = True,
                lr_mul: float = 1, ) -> None:
        super().__init__(
            weight_shape=(out_features, in_features),
            equalize_lr = equalize_lr,
            lr_mul=lr_mul, 
            nonlinearity='leaky_relu')

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features).fill_(bias_init))

        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.get_weight()
        bias = self.bias * self.lr_mul if self.bias is not None else None
        return  F.linear(x, w, bias=bias)

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return x + self.weight * noise