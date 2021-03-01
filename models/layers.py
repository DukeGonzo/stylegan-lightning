import math
from math import hypot
from typing import Any, Optional, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import conv2d_gradfix

class ConstantLayer(pl.LightningModule):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size, device=self.device))

    def forward(self, batch_size: int) -> torch.Tensor:
        assert batch_size > 0
        out = self.input.repeat(batch_size, 1, 1, 1)

        return out

class EqualizedLrLayer(pl.LightningModule):
    def __init__(self, 
                weight_shape: tuple, 
                equalize_lr: bool = True, 
                lr_mul: float = 1,
                nonlinearity = 'leaky_relu', 
                batch_dim = False) -> None:  # TODO: remove nonlinearity arg?
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
        self.weight = nn.Parameter(torch.randn(weight_shape, device=self.device).div_(lr_mul))

        if not equalize_lr:
            # TODO REMOVE batch_dim! 
            nn.init.kaiming_normal_(self.weight[0] if batch_dim else self.weight, a=0.2, mode='fan_in', nonlinearity=nonlinearity)
            self.scale = 1
        else:
            fan = nn.init._calculate_correct_fan(self.weight[0] if batch_dim else self.weight, 'fan_in') 
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
            weight_shape=(1, out_channels, in_channels, filter_size, filter_size),
            equalize_lr = equalize_lr,
            lr_mul=lr_mul, nonlinearity='leaky_relu', batch_dim=True)

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
        # expanded_kernel = kernel.view(1, self.out_channels, self.in_channels, self.filter_size, self.filter_size)
        
        # modulation
        kernel = kernel * expanded_style 

        if self.demodulate:
            demodulation_coefficient = torch.linalg.norm(kernel, dim=(2, 3, 4), keepdim=True) + self.eps
            kernel = kernel / demodulation_coefficient

        return kernel
    
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


    def forward(self, x, style: torch.Tensor, fused_bias: Optional[torch.Tensor] = None):
        batch_size, c , h, w = x.shape

        assert c == self.in_channels
        
        kernel = self.get_weight()

        # x = x.type_as(kernel)
        # style = style.type_as(kernel)

        kernel = self.modulate(kernel, style)

        # reshape to represent batch dimention as channel groups
        # unique kernel for each object in batch 
        x = x.view(1, -1, h, w) 
        _, _, *ws = kernel.shape
        kernel = kernel.view(batch_size * self.out_channels, *ws)

        if fused_bias is not None:
            # fused_bias = torch.repeat_interleave(fused_bias, batch_size)
            fused_bias = fused_bias.expand(batch_size, -1).flatten()

        padding = self._get_same_padding(h)
        x = conv2d_gradfix.conv2d(x, kernel, padding=padding, groups=batch_size, bias=fused_bias)

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


    def forward(self, x, style: torch.Tensor, fused_bias: Optional[torch.Tensor] = None):
        batch_size, c , h, w = x.shape

        assert c == self.in_channels
        
        kernel = self.get_weight()

        # x = x.type_as(kernel)
        # style = style.type_as(kernel)

        expanded_kernel = self.modulate(kernel, style)

        expanded_kernel = expanded_kernel.transpose(1, 2) 

        x = x.view(1, -1, h, w) 
        _, _, *ws = expanded_kernel.shape
        kernel = expanded_kernel.reshape(batch_size * self.in_channels, *ws) #SUSPICIOUS 

        if fused_bias is not None:
            # fused_bias = torch.repeat_interleave(fused_bias, batch_size)
            fused_bias = fused_bias.expand(batch_size, -1).flatten()

        x = conv2d_gradfix.conv_transpose2d(x, kernel, padding=0, stride=2, groups=batch_size, bias=fused_bias) # TODO: remove hardcode, calculate padding properly
        _,_, h,w = x.shape
        return x.view(batch_size, self.out_channels, h, w)

class UpsampleZeros(pl.LightningModule):
    def __init__(self, upsample_factor: int = 2):
        super().__init__()

        self.upsample_factor = upsample_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, in_c, in_h, in_w = x.shape

        x = x.view(-1, in_c, in_h, 1, in_w, 1)
        x = F.pad(x, [0, self.upsample_factor - 1, 0, 0, 0, self.upsample_factor - 1])
        x = x.view(-1, in_c, in_h * self.upsample_factor, in_w * self.upsample_factor)

        return x

class BilinearFilter(pl.LightningModule):
    def __init__(self, channels: int, 
                       kernel: Union[List[float], np.ndarray] = [1.,3.,3.,1.], 
                       scaling_factor: Optional[int] = None,
                       padding: Union[str, int] = 'SAME'):
        super().__init__()

        self.channels = channels
        self.scaling_factor = scaling_factor if scaling_factor is not None else 1

        kernel = self._make_kernel(kernel)

        kernel *= (self.scaling_factor ** 2)
        kernel = kernel[None, None, :, :].expand((self.channels, -1, -1, -1))

        self.register_buffer('kernel', kernel, persistent = False) # TODO: maybe it's a not good idea

        if padding == 'SAME':
            self.pad0, self.pad1 = self._calculate_padding()
        else:
            self.pad0, self.pad1 = padding, padding

    @staticmethod
    def _make_kernel(kernel):
        kernel = torch.tensor(kernel)

        if kernel.ndim == 1:
            kernel = kernel[None, :] * kernel[:, None]

        kernel /= kernel.sum()

        return kernel

    def _calculate_padding(self) -> Tuple[int]: # TODO: CHECK! It could easily be wrong! 
        factor = self.scaling_factor if self.scaling_factor is not None else 1
        padding = self.kernel.shape[2] - factor

        pad0 = (padding + 1) // 2 + factor - 1
        pad1 = padding // 2

        return pad0, pad1

    def forward(self, x: torch.Tensor, fused_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        kernel = self.kernel

        pad0, pad1 = self.pad0, self.pad1
       
        # x = F.pad(x, [pad0, pad1, pad0, pad1])
        x = conv2d_gradfix.conv2d(x, kernel, groups=self.channels, bias=fused_bias, padding=[pad0, pad1])

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
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.activation = ScaledLeakyReLU(0.2, inplace=True)

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
        x = self.conv.forward(x, style, fused_bias=self.bias)
        x = self.inject_noise.forward(x, noise)
        # x += self.bias[np.newaxis, :, np.newaxis, np.newaxis]
        return self.activation(x)

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
        self.filter_bilinear = BilinearFilter(out_channels, scaling_factor= 2, padding=1)
        
        self.inject_noise = NoiseInjection()


    def forward(self, x: torch.Tensor, latent: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:       
        style = self.latent2style.forward(latent)
        x = self.conv.forward(x, style)
        x = self.filter_bilinear.forward(x, fused_bias=self.bias)
        x = self.inject_noise.forward(x, noise)
        return self.activation(x)


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
        x = self.conv.forward(x, style, fused_bias=self.bias)
        # x += self.bias[np.newaxis, :, np.newaxis, np.newaxis]
        return x

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

class EqualizedConv(EqualizedLrLayer):
    def __init__(self, in_channels: int, 
                       out_channels: int, 
                       kernel_size: int, 
                       stride: int=1,
                       padding: Union[str, int] = 0, 
                       dilation: int=1,
                       bias: bool = True,
                       equalize_lr: bool = True,
                       lr_mul: float = 1):
        super().__init__(
            weight_shape=(out_channels, in_channels, kernel_size, kernel_size),
            equalize_lr = equalize_lr,
            lr_mul=lr_mul, nonlinearity='leaky_relu', batch_dim=False)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.bias = None
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))


    def _get_same_padding(self, input_size):
        return ((input_size - 1) * (self.stride - 1) + self.dilation * (self.kernel_size - 1)) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.get_weight()
        _, _, h, w = x.shape

        if self.padding == 'SAME':
            padding = self._get_same_padding(h)
        else:
            padding = self.padding 

        x = conv2d_gradfix.conv2d(x, 
                     weight = kernel, 
                     bias = self.bias,
                     stride=self.stride,
                     padding=padding,
                     dilation = self.dilation
                    )
        return x

class AddStdChannel(pl.LightningModule):
    def __init__(self, stddev_group: int = 4, stddev_feat: int = 1) -> None:
        super().__init__()

        self.stddev_group = stddev_group
        self.stddev_feat = stddev_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channel, height, width = x.shape
        group = min(batch_size, self.stddev_group)
        stddev = x.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        return torch.cat([x, stddev], 1)

# @torch.jit.script
# def fused_leaky_relu(x):
#     return x * 0.5 * (1.0 + torch.erf(x / 1.41421))      

class ScaledLeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope: float, inplace: bool = False, scale: Optional[float] = math.sqrt(2)) -> None:
        super().__init__(negative_slope=negative_slope, inplace = inplace)

        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mul_(self.scale)
        return super().forward(x) 

