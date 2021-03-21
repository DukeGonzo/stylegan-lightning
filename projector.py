from itertools import chain
import math
from typing import Optional
import random


import torch
from torch import optim, nn
from torch import autograd
from torch.autograd.grad_mode import no_grad
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.nn.modules import distance

from stylegan import GanTask
import numpy as np
import lpips
from PIL import Image as pil
from torchvision import transforms
from tqdm import tqdm
import torchvision.models as models

import conv2d_gradfix

conv2d_gradfix.enabled = True

class Projector():
    def __init__(self, gan: GanTask):
        self.gan = gan
        self._lpips_loss = lpips.LPIPS(net='vgg').to(gan.device)
        self._vgg = models.vgg16(pretrained=True).to(self.gan.device)
        self._num_lables = gan.label_size

        latents_statistics = []
        num_samples = 512
        
        with torch.no_grad():
            if self._num_lables > 0:
                for label in range(self._num_lables):
                    z = torch.randn([num_samples, 512]).float().to(gan.device)
                    labels = F.one_hot(torch.ones((num_samples,)).long() * label, self._num_lables).float().to(gan.device)
                    latents = gan.mapping_net.forward(z, labels)
                    latent_mean = torch.mean(latents, 0).detach()
                    latent_std = torch.std(latents, 0).detach()
                    latents_statistics.append((latent_mean, latent_std))
            else:
                z = torch.randn([num_samples, 512]).float().to(gan.device)
                latents = gan.mapping_net.forward(z)
                latent_mean = torch.mean(latents, 0).detach()
                latent_std = torch.std(latents, 0).detach()
                latents_statistics.append((latent_mean, latent_std))
        
        self._latents_statistics = latents_statistics

        # copy-pasted from data_module TODO: refactor
        self._transforms = transforms.Compose(
            [
                transforms.Resize(gan.resolution), # TODO skip if dump_on_disk 
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
            ]
        )

        self._vgg_mean = torch.tensor([0.485, 0.456, 0.406], device=gan.device).view(1,3,1,1)
        self._vgg_std = torch.tensor([0.229, 0.224, 0.225], device=gan.device).view(1,3,1,1)

    @staticmethod
    def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)

        return initial_lr * lr_ramp

    def _downsample_to_vgg(self, image: torch.Tensor):
        x = F.interpolate(image, (256, 256), mode='bilinear', align_corners=True)
        return x

    def _normalize_for_vgg(self, image: torch.Tensor):
        return ((image + 1) / 2 - self._vgg_mean) / self._vgg_std

    @staticmethod
    def whitening(tensor: torch.Tensor) -> torch.Tensor:
        mu = torch.mean(tensor, dim=[2,3], keepdim=True)
        sigma = torch.std(tensor, dim=[2,3], keepdim=True)

        return (tensor - mu) / sigma

    def project(self, 
                image: pil.Image, 
                num_steps: int, 
                label: Optional[int], 
                lpips_weight: float = 1.0,
                vgg_weight: float = 1.0,
                critic_weight: float = 1.0,
                vgg_layer: int = 15,
                lr_init: float = 0.1,
                noise_weight: float = 0.05,
                noise_ramp: float = 0.7) -> torch.Tensor:
        assert label is None or label >= 0

        source_image = torch.unsqueeze(self._transforms(image), 0).to(self.gan.device)
        source_image_vgg = self._downsample_to_vgg(source_image)

        # get precalculated statistics
        latent_mean, latent_std = self._latents_statistics[label]

        # go to expanded latent space
        latent_in = latent_mean[None,None, :].repeat(1, self.gan.synthesis_net_ema.num_layers, 1).detach()

        # apply gradients
        latent_in.requires_grad = True
        self.gan.synthesis_net_ema.requires_grad_(False)
        self.gan.synthesis_net_ema = self.gan.synthesis_net_ema.eval()

        optimizer = optim.Adam([latent_in], lr=lr_init, betas=(0.9, 0.999))

        progress_bar = tqdm(range(num_steps))
        
            
        for i in progress_bar:
            message = f' iter {i}, '
            t = i / num_steps
            lr = self.get_lr(t, lr_init)            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # generate latent noise
            noise_strength = latent_std * noise_weight * max(0, 1 - t / noise_ramp) ** 2
#             noise_strength = 0.
            noisy_latent = latent_in + torch.randn_like(latent_in) * noise_strength
            spatial_noise = self.gan.synthesis_net_ema.make_noise(1)

            fake_images = self.gan.synthesis_net_ema.forward(noisy_latent, spatial_noise)
            fake_images_vgg = self._downsample_to_vgg(fake_images)

            loss = 0.0

            if lpips_weight > 0:
                lpips_loss_value = torch.sum(self._lpips_loss(fake_images_vgg, source_image_vgg))
                loss += lpips_weight * lpips_loss_value
                message += f', lpips_loss: {lpips_loss_value.item()}'

            if vgg_weight > 0:
                vgg_subnet = self._vgg.features[: vgg_layer]
                fake_images_vgg = self._normalize_for_vgg(fake_images_vgg)
                norm_source_image_vgg = self._normalize_for_vgg(source_image_vgg)
                fake_features = self.whitening(vgg_subnet.forward(fake_images_vgg))
                source_features = self.whitening(vgg_subnet.forward(norm_source_image_vgg))
                vgg_loss_value = torch.sum(F.mse_loss(fake_features, source_features))
                loss += vgg_weight * vgg_loss_value
                message += f', vgg_loss: {vgg_loss_value.item()}'

            if critic_weight > 0:
                critic_score = self.gan.critic_net.forward(fake_images, F.one_hot(torch.ones((1,)).long() * label, self._num_lables).to(self.gan.device))
                loss -= critic_weight * torch.sum(critic_score)
                message += f', critic_score: {critic_score.item()}'

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            progress_bar.set_description(message, refresh=True)
            # progress_bar.refresh() # to show immediately the update
                # print(f'iter {i}, loss {loss.view(-1).data.cpu().numpy()[0]}, lr {lr}')
        
        return latent_in.detach()