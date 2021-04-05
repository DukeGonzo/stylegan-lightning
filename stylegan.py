from itertools import chain
import math
from typing import List, Optional, Tuple
import random
import numpy as np
from functools import partial


import torch
from torch import optim
from torch import autograd
from torch.nn import functional as F
import torch.nn as nn
import pytorch_lightning as pl

from models.generator import SynthesisNetwork
from models.critic import CriticNetwork, MultiResCriticNetwork
from models.mapping_network import MappingNetwork
import conv2d_gradfix

conv2d_gradfix.enabled = True


class GanTask(pl.LightningModule):
    def __init__(
        self,
        gamma: float = 10,
        ppl_reg_every: int = 4,
        penalize_d_every: int = 16,
        ppl_weight: float = 2,
        resolution: int = 512,
        use_ema: bool = True,
        ema_beta: float = 0.99,
        latent_size: int = 512,
        label_size: int = 2,
        use_top_k: bool = True,  # https://arxiv.org/abs/2002.06224
        top_k_decay_rate: float = 0.94,
        use_anycost_gan: bool = False,
    ):
        super().__init__()
        self.gamma = gamma
        self.ppl_reg_every = ppl_reg_every
        self.penalize_d_every = penalize_d_every
        self.ppl_weight = ppl_weight
        self.resolution = resolution
        self._log_resolution = int(np.log2(resolution))
        self.use_ema = use_ema
        self._mean_path_length = 0
        self.label_size = label_size
        self.use_top_k = use_top_k
        self.top_k_decay_rate = top_k_decay_rate
        self.register_buffer("k", torch.scalar_tensor(1.0), persistent=True)
        self.use_anycost_gan = use_anycost_gan

        self.mapping_net = MappingNetwork(
            input_size=512, state_size=512, latent_size=latent_size, label_size=label_size, lr_mul=0.01
        )
        self.synthesis_net = SynthesisNetwork(resolution=resolution, latent_size=latent_size, channel_multiplier=1)

        if use_anycost_gan:
            self.critic_net = MultiResCriticNetwork(resolution=resolution, label_size=label_size, channel_multiplier=1)
        else:
            self.critic_net = CriticNetwork(resolution=resolution, label_size=label_size, channel_multiplier=1)

        if use_ema:
            import copy

            self.synthesis_net_ema = copy.deepcopy(self.synthesis_net).eval()
            self.mapping_net_ema = copy.deepcopy(self.mapping_net).eval()
            self.ema_beta = ema_beta

    def generate_latents(self, batch_size: int, labels: Optional[torch.Tensor]) -> torch.Tensor:
        # sample noise
        z = torch.randn(batch_size, self.mapping_net.input_size, device=self.device)
        # TODO: not sure about this govnokod
        z = z.type_as(self.synthesis_net.style_conv.bias)

        w = self.mapping_net.forward(z, labels)

        return w[:, None, :].expand(-1, self.synthesis_net.num_layers, -1)

    def forward(
        self, 
        batch_size: int, 
        labels: Optional[torch.Tensor], 
        mix_prob: float = 0.9, 
        target_resolution: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch_size (int): number of samples
            labels (Optional[torch.Tensor]): one hot vector of labels
            mix_prob (float, optional): Probability of crossover regularization. Defaults to 0.9.
            target_resolution (Optional[int], optional): Target resolution, specify if you use anycost gan https://arxiv.org/abs/2103.03243 . Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        w_plus = self.generate_latents(batch_size, labels)

        if mix_prob > 0 and random.random() < mix_prob:
            # TODO: consider to put random labels and use mixup in discriminator
            w_plus_ = self.generate_latents(batch_size, labels)
            # TODO: consider more radical mixing
            w_plus = self.mix_latents(w_plus, w_plus_)

        return self.synthesis_net.forward(w_plus, None, target_resolution), w_plus

    @staticmethod
    def generator_non_saturating_gan_loss(fake_scores: torch.Tensor) -> torch.Tensor:
        maximize_fakes = F.softplus(-fake_scores).mean()
        return maximize_fakes

    @staticmethod
    def critic_non_saturating_gan_loss(fake_scores: torch.Tensor, real_scores: torch.Tensor) -> torch.Tensor:
        maximize_reals = F.softplus(-real_scores).mean()
        minimaze_fakes = F.softplus(fake_scores).mean()

        critic_loss = maximize_reals + minimaze_fakes
        return critic_loss

    @staticmethod
    def r1_penalty(real_score: torch.Tensor, real_input: torch.Tensor) -> torch.Tensor:
        (grad_real,) = autograd.grad(outputs=real_score.sum(), inputs=real_input, retain_graph=True, create_graph=True)
        grad_penalty = torch.sum(grad_real.pow(2), dim=(1, 2, 3))

        return grad_penalty.mean()

    def ppl_regularization(self, fake_img: torch.Tensor, latents: torch.Tensor, decay: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_path_length = self._mean_path_length

        noise = torch.randn_like(fake_img, device=self.device) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])

        (grad,) = autograd.grad(
            outputs=(fake_img * noise).sum(), inputs=latents, retain_graph=True, create_graph=True, only_inputs=True
        )
        path_lengths = torch.sqrt(grad.square().sum(2).mean(1))

        path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
        path_penalty = (path_lengths - path_mean).square().mean()
        self._mean_path_length = path_mean.detach()

        return path_penalty, path_lengths

    def configure_optimizers(self) -> List[optim.Optimizer]:
        generator_parameters = chain(self.synthesis_net.parameters(), self.mapping_net.parameters())
        generator_opt = optim.Adam(generator_parameters, lr=0.0025, betas=(0.0, 0.99))
        critic_opt = optim.Adam(self.critic_net.parameters(), lr=0.0025, betas=(0.0, 0.99))

        return [generator_opt, critic_opt]

    def get_optimizers(self) -> List[optim.Optimizer]:
        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            return optimizers
        return [optimizers]

    @staticmethod
    def mix_latents(w: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        layer_num = w.shape[1]
        index = torch.randint(1, layer_num, (1,))
        return torch.cat([w[:, :index], w2[:, index:]], dim=1)

    def switch_to_critic(self) -> None:
        """ 
        disabling gradient propagatation through generator  
        """
        self.critic_net.requires_grad_(True)
        self.synthesis_net.requires_grad_(False)
        self.mapping_net.requires_grad_(False)
        

    def switch_to_generator(self) -> None:
        self.critic_net.requires_grad_(False)
        self.synthesis_net.requires_grad_(True)
        self.mapping_net.requires_grad_(True)

    def update_ema(self) -> None:
        with torch.no_grad():
            for p_ema, p in zip(self.synthesis_net_ema.parameters(), self.synthesis_net.parameters()):
                p_ema.copy_(p.lerp(p_ema, self.ema_beta))

            for p_ema, p in zip(self.mapping_net_ema.parameters(), self.mapping_net.parameters()):
                p_ema.copy_(p.lerp(p_ema, self.ema_beta))

    def training_step(self, batch, batch_idx, optimizer_idx):
        # fetch data
        real_images, labels = batch

        labels = F.one_hot(labels.long(), self.mapping_net.label_size)  # TODO: remove
        # TODO: not sure about this gonnokod
        labels = labels.type_as(self.synthesis_net.style_conv.bias)

        batch_size = real_images.shape[0]

        get_fake_images = self.forward
        get_critic_scores = self.critic_net.forward
        sampled_resolution = None
        
        if self.use_anycost_gan:
            sampled_resolution_log = np.random.randint(2, self._log_resolution)
            sampled_resolution = 2 ** sampled_resolution_log
            get_fake_images = partial(self.forward, target_resolution=sampled_resolution)
            get_critic_scores = partial(self.critic_net.forward, target_resolution=sampled_resolution)
            real_images = F.interpolate(real_images, (sampled_resolution, sampled_resolution))

        fake_images, latents = get_fake_images(batch_size, labels)

        # get optimizers
        generator_opt, critic_opt = self.get_optimizers()

        # DEAL WITH CRITIQUE
        self.switch_to_critic()
        real_images.requires_grad_(True) # needed to perform regularization

        real_scores = get_critic_scores(real_images, labels)
        fake_scores_no_gen = get_critic_scores(fake_images.detach(), labels)
        critic_loss = self.critic_non_saturating_gan_loss(fake_scores_no_gen, real_scores)

        self.log("real_scores", real_scores.mean(), prog_bar=True)
        self.log("fake_scores", fake_scores_no_gen.mean(), prog_bar=True)

        # add gradient penalty
        if batch_idx % self.penalize_d_every == 0:
            r1_penalty = self.r1_penalty(real_scores, real_images)
            self.log("r1_penalty", r1_penalty, prog_bar=True)

            r1_loss = self.gamma * self.penalize_d_every / 2.0 * r1_penalty
            self.log("r1_loss", r1_loss, prog_bar=True)

            critic_loss = critic_loss + r1_loss

        self.log("critic_loss", critic_loss, prog_bar=True)
        self.manual_backward(critic_loss, critic_opt)
        critic_opt.step()
        critic_opt.zero_grad(set_to_none=True)  # type: ignore  # Pylance bug

        # DEAL WITH GENERATOR
        self.switch_to_generator()

        # make optimization steps
        fake_scores = get_critic_scores(fake_images, labels, return_activations=False)

        if self.use_top_k:
            k = math.ceil(max(self.k.item(), 0.5) * batch_size)
            fake_scores, _ = torch.topk(fake_scores, k, dim=0)

        generator_loss = self.generator_non_saturating_gan_loss(fake_scores)

        self.log("generator_loss", generator_loss, prog_bar=True)

        if batch_idx % self.ppl_reg_every == 0:
            path_loss, path_lengths = self.ppl_regularization(fake_images, latents)
            self.log("path_lengths", path_lengths.mean(), prog_bar=True)

            generator_loss += self.ppl_weight * self.ppl_reg_every * path_loss

        self.manual_backward(generator_loss, generator_opt)
        generator_opt.step()

        generator_opt.zero_grad(set_to_none=True)

        if self.use_ema:
            self.update_ema()

        return [critic_loss, generator_loss]

    def training_epoch_end(self, _) -> None:
        if self.use_top_k:
            self.k *= self.top_k_decay_rate
