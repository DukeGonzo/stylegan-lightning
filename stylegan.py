from itertools import chain
import math
from typing import Optional
import random


import torch
from torch import optim, nn
from torch import autograd
from torch.nn import functional as F
import pytorch_lightning as pl

from models.generator import SynthesisNetwork
from models.critic import CriticNetwork
from models.mapping_network import MappingNetwork

class GanTask(pl.LightningModule):
    def __init__(self,
                gamma: float,
                ppl_reg_every: int = 4,
                penalize_d_every: int =16,
                ppl_weight: float = 2,
                ):
        super().__init__()
        self.gamma = gamma # TODO: check the value
        self.ppl_reg_every = ppl_reg_every
        self.penalize_d_every = penalize_d_every
        self.ppl_weight = ppl_weight

        self._mean_path_length = 0

        self.mapping_net = MappingNetwork(input_size = 512, state_size= 512, latent_size = 512, label_size = 2, lr_mul = 0.001)
        self.synthesis_net = SynthesisNetwork(resolution=512, latent_size=512, channel_multiplier=1)
        self.critic_net = CriticNetwork(resolution=512, label_size=2, channel_multiplier=1)

    def generate_latents(self, batch_size: int, labels: Optional[torch.Tensor]) -> torch.Tensor:
        # sample noise
        z = torch.randn(batch_size, self.mapping_net.input_size) 
        z = z.type_as(self.synthesis_net.style_conv.bias) #TODO: not sure about this gonnokod

        w = self.mapping_net.forward(z, labels)

        # return w[:, None, :].expand(-1, self.synthesis_net.num_layers, -1)
        return torch.repeat_interleave(w[:, None, :], self.synthesis_net.num_layers, dim=1)

    def forward(self, batch_size: int, labels: Optional[torch.Tensor], mix_prob: float = 0.9) -> torch.Tensor:
        w_plus = self.generate_latents(batch_size, labels)
        if mix_prob > 0 and random.random() < mix_prob: 
            w_plus_ = self.generate_latents(batch_size, labels) # TODO: consider to put random labels and use mixup in discriminator
            w_plus = self.mix_latents(w_plus, w_plus_) # TODO: consider more radical mixing
        return self.synthesis_net.forward(w_plus), w_plus

    def ppl_regularization(self, fake_img: torch.Tensor, latents: torch.Tensor, decay: float=0.01) -> torch.Tensor:
        mean_path_length = self._mean_path_length

        noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
        grad, = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

        path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

        path_penalty = (path_lengths - path_mean).pow(2).mean()

        # TODO: maybe this side effect is not a good idea
        self._mean_path_length = path_mean.detach()

        return path_penalty, path_lengths
    
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
        grad_real, = autograd.grad(outputs=real_score.sum(), inputs=real_input, create_graph=True)
        grad_penalty =  torch.sum(grad_real.pow(2), dim=(1,2,3))

        return grad_penalty.mean()

    def configure_optimizers(self):
         
        generator_opt = optim.Adam(chain(self.synthesis_net.parameters(), self.mapping_net.parameters()), lr=0.01, betas=(0.0, 0.99))
        critic_opt = optim.Adam(self.critic_net.parameters(), lr=0.02)

        gen_scheduler = {'scheduler': optim.lr_scheduler.ExponentialLR(generator_opt, 0.99),
                 'interval': 'step'} 
        critic_scheduler = {'scheduler': optim.lr_scheduler.ExponentialLR(critic_opt, 0.99),
                 'interval': 'step'}   
        
        return [generator_opt, critic_opt], [gen_scheduler, critic_scheduler]

    def mix_latents(self, w: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        layer_num = w.shape[1]
        index = torch.randint(1, layer_num, (1,))
        return torch.cat([w[:, :index], w2[:, index:]], dim=1)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images, labels = batch

        labels = F.one_hot(labels.long(), self.mapping_net.label_size) #TODO: remove
        labels = labels.type_as(self.synthesis_net.style_conv.bias) #TODO: not sure about this gonnokod

        batch_size = real_images.shape[0]

        # get optimizers 
        (generator_opt, critic_opt) = self.optimizers()

        # DEAL WITH GENERATOR
        self.critic_net.requires_grad_(False)
        self.synthesis_net.requires_grad_(True)
        self.mapping_net.requires_grad_(True)

        # make optimization steps
        fake_images, latents = self.forward(batch_size, labels)
        fake_scores = self.critic_net.forward(fake_images, labels)
        generator_loss = self.generator_non_saturating_gan_loss(fake_scores)

        # add ppl penalty
        if batch_idx % self.ppl_reg_every == 0: # TODO: Check it! Rosinality line 258. For some reasons guy zeroing grads and making extra step
            path_loss, path_lengths = self.ppl_regularization(fake_images, latents)

            generator_loss = generator_loss + self.ppl_weight * self.ppl_reg_every * path_loss

        self.log('generator_loss', generator_loss, prog_bar=True)

        self.manual_backward(generator_loss, generator_opt)
        generator_opt.step()
        generator_opt.zero_grad()


        # DEAL WITH CRITIQUE
        self.critic_net.requires_grad_(True)
        self.synthesis_net.requires_grad_(False)
        self.mapping_net.requires_grad_(False)
        real_images.requires_grad_(True)

        real_scores = self.critic_net.forward(real_images, labels)
        fake_scores_no_gen = self.critic_net.forward(fake_images.detach(), labels) # DETACH is used to prevent extra computations 
        critic_loss = self.critic_non_saturating_gan_loss(fake_scores_no_gen, real_scores)

        # add gradient penalty
        if batch_idx % self.penalize_d_every == 0:
            r1_penalty = self.r1_penalty(real_scores, real_images)
            critic_loss = critic_loss + self.gamma * self.penalize_d_every * r1_penalty

        self.log('critic_loss', critic_loss, prog_bar=True)

        self.manual_backward(critic_loss, critic_opt)
        critic_opt.step()
        critic_opt.zero_grad()