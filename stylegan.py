from itertools import chain
import math
from typing import Optional
import random


import torch
from torch import optim, nn
from torch import autograd
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.nn.modules import distance

from models.generator import SynthesisNetwork, AuxiliaryL2Projector
from models.critic import CriticNetwork
from models.mapping_network import MappingNetwork
import numpy as np

class GanTask(pl.LightningModule):
    def __init__(self,
                gamma: float = 10,
                ppl_reg_every: int = 4,
                penalize_d_every: int =16,
                ppl_weight: float = 2,
                resolution=512
                ):
        super().__init__()
        self.gamma = gamma # TODO: check the value
        self.ppl_reg_every = ppl_reg_every
        self.penalize_d_every = penalize_d_every
        self.ppl_weight = ppl_weight
        self.resolution = resolution

        self._mean_path_length = 0

        self.mapping_net = MappingNetwork(input_size = 512, state_size= 512, latent_size = 512, label_size = 2, lr_mul = 0.01)
        self.synthesis_net = SynthesisNetwork(resolution=resolution, latent_size=512, channel_multiplier=1)
        self.critic_net = CriticNetwork(resolution=resolution, label_size=2, channel_multiplier=1)
        self.auxiliary_projector = AuxiliaryL2Projector(lr_mul=1)

    def generate_latents(self, batch_size: int, labels: Optional[torch.Tensor]) -> torch.Tensor:
        # sample noise
        z = torch.randn(batch_size, self.mapping_net.input_size) 
        z = z.type_as(self.synthesis_net.style_conv.bias) #TODO: not sure about this gonnokod

        w = self.mapping_net.forward(z, labels)

        return w[:, None, :].expand(-1, self.synthesis_net.num_layers, -1)
        # return torch.repeat_interleave(w[:, None, :], self.synthesis_net.num_layers, dim=1)

    def forward(self, batch_size: int, labels: Optional[torch.Tensor], mix_prob: float = 0.9) -> torch.Tensor:
        w_plus = self.generate_latents(batch_size, labels)
        if mix_prob > 0 and random.random() < mix_prob: 
            w_plus_ = self.generate_latents(batch_size, labels) # TODO: consider to put random labels and use mixup in discriminator
            w_plus = self.mix_latents(w_plus, w_plus_) # TODO: consider more radical mixing

        return self.synthesis_net.forward(w_plus), w_plus
    
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
        grad_real, = autograd.grad(outputs=real_score.sum(), inputs=real_input, retain_graph=True, create_graph=True)
        grad_penalty =  torch.sum(grad_real.pow(2), dim=(1,2,3))

        return grad_penalty.mean()

    def ppl_regularization(self, fake_img: torch.Tensor, latents: torch.Tensor, decay: float=0.01) -> torch.Tensor:
        mean_path_length = self._mean_path_length
        
        noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
        grad, = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, retain_graph=True, create_graph=True)
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

        path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

        path_penalty = (path_lengths - path_mean).pow(2).mean()

        # TODO: maybe this side effect is not a good idea
        self._mean_path_length = path_mean.detach()

        return path_penalty, path_lengths

    def configure_optimizers(self):
         
        generator_opt = optim.Adam(chain(self.synthesis_net.parameters(), self.mapping_net.parameters(), self.auxiliary_projector.parameters()), lr=0.0025, betas=(0.0, 0.99))
        critic_opt = optim.Adam(self.critic_net.parameters(), lr=0.0025, betas=(0.0, 0.99))

        # gen_scheduler = {'scheduler': optim.lr_scheduler.ExponentialLR(generator_opt, 0.99),
        #          'interval': 'step'} 
        # critic_scheduler = {'scheduler': optim.lr_scheduler.ExponentialLR(critic_opt, 0.99),
        #          'interval': 'step'}   
        
        return [generator_opt, critic_opt] #, [gen_scheduler, critic_scheduler]

    def mix_latents(self, w: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        layer_num = w.shape[1]
        index = torch.randint(1, layer_num, (1,))
        return torch.cat([w[:, :index], w2[:, index:]], dim=1)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images, labels = batch

        labels = F.one_hot(labels.long(), self.mapping_net.label_size) #TODO: remove
        labels = labels.type_as(self.synthesis_net.style_conv.bias) #TODO: not sure about this gonnokod

        batch_size = real_images.shape[0]

        fake_images, latents = self.forward(batch_size, labels)


        # get optimizers 
        (generator_opt, critic_opt) = self.optimizers()

        # DEAL WITH CRITIQUE
        self.critic_net.requires_grad_(True)
        self.synthesis_net.requires_grad_(False)
        self.mapping_net.requires_grad_(False)
        real_images.requires_grad_(True)

        real_scores = self.critic_net.forward(real_images, labels)
        fake_scores_no_gen = self.critic_net.forward(fake_images.detach(), labels) # DETACH is used to prevent extra computations 
        critic_loss = self.critic_non_saturating_gan_loss(fake_scores_no_gen, real_scores)

        self.log('real_scores', real_scores.mean(), prog_bar=True)
        self.log('fake_scores', fake_scores_no_gen.mean(), prog_bar=True)

        #add gradient penalty
        if batch_idx % self.penalize_d_every == 0:
            r1_penalty = self.r1_penalty(real_scores, real_images)
            self.log('r1_penalty', r1_penalty, prog_bar=True)
            critic_loss = critic_loss + self.gamma / 2. * r1_penalty

        self.log('critic_loss', critic_loss, prog_bar=True)

        self.manual_backward(critic_loss, critic_opt)
        critic_opt.step()

        # critic_grad = np.mean([torch.linalg.norm(i.grad).item() for i in self.critic_net.parameters()])
        # self.log('critic_grad', critic_grad, prog_bar=True)

        # synthesis_grad_crit = np.mean([torch.linalg.norm(i.grad).item() for i in self.synthesis_net.parameters()])
        # self.log('synthesis_grad_crit', synthesis_grad_crit, prog_bar=True)

        # mapping_net_grad_crit = np.mean([torch.linalg.norm(i.grad).item() for i in self.mapping_net.parameters()])
        # self.log('mapping_net_grad_crit', mapping_net_grad_crit, prog_bar=True)

        critic_opt.zero_grad()

        # DEAL WITH GENERATOR
        self.critic_net.requires_grad_(False)
        self.synthesis_net.requires_grad_(True)
        self.mapping_net.requires_grad_(True)

        # make optimization steps
        fake_scores, activations = self.critic_net.forward(fake_images, labels, return_activations=True)
        generator_loss = self.generator_non_saturating_gan_loss(fake_scores)

        self.log('generator_loss', generator_loss, prog_bar=True)


        ###### SHITTY PPL
        # if batch_idx > 32:
        projection = self.auxiliary_projector.forward(activations[-1])
        permutation = torch.randperm(batch_size)
        flatents = torch.flatten(latents, 1)
        latent_distances = torch.pairwise_distance(flatents, flatents[permutation])
        projection_distances = torch.pairwise_distance(projection, projection[permutation])
        distance_regularization = F.mse_loss(latent_distances, projection_distances)

        self.log('distance_reg', distance_regularization, prog_bar=True)

        generator_loss += 0.0001 * distance_regularization


        self.manual_backward(generator_loss, generator_opt)
        generator_opt.step()

        # critic_grad_gen = np.mean([torch.linalg.norm(i.grad).item() for i in self.critic_net.parameters()])
        # self.log('critic_grad_gen', critic_grad_gen, prog_bar=True)

        # synthesis_grad_gen = np.mean([torch.linalg.norm(i.grad).item() for i in self.synthesis_net.parameters()])
        # self.log('synthesis_grad_gen', synthesis_grad_gen, prog_bar=True)

        # mapping_net_grad_gen = np.mean([torch.linalg.norm(i.grad).item() for i in self.mapping_net.parameters()])
        # self.log('mapping_net_grad_gen', mapping_net_grad_gen, prog_bar=True)

        generator_opt.zero_grad()

        # add ppl penalty

        # if batch_idx % self.ppl_reg_every == 0: # TODO: Check it! Rosinality line 258. For some reasons guy zeroing grads and making extra step
        #     fake_images, latents = self.forward(2, labels[:2])
        #     path_loss, path_lengths = self.ppl_regularization(fake_images, latents)
        #     self.log('path_lengths', path_lengths.mean(), prog_bar=True)

        #     path_loss = self.ppl_weight * path_loss
        
        #     self.manual_backward(path_loss, generator_opt)
        #     generator_opt.step()
        #     generator_opt.zero_grad()


        return [critic_loss, generator_loss]