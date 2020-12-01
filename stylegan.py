from typing import Optional
from itertools import chain

import torch
from torch import optim, nn
from torch import autograd
from torch.nn import functional as F
import pytorch_lightning as pl

from model.generator import SynthesisNetwork
from model.critic import CriticNetwork
from model.mapping_network import MappingNetwork

class GanTask(pl.LightningModule):
    def __init__(self
                ):
        super().__init__()
        self.gamma = 0.01 # TODO: Add as parameter, check the value
        self.mapping_net = MappingNetwork(latent_size = 512, state_size= 512, style_size = 512, label_size = 2, lr_mul = 0.001)
        self.synthesis_net = SynthesisNetwork(resolution=512, latent_size=512, channel_multiplier=1)
        self.critic_net = CriticNetwork(resolution=512, label_size=2, channel_multiplier=1)


    def forward(self, z: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        latent_vector = self.mapping_net.forward(z, labels)
        latent_vector = torch.repeat_interleave(latent_vector[:, :, None], 16, dim=1)

        return self.synthesis_net.forward(latent_vector)

    @staticmethod
    def non_saturating_gan_loss(fake_scores: torch.Tensor, real_scores: torch.Tensor) -> torch.Tensor:
        maximize_reals = F.softplus(-real_scores).mean()
        minimaze_fakes = F.softplus(fake_scores).mean()
        maximize_fakes = F.softplus(-fake_scores).mean()

        critic_loss = maximize_reals + minimaze_fakes  
        generator_loss = maximize_fakes
        return critic_loss, generator_loss

    @staticmethod
    def r1_penalty(real_score: torch.Tensor, real_input: torch.Tensor) -> torch.Tensor:
        grad_real, = autograd.grad(outputs=real_score.sum(), inputs=real_input, create_graph=True)
        grad_penalty =  torch.sum(grad_real.pow(2), dim=(1,2,3))

        return grad_penalty.mean()

    # def path_len_regularizer(self, ):

    def configure_optimizers(self):
         
        generator_opt = optim.Adam(chain(self.synthesis_net.parameters(), self.mapping_net.parameters()), lr=0.01, betas=(0.0, 0.99))
        critic_opt = optim.Adam(self.critic_net.parameters(), lr=0.02)

        gen_scheduler = {'scheduler': optim.lr_scheduler.ExponentialLR(generator_opt, 0.99),
                 'interval': 'step'} 
        critic_scheduler = {'scheduler': optim.lr_scheduler.ExponentialLR(critic_opt, 0.99),
                 'interval': 'step'}   
        
        return [generator_opt, critic_opt], [gen_scheduler, critic_scheduler]

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images, labels = batch

        # sample noise
        z = torch.randn(real_images.shape[0], 512) # TODO: 512 to parameters
        z = z.type_as(real_images)

        # generate fakes and scores
        fake_images = self.forward(z, labels)
        real_scores = self.critic_net.forward(real_images)
        fake_scores = self.critic_net.forward(fake_images)

        # get losses
        critic_loss, generator_loss = self.non_saturating_gan_loss(fake_scores, real_scores)

        # add gradient penalty
        r1_penalty = self.r1_penalty(real_scores, real_images)
        critic_loss = critic_loss + self.gamma * r1_penalty

        # TODO: CRITICAL ADD PERCEPTUAL PATH LEN REGULARIZER!!!!!!!!!!!!!!!

        # get optimizers 
        (generator_opt, critic_opt) = self.optimizers()

        # make optimization steps
        self.manual_backward(generator_loss, generator_opt)
        generator_opt.step()
        generator_opt.zero_grad()

        self.manual_backward(critic_loss, critic_opt)
        critic_opt.step()
        critic_opt.zero_grad()



    

