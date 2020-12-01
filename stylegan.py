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

        self.mapping_net = MappingNetwork(latent_size = 512, state_size= 512, style_size = 512, label_size = 2, lr_mul = 0.001)
        self.synthesis_net = SynthesisNetwork(resolution=512, latent_size=512, channel_multiplier=1)
        self.critic_net = CriticNetwork(resolution=512, label_size=2, channel_multiplier=1)


    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        latent_vector = self.mapping_net.forward(x, labels)
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

    

    

