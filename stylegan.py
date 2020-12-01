from typing import Optional
import torch
from torch import optim
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

    def configure_optimizers(self):
        generator_opt = optim.Adam(self.model_gen.parameters(), lr=0.01)
        critic_opt = optim.Adam(self.model_disc.parameters(), lr=0.02)

        gen_scheduler = {'scheduler': optim.lr_scheduler.ExponentialLR(generator_opt, 0.99),
                 'interval': 'step'} 
        critic_scheduler = {'scheduler': optim.lr_scheduler.ExponentialLR(critic_opt, 0.99),
                 'interval': 'step'}   
        
        return [generator_opt, critic_opt], [gen_scheduler, critic_scheduler]

    

    

