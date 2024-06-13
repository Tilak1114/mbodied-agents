from collections import OrderedDict
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from gym import spaces
from mbodied_agents.agents.motion.vit_rt.dataset.dataloader import get_data_loader
from mbodied_agents.agents.motion.vit_rt.transformer_network import TransformerNetwork
from torch.optim import Adam


class RTLightningModule(pl.LightningModule):
    """A PyTorch Lightning module for the TransformerNetwork."""

    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Dict, **kwargs):
        super().__init__()
        self.model = TransformerNetwork(
            observation_space, action_space, **kwargs)

    def forward(self, observations: Dict[str, torch.Tensor]) -> tuple:
        return self.model(observations)

    def training_step(self, batch, batch_idx):
        observations, actions, target = batch
        output = self(observations)

        # Calculate the loss
        loss = self.model.loss_object(output, target).mean()

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)


def main():
    # Set up the data loader
    data_loader = get_data_loader()

    observation_space = spaces.Dict(
        {
            "image_primary": spaces.Box(low=0.0, high=1.0, shape=(3, 224, 224), dtype=np.float32),
            "natural_language_embedding": spaces.Box(low=-np.inf, high=np.inf, shape=[768], dtype=np.float32),
        },
    )
    action_space = spaces.Dict(
        OrderedDict(
            [
                (
                    "xyz",
                    spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                ),
                (
                    "rpy",
                    spaces.Box(low=-np.pi, high=np.pi,
                               shape=(3,), dtype=np.float32),
                ),
                (
                    "grasp",
                    spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                ),
            ],
        ),
    )
    model = RTLightningModule(observation_space, action_space)

    # Initialize the Trainer with multi-GPU setup
    trainer = pl.Trainer(
        devices=-1,
        strategy='ddp',
    )

    # Train the model
    trainer.fit(model, data_loader)


if __name__ == "__main__":
    main()
