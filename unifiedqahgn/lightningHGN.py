import torch
from torch import nn
import pytorch_lightning as pl

class lightningHGN(pl.LightningModule):

    def __init__(self, args):
        super(lightningHGN, self).__init__()

    def forward(self):
        return

    def training_step(self):
        return

    def validation_step_end(self):
        return

    def validation_epoch_end(self, validation_step_outputs):
        for out in validation_step_outputs:
            break
        return
    def configure_optimizers(self):
        return