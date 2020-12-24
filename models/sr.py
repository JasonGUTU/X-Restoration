import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from argparse import ArgumentParser

import pytorch_lightning as pl


class SuperResolution(pl.LightningModule):
    def __init__(self):
        pass

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dvs_like', type=bool, default=True)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--c_dvs', type=float, default=0.3)
        parser.add_argument('--noise', type=float, default=0.04)
        return parser

    @staticmethod
    def from_argparser_args(args):
        
        return args

    def forward(self, x):
        pass

    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        pass



