#================================ Imports
import os
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from models import load_sr_model
from networks import load_network
from dataset import load_srdata


if __name__ == "__main__":
    #================================ ArgParsing
    # fix the seed for reporducing
    pl.seed_everything(1234)
    # init Argment Parser
    parser = ArgumentParser()
    # add all the available trainer options to argparse, Check trainer's paras for help
    parser = pl.Trainer.add_argparse_args(parser)
    # figure out which model to use
    parser.add_argument('--model_type', type=str, default='gan', help='gan or mnist')
    parser.add_argument('--network_name', type=str, default='SRCNN', help='gan or mnist')
    temp_args, _ = parser.parse_known_args()
    # add model specific args
    SR_Model = load_sr_model(temp_args.model_type)
    SR_Nets = load_sr_model(temp_args.network_name)


    
    args = parser.parse_args()







    #================================ Initilizing


    #================================ Train