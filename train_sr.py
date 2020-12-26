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

from models import load_model
from networks import load_sr_network
# from srdata import load_srdata


def main(args):
    # ================================ Initilizing
    # init network and model
    net = SR_Nets.from_namespace(args)
    model = SR_Model.from_namespace(args)
    model.set_network(net)

    # init dataloader
    train_dataloader =
    test_dataloader =

    # init logger
    logger =

    # init trainer
    trainer = pl.Trainer()
    # ================================ Train


if __name__ == "__main__":
    # fix the seed for reporducing
    pl.seed_everything(1)
    #================================ ArgParsing
    # init Argment Parser
    parser = ArgumentParser()
    # add all the available trainer options to argparse, Check trainer's paras for help
    parser = pl.Trainer.add_argparse_args(parser)
    # figure out which model to use
    parser.add_argument('--model_type', type=str, default='sr', help='Perceptual [percept] or PSNR [sr] oriented SR')
    parser.add_argument('--network_name', type=str, default='FSRCNN', help='Name of your output')
    temp_args, _ = parser.parse_known_args()
    # add model specific args
    SR_Model = load_model(temp_args.model_type)
    SR_Nets = load_sr_network(temp_args.network_name)
    parser = SR_Model.add_model_specific_args(parser)
    parser = SR_Nets.add_model_specific_args(parser)
    # add training data specific args
    
    args = parser.parse_args()
    main(args)

