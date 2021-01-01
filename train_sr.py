#================================ Imports
import os
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from models import load_model
from models import SRLoggingCallback
from networks import load_sr_network
from srdata.dataloader import SRDataLoader
from utils.trainer_args import sr_Trainer_Default


def main(args):
    # ================================ Initializing
    # init network and model
    net = SR_Nets.from_namespace(args)
    model = SR_Model.from_namespace(args)
    model.set_network(net)

    # init dataloader
    pl_srdata = SRDataLoader.from_namespace(args)
    pl_srdata.setup()
    train_dataloader = pl_srdata.train_dataloader()
    test_dataloader = pl_srdata.test_dataloader()

    # init logger
    logger = TensorBoardLogger(os.path.join(args.logger_dir, args.exp_name), name=args.model_type)

    # model saving
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.default_root_dir,
        verbose=True,
        save_last=True,
        prefix=args.exp_name,
        period=args.check_val_every_n_epoch,  # this is to save model to `last`, in callback, we save model in separate files
        filename='{epoch}-{step}',
    )

    # init trainer
    trainer = pl.Trainer.from_argparse_args(args,
                                            gpus=1,
                                            logger=logger,
                                            callbacks=[model_checkpoint_callback, SRLoggingCallback()])


    # ================================ Train
    trainer.fit(model,
                train_dataloader=train_dataloader,
                val_dataloaders=test_dataloader)

    # ================================ Save Final Model
    trainer.save_checkpoint(f'{args.default_root_dir}/{args.exp_name}-final.ckpt')


if __name__ == "__main__":
    # fix the seed for reproducing
    pl.seed_everything(1)

    #================================ ArgParsing
    # init Argment Parser
    parser = ArgumentParser()
    # add all the available trainer options to argparse, Check trainer's paras for help
    parser = pl.Trainer.add_argparse_args(parser)
    # figure out which model to use
    parser.add_argument('--model_type', type=str, default='sr', help='Perceptual [percept] or PSNR [sr] oriented SR')
    parser.add_argument('--network_name', type=str, default='FSRCNN', help='Name of your output')
    parser.add_argument('--logger_dir', type=str, default='./EXPs/tb_logs', help='logging path')

    temp_args, _ = parser.parse_known_args()

    # add model specific args
    SR_Model = load_model(temp_args.model_type)
    SR_Nets = load_sr_network(temp_args.network_name)
    parser = SR_Model.add_model_specific_args(parser)
    parser = SR_Nets.add_model_specific_args(parser)
    # add training data specific args
    parser = SRDataLoader.add_data_specific_args(parser)
    
    args = parser.parse_args()
    sr_Trainer_Default(args)

    main(args)

