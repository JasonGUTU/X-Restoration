import os
import time
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.utilities import argparse_utils

from networks import print_network
from utils.augmentation import *
from metrics.psnr import psnr_base


class SuperResolution(pl.LightningModule):
    """
    Base Model for PSNR-oriented Super-Resolution
    Including:
        1. training code
        2. validation code
        3. testing code
    Note:
        1. the image is ranging from 0 to 1.
    """
    def __init__(self,
                 optimizer='adam',
                 lr=0.0001,
                 adam_beta1=0.9,
                 adam_beta2=0.999,
                 scheduler='step',
                 scheduler_step_epoch=400,
                 scheduler_step_gamma=0.5,
                 scheduler_cos_T_max=20,
                 scheduler_cos_eta_min=0.000001,
                 loss='l1',
                 v_flip=0.5,
                 h_flip=0.5,
                 mixup_prop=0.0,
                 mixup_alpha=1.0,
                 rgb_permute_prop=0.0,
                 visualize_validation=True,
                 visualize_gt=True,
                 exp_name='FSRCNN'):
        super(SuperResolution, self).__init__()
        self.save_hyperparameters()
            # 'optimizer', 'lr', 'adam_beta1', 'adam_beta2', 'scheduler', 'scheduler_decay_epoch'
        # )
        self.network = None
        self.network_cls = None
        self.loss = nn.L1Loss() if self.hparams.loss == 'l1' else nn.MSELoss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # ================= Optimizer
        parser.add_argument('--optimizer', type=str, default='adam', help='Used optimizer, [`adam`, `sgd`]')
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--adam_beta1', type=float, default=0.9)
        parser.add_argument('--adam_beta2', type=float, default=0.999)
        # ================= Optimizer, Learning rate scheduler
        parser.add_argument('--scheduler', type=str, default='step', help='Learning rate scheduler, [`no`, `step`, `exp`, `linear`, `cos`]')
        parser.add_argument('--scheduler_step_epoch', type=int, default=3000)
        parser.add_argument('--scheduler_step_gamma', type=float, default=0.5)
        parser.add_argument('--scheduler_cos_T_max', type=int, default=20)
        parser.add_argument('--scheduler_cos_eta_min', type=float, default=0.000001)
        # ================= Loss Function
        parser.add_argument('--loss', type=str, default='l1', help='Loss function, [`l1`, `l2`]')
        # Augmentation
        parser.add_argument('--v_flip', type=float, default=0.5, help='vertical flip, 0 for no flip, 0.3 for 30% image to be flipped, randomly')
        parser.add_argument('--h_flip', type=float, default=0.5, help='horizontal flip, 0 for no flip, 0.3 for 30% image to be flipped, randomly')
        parser.add_argument('--mixup_prop', type=float, default=0.0, help='proportion of the mixup-ed pairs')
        parser.add_argument('--mixup_alpha', type=float, default=1.0, help='mixup parameter of beta distribution')
        parser.add_argument('--rgb_permute_prop', type=float, default=0.0, help='proportion of the RGB permuted pairs')
        # Testing Tricks
        parser.add_argument('--visualize_validation', action='store_false', help='proportion of the RGB permuted pairs')
        parser.add_argument('--visualize_gt', action='store_false', help='proportion of the RGB permuted pairs')
        # Model saving
        parser.add_argument('--exp_name', type=str, default='FSRCNN', help='Experiment name')

        return parser

    @staticmethod
    def from_namespace(args):
        instance = SuperResolution(
            optimizer=args.optimizer,
            lr=args.lr,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            scheduler=args.scheduler,
            scheduler_step_epoch=args.scheduler_step_epoch,
            scheduler_step_gamma=args.scheduler_step_gamma,
            scheduler_cos_T_max=args.scheduler_cos_T_max,
            scheduler_cos_eta_min=args.scheduler_cos_eta_min,
            loss=args.loss,
            v_flip=args.v_flip,
            h_flip=args.h_flip,
            mixup_prop=args.mixup_prop,
            mixup_alpha=args.mixup_alpha,
            rgb_permute_prop=args.rgb_permute_prop,
            visualize_validation=args.visualize_validation,
            visualize_gt=args.visualize_gt,
            exp_name=args.exp_name
        )
        return instance

    def _augmentation(self, hr_batch, lr_batch ):
        # RGB permute
        hr_batch, lr_batch = aug_RGB_perm(hr_batch, lr_batch, self.hparams.rgb_permute_prop)
        # Flips
        hr_batch, lr_batch = aug_filp(hr_batch, lr_batch, self.hparams.v_flip, self.hparams.h_flip)
        # Mixup
        hr_batch, lr_batch = aug_mixup(hr_batch, lr_batch, self.hparams.mixup_alpha, self.hparams.mixup_prop)
        return hr_batch, lr_batch

    def set_network(self, network):
        self.network = network
        self.network_cls = type(network).__name__
        print(f'Successfully load SR network {self.network_cls}')
        print_network(network)

    def check_network(self):
        if self.network_cls == None:
            raise Exception('No SR network loaded...')

    def pre_processing(self, x):
        if x.dim == 3:
            x.unsqueeze(0)
        x = (x - x.min()) / x.max()
        return x

    def post_processing(self, x):
        return torch.clamp(x, 0., 1.)

    def forward(self, x):
        if isinstance(x, list):
            x = x[1]
        self.check_network()
        x = self.pre_processing(x)
        x_sr = self.network(x)
        x_sr = self.post_processing(x_sr)
        return x_sr

    def on_save_checkpoint(self, checkpoint):
        checkpoint['hparams'] = dict(self.hparams)
        checkpoint['global_step'] = self.global_step
        checkpoint['global_epoch'] = self.current_epoch

    def on_load_checkpoint(self, checkpoint):
        self.network.load_state_dict(checkpoint['state_dict'])
        self.hparams.update(checkpoint['hparams'])

    def configure_scheduler(self, optimizer):
        if self.hparams.scheduler == 'no':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: self.hparams.lr)
        elif self.hparams.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.scheduler_step_epoch, gamma=self.hparams.scheduler_step_gamma)
        elif self.hparams.scheduler == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.scheduler_cos_T_max, eta_min=self.hparams.scheduler_cos_eta_min)
        else:
            print(f'Wrong scheduler parameter {self.hparams.scheduler}, using no scheduler')
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: self.hparams.lr)
        return scheduler

    def configure_optimizers(self):
        params = self.network.parameters()
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.hparams.lr, betas=(self.hparams.adam_beta1, self.hparams.adam_beta2))
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.hparams.lr)
        else:
            print(f'Wrong optimizer parameter {self.hparams.optimizer}, using Adam')
            optimizer = torch.optim.Adam(params, lr=self.hparams.lr, betas=(self.hparams.adam_beta1, self.hparams.adam_beta2))
        scheduler = self.configure_scheduler(optimizer)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        hr, lr = batch
        hr, lr = self._augmentation(hr, lr)
        sr = self.network(lr)

        loss = self.loss(sr, hr)

        # self.logger.experiment.add_image(f'vis_training_data_hr', make_grid(hr), self.global_step)
        # self.logger.experiment.add_image(f'vis_training_data_lr', make_grid(lr), self.global_step)

        self.logger.experiment.add_scalar('train_loss', loss, self.global_step)
        return loss

    def _PSNR_test(self, hr, lr):
        result = self.forward(lr)
        psnr = psnr_base(result, hr)
        return result, psnr

    def test_step(self, batch, batch_idx):
        test_psnr = self._PSNR_test(batch[0], batch[1])
        return test_psnr

    def test_epoch_end(self, outputs):
        return torch.mean(torch.Tensor(outputs))

    def validation_step(self, batch, batch_idx):
        valid_result, valid_psnr = self._PSNR_test(batch[0], batch[1])
        return valid_result, valid_psnr

    def validation_epoch_end(self, val_step_outputs):
        valid_psnrs = []
        for idx, valid_data in enumerate(val_step_outputs):
            result, psnr = valid_data
            if self.hparams.visualize_validation:
                self.logger.experiment.add_image(f'image_{idx}_validation_results', make_grid(result), self.global_step + 1)
            valid_psnrs.append(psnr)

        self.logger.experiment.add_scalar(f'validation_psnr', torch.mean(torch.Tensor(valid_psnrs)), self.global_step)
        print(f'Validate at epoch {self.current_epoch}, the psnr is {torch.mean(torch.Tensor(valid_psnrs)):.2f} db')



