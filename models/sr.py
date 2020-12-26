import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from argparse import ArgumentParser

import pytorch_lightning as pl


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
                 scheduler='half',
                 scheduler_decay_epoch=400):
        super(SuperResolution, self).__init__()
        self.save_hyperparameters()
        self.network = None
        self.network_cls = None
        self.loss = nn.L1Loss() if self.hparams.loss == 'l1' else nn.MSELoss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # ================= Optimizer
        parser.add_argument('--optimizer', type=str, default='adam', description='Used optimizer, [`adam`, `sgd`]')
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--adam_beta1', type=float, default=0.9)
        parser.add_argument('--adam_beta2', type=float, default=0.999)
        # ================= Optimizer, Learning rate scheduler
        parser.add_argument('--scheduler', type=str, default='half', description='Learning rate scheduler, [`no`, `step`, `exp`, `linear`, `cos`]')
        parser.add_argument('--scheduler_step_epoch', type=int, default=400)
        parser.add_argument('--scheduler_step_gamma', type=float, default=0.5)
        parser.add_argument('--scheduler_cos_T_max', type=int, default=20)
        parser.add_argument('--scheduler_cos_eta_min', type=float, default=0.000001)
        # ================= Loss Function
        parser.add_argument('--loss', type=str, default='l1', description='Loss function, [`l1`, `l2`]')

        return parser

    @staticmethod
    def from_namespace(args):
        instance = SuperResolution()
        return instance

    def set_network(self, network):
        self.network = network
        self.network_cls = type(network).__name__
        print(f'Successfully load SR network {self.network_cls}')
        network.print_network()

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
        self.check_network()
        x = self.pre_processing(x)
        x_sr = self.network(x)
        x_sr = self.post_processing(x_sr)
        return x_sr

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
        params = self.model
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
        x_sr = self.network(batch)
        loss = self.loss(x_sr, batch)

        self.log('train_loss', loss, prog_bar=True, logger=True, )

        return loss

    def _PSNR_test(self, x):
        result = self.forward(x)
        psnr = self.loss(result, x)
        return psnr

    def test_step(self, batch, batch_idx):
        test_psnr = self._PSNR_test(batch)

        # TODO calculate PSNR

        return test_psnr

    def test_epoch_end(self, outputs):
        return torch.mean(outputs)

    def validation_step(self, batch, batch_idx):
        valid_psnr = self._PSNR_test(batch)

        # TODO calculate PSNR
        # TODO: inperceptual SR, visualize results

        self.log('validation_psnr', valid_psnr, logger=True, on_epoch=True)

        return valid_psnr

    def validation_epoch_end(self, val_step_outputs):
        return torch.mean(val_step_outputs)



