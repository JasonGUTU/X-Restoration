import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader

import pytorch_lightning as pl


from srdata.datasets import SRImageFolder, SRImdb, SRTest


class SRDataLoader(pl.LightningDataModule):
    """"""
    def __init__(self,
                 hr_path,
                 lr_path,
                 lr_size=64,
                 sr_factor=4,
                 img_format='RGB',
                 type='lmdb',
                 image_loader='cv2',
                 no_shuffle=False,
                 num_workers=8,
                 batch_size=32,
                 test_hr_path='',
                 test_lr_path='',
                 ):
        super().__init__()
        self.hr_path = hr_path
        self.lr_path = lr_path
        self.lr_patch_size = lr_size
        self.sr_factor = sr_factor
        self.img_format = img_format
        self.type = type
        self.image_loader = image_loader
        self.no_shuffle = no_shuffle
        self.num_workers = num_workers
        self.test_hr_path = test_hr_path
        self.test_lr_path = test_lr_path
        self.batch_size = batch_size

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # Training dataset type
        parser.add_argument('--type', type=str, default='lmdb', help='Image Format, `RGB` or `Y`')
        # Common args
        parser.add_argument('--hr_path', type=str, default='/home/mist/Data/SRData/DIV2K_train_HR_slice4.lmdb')
        parser.add_argument('--lr_path', type=str, default='/home/mist/Data/SRData/DIV2K_train_BILRX4_slice4.lmdb')
        parser.add_argument('--lr_patch_size', type=int, default=64)
        parser.add_argument('--sr_factor', type=int, default=4)
        parser.add_argument('--img_format', type=str, default='RGB', help='Image Format, `RGB` or `Y`')
        # args for `folder`
        parser.add_argument('--image_loader', type=str, default='cv2', help='Load image using PIL.Image or Opencv')
        # args for dataloader
        parser.add_argument('--no_shuffle', action='store_false', help='--no_shuffle for no shuffle')
        parser.add_argument('--num_workers', type=int, default=12)
        parser.add_argument('--batch_size', type=int, default=32)
        # validation dataset
        parser.add_argument('--test_hr_path', type=str, default='/home/mist/Data/SRData/Testsets/HR/B100/x4')
        parser.add_argument('--test_lr_path', type=str, default='/home/mist/Data/SRData/Testsets/LR/LRBI/B100/x4')

        return parser

    @staticmethod
    def from_namespace(args):
        instance = SRDataLoader(
            hr_path=args.hr_path,
            lr_path=args.lr_path,
            lr_size=args.lr_patch_size,
            sr_factor=args.sr_factor,
            img_format=args.img_format,
            type=args.type,
            image_loader=args.image_loader,
            no_shuffle=args.no_shuffle,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            test_hr_path=args.test_hr_path,
            test_lr_path=args.test_lr_path,
        )
        return instance

    def setup(self, stage=None):
        if self.type == 'lmdb':
            self.sr_dataset = SRImdb(
                hr_path=self.hr_path,
                lr_path=self.lr_path,
                lr_patch_size=self.lr_patch_size,
                sr_factor=self.sr_factor,
                img_format=self.img_format
            )
        elif self.type == 'folder':
            self.sr_dataset = SRImageFolder(
                hr_path=self.hr_path,
                lr_path=self.lr_path,
                lr_patch_size=self.lr_patch_size,
                sr_factor=self.sr_factor,
                img_format=self.img_format,
                image_loader=self.image_loader,
            )
        else:
            raise NotImplementedError('Please check your `type` args.')

        if self.test_hr_path != '' and self.test_lr_path != '':
            self.test_dataset = SRTest(
                hr_path=self.test_hr_path,
                lr_path=self.test_lr_path,
                sr_factor=self.sr_factor,
                image_loader=self.image_loader,
                img_format=self.img_format
            )

    def train_dataloader(self):
        return DataLoader(
            self.sr_dataset,
            batch_size=self.batch_size,
            shuffle=self.no_shuffle,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        if self.test_hr_path != '' and self.test_lr_path != '':
            return DataLoader(
                self.test_dataset, batch_size=1, shuffle=False, num_workers=4)
        else:
            return None

    def test_dataloader(self):
        if self.test_hr_path != '' and self.test_lr_path != '':
            return DataLoader(
                self.test_dataset, batch_size=1, shuffle=False, num_workers=4)
        else:
            return None

