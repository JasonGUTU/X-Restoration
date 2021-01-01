import glob
import os
import cv2
import lmdb
from argparse import ArgumentParser

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

from .data_utils import *


class SRImageFolder(Dataset):
    def __init__(self,
                 hr_path,
                 lr_path,
                 lr_patch_size=64,
                 sr_factor=4,
                 image_loader='cv2',
                 img_format='RGB'
                 ):
        super(SRImageFolder, self).__init__()

        self.hr_img_path_list = image_files(hr_path)
        self.lr_img_path_list = image_files(lr_path)
        assert len(self.hr_img_path_list) == len(self.lr_img_path_list), "Check your image path list"

        self.loader = cv2_load_as_tensor if image_loader == 'cv2' else pil_load_as_tensor
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * sr_factor
        self.sr_factor = sr_factor
        self.mode = img_format

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hr_path', type=str, default='')
        parser.add_argument('--lr_path', type=str, default='')
        parser.add_argument('--lr_patch_size', type=int, default=64)
        parser.add_argument('--sr_factor', type=int, default=4)
        parser.add_argument('--image_loader', type=str, default='cv2', description='Load image using PIL.Image or Opencv')
        parser.add_argument('--img_format', type=str, default='RGB', description='Image Format, `RGB` or `Y`')

        return parser

    @staticmethod
    def from_namespace(args):
        instance = SRImageFolder(
            hr_path=args.hr_path,
            lr_path=args.lr_path,
            lr_patch_size=args.lr_patch_size,
            sr_factor=args.sr_factor,
            image_loader=args.image_loader,
            img_format=args.img_format
        )
        return instance

    def __getitem__(self, item):
        hr_image_path = self.hr_img_path_list[item]
        lr_image_path = self.lr_img_path_list[item]

        hr_image = self.loader(hr_image_path, mode=self.mode)
        lr_image = self.loader(lr_image_path, mode=self.mode)

        # Crop Image patch
        _, H, W = lr_image.size()
        assert H >= self.lr_patch_size and W >= self.lr_patch_size, 'Your input images must bigger than the image patch size'
        H_start = 0 if H == self.lr_patch_size else np.random.randint(0, H - self.lr_patch_size)
        W_start = 0 if W == self.lr_patch_size else np.random.randint(0, W - self.lr_patch_size)

        lr_image_patch = lr_image[:,
                         H_start: H_start + self.lr_patch_size,
                         W_start: W_start + self.lr_patch_size]
        hr_image_patch = hr_image[:,
                         H_start * self.sr_factor: H_start * self.sr_factor + self.hr_patch_size,
                         W_start * self.sr_factor: W_start * self.sr_factor + self.hr_patch_size]

        return hr_image_patch, lr_image_patch


    def __len__(self):
        return len(self.hr_img_path_list)


class SRImdb(Dataset):
    def __init__(self,
                 hr_path,
                 lr_path,
                 lr_patch_size=64,
                 sr_factor=4,
                 img_format='RGB'):
        super(SRImdb, self).__init__()

        assert hr_path.endswith('lmdb') and lr_path.endswith('lmdb'), 'When using lmdb dataset, the dataset path should end with lmdb'
        self.HR_env = lmdb.open(hr_path,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
        self.paths_HR, self.sizes_HR = get_paths_from_lmdb(hr_path)
        self.LR_env = lmdb.open(lr_path,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
        self.paths_LR, self.sizes_LR = get_paths_from_lmdb(lr_path)
        assert len(self.paths_HR) == len(self.paths_LR), 'Check wheter your hr lmdb and hr lmdb dataset are matched'
        self.sr_factor = sr_factor
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * sr_factor
        self.mode = img_format

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hr_path', type=str, default='')
        parser.add_argument('--lr_path', type=str, default='')
        parser.add_argument('--lr_patch_size', type=int, default=64)
        parser.add_argument('--sr_factor', type=int, default=4)
        parser.add_argument('--img_format', type=str, default='RGB', description='Image Format, `RGB` or `Y`')
        return parser

    @staticmethod
    def from_namespace(args):
        instance = SRImdb(
            hr_path=args.hr_path,
            lr_path=args.lr_path,
            lr_patch_size=args.lr_patch_size,
            sr_factor=args.sr_factor,
            img_format=args.img_format
        )
        return instance

    def __getitem__(self, item):
        # get HR image
        HR_path = self.paths_HR[item]
        LR_path = self.paths_LR[item]
        assert HR_path == LR_path, 'check your lmdb keys'
        resolution_HR = [int(s) for s in self.sizes_HR[item].split('_')]
        resolution_LR = [int(s) for s in self.sizes_LR[item].split('_')]
        assert resolution_HR[1] == resolution_LR[1] * self.sr_factor, 'check your lmdb, the resolution is wrong'
        assert resolution_HR[2] == resolution_LR[2] * self.sr_factor, 'check your lmdb, the resolution is wrong'

        hr_image = read_img_lmdb(self.HR_env, HR_path, resolution_HR, mode=self.mode)
        lr_image = read_img_lmdb(self.LR_env, LR_path, resolution_LR, mode=self.mode)

        # Crop Image patch
        _, H, W = lr_image.size()
        assert H >= self.lr_patch_size and W >= self.lr_patch_size, 'Your input images must bigger than the image patch size'
        H_start = 0 if H == self.lr_patch_size else np.random.randint(0, H - self.lr_patch_size)
        W_start = 0 if W == self.lr_patch_size else np.random.randint(0, W - self.lr_patch_size)

        lr_image_patch = lr_image[:,
                         H_start: H_start + self.lr_patch_size,
                         W_start: W_start + self.lr_patch_size]
        hr_image_patch = hr_image[:,
                         H_start * self.sr_factor: H_start * self.sr_factor + self.hr_patch_size,
                         W_start * self.sr_factor: W_start * self.sr_factor + self.hr_patch_size]

        return hr_image_patch, lr_image_patch


    def __len__(self):
        return len(self.paths_LR)


class SRTest(Dataset):
    def __init__(self,
                hr_path,
                lr_path,
                sr_factor=4,
                image_loader='cv2',
                img_format='RGB'):
        super(SRTest, self).__init__()
        self.hr_img_path_list = image_files(hr_path)
        self.lr_img_path_list = image_files(lr_path)
        assert len(self.hr_img_path_list) == len(self.lr_img_path_list), "Check your image path list"

        self.loader = cv2_load_as_tensor if image_loader == 'cv2' else pil_load_as_tensor
        self.sr_factor = sr_factor
        self.mode = img_format
        self.lr_imgs = []
        self.hr_imgs = []

        for idx in range(len(self.hr_img_path_list)):
            lr_img = self.loader(self.lr_img_path_list[idx], mode=self.mode)
            hr_img = self.loader(self.hr_img_path_list[idx], mode=self.mode)
            hr_img = hr_img[:, : - hr_img.shape[1] % self.sr_factor, :] if hr_img.shape[1] % self.sr_factor != 0 else hr_img
            hr_img = hr_img[:, :, : - hr_img.shape[2] % self.sr_factor] if hr_img.shape[2] % self.sr_factor != 0 else hr_img
            c, h, w = lr_img.shape
            C, H, W = hr_img.shape
            assert c == C and h * self.sr_factor == H and w * self.sr_factor == W, 'Check your test images size'
            self.lr_imgs.append(lr_img)
            self.hr_imgs.append(hr_img)

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hr_path', type=str, default='')
        parser.add_argument('--lr_path', type=str, default='')
        parser.add_argument('--sr_factor', type=int, default=4)
        parser.add_argument('--image_loader', type=str, default='cv2', help='Load image using PIL.Image or Opencv')
        parser.add_argument('--img_format', type=str, default='RGB', help='Image Format, `RGB` or `Y`')
        return parser

    @staticmethod
    def from_namespace(args):
        instance = SRTest(
            hr_path=args.hr_path,
            lr_path=args.lr_path,
            sr_factor=args.sr_factor,
            image_loader=args.image_loader,
            img_format=args.img_format
        )
        return instance

    def __getitem__(self, item):
        return self.hr_imgs[item], self.lr_imgs[item]

    def __len__(self):
        return len(self.lr_imgs)



