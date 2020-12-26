import glob
import os
from argparse import ArgumentParser

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader



class SRImageFolder(Dataset):
    def __init__(self,
                 hr_path,
                 lr_path,
                 sr_factor=4,
                 v_flip=0.5,
                 h_flip=0.5,
                 mixup_prop=0.0,
                 mixup_alpha=1.0,
                 rgb_permute_prop=0.0):
        super(SRImageFolder, self).__init__()

        self.hr_img_path_list = sorted(glob.glob(f'{os.path.join(os.path.abspath(hr_path), "*")}'))
        self.lr_img_path_list = sorted(glob.glob(f'{os.path.join(os.path.abspath(lr_path), "*")}'))
        assert len(self.hr_img_path_list) == len(self.lr_img_path_list), "Check your image path list"
    # def mixup_data(x, y, alpha=1.0, use_cuda=True):
    #
    #     '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    #     if alpha > 0.:
    #         lam = np.random.beta(alpha, alpha)
    #     else:
    #         lam = 1.
    #     batch_size = x.size()[0]
    #     if use_cuda:
    #         index = torch.randperm(batch_size).cuda()
    #     else:
    #         index = torch.randperm(batch_size)
    #
    #     mixed_x = lam * x + (1 - lam) * x[index, :]
    #     y_a, y_b = y, y[index]
    #     return mixed_x, y_a, y_b, lam

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hr_path', type=str, default='')
        parser.add_argument('--lr_path', type=str, default='')
        parser.add_argument('--lr_size', type=int, default=64)
        parser.add_argument('--sr_factor', type=int, default=4)
        # Augmentation
        parser.add_argument('--v_flip', type=float, default=0.5, description='vertical flip, 0 for no flip, 0.3 for 30% image to be flipped, randomly')
        parser.add_argument('--h_flip', type=float, default=0.5, description='horizontal flip, 0 for no flip, 0.3 for 30% image to be flipped, randomly')
        parser.add_argument('--mixup_prop', type=float, default=0.0, description='proportion of the mixup-ed pairs')
        parser.add_argument('--mixup_alpha', type=float, default=1.0, description='mixup parameter of beta distribution')
        parser.add_argument('--rgb_permute_prop', type=float, default=0.0, description='proportion of the RGB permuted pairs')

        return parser

    @staticmethod
    def from_namespace(args):
        instance = SRImageFolder()
        return instance


    def _augmentation(self, hr_sample, lr_sample):
        # RGB permute

        # Flips

        # Mixup

        return

    def __getitem__(self, item):

    def __len__(self):