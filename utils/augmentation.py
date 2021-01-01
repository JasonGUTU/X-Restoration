import numpy as np

import torch
import torchvision.transforms.functional


def aug_filp(hr_batch, lr_batch, v_flip=0.5, h_flip=0.5):
    if np.random.binomial(1, v_flip):
        hr_batch.flip(3)
        lr_batch.flip(3)
    if np.random.binomial(1, h_flip):
        hr_batch.flip(2)
        lr_batch.flip(2)
    return hr_batch, lr_batch


def aug_RGB_perm(hr_batch, lr_batch, prop=0.0):
    if prop != 0.:
        prop = np.clip(prop, 0., 1.)
        batch_size = lr_batch.size()[0]
        premed_number = int(batch_size * prop)
        perm = torch.randperm(3)

        hr_permute, hr_nonpermute = torch.split(hr_batch, [premed_number, batch_size - premed_number])
        lr_permute, lr_nonpermute = torch.split(lr_batch, [premed_number, batch_size - premed_number])

        hr_permute = hr_permute[:, perm, :, :]
        lr_permute = lr_permute[:, perm, :, :]

        return torch.cat([hr_permute, hr_nonpermute], dim=0), \
               torch.cat([lr_permute, lr_nonpermute], dim=0)
    else:
        return hr_batch, lr_batch


def aug_mixup(hr_batch, lr_batch, alpha=1.0, prop=0.0):
    if prop != 0.:
        lam = np.random.beta(alpha, alpha) if alpha > 0. else 1.
        prop = np.clip(prop, 0., 1.)
        batch_size = lr_batch.size()[0]
        mixed_number = int(batch_size * prop)

        hr_mixup, hr_nonmixup = torch.split(hr_batch, [mixed_number, batch_size - mixed_number])
        lr_mixup, lr_nonmixup = torch.split(lr_batch, [mixed_number, batch_size - mixed_number])

        index = torch.randperm(mixed_number)
        hr_mixup = lam * hr_mixup + (1 - lam) * hr_mixup[index, :, :, :]
        lr_mixup = lam * lr_mixup + (1 - lam) * lr_mixup[index, :, :, :]

        return torch.cat([hr_mixup, hr_nonmixup], dim=0), \
               torch.cat([lr_mixup, lr_nonmixup], dim=0)
    else:
        return hr_batch, lr_batch

