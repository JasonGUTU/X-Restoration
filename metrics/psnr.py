import torch
import numpy as np
import math


# def psnr_base(a, b, max_val=1.):
#     """ a: First set of images.
#         b: Second set of images.
#         max_val: The dynamic range of the images (i.e., the difference between the
#             maximum the and minimum allowed values).
#         name: Namespace to embed the computation in.
#         Returns:
#             The scalar PSNR between a and b. The returned tensor has type `tf.float32`
#             and shape [batch_size, 1].
#     """
#     # Need to convert the images to float32.  Scale max_val accordingly so that
#     # PSNR is computed correctly.
#     a = a.type(torch.FloatTensor)
#     b = b.type(torch.FloatTensor)
#     assert a.shape == b.shape, 'Shape must be same for calculating psnr'

#     mse = torch.mean(torch.pow(a - b, 2).view((a.shape[0], -1)), dim=1)

#     psnr_val = torch.sub(
#         20 * np.log(max_val) / np.log(10.),
#         np.float32(10 / np.log(10)) * torch.log(mse),
#     )

#     return psnr_val



def psnr_base(sr, hr, scale=4, rgb_range=1., benchmark=False):

    diff = (sr - hr).data.div(rgb_range)
    shave = scale
    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)
    '''
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    '''
    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)
