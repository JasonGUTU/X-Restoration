"""Create lmdb files for training image folder"""
import argparse
import sys
import os.path as osp
import glob
import pickle
from multiprocessing import Pool
import numpy as np
import lmdb
import cv2
from tqdm import tqdm

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from srdata.data_utils import *


def read_image_worker(path, key):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return (key, img)


def image_slicer(cv2_image, slice, lr, sr_factor=4, verbose=True):
    sliced_list = []
    slice_h_idx = []
    slice_w_idx = []

    if lr:
        H, W, C = cv2_image.shape
        for n in range(slice):
            slice_h_idx.append((H // slice) * n)
            slice_w_idx.append((W // slice) * n)
        slice_h_idx.append(H)
        slice_w_idx.append(W)
    else:
        H_hr, W_hr, C = cv2_image.shape
        H, W = H_hr // sr_factor, W_hr // sr_factor
        for n in range(slice):
            slice_h_idx.append((H // slice) * n * sr_factor)
            slice_w_idx.append((W // slice) * n * sr_factor)
        slice_h_idx.append(H * sr_factor)
        slice_w_idx.append(W * sr_factor)

    if verbose:
        print(f'Slices:', end=' ')

    for i in range(slice):
        for j in range(slice):
            sliced_list.append(
                np.ascontiguousarray(cv2_image[slice_h_idx[i]: slice_h_idx[i + 1], slice_w_idx[j]: slice_w_idx[j + 1], :])
            )
            if verbose:
                print(f'[{slice_h_idx[i]}:{slice_h_idx[i + 1]}, {slice_w_idx[j]}:{slice_w_idx[j + 1]}]', end=', ')
    if verbose:
        print()

    return sliced_list


def general_image_folder(name, img_folder, lmdb_save_path, BATCH=3000, slice=2, lr=True, sr_factor=4):
    """Create lmdb for general image folders
    Users should define the keys, such as: '0321_s035' for DIV2K sub-images
    If all the images have the same resolution, it will only store one copy of resolution info.
        Otherwise, it will store every resolution info.

        When slice is 2, the big image will be sliced into 2**2 sub-images.
    """
    meta_info = {'name': name}
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    if osp.exists(lmdb_save_path):
        print(f'Folder {lmdb_save_path} already exists. Exit...')
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    all_img_list = image_files(img_folder)
    keys = []
    for idx, img_path in enumerate(all_img_list):
        keys.append(idx*(slice**2))

    #### create lmdb environment
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    #### write data to lmdb
    txn = env.begin(write=True)
    resolutions = []
    commited_images_num = 0
    commited_keys = []
    for idx, (path, key) in enumerate(zip(all_img_list, keys)):
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if data.ndim == 2:
            np.expand_dims(data, 2)

        H, W, C = data.shape
        print(f'Processing image {path}, size H: {H}, W: {W}, C: {C}', end='; ')
        sliced = image_slicer(data, slice, lr=lr, sr_factor=sr_factor)

        for slice_idx, image in enumerate(sliced):
            commited_keys.append(f'{key + slice_idx}')
            key_byte = f'{key + slice_idx}'.encode('ascii')
            txn.put(key_byte, image)
            H, W, C = image.shape
            resolutions.append('{:d}_{:d}_{:d}'.format(C, H, W))
            commited_images_num += 1
            if commited_images_num % BATCH == 0:
                txn.commit()
                txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    #### create meta information
    # check whether all the images are the same size
    assert len(commited_keys) == len(resolutions)
    meta_info['resolution'] = resolutions
    meta_info['keys'] = commited_keys
    print('Not all images have the same resolution. Save meta info for each image.')

    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create lmdb dataset for a image folder.')
    parser.add_argument('-n', '--dataset_name',
                        default='DIV2K_HR_slice4',
                        help='name of the lmdb dataset')
    parser.add_argument('-i', '--target_images',
                        default='/home/mist/Data/SRData/DIV2K/DIV2K_train_LR_bicubic/X4',
                        help='Target images to create.')
    parser.add_argument('-o', '--ouput_path',
                        default='/home/mist/Data/SRData/DIV2K_train_BILRX4_slice4.lmdb',
                        help='the output lmdb path')
    parser.add_argument('--sr_factor',
                        default=4,
                        type=str)
    parser.add_argument('--slice',
                        default=2,
                        type=str)
    parser.add_argument('--lr',
                        action='store_true')

    args, other_args = parser.parse_known_args()
    general_image_folder(args.dataset_name,
                         img_folder=args.target_images,
                         lmdb_save_path=args.ouput_path,
                         sr_factor=args.sr_factor,
                         slice=args.slice,
                         lr=args.lr)
