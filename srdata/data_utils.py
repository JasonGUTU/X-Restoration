import os
from PIL import Image
import torchvision
import cv2
import numpy as np
import torch
import pickle

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def pil_loader(path, mode='RGB'):
    """
    open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    :param path: image path
    :return: PIL.Image
    """
    assert _is_image_file(path), "%s is not an image" % path
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert(mode)


def get_paths_from_lmdb(dataroot):
    """get image path list from lmdb meta info"""
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    return paths, sizes


def read_img_lmdb(env, key, size, mode='RGB'):
    """read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple"""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    img = img.astype(np.float32)

    if mode != 'Y':
        img_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
        img_tensor = img_tensor[[2, 1, 0], :, :]  / 255.
    else:
        img = (np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0) / 255.
        img_tensor = torch.from_numpy(np.ascontiguousarray(np.expand_dims(img, axis=0))).float()

    return img_tensor


def cv2_load_as_tensor(path, mode='RGB'):
    """read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]

    if mode != 'Y':
        img_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
        img_tensor = img_tensor[[2, 1, 0], :, :] / 255.
    else:
        img = (np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0) / 255.
        img_tensor = torch.from_numpy(np.ascontiguousarray(np.expand_dims(img, axis=0))).float()

    return img_tensor


def pil_load_as_tensor(path, mode='RGB'):
    """
    Load image to tensor
    :param path: image path
    :param mode: 'Y' returns 1 channel tensor, 'RGB' returns 3 channels, 'RGBA' returns 4 channels, 'YCbCr' returns 3 channels
    :return: 3D tensor
    """
    if mode != 'Y':
        return PIL2Tensor(pil_loader(path, mode=mode))
    else:
        return PIL2Tensor(pil_loader(path, mode='YCbCr'))[:1]


def PIL2Tensor(pil_image):
    return torchvision.transforms.functional.to_tensor(pil_image)


def Tensor2PIL(tensor_image, mode='RGB'):
    if len(tensor_image.size()) == 4 and tensor_image.size()[0] == 1:
        tensor_image = tensor_image.view(tensor_image.size()[1:])
    return torchvision.transforms.functional.to_pil_image(tensor_image.detach(), mode=mode)


def _is_image_file(filename):
    """
    judge if the file is an image file
    :param filename: path
    :return: bool of judgement
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def image_files(path):
    """
    return list of images in the path

        # self.hr_img_path_list = sorted(glob.glob(f'{os.path.join(os.path.abspath(hr_path), "*")}'))
        # self.lr_img_path_list = sorted(glob.glob(f'{os.path.join(os.path.abspath(lr_path), "*")}'))
    :param path: path to Data Folder, absolute path
    :return: 1D list of image files absolute path
    """
    abs_path = os.path.abspath(path)
    image_files = os.listdir(abs_path)
    for i in range(len(image_files)):
        if (not os.path.isdir(image_files[i])) and (_is_image_file(image_files[i])):
            image_files[i] = os.path.join(abs_path, image_files[i])
    return sorted(image_files)


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

