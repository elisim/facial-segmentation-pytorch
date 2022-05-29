import glob
import os

import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

from .utils import *


NUM_LABELS = 6

LABEL_CODES = [
    (255, 0, 0),  # background
    (127, 0, 0),  # hair
    (255, 255, 0),  # skin
    (0, 0, 255),  # eyes
    (0, 255, 255),  # nose
    (0, 255, 0),  # mouth
]

COLOR_MAP = {
    label_code: label_ for label_code, label_ in zip(LABEL_CODES, range(NUM_LABELS))
}


class FacialSegmentationDataset(data.Dataset):
    def __init__(self, root_dir, train, transform=None):
        if not os.path.isdir(root_dir):
            raise ValueError("Please download the dataset first, using `download_dataset.sh`")
        
        self.transform = transform

        if train:
            self.img_files = glob.glob(os.path.join(root_dir, "Train_RGB", "*.bmp"))
            self.mask_files = glob.glob(os.path.join(root_dir, "Train_Labels", "*.bmp"))
        else:
            self.img_files = glob.glob(os.path.join(root_dir, "Test_RGB", "*.bmp"))
            self.mask_files = glob.glob(os.path.join(root_dir, "Test_Labels", "*.bmp"))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        sample = Image.open(img_path), Image.open(mask_path)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_files)


class Resize(object):
    """
    Rescale the image to a given size.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, mask = sample  # PIL images
        resize = transforms.Resize(self.size)
        img, mask = resize(img), resize(mask)
        return img, mask


class MaskToLabel(object):
    def __call__(self, sample):
        img, mask = sample
        img = np.array(img, dtype=np.float32)
        mask = np.array(mask)  # mask.shape = (H, W, 3)
        mask_labels = mask2label(mask, colormap=COLOR_MAP)
        return img, mask_labels


class ToTensor(object):
    def __call__(self, sample):
        img, mask = sample

        # swap color axis because:
        # img is numpy image: H x W x C and torch image: C X H X W
        # mask is 2D tensor: (H x W) so no swap axis
        img = img.transpose((2, 0, 1))
        img, mask = torch.from_numpy(img), torch.from_numpy(mask)
        return img, mask
