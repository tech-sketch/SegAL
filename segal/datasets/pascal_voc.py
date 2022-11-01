from typing import Callable, List, Optional, Tuple

import albumentations as albu
import cv2
import numpy as np
from albumentations import BaseCompose
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


class VOCDataset(Dataset):

    # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html
    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boad",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "dining table",
        "dog",
        "horse",
        "motor bike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
        "void",
    ]
    cmap = voc_cmap()

    def __init__(
        self,
        images_path,
        masks_path,
        classes,
        augmentation=None,
        preprocessing: Optional[BaseCompose] = None,
    ):

        self.images_path = images_path
        self.masks_path = masks_path
        # convert str names to class values on masks
        # self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, i):

        image = Image.open(self.images_path[i])
        mask = Image.open(self.masks_path[i])

        # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype("float")

        if self.augmentation:
            image, mask = self.augmentation(image, mask)

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]


# def get_augmentation(phase):
#     if phase == "train":
#         train_transform = [
#             albu.HorizontalFlip(p=0.5),
#             albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
#             albu.RandomBrightnessContrast()
#         ]
#         return albu.Compose(train_transform)

#     if phase=="val" or phase=="test":
#         return None

# def to_tensor(x, **kwargs):
#     return x.transpose(2, 0, 1).astype("float32")

# def get_preprocessing(preprocessing_fn):
#     _transform = [
#         albu.Lambda(image=preprocessing_fn),
#         albu.Lambda(image=to_tensor, mask=to_tensor),
#     ]
#     return albu.Compose(_transform)

# def crop_to_square(image):
#     size = min(image.size)
#     left, upper = (image.width - size) // 2, (image.height - size) // 2
#     right, bottom = (image.width + size) // 2, (image.height + size) // 2
#     return image.crop((left, upper, right, bottom))


class VOCDataset(Dataset):

    # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html
    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boad",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "dining table",
        "dog",
        "horse",
        "motor bike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
    ]

    def __init__(
        self, image_paths, mask_paths, classes, augmentation=None, preprocessing=None
    ):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.image_paths[i])  # Read as BGR, 0~255
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB, 0~1
        mask = cv2.imread(self.mask_paths[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype("float")

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask


# class VOCTestDataset(Dataset):
#     """VOC Dataset. Read images, apply augmentation and preprocessing transformations.

#     Args:
#         images_dir (str): path to images folder
#         augmentation (albumentations.Compose): data transfromation pipeline
#             (e.g. flip, scale, etc.)
#         preprocessing (albumentations.Compose): data preprocessing
#             (e.g. noralization, shape manipulation, etc.)
#     """

#     def __init__(
#             self,
#             images_path,
#             masks_path,
#             augmentation=None,
#             preprocessing=None,
#     ):
#         self.images_path = images_path
#         self.masks_path = masks_path

#         self.augmentation = augmentation
#         self.preprocessing = preprocessing

#     def __getitem__(self, i):

#         # read data
#         image = cv2.imread(self.images_path[i])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # apply augmentations
#         if self.augmentation:
#             sample = self.augmentation(image=image)
#             image = sample['image']

#         # apply preprocessing
#         if self.preprocessing:
#             sample = self.preprocessing(image=image)
#             image = sample['image']

#         return image, self.masks_path[i]

#     def __len__(self):
#         return len(self.images_path)
