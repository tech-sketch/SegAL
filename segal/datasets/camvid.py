from typing import Callable, List, Optional, Tuple

import albumentations as albu
import cv2
import numpy as np
from albumentations import BaseCompose
from torch import Tensor
from torch.utils.data import Dataset


def get_training_augmentation() -> BaseCompose:
    """Set up training augmentation workflow.

    Returns:
        BaseCompose: The augmentation transform.
    """
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
        ),
        albu.PadIfNeeded(
            min_height=320, min_width=320, always_apply=True, border_mode=0
        ),
        albu.RandomCrop(height=320, width=320, always_apply=True),
        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation() -> BaseCompose:
    """Set up training augmentation workflow.

    Returns:
        BaseCompose: The augmentation transform.

    PS:
        Add paddings to make image shape divisible by 32
    """
    test_transform = [albu.PadIfNeeded(384, 480)]
    return albu.Compose(test_transform)


def to_tensor(x: Tensor, **kwargs) -> Tensor:
    """Convert to tensor.

    Args:
        x (Tensor): Input image tensor.

    Returns:
        Tensor: Image tensor after transposed.
    """
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn: Callable) -> BaseCompose:
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        BaseCompose: The preprocess transform.
    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


class CamvidDataset(Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        image_paths (List[str]): path to images
        mask_paths (List[str]): path to segmentation masks
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transformation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = [
        "sky",
        "building",
        "pole",
        "road",
        "pavement",
        "tree",
        "signsymbol",
        "fence",
        "car",
        "pedestrian",
        "bicyclist",
        "unlabelled",
    ]

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        classes: Optional[str] = None,
        augmentation: Optional[BaseCompose] = None,
        preprocessing: Optional[BaseCompose] = None,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i) -> Tuple[Tensor, Tensor]:
        """Get data by index.

        Args:
            i (int): Index.

        Returns:
            Tensor, Tensor: image and mask tensor.
        """

        # read data
        image = cv2.imread(self.image_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

    def __len__(self) -> int:
        """return size of dataset.

        Returns:
            int: size of dataset.
        """
        return len(self.image_paths)
