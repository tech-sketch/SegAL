from typing import List, Optional, Tuple

import albumentations as albu
import cv2
import numpy as np
from albumentations import BaseCompose
from torch import Tensor
from torch.utils.data import Dataset


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

    def __len__(self) -> int:
        """return size of dataset.

        Returns:
            int: size of dataset.
        """
        return len(self.image_paths)
