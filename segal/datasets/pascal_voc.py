from typing import List, Optional

import cv2
import numpy as np
from albumentations import BaseCompose
from torch.utils.data import Dataset

from segal.utils import is_list_of_strings


class VOCDataset(Dataset):
    """VOCDataset Dataset.

    Args:
        image_paths (List[str]): path to images
        mask_paths (List[str]): path to segmentation masks
        class_values (Optional[List[str]]): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transformation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """

    # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html
    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boad",
        "bottle",
        "bus",
        "cat",
        "car",
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
        self,
        image_paths: List[str],
        mask_paths: List[str],
        classes: Optional[List[str]] = None,
        augmentation: Optional[BaseCompose] = None,
        preprocessing: Optional[BaseCompose] = None,
    ):
        if not all([is_list_of_strings(image_paths), is_list_of_strings(mask_paths)]):
            raise TypeError("Images paths must be a list of string!")
        if not is_list_of_strings(classes):
            raise TypeError("classes must be a numpy array!")
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        """return size of dataset.

        Returns:
            int: size of dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, i):
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
