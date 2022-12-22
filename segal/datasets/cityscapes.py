from typing import List, Optional, Tuple

import cv2
import numpy as np
from albumentations import BaseCompose
from torch import Tensor

from segal.datasets.base_dataset import BaseDataset


class CityscapesDataset(BaseDataset):
    """Cityscapes Dataset.

    Args:
        image_paths (List[str]): path to images
        mask_paths (List[str]): path to segmentation masks
        class_values ([List[int]]): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transformation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """

    mapping = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 1,
        8: 2,
        9: 0,
        10: 0,
        11: 3,
        12: 4,
        13: 5,
        14: 0,
        15: 0,
        16: 0,
        17: 6,
        18: 0,
        19: 7,
        20: 8,
        21: 9,
        22: 10,
        23: 11,
        24: 12,
        25: 13,
        26: 14,
        27: 15,
        28: 16,
        29: 0,
        30: 0,
        31: 17,
        32: 18,
        33: 19,
        -1: 0,
    }

    # The values above are remapped to the following
    CLASSES = [
        "unlabeled",
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic_light",
        "traffic_sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        class_values: List[int],
        augmentation: Optional[BaseCompose] = None,
        preprocessing: Optional[BaseCompose] = None,
    ):
        super(CityscapesDataset, self).__init__(
            image_paths,
            mask_paths,
            class_values,
            augmentation,
            preprocessing,
        )

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

        # Remap label
        mask = self.encode_labels(mask)
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

    def encode_labels(self, mask: np.ndarray) -> np.ndarray:
        """Convert 33 classes to 20 classes

        Args:
            mask (np.ndarray): mask with 33 classes id

        Returns:
            np.ndarray: mask with 20 classes id
        """
        label_mask = np.zeros_like(mask)
        for k in self.mapping:
            label_mask[mask == k] = self.mapping[k]
        return label_mask
