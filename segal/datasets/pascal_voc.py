from typing import List, Optional

from albumentations import BaseCompose

from segal.datasets.base_dataset import BaseDataset


class VOCDataset(BaseDataset):
    """VOCDataset Dataset.

    Args:
        image_paths (List[str]): path to images
        mask_paths (List[str]): path to segmentation masks
        class_values ([List[int]]): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transformation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """

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
        class_values: List[int],
        augmentation: Optional[BaseCompose] = None,
        preprocessing: Optional[BaseCompose] = None,
    ):
        super(VOCDataset, self).__init__(
            image_paths,
            mask_paths,
            class_values,
            augmentation,
            preprocessing,
        )
