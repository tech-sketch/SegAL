import cv2
import numpy as np
from torch.utils.data import Dataset


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
