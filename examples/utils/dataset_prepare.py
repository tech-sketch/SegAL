import os
import ssl
import tarfile
import urllib.request
from datetime import datetime
from glob import glob
from typing import Callable, List

import albumentations as albu
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations import BaseCompose

from segal.datasets import CamvidDataset, CityscapesDataset, VOCDataset

ssl._create_default_https_context = ssl._create_unverified_context


CAMVID_CLASSES = [
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


CITYSCAPES_CLASSES = [
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

VOC_CLASSES = [
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


def to_tensor(x: torch.Tensor, **kwargs) -> torch.Tensor:
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


def get_classes(dataset: str) -> List[int]:
    """Convert str names to class values on masks

    Args:
        dataset (str): dataset

    Returns:
        List[int]: class values
    """
    if dataset == "CamVid":
        class_values = [CAMVID_CLASSES.index(cls.lower()) for cls in CAMVID_CLASSES]
    elif dataset == "VOC":
        class_values = [VOC_CLASSES.index(cls.lower()) for cls in VOC_CLASSES]
    elif dataset == "CityScapes":
        class_values = [
            CITYSCAPES_CLASSES.index(cls.lower()) for cls in CITYSCAPES_CLASSES
        ]
    return class_values


def get_paths(DATASET: str, DATA_DIR: str, FULL_MODE: bool) -> List[List[str]]:
    """Get image paths of train, val, test

    Args:
        DATASET (str): dataset
        DATA_DIR (str): directory of dataset
        FULL_MODE (bool): train on full dataset or nor

    Raises:
        AttributeError: raise error if dataset is not supported

    Returns:
        List[List[str]]: paths of train, val, test
    """
    if DATASET == "CamVid":
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
            print("Loading data...")
            os.system(
                "git clone https://github.com/alexgkendall/SegNet-Tutorial ./data"
            )
            print("Done!")

        # Load data paths
        pool_images = sorted(glob(os.path.join(DATA_DIR, "train/*")))
        pool_labels = sorted(glob(os.path.join(DATA_DIR, "trainannot/*")))

        val_images = sorted(glob(os.path.join(DATA_DIR, "val/*")))
        val_labels = sorted(glob(os.path.join(DATA_DIR, "valannot/*")))

        test_images = sorted(glob(os.path.join(DATA_DIR, "test/*")))
        test_labels = sorted(glob(os.path.join(DATA_DIR, "testannot/*")))

    elif DATASET == "VOC":
        data_dir = "./data/"
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        # If "./data/VOC2012_trainval" not exist, download tar file
        if not os.path.exists(DATA_DIR):
            print("Loading data...")
            trainval_dir = "VOC2012_trainval"

            trainval_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

            trainval_target_path = os.path.join(data_dir, trainval_dir)
            trainval_tar_path = os.path.join(data_dir, "VOC2012_trainval.tar")

            # Download trainval
            time_start = datetime.now()
            if not os.path.exists(trainval_target_path):
                urllib.request.urlretrieve(trainval_url, trainval_tar_path)
                tar = tarfile.TarFile(trainval_tar_path)
                tar.extractall(trainval_target_path)
                tar.close()
                time_end = datetime.now() - time_start
                print(f"Download trianval: {time_end}")
        else:
            trainval_target_path = DATA_DIR  # "./data/VOC2012_trainval"

        # Load training data paths
        trainval_path = os.path.join(trainval_target_path, "VOCdevkit/VOC2012/")
        pool_file = trainval_path + "ImageSets/Segmentation/train.txt"
        with open(pool_file, "r") as f:
            pool_file_names = [x.strip() for x in f.readlines()]
        pool_images = [
            os.path.join(trainval_path, f"JPEGImages/{file_name}.jpg")
            for file_name in pool_file_names
        ]
        pool_labels = [
            os.path.join(trainval_path, f"SegmentationClass/{file_name}.png")
            for file_name in pool_file_names
        ]

        # Load val data paths
        val_file = trainval_path + "ImageSets/Segmentation/val.txt"
        with open(val_file, "r") as f:
            val_file_names = [x.strip() for x in f.readlines()]
        val_images = [
            trainval_path + f"JPEGImages/{file_name}.jpg"
            for file_name in val_file_names
        ]
        val_labels = [
            trainval_path + f"SegmentationClass/{file_name}.png"
            for file_name in val_file_names
        ]

        # If using full mode, there is no test dataset
        # If not using full mode (active learning mode), 10% of training data will be used as validation data.
        # The validation data will be used as test data.
        if FULL_MODE:
            test_images = []  # placeholder
            test_labels = []  # placeholder
        else:
            message = (
                "Parameter 'full' have to be True. Because PASCAL VOC 2012 test data does not have ground truth,"
                " it can not evaluate test data in active learning cycle. "
                "SegAL only support train the model on full training dataset."
            )
            print(message)
            test_images = val_images.copy()
            test_labels = val_labels.copy()

            pool_images_all = pool_images.copy()
            pool_labels_all = pool_labels.copy()
            n_pool = len(pool_images_all)
            idxs_train = np.arange(n_pool)
            np.random.shuffle(idxs_train)

            idxs_val = idxs_train[: int(n_pool * 0.1)]
            idxs_train = idxs_train[int(n_pool * 0.1) :]

            val_images = [pool_images_all[x] for x in idxs_val]
            val_labels = [pool_labels_all[x] for x in idxs_val]

            pool_images = [pool_images_all[x] for x in idxs_train]
            pool_labels = [pool_labels_all[x] for x in idxs_train]

    elif DATASET == "CityScapes":
        pool_images = sorted(
            glob(
                os.path.join(
                    os.path.join(DATA_DIR, "leftImg8bit/train"), "*/*_leftImg8bit.png"
                )
            )
        )
        pool_labels = sorted(
            glob(
                os.path.join(
                    os.path.join(DATA_DIR, "gtFine/train"), "*/*_gtFine_labelIds.png"
                )
            )
        )

        val_images = sorted(
            glob(
                os.path.join(
                    os.path.join(DATA_DIR, "leftImg8bit/val"), "*/*_leftImg8bit.png"
                )
            )
        )
        val_labels = sorted(
            glob(
                os.path.join(
                    os.path.join(DATA_DIR, "gtFine/val"), "*/*_gtFine_labelIds.png"
                )
            )
        )

        test_images = sorted(
            glob(
                os.path.join(
                    os.path.join(DATA_DIR, "leftImg8bit/test"), "*/*_leftImg8bit.png"
                )
            )
        )
        test_labels = sorted(
            glob(
                os.path.join(
                    os.path.join(DATA_DIR, "gtFine/test"), "*/*_gtFine_labelIds.png"
                )
            )
        )

    else:
        raise AttributeError("Dataset is not supported. Please add the custom dataset")

    return [pool_images, pool_labels, val_images, val_labels, test_images, test_labels]


def get_dataset(DATASET: str, ENCODER: str, ENCODER_WEIGHTS: str, CROP_SIZE: int):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    train_transform = albu.Compose(
        [
            albu.RandomScale((-0.1, 1.0)),
            albu.PadIfNeeded(
                min_height=CROP_SIZE,
                min_width=CROP_SIZE,
                always_apply=True,
                border_mode=0,
            ),
            albu.RandomCrop(CROP_SIZE, CROP_SIZE),
            albu.HorizontalFlip(),
            albu.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            # ToTensorV2(),
        ]
    )
    val_transform = albu.Compose(
        [
            albu.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            albu.PadIfNeeded(
                min_height=CROP_SIZE,
                min_width=CROP_SIZE,
                always_apply=True,
                border_mode=0,
            ),
        ]
    )

    if DATASET == "CamVid":

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

        dataset_params = {
            "training_augmentation": get_training_augmentation(),
            "validation_augmentation": get_validation_augmentation(),
            "preprocessing": get_preprocessing(preprocessing_fn),
            "class_values": get_classes(DATASET),
        }
        DatasetClass = CamvidDataset

    elif DATASET == "VOC":
        dataset_params = {
            "training_augmentation": train_transform,
            "validation_augmentation": val_transform,
            "preprocessing": get_preprocessing(preprocessing_fn),
            "class_values": get_classes(DATASET),
        }
        DatasetClass = VOCDataset

    elif DATASET == "CityScapes":
        dataset_params = {
            "training_augmentation": train_transform,
            "validation_augmentation": val_transform,
            "preprocessing": get_preprocessing(preprocessing_fn),
            "class_values": get_classes(DATASET),
        }
        DatasetClass = CityscapesDataset

    return dataset_params, DatasetClass
