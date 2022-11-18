import argparse
import math
import os
import random
import ssl
import tarfile
import urllib.request
from datetime import datetime
from glob import glob
from typing import Callable

import albumentations as albu
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations import BaseCompose

from segal import strategies, utils
from segal.datasets import CamvidDataset, VOCDataset, CityscapesDataset

ssl._create_default_https_context = ssl._create_unverified_context

###############################################################################
parser = argparse.ArgumentParser()
# Dataset
parser.add_argument("--dataset", help="dataset", type=str, default="CamVid")
parser.add_argument("--data_path", help="data path", type=str, default="./data/CamVid/")
parser.add_argument("--crop_size", help="crop size", type=int, default=512)

# Model
parser.add_argument(
    "--model_name", help="model - FPN, Unet, DeepLabV3Plus", type=str, default="Unet"
)
parser.add_argument(
    "--encoder",
    help="encoder - resnet34, se_resnext50_32x4d",
    type=str,
    default="resnet34",
)
parser.add_argument(
    "--encoder_weights", help="encoder weights - imagenet", type=str, default="imagenet"
)
parser.add_argument("--num_classes", help="number of classes", type=int, default=12)

# Active Learning Setup
parser.add_argument(
    "--seed_ratio", help="percentage of seed data", type=float, default=0.4
)
parser.add_argument(
    "--query_ratio", help="percentage of query data", type=float, default=0.4
)
parser.add_argument(
    "--n_epoch", help="number of training epochs in each iteration", type=int, default=2
)
parser.add_argument("--random_seed", help="manual random seed", type=int, default=0)

# Strategy
parser.add_argument(
    "--strategy",
    help="acquisition algorithm, - RandomSampling, LeastConfidence, MarginSampling, EntropySampling",
    type=str,
    default="RandomSampling",
)

# Mode
parser.add_argument("--full", help="train model on full data", action="store_true")

args = parser.parse_args()


# Seed
random_seed = args.random_seed
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=True)


# Get data
DATASET = args.dataset
DATA_DIR = args.data_path

# load repo with data if it is not exists
if DATASET == "CamVid":
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
        print("Loading data...")
        os.system("git clone https://github.com/alexgkendall/SegNet-Tutorial ./data")
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

    # If "./data/VOC2012_trainval" not exist, donwload tar file
    if not os.path.exists(DATA_DIR):
        print("Loading data...")
        trainval_dir = "VOC2012_trainval"

        trainval_url = (
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
        )

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
    with open(pool_file, "r") as f:
        val_file_names = [x.strip() for x in f.readlines()]
    val_images = [
        trainval_path + f"JPEGImages/{file_name}.jpg" for file_name in val_file_names
    ]
    val_labels = [
        trainval_path + f"SegmentationClass/{file_name}.png"
        for file_name in val_file_names
    ]

    # If using full mode, there is no test dataset
    # If not using full mode (active learning mode), 10% of training data will be used as validation data.
    # The validation data will be used as test data.
    if args.full:
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

    # # Test on small data
    # pool_images = pool_images[:16]
    # pool_labels = pool_labels[:16]
    # val_images = val_images[:16]
    # val_labels = val_labels[:16]
    # test_images = test_images[:16]
    # test_labels = test_labels[:16]

elif DATASET == "CityScapes":
    images_dir = os.path.join(DATA_DIR, "leftImg8bit")
    labels_dir = os.path.join(DATA_DIR, "gtFine")

    pool_images = sorted(glob(os.path.join(os.path.join(DATA_DIR, "leftImg8bit/train"), "*/*_leftImg8bit.png")))
    pool_labels = sorted(glob(os.path.join(os.path.join(DATA_DIR, "gtFine/train"), "*/*_gtFine_labelIds.png")))

    val_images = sorted(glob(os.path.join(os.path.join(DATA_DIR, "leftImg8bit/val"), "*/*_leftImg8bit.png")))
    val_labels = sorted(glob(os.path.join(os.path.join(DATA_DIR, "gtFine/val"), "*/*_gtFine_labelIds.png")))

    test_images = sorted(glob(os.path.join(os.path.join(DATA_DIR, "leftImg8bit/test"), "*/*_leftImg8bit.png")))
    test_labels = sorted(glob(os.path.join(os.path.join(DATA_DIR, "gtFine/test"), "*/*_gtFine_labelIds.png")))

    # Test on small data
    # pool_images = pool_images[:16]
    # pool_labels = pool_labels[:16]
    # val_images = val_images[:4]
    # val_labels = val_labels[:4]
    # test_images = test_images[:4]
    # test_labels = test_labels[:4]
else:
    raise AttributeError("Dataset is not supported. Please add the custom dataset")

# Output size of dataset
print(f"Number of training data: {len(pool_images)}")
print(f"Number of val data: {len(val_images)}")
print(f"Number of test data: {len(test_images)}")

# Calculate NUM_INIT_LB, NUM_QUERY, NUM_ROUND
if args.full:
    NUM_INIT_LB = len(pool_images)
    NUM_QUERY = 0
    NUM_ROUND = 0
else:
    seed_ratio = args.seed_ratio
    query_ratio = args.query_ratio
    NUM_INIT_LB = math.ceil(len(pool_images) * seed_ratio)
    NUM_QUERY = math.ceil(len(pool_images) * query_ratio)
    NUM_ROUND = math.ceil(len(pool_images) / NUM_QUERY) - 1

print(f"Initialize model with {NUM_INIT_LB} images.")
print(f"Query {NUM_QUERY} images in each round.")
print(f"Run {NUM_ROUND} rounds.")


# Get model
MODEL_NAME = args.model_name
ENCODER = args.encoder
ENCODER_WEIGHTS = args.encoder_weights
NUM_CLASSES = args.num_classes
model_params = {
    "MODEL_NAME": MODEL_NAME,
    "ENCODER": ENCODER,
    "ENCODER_WEIGHTS": ENCODER_WEIGHTS,
    "NUM_CLASSES": NUM_CLASSES,
}
print(f"Model setup: {model_params}")


# Prepare augmentation and preprocess methods
if DATASET == "CamVid":
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

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

    dataset_params = {
        "training_augmentation": get_training_augmentation(),
        "validation_augmentation": get_validation_augmentation(),
        "preprocessing": get_preprocessing(preprocessing_fn),
        "classes": CamvidDataset.CLASSES,
    }
    DatasetClass = CamvidDataset

elif DATASET == "VOC":
    crop_size = args.crop_size
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    train_transform = albu.Compose(
        [
            albu.RandomScale((-0.1, 1.0)),
            albu.PadIfNeeded(
                min_height=crop_size,
                min_width=crop_size,
                always_apply=True,
                border_mode=0,
            ),
            albu.RandomCrop(crop_size, crop_size),
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
                min_height=crop_size,
                min_width=crop_size,
                always_apply=True,
                border_mode=0,
            ),
        ]
    )

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

    dataset_params = {
        "training_augmentation": train_transform,
        "validation_augmentation": val_transform,
        "preprocessing": get_preprocessing(preprocessing_fn),
        "classes": VOCDataset.CLASSES,
    }
    DatasetClass = VOCDataset

elif DATASET == "CityScapes":
    crop_size = args.crop_size
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    train_transform = albu.Compose(
        [
            albu.RandomScale((-0.1, 1.0)),
            albu.PadIfNeeded(
                min_height=crop_size,
                min_width=crop_size,
                always_apply=True,
                border_mode=0,
            ),
            albu.RandomCrop(crop_size, crop_size),
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
                min_height=crop_size,
                min_width=crop_size,
                always_apply=True,
                border_mode=0,
            ),
        ]
    )

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

    dataset_params = {
        "training_augmentation": train_transform,
        "validation_augmentation": val_transform,
        "preprocessing": get_preprocessing(preprocessing_fn),
        "classes": CityscapesDataset.CLASSES,
    }
    DatasetClass = CityscapesDataset


# Set up index recorder
n_pool = len(pool_images)
idxs_lb = np.zeros(n_pool, dtype=bool)  # number of labeled image index
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True


strategy_name = args.strategy
strategy = strategies.__dict__[strategy_name](
    pool_images,
    pool_labels,
    val_images,
    val_labels,
    test_images,
    test_labels,
    idxs_lb,
    model_params,
    DatasetClass,
    dataset_params,
)

time_start = datetime.now()
n_epoch = args.n_epoch

if args.full is True and DATASET == "VOC":
    print("Train on full training data")
    val_performance = strategy.train(n_epoch)
    print(f"val_performance: {val_performance}")
    save_path = f"./output/{DATASET}_{MODEL_NAME}_epoch_{n_epoch}_{ENCODER}_full_val_result.json"
    utils.save_json(val_performance, save_path)

elif args.full is True and DATASET != "VOC":
    print("Train on full training data")
    val_performance = strategy.train(n_epoch)
    test_performance = strategy.evaluate()
    print(f"test_performance: {test_performance}")
    save_path = f"./output/{DATASET}_{MODEL_NAME}_epoch_{n_epoch}_{ENCODER}_full_test_result.json"
    utils.save_json(test_performance, save_path)

else:
    print("Round 0: initialize model")
    val_performance = strategy.train(n_epoch)
    test_performance = strategy.evaluate()
    print(f"test_performance: {test_performance}")

    # Active Learning Cycle
    for round in range(1, NUM_ROUND + 1):
        time_round_start = datetime.now()

        print(f"Round: {round}")
        labeled = len(
            np.arange(n_pool)[idxs_lb]
        )  # Mark the index of seed data as labeled
        print(f"Number of labeled data: {labeled}")
        print(f"Rest of unlabeled data: {n_pool - labeled}")
        if NUM_QUERY > n_pool - labeled:
            NUM_QUERY = n_pool - labeled
        print(f"Number of queried data in this round: {NUM_QUERY}")

        idxs_queried = strategy.query(NUM_QUERY)
        idxs_lb[idxs_queried] = True

        # update labeled data
        strategy.update(idxs_lb)
        print(f"Number of updated labeled data: {sum(idxs_lb)}")

        # retrain model
        print("Retrain model:")
        test_performance = strategy.train(n_epoch)
        print(test_performance)

        time_round_end = datetime.now() - time_round_start
        print(f"This round takes {time_round_end}")
        print()
        print()

    time_end = datetime.now() - time_start
    print(f"All rounds take {time_end}")

    for round, round_log in enumerate(strategy.test_logs):
        print(
            f'Round: {round}, dice loss: {round_log["dice_loss"]}, mIoU: {round_log["iou_score"]}'
        )

    save_path = f"./output/{DATASET}_{MODEL_NAME}_{ENCODER}_{strategy_name}_epochs_{n_epoch}_test_result.json"
    utils.save_json(strategy.test_logs, save_path)
