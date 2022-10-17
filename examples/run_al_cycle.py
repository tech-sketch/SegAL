import argparse
import math
import os
import random
import ssl
from datetime import datetime
from glob import glob

import numpy as np
import segmentation_models_pytorch as smp
import torch

from segal import strategies, utils
from segal.datasets.camvid import (
    CamvidDataset,
    get_preprocessing,
    get_training_augmentation,
    get_validation_augmentation,
)

ssl._create_default_https_context = ssl._create_unverified_context

###############################################################################
parser = argparse.ArgumentParser()
# Dataset
parser.add_argument("--dataset", help="dataset", type=str, default="CamVid")
parser.add_argument("--data_path", help="data path", type=str, default="./data/CamVid/")

# Model
parser.add_argument(
    "--model_name", help="model - FPN, Unet, DeepLabV3", type=str, default="Unet"
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

# Mode
parser.add_argument("--full", help="train model on full data", type=bool, default=False)

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

args = parser.parse_args()


# Seed
random_seed = args.random_seed
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

# Get data
DATASET = args.dataset
DATA_DIR = args.data_path

# load repo with data if it is not exists
if DATASET == "CamVid" and not os.path.exists(DATA_DIR):
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


# if dataset is CamVid
if DATASET == "CamVid":
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    dataset_params = {
        "training_augmentation": get_training_augmentation(),
        "validation_augmentation": get_validation_augmentation(),
        "preprocessing": get_preprocessing(preprocessing_fn),
        "classes": CamvidDataset.CLASSES,
    }

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
    CamvidDataset,
    dataset_params,
)

time_start = datetime.now()

print("Round 0: initialize model")
n_epoch = args.n_epoch
test_performance = strategy.train(n_epoch)
print(test_performance)

if args.full:
    save_path = f"./output/{DATASET}_{MODEL_NAME}_epoch_{n_epoch}_{ENCODER}_full_test_result.json"
    utils.save_json(test_performance, save_path)


for round in range(1, NUM_ROUND + 1):
    time_round_start = datetime.now()

    print(f"Round: {round}")
    labeled = len(np.arange(n_pool)[idxs_lb])  # Mark the index of seed data as labeled
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
    print(f"This round take {time_round_end}")
    print()
    print()

time_end = datetime.now() - time_start
print(f"All rounds take {time_end}")

for round, round_log in enumerate(strategy.test_logs):
    print(
        f'Round: {round}, dice loss: {round_log["dice_loss"]}, mIoU: {round_log["iou_score"]}'
    )

if not args.full:
    save_path = (
        f"./output/{DATASET}_{MODEL_NAME}_{ENCODER}_{strategy_name}_test_result.json"
    )
    utils.save_json(strategy.test_logs, save_path)
