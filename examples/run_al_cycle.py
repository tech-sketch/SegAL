import argparse
import math
import random
from datetime import datetime

import numpy as np
import torch
from utils import dataset_prepare

from segal import strategies, utils

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
FULL_MODE = args.full
# load repo with data if it is not exists
(
    pool_images,
    pool_labels,
    val_images,
    val_labels,
    test_images,
    test_labels,
) = dataset_prepare.get_paths(DATASET, DATA_DIR, FULL_MODE)

# Test on small data
# pool_images = pool_images[:16]
# pool_labels = pool_labels[:16]
# val_images = val_images[:4]
# val_labels = val_labels[:4]
# test_images = test_images[:4]
# test_labels = test_labels[:4]

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
CROP_SIZE = args.crop_size
dataset_params, DatasetClass = dataset_prepare.get_dataset(
    DATASET, ENCODER, ENCODER_WEIGHTS, CROP_SIZE
)

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
