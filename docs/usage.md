# Usage

This tutorial explain how SegAL performs active learning cycle for semantic segmentation.


## Run Active Learning Cycle by Command Line

```
python examples/run_al_cycle.py --dataset CamVid  --data_path ./data/CamVid/ --model_name Unet --encoder resnet34 --encoder_weights imagenet --num_classes 12 --strategy LeastConfidence --seed_ratio 0.02 --query_ratio 0.02 --n_epoch 1
```

- `dataset`: which dataset to use, [`CamVid`](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)、[`VOC`](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)、[`CityScapes`](https://www.cityscapes-dataset.com/). The script will download `CamVid` and `VOC` dataset automatically. Notice that `VOC` does not release the test dataset, so we use the validation dataset as test dataset and 10% of training dataset as the validation dataset. The user should download `CityScapes` dataset manually and make user that it contains the `gtFine` and `leftImg8bit` folders. 
- `data_path`: the path where the data store
- `crop_size`: corp size. Corp big size image to train faster. This parameter will be used for `VOC` and `CityScapes` datasets, because the size of these datasets are big.
- `num_classes`: number of classes. If you do not know how many classes, [this](https://github.com/tech-sketch/SegAL/blob/main/examples/utils/dataset_prepare.py) would be helpful.
- `model_name`: name of segmentation model. More model names can be found in [architectures](https://github.com/qubvel/segmentation_models.pytorch#architectures-)
- `encoder`: name of encoder used in model. More encoder names can be found in [encoders](https://github.com/qubvel/segmentation_models.pytorch#encoders-)
- `encoder_weights`: pretrained weights. See [encoder table](https://github.com/qubvel/segmentation_models.pytorch#encoders-) with available weights for each encoder
- `strategy`: name of sampling strategy. Available strategies: `RandomSampling`, `LeastConfidence`, `MarginSampling`, `EntropySampling`, `CealSampling`, `VoteSampling`. You can find the papers for these strategy in [here](https://github.com/cure-lab/deep-active-learning/tree/main#deep-active-learning-strategies)
- `seed_ratio`: percentage of seed data. The  used for initial training. We usually set 0.02, which means take 2% of the training dataset as the seed dataset.
- `query_ratio`: percentage of queried data in each round. We usually set 0.02, which means take 2% of the training dataset in each round.
- `n_epoch`: number of epoch in each round
- `random_seed`: random seed

## Explanation

Below is a active learning workflow that SegAL controls.

![al_cycle_simulation](./images/al_cycle_simulation.png)


To understand what SegAL have done with above command, we will explain the script in this section. 

First, we import libraries and read the parameters from the command line.

```python
import argparse
import math
import os
import ssl
from glob import glob

import numpy as np
import segmentation_models_pytorch as smp

from segal import strategies
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
parser.add_argument("--crop_size", help="crop size", type=int, default=512)

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
    "--strategy", help="acquisition algorithm", type=str, default="EntropySampling"
)


args = parser.parse_args()
```

Then we get dataset and load the images paths. All image data should contain labels. For example, the `CamVid` dataset contains 6 folders. The 3 folders for original training data, validation data and test data. Another 3 folders for the corresponding label data. 


```python

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
```

Next we calculate `NUM_INIT_LB`, `NUM_QUERY`, `NUM_ROUND`.

- `NUM_INIT_LB`: Number of initial labeled data.
- `NUM_QUERY`: Number of queried data in each round.
- `NUM_ROUND`: Number of round.

```python
# Calculate NUM_INIT_LB, NUM_QUERY, NUM_ROUND
seed_ratio = args.seed_ratio
query_ratio = args.query_ratio
NUM_INIT_LB = math.ceil(len(pool_images) * seed_ratio)
NUM_QUERY = math.ceil(len(pool_images) * query_ratio)
NUM_ROUND = math.ceil((1 - seed_ratio) / query_ratio)

print(f"Initialize model with {NUM_INIT_LB} images.")
print(f"Query {NUM_QUERY} images in each round.")
print(f"Run {NUM_ROUND} rounds.")
```

According to the parameters of `model_name`, `encoder`, `encoder_weights`, `num_classes`, we create the model. Available models, encoders, encoder weights can be found in [here](https://github.com/qubvel/segmentation_models.pytorch#-models-)


```python
# Get model
MODEL_NAME = args.model_name
ENCODER = args.encoder
ENCODER_WEIGHTS = args.encoder_weights
NUM_CLASSES = args.num_classes
print(ENCODER, ENCODER_WEIGHTS)
model = smp.__dict__[MODEL_NAME](
    encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=NUM_CLASSES
)
```

Because we are dealing with image data, we have to preprocess it or augmente it. For example, we create the [processing pipeline](../segal/datasets/camvid.py) for `CamVid`. The user can write their own pipeline for the custom dataset.


```python
# if dataset is CamVid
if DATASET == "CamVid":
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    dataset_params = {
        "training_augmentation": get_training_augmentation(),
        "validation_augmentation": get_validation_augmentation(),
        "preprocessing": get_preprocessing(preprocessing_fn),
        "classes": CamvidDataset.CLASSES,
    }
```


In order to remember which data have be stored in labeled data (training + queired data), we use a list of bool `idxs_lb` to save the information. If one element is `True`, it means the data in this index is marked as labeled data.


```python
# Set up index recorder
n_pool = len(pool_images)
idxs_lb = np.zeros(n_pool, dtype=bool)  # number of labeled image index
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True
```

Next we will initialize the `strategy`. Available strategies can be found in [here](../segal/strategies/__init__.py). 


```python
strategy_name = args.strategy
strategy = strategies.__dict__[strategy_name](
    pool_images,
    pool_labels,
    val_images,
    val_labels,
    test_images,
    test_labels,
    idxs_lb,
    model,
    CamvidDataset,
    dataset_params,
)
```

We train the model with specified epochs and print out the performance on test data. 

```python
print("Round 0: initialize model")
n_epoch = args.n_epoch
test_performance = strategy.train(n_epoch)
print(test_performance)
```

After initializing model, we can run the active learning cycle. `NUM_QUERY` will recorder the number of queried data in each round. `strategy.query(NUM_QUERY)` will return the indices of queried data. `strategy.update(idxs_lb)` will update the labeled data, which simulates that adding queired data to labeled data. After that, we call `strategy.train(n_epoch)` to retrain the model.

```python
for round in range(1, NUM_ROUND + 1):
    print(f"Round: {round}")

    labeled = len(np.arange(n_pool)[idxs_lb])  # Mark the index of seed data as labeled
    print(f"Num of labeled data: {labeled}")
    if NUM_QUERY > n_pool - labeled:
        NUM_QUERY = n_pool - labeled

    idxs_queried = strategy.query(NUM_QUERY)
    idxs_lb[idxs_queried] = True

    # update labeled data
    strategy.update(idxs_lb)

    # retrain model
    test_performance = strategy.train(n_epoch)
    print(test_performance)


for round, round_log in enumerate(strategy.test_logs):
    print(
        f'Round: {round}, dice loss: {round_log["dice_loss"]}, mIoU: {round_log["iou_score"]}'
    )

```

`strategy.test_logs` will save the test performance in each round. We can print it out after all rounds run. 