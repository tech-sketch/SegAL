# SegAL

SegAL is a semantice segmentation active learning tool.

## Installation

SegAL is available on PyPI:

`pip install seqal`

SegAL officially supports Python 3.8+.

## Usage

```
python examples/run_al_cycle.py --dataset CamVid  --data_path ./data/CamVid/ --model_name Unet --encoder resnet34 --encoder_weights imagenet --num_classes 12 --strategy LeastConfidence --seed_ratio 0.02 --query_ratio 0.02 --n_epoch 1
```