from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def fixture_path() -> Path:
    """Path to save file"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_image_paths(fixture_path) -> Path:
    """Paths of images"""
    iamge_path = fixture_path / "image.png"
    label_path = fixture_path / "label.png"
    return (str(iamge_path.absolute()), str(label_path.absolute()))


@pytest.fixture
def processed_images(fixture_image_paths) -> Path:
    """Processed Images"""
    image = cv2.imread(fixture_image_paths[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    class_values = [8]
    mask = cv2.imread(fixture_image_paths[1], 0)
    masks = [(mask == v) for v in class_values]
    mask = np.stack(masks, axis=-1).astype("float")

    return (image, mask)


@pytest.fixture
def processed_images_cityscapes(fixture_image_paths) -> Path:
    """Processed Images"""
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
    image = cv2.imread(fixture_image_paths[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    class_values = [8]

    mask = cv2.imread(fixture_image_paths[1], 0)
    label_mask = np.zeros_like(mask)
    for k in mapping:
        label_mask[mask == k] = mapping[k]

    label_mask = [(label_mask == v) for v in class_values]
    label_mask = np.stack(label_mask, axis=-1).astype("float")

    return (image, label_mask)


@pytest.fixture
def probs() -> np.array:
    """Probs for strategies test"""
    probs = np.array(
        [
            [[[1, 9], [8, 6]], [[6, 6], [7, 0]], [[8, 5], [5, 7]]],
            [[[1, 9], [2, 3]], [[1, 9], [7, 9]], [[7, 4], [9, 8]]],
            [[[2, 7], [1, 3]], [[3, 8], [3, 2]], [[7, 6], [1, 2]]],
            [[[0, 6], [3, 6]], [[9, 8], [4, 8]], [[9, 3], [1, 5]]],
            [[[7, 6], [5, 1]], [[3, 8], [8, 1]], [[5, 2], [6, 4]]],
        ]
    )  # B,C,H,W = 5, 3, 2, 2

    return probs
