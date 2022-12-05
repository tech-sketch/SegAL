import numpy as np
import pytest

from segal.datasets import VOCDataset


@pytest.fixture()
def dataset(fixture_image_paths, scope="function"):
    """Dataset instance"""
    image_paths = [fixture_image_paths[0]]
    mask_paths = [fixture_image_paths[1]]
    classes = ["car"]
    dataset = VOCDataset(
        image_paths, mask_paths, classes=classes, preprocessing=None, augmentation=None
    )
    return dataset


class TestVOCDataset:
    """Test VOCDataset class"""

    def test_getitem(self, dataset, processed_images) -> None:
        """Test VOCDataset.__getitem__ function"""
        # Act
        image, mask = dataset[0]

        # Assert
        assert np.array_equal(image, processed_images[0]) is True
        assert np.array_equal(mask, processed_images[1]) is True

    def test_len_(self, dataset) -> None:
        """Test VOCDataset.__len__"""
        # Act
        length = len(dataset)

        # Assert
        assert length == 1
