import numpy as np
import pytest

from segal.datasets import CamvidDataset


@pytest.fixture()
def dataset(fixture_image_paths, scope="function"):
    """Dataset instance"""
    image_paths = [fixture_image_paths[0]]
    mask_paths = [fixture_image_paths[1]]
    classes = ["car"]
    dataset = CamvidDataset(
        image_paths, mask_paths, classes=classes, preprocessing=None, augmentation=None
    )
    return dataset


class TestCamvidDataset:
    """Test CamvidDataset class"""

    def test_getitem(self, dataset, processed_images) -> None:
        """Test CamvidDataset.__getitem__ function"""
        # Act
        image, mask = dataset[0]

        # Assert
        assert np.array_equal(image, processed_images[0]) is True
        assert np.array_equal(mask, processed_images[1]) is True

    def test_len_(self, dataset) -> None:
        """Test CamvidDataset.__len__"""
        # Act
        length = len(dataset)

        # Assert
        assert length == 1
