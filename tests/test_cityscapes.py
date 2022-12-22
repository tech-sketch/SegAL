import numpy as np
import pytest

from segal.datasets import CityscapesDataset


@pytest.fixture()
def dataset(fixture_image_paths, scope="function"):
    """Dataset instance"""
    image_paths = [fixture_image_paths[0]]
    mask_paths = [fixture_image_paths[1]]
    class_values = [8]
    dataset = CityscapesDataset(
        image_paths,
        mask_paths,
        class_values=class_values,
        preprocessing=None,
        augmentation=None,
    )
    return dataset


class TestCityscapesDataset:
    """Test CityscapesDataset class"""

    def test_getitem(self, dataset, processed_images_cityscapes) -> None:
        """Test CityscapesDataset.__getitem__ function"""
        # Act
        image, mask = dataset[0]

        # Assert
        assert np.array_equal(image, processed_images_cityscapes[0]) is True
        assert np.array_equal(mask, processed_images_cityscapes[1]) is True

    def test_len_(self, dataset) -> None:
        """Test CityscapesDataset.__len__"""
        # Act
        length = len(dataset)

        # Assert
        assert length == 1
