from unittest.mock import MagicMock

import numpy as np
import pytest

from segal.strategies import Strategy


@pytest.fixture()
def sampling(scope="function"):
    """Strategy instance"""
    pool_images = ["path1", "path2", "path3", "path4", "path5"]
    pool_labels = ["path1", "path2", "path3", "path4", "path5"]
    val_images = ["path1", "path2", "path3", "path4", "path5"]
    val_labels = ["path1", "path2", "path3", "path4", "path5"]
    test_images = ["path1", "path2", "path3", "path4", "path5"]
    test_labels = ["path1", "path2", "path3", "path4", "path5"]
    n_pool = len(pool_images)
    idxs_lb = np.zeros(n_pool, dtype=bool)  # number of labeled image index
    idxs_tmp = np.arange(n_pool)
    idxs_lb[idxs_tmp[:2]] = True  # [ True,  True, False, False, False]
    DatasetClass = None
    dataset_params = None
    model_params = {"NUM_CLASSES": 3}

    sampling = Strategy(
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
    return sampling


class TestStrategy:
    """Test Strategy class"""

    def test_update(self, sampling) -> None:
        """Test Strategy.update"""
        # Arrange
        idxs_lb = np.array([4, 5, 1, 6])

        # Act
        sampling.update(idxs_lb)

        # Assert
        assert np.array_equal(idxs_lb, sampling.idxs_lb) is True

    def test_query(self, sampling) -> None:
        """Test Strategy.query"""
        # Arrange
        sampling.query = MagicMock()

        # Act
        sampling.query()

        # Assert
        assert sampling.query.called

    def test_train(self, sampling) -> None:
        """Test Strategy.train"""
        # Arrange
        sampling.train = MagicMock()

        # Act
        sampling.train()

        # Assert
        assert sampling.train.called

    def test_evaluate(self, sampling) -> None:
        """Test Strategy.evaluate"""
        # Arrange
        sampling.evaluate = MagicMock()

        # Act
        sampling.evaluate()

        # Assert
        assert sampling.evaluate.called

    def test_predict_prob(self, sampling) -> None:
        """Test Strategy.predict_prob"""
        # Arrange
        sampling.predict_prob = MagicMock()

        # Act
        sampling.predict_prob()

        # Assert
        assert sampling.predict_prob.called
