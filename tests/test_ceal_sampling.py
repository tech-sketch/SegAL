from unittest.mock import MagicMock

import numpy as np
import pytest

from segal.strategies import CealSampling
from segal.strategies.ceal_sampling import (
    get_high_confidence_samples,
    get_uncertain_samples_idxs,
)


@pytest.mark.parametrize(
    "criteria,excepted",
    [("lc", np.array([2, 4])), ("ms", np.array([3, 0])), ("en", np.array([2, 4]))],
)
def test_get_uncertain_samples_idxs(probs, criteria, excepted) -> None:
    """Test get_uncertain_samples_idxs function"""
    # Act
    topk_idxs = get_uncertain_samples_idxs(probs, 2, criteria)

    # Assert
    assert np.array_equal(topk_idxs, excepted) is True


def test_get_high_confidence_samples(probs) -> None:
    """Test get_high_confidence_samples function"""
    # Arrange
    delta = 0.0275
    k = 2
    expected = (np.array([2, 4]), np.array([[[2, 1], [1, 0]], [[0, 1], [1, 2]]]))

    # Act
    topk_idxs, labels = get_high_confidence_samples(probs, delta, k)

    # Assert
    assert np.array_equal(topk_idxs, expected[0]) is True
    assert np.array_equal(labels, expected[1]) is True


class TestCealSampling:
    """Test CealSampling class"""

    def test_cal_scores_return_correct_result(self, probs) -> None:
        """Test CealSampling.cal_scores function return correct result if log_probability runs after prediction"""
        # Arrange
        scores_expected = np.array([-46.3952, -48.0438, -24.7694, -41.2994, -34.1613])

        # Act
        scores, entropy_maps = CealSampling.cal_scores(probs)
        scores = np.around(scores, decimals=4)

        # Assert
        assert np.array_equal(scores, scores_expected) is True
        assert len(entropy_maps) == 5 and entropy_maps[0].shape == (3, 2, 2)

    def test_get_topk_idxs(self) -> None:
        """Test CealSampling.get_topk_idxs"""
        # Arrange
        scores = np.array([4, 5, 1, 6])
        k = 3
        expected = np.array([3, 1, 0])

        # Act
        topk_idxs = CealSampling.get_topk_idxs(scores, k)

        # Assert
        assert np.array_equal(topk_idxs, expected) is True

    def test_query(self, probs) -> None:
        """Test CealSampling.query function"""
        # Arrange
        pool_images = ["path1", "path2", "path3", "path4", "path5", "path6", "path7"]
        pool_labels = ["path1", "path2", "path3", "path4", "path5", "path6", "path7"]
        val_images = ["path1", "path2", "path3", "path4", "path5", "path6", "path7"]
        val_labels = ["path1", "path2", "path3", "path4", "path5", "path6", "path7"]
        test_images = ["path1", "path2", "path3", "path4", "path5", "path6", "path7"]
        test_labels = ["path1", "path2", "path3", "path4", "path5", "path6", "path7"]
        n_pool = len(pool_images)
        idxs_lb = np.zeros(n_pool, dtype=bool)  # number of labeled image index
        idxs_tmp = np.arange(n_pool)
        idxs_lb[idxs_tmp[:2]] = True  # [ True,  True, False, False, False]
        DatasetClass = None
        dataset_params = {}
        model_params = {}

        sampling = CealSampling(
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
        sampling.predict_prob = MagicMock(return_value=probs)
        expected = np.array([4, 6])

        # Act
        idxs_queried = sampling.query(n=2)

        # Assert
        assert np.array_equal(idxs_queried, expected) is True
