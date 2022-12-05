from unittest.mock import MagicMock

import numpy as np

from segal.strategies import LeastConfidence


class TestLeastConfidence:
    """Test LeastConfidence class"""

    def test_cal_scores_return_correct_result(self, probs) -> None:
        """Test LeastConfidence.cal_scores function return correct result if log_probability runs after prediction"""
        # Arrange
        expected = np.array([-8.0, -8.5, -5.25, -7.25, -6.75])

        # Act
        scores = LeastConfidence.cal_scores(probs)

        # Assert
        assert np.array_equal(scores, expected) is True

    def test_get_topk_idxs(self) -> None:
        """Test LeastConfidence.get_topk_idxs"""
        # Arrange
        scores = np.array([4, 5, 1, 6])
        k = 3
        expected = np.array([3, 1, 0])

        # Act
        topk_idxs = LeastConfidence.get_topk_idxs(scores, k)

        # Assert
        assert np.array_equal(topk_idxs, expected) is True

    def test_query(self) -> None:
        """Test LeastConfidence.query function"""
        # Arrange
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
        model_params = None

        sampling = LeastConfidence(
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
        sampling.predict_prob = MagicMock(return_value=None)
        sampling.cal_scores = MagicMock(return_value=np.array([3, 4, 1]))
        expected = np.array([3, 2])

        # Act
        idxs_queried = sampling.query(n=2)

        # Assert
        assert np.array_equal(idxs_queried, expected) is True
