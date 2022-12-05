import numpy as np

from segal.strategies import RandomSampling


class TestRandomSampling:
    """Test RandomSampling class"""

    def test_query(self) -> None:
        """Test RandomSampling.query function"""
        # Arrange
        np.random.seed(2)
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

        sampling = RandomSampling(
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
        unexpected = [2, 3, 4]

        # Act
        idxs_queried = sampling.query(n=3)

        # Assert
        assert np.array_equal(idxs_queried, unexpected) is False
