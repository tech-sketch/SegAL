import numpy as np

from .strategy import Strategy


class RandomSampling(Strategy):
    def __init__(
        self,
        pool_images,
        pool_labels,
        val_images,
        val_labels,
        test_images,
        test_labels,
        idxs_lb,
        model,
        dataset,
        dataset_params,
    ):
        super(RandomSampling, self).__init__(
            pool_images,
            pool_labels,
            val_images,
            val_labels,
            test_images,
            test_labels,
            idxs_lb,
            model,
            dataset,
            dataset_params,
        )

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[
            ~self.idxs_lb
        ]  # reserve the index of unlabeled data
        np.random.shuffle(idxs_unlabeled)
        idxs_queried = idxs_unlabeled[:n]  # idxs_queried: index in pool_images
        return idxs_queried
