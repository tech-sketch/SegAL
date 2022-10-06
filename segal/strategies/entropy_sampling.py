import numpy as np

from .strategy import Strategy


class EntropySampling(Strategy):
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
        super(EntropySampling, self).__init__(
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

    def get_topk_idxs(self, scores, k):
        return scores.argsort()[::-1][:k]

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[
            ~self.idxs_lb
        ]  # reserve the index of unlabeled data
        probs = self.predict_prob(idxs_unlabeled)
        scores = self.cal_scores(probs)  # The larger the better
        topk_idxs = self.get_topk_idxs(scores, n)  # index in scores
        idxs_queried = idxs_unlabeled[topk_idxs]  # idxs_queried: index in pool_images
        return idxs_queried

    @staticmethod
    def cal_scores(probs):  # B,C,H,W
        scores = []
        for i in range(len(probs)):  # one img prob
            entropy = np.mean(
                -np.nansum(np.multiply(probs[i], np.log2(probs[i] + 1e-12)), axis=0)
            )
            scores.append(entropy)
        return np.array(scores)
