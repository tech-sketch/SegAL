import numpy as np

from .strategy import Strategy


class MarginSampling(Strategy):
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
        super(MarginSampling, self).__init__(
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
        if isinstance(scores, list):
            scores = np.array(scores)
        return scores.argsort()[::-1][:k]

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[
            ~self.idxs_lb
        ]  # reserve the index of unlabeled data
        probs = self.predict_prob(idxs_unlabeled)
        scores = self.cal_scores(probs)
        topk_idxs = self.get_topk_idxs(scores, n)  # index in scores
        idxs_queried = idxs_unlabeled[topk_idxs]  # idxs_queried: index in pool_images
        return idxs_queried

    @staticmethod
    def cal_scores(probs):  # B,C,H,W
        scores = []
        for prob in probs:
            most_confident_scores = np.max(np.squeeze(prob), axis=0)
            ndx = np.indices(prob.shape)
            second_most_confident_scores = prob[prob.argsort(0), ndx[1], ndx[2]][-2]
            margin = most_confident_scores - second_most_confident_scores
            scores.append(np.mean(margin))

        scores = (
            np.array(scores) * -1
        )  # the smaller the better (margin is low) -> Reverse it makes the larger the better
        return scores
