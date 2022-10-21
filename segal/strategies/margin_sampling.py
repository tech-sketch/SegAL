from typing import List

import numpy as np
from torch.utils.data import Dataset

from .strategy import Strategy


class MarginSampling(Strategy):
    """Margin Sampling Class.

    Args:
        pool_images (List[str]): List of pool image paths.
        pool_labels (List[str]): List of pool label paths.
        val_images (List[str]): List of validation image paths.
        val_labels (List[str]): List of validation label paths.
        test_images (List[str]): List of test image paths.
        test_labels (List[str]): List of test label paths.
        idxs_lb (List[bool]): List of bool type to record labeled data.
        model_params (dict): Model parameters.
                            e.g. model_params = {
                                    "MODEL_NAME": MODEL_NAME,
                                    "ENCODER": ENCODER,
                                    "ENCODER_WEIGHTS": ENCODER_WEIGHTS,
                                    "NUM_CLASSES": NUM_CLASSES,
                                }
        dataset (Dataset): Dataset class.
        dataset_params (dict): Dataset parameters.
                            e.g.     dataset_params = {
                                            "training_augmentation": get_training_augmentation(),
                                            "validation_augmentation": get_validation_augmentation(),
                                            "preprocessing": get_preprocessing(preprocessing_fn),
                                            "classes": CamvidDataset.CLASSES,
                                        }
    """

    def __init__(
        self,
        pool_images: List[str],
        pool_labels: List[str],
        val_images: List[str],
        val_labels: List[str],
        test_images: List[str],
        test_labels: List[str],
        idxs_lb: List[bool],
        model_params: dict,
        dataset: Dataset,
        dataset_params: dict,
    ):
        super(MarginSampling, self).__init__(
            pool_images,
            pool_labels,
            val_images,
            val_labels,
            test_images,
            test_labels,
            idxs_lb,
            model_params,
            dataset,
            dataset_params,
        )

    def get_topk_idxs(self, scores: np.array, k: int) -> List[int]:
        """Get top k indices."""
        if isinstance(scores, list):
            scores = np.array(scores)
        return scores.argsort()[::-1][:k]

    def query(self, n: int) -> List[int]:
        """Query data.

        Args:
            n (int): Number of data to query.

        Returns:
            List[int]: Indices of queried data.
        """
        idxs_unlabeled = np.arange(self.n_pool)[
            ~self.idxs_lb
        ]  # reserve the index of unlabeled data
        probs = self.predict_prob(idxs_unlabeled)
        scores = self.cal_scores(probs)
        topk_idxs = self.get_topk_idxs(scores, n)  # index in scores
        idxs_queried = idxs_unlabeled[topk_idxs]  # idxs_queried: index in pool_images
        return idxs_queried

    @staticmethod
    def cal_scores(probs: np.array) -> np.array:  # B,C,H,W
        """Calculate score by probability.

        Args:
            probs (np.array): Probability.

        Returns:
            np.array: Image score.
        """
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
