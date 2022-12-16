from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from .strategy import Strategy


class VoteSampling(Strategy):
    """Vote Sampling Class.

    Args:
        pool_images (List[str]): List of pool image paths.
        pool_labels (List[str]): List of pool label paths.
        val_images (List[str]): List of validation image paths.
        val_labels (List[str]): List of validation label paths.
        test_images (List[str]): List of test image paths.
        test_labels (List[str]): List of test label paths.
        idxs_lb (np.ndarray): Array of bool type to record labeled data.
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
        idxs_lb: np.ndarray,
        model_params: dict,
        dataset: Dataset,
        dataset_params: dict,
    ):
        super(VoteSampling, self).__init__(
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
    def get_topk_idxs(scores: np.ndarray, k: int) -> np.ndarray:
        """Get top k indices.

        Args:
            scores (np.ndarray): scores of batch data
            k (int): num of data to query

        Returns:
            np.ndarray: index of queried data
        """
        return scores.argsort()[::-1][:k]

    def cal_scores(self, probs: np.ndarray, steps: int = 20) -> np.ndarray:  # B,C,H,W
        """Calculate score by probability.

        Args:
            probs (np.ndarray): probability
            steps (int, optional): num of steps. Defaults to 20.

        Returns:
            np.ndarray: scores
        """
        num_classes = self.model_params["NUM_CLASSES"]
        outputs = torch.FloatTensor(
            probs.shape[0], num_classes, probs.shape[2], probs.shape[3]
        ).fill_(0)

        with torch.no_grad():
            for _ in range(steps):
                outputs[:, :, :, :] += probs

        outputs /= steps
        scores = []
        for i in range(probs.shape[0]):
            entropy_map = torch.FloatTensor(probs.shape[2], probs.shape[3]).fill_(0)
            for c in range(num_classes):
                entropy_map = entropy_map - (
                    outputs[i, c, :, :] * torch.log2(outputs[i, c, :, :] + 1e-12)
                )
            score = np.mean(-np.nansum(entropy_map.cpu().numpy(), axis=0))
            scores.append(score)

        return np.array(scores)
