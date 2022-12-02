from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset

from .entropy_sampling import EntropySampling
from .least_confidence import LeastConfidence
from .margin_sampling import MarginSampling
from .strategy import Strategy


def get_uncertain_samples_idxs(probs: np.ndarray, n: int, criteria: str) -> np.ndarray:
    """
    Get the K most informative samples based on the criteria
    Parameters
    ----------
    probs : np.ndarray
        prediction probability of x_i with dimension (batch x n_class)
    n: int
    criteria: str
        `lc` : least_confidence
        `ms` : margin_sampling
        `en` : entropy
    Returns
    -------
    tuple(np.ndarray, np.ndarray)
    """
    if criteria == "lc":
        scores = LeastConfidence.cal_scores(probs)
        topk_idxs = LeastConfidence.get_topk_idxs(scores, n)
    elif criteria == "ms":
        scores = MarginSampling.cal_scores(probs)
        topk_idxs = MarginSampling.get_topk_idxs(scores, n)
    elif criteria == "en":
        scores = EntropySampling.cal_scores(probs)
        topk_idxs = EntropySampling.get_topk_idxs(scores, n)
    else:
        raise ValueError("criteria {} not found !".format(criteria))
    return topk_idxs


def get_high_confidence_samples(
    probs: np.ndarray, delta: float, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select high confidence samples from `D^U` whose entropy is smaller than
     the threshold
    `delta`.
    Parameters
    ----------
    probs : np.ndarray
        prediction probability of x_i with dimension (batch x n_class)
    delta : float
        threshold
    Returns
    -------
    np.array with dimension (K x 1)  containing the indices of the K
        most informative samples.
    np.array with dimension (K x 1) containing the predicted classes of the
        k most informative samples
    """
    scores, entropy_maps = CealSampling.cal_scores(probs)
    topk_idxs = CealSampling.get_topk_idxs(scores, k)
    pred_class = np.argmax(probs, axis=1)
    for idx, entropy_map in entropy_maps.items():
        entropy_map = np.argmax(entropy_map, axis=0)
        qualified_area = entropy_map <= delta
        if np.any(qualified_area):
            pred_class[idx][qualified_area is False] = 255
    labels = pred_class[topk_idxs]
    return topk_idxs, labels


class CealSampling(Strategy):
    """CEAL Sampling Class.

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
        idxs_lb: np.ndarray,
        model_params: dict,
        dataset: Dataset,
        dataset_params: dict,
        start_entropy_threshold: float = 0.0275,
        entropy_change_per_selection: float = 0.001815,
    ):
        super(CealSampling, self).__init__(
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

        self.current_entropy_threshold = start_entropy_threshold
        self.entropy_change_per_selection = entropy_change_per_selection

    def query(self, n: int, criteria: str = "lc") -> List[int]:
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
        us_idxs = get_uncertain_samples_idxs(probs, n, criteria)
        us_idxs = idxs_unlabeled[us_idxs]  # idxs_queried: index in pool_images
        hcs_idx, _ = get_high_confidence_samples(
            probs, self.current_entropy_threshold, n
        )
        hcs_idx = idxs_unlabeled[hcs_idx]  # idxs_queried: index in pool_images
        idxs_queried = list(set(us_idxs) | set(hcs_idx))
        self.current_entropy_threshold -= self.entropy_change_per_selection
        return idxs_queried

    @staticmethod
    def get_topk_idxs(scores: np.ndarray, k: int) -> np.ndarray:
        """Get top k indices."""
        if isinstance(scores, list):
            scores = np.array(scores)
        return scores.argsort()[::-1][:k]

    @staticmethod
    def cal_scores(probs: np.ndarray) -> Tuple[np.ndarray, dict]:  # B,C,H,W
        """Calculate score by probability.

        Args:
            probs (np.ndarray): Probability.

        Returns:
            np.ndarray: Image score.
        """
        scores = []
        entropy_maps = {}
        for i in range(len(probs)):  # one img prob
            entropy = np.multiply(probs[i], np.log2(probs[i] + 1e-12))
            score = np.mean(-np.nansum(entropy, axis=0))
            scores.append(score)
            entropy_maps[i] = entropy
        return np.array(scores), entropy_maps
