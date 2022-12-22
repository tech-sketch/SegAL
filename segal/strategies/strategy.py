from pathlib import Path
from typing import List, Optional

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from segmentation_models_pytorch import utils
from torch.utils.data import DataLoader, Dataset

from segal.utils import is_array_of_bools, is_list_of_strings


class Strategy:
    """Base Strategy class.

    Attributes:
        pool_images (List[str]): List of pool image paths.
        pool_labels (List[str]): List of pool label paths.
        val_images (List[str]): List of validation image paths.
        val_labels (List[str]): List of validation label paths.
        test_images (List[str]): List of test image paths.
        test_labels (List[str]): List of test label paths.
        idxs_lb (np.ndarray): Array of bool type to record labeled data.
        model_params (dict): Model parameters.
        model_params (dict): Model parameters.
        dataset (Dataset): Dataset class.
        dataset_params (dict): Dataset parameters.
        val_dataset (Union[Dataset, None]): Validation dataset.
        test_dataset (Union[Dataset, None]): Validation dataset.
        n_pool (int): Number of pool data.
        device (str): "cuda" or "cpu".
        best_model (torch.nn.Module): Segmentation model.
        train_logs (List[dict]): List of training logs.
        val_logs (List[dict]): List of validation logs.
        test_logs (List[dict]): List of test logs.
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
        """Initialize Strategy class.

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
        if not all(
            [
                is_list_of_strings(pool_images),
                is_list_of_strings(pool_labels),
                is_list_of_strings(val_images),
                is_list_of_strings(val_labels),
                is_list_of_strings(test_images),
                is_list_of_strings(test_images),
                is_list_of_strings(test_labels),
            ]
        ):
            raise TypeError("Images paths must be a list of string!")
        if not is_array_of_bools(idxs_lb):
            raise TypeError("idxs_lb must be a numpy array of bool!")
        if not isinstance(dataset_params, dict):
            raise TypeError("dataset_params must be a dict!")
        if not isinstance(model_params, dict):
            raise TypeError("model_params must be a numpy array!")

        self.pool_images: List[str] = pool_images
        self.pool_labels: List[str] = pool_labels
        self.val_images: List[str] = val_images
        self.val_labels: List[str] = val_labels
        self.test_images: List[str] = test_images
        self.test_labels: List[str] = test_labels
        self.idxs_lb: np.ndarray = idxs_lb
        self.dataset: Dataset = dataset
        self.dataset_params: dict = dataset_params
        self.model_params: dict = model_params
        self.val_dataset = None
        self.test_dataset = None
        self.n_pool = len(pool_images)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.best_model: torch.nn.Module = torch.nn.Module()
        self.train_logs: List[dict] = []
        self.val_logs: List[dict] = []
        self.test_logs: List[dict] = []

    def query(self, n: int) -> List[int]:
        """Query data

        Args:
            n (int): num of query data

        Returns:
            List[int]: index of query data
        """
        pass

    def update(self, idxs_lb: np.ndarray) -> None:
        """Update labeled data index

        Args:
            idxs_lb (np.ndarray): array of bool type to record labeled data.
        """
        self.idxs_lb = idxs_lb

    def train(
        self,
        n_epoch: int = 10,
        activation: str = "softmax2d",
        save_path: str = "output",
    ) -> dict:
        """Train model and return performance on test data

        Args:
            n_epoch (int, optional): num of epochs. Defaults to 10.
            activation (str, optional): activation function. Defaults to "softmax2d".
            save_path (str, optional): save path of result. Defaults to "output".

        Returns:
            dict: a dict with log of performance
        """
        model = smp.__dict__[self.model_params["MODEL_NAME"]](
            encoder_name=self.model_params["ENCODER"],
            encoder_weights=self.model_params["ENCODER_WEIGHTS"],
            classes=self.model_params["NUM_CLASSES"],
        )
        model.to(self.device)

        base_path = Path(save_path)
        if not base_path.exists():
            base_path.mkdir(exist_ok=True, parents=True)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]

        train_images = [self.pool_images[idx] for idx in idxs_train]
        train_labels = [self.pool_labels[idx] for idx in idxs_train]

        train_dataset: Dataset = self.dataset(
            train_images,
            train_labels,
            augmentation=self.dataset_params["training_augmentation"],
            preprocessing=self.dataset_params["preprocessing"],
            class_values=self.dataset_params["class_values"],
        )  # type: ignore[operator]

        if self.val_dataset is None:
            valid_dataset = self.dataset(
                self.val_images,
                self.val_labels,
                augmentation=self.dataset_params["validation_augmentation"],
                preprocessing=self.dataset_params["preprocessing"],
                class_values=self.dataset_params["class_values"],
            )  # type: ignore[operator]
            self.valid_dataset = valid_dataset
        else:
            valid_dataset = self.valid_dataset

        if self.device == "cuda":
            train_loader = DataLoader(
                train_dataset, batch_size=4, shuffle=True, num_workers=4
            )
            valid_loader = DataLoader(
                valid_dataset, batch_size=1, shuffle=False, num_workers=2
            )
        else:
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

        loss = utils.losses.DiceLoss(activation=activation)
        metrics = [
            utils.metrics.IoU(threshold=0.5, activation=activation),
        ]

        optimizer = torch.optim.Adam(
            [
                dict(params=model.parameters(), lr=0.0001),
            ]
        )

        train_epoch = utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=self.device,
            verbose=True,
        )

        valid_epoch = utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device=self.device,
            verbose=True,
        )

        max_score = 0
        for i in range(0, n_epoch):

            print("\nEpoch: {}".format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            self.train_logs.append(train_logs)
            self.val_logs.append(valid_logs)

            # Save best model
            if max_score < valid_logs["iou_score"]:
                max_score = valid_logs["iou_score"]
                self.best_model = model
                model_path = base_path / "best_model.pth"
                torch.save(model, model_path)
                print("Model saved!")

        return valid_logs

    def evaluate(
        self, activation: str = "softmax2d", check_point: Optional[str] = None
    ):
        """Evaluate on dataset

        Args:
            activation (str, optional): activation function. Defaults to "softmax2d".
            check_point (Optional[str], optional): saved model. Defaults to None.
        """
        loss = utils.losses.DiceLoss(activation=activation)
        metrics = [
            utils.metrics.IoU(threshold=0.5, activation=activation),
        ]
        if check_point:
            model = torch.load(check_point)
        else:
            model = self.best_model

        # Evaluate model on test data
        if self.test_dataset is None:
            test_dataset = self.dataset(
                self.test_images,
                self.test_labels,
                augmentation=self.dataset_params["validation_augmentation"],
                preprocessing=self.dataset_params["preprocessing"],
                class_values=self.dataset_params["class_values"],
            )  # type: ignore[operator]
            self.test_dataset = test_dataset
        else:
            test_dataset = self.test_dataset

        print("Evaluate on test data")
        test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)
        test_epoch = utils.train.ValidEpoch(
            model=model,
            loss=loss,
            metrics=metrics,
            device=self.device,
        )

        round_test_log = test_epoch.run(test_dataloader)
        self.test_logs.append(round_test_log)

        return round_test_log

    def predict_prob(self, idxs_unlabeled: List[int]) -> np.ndarray:
        """Predict on unlabeled data.

        Args:
            idxs_unlabeled (List[int]): List of unlabeled data indices.

        Returns:
            np.ndarray: Probability of classes.
        """
        model = self.best_model
        # model:torch.nn.Module = self.best_model
        unlabeled_images = [self.pool_images[idx] for idx in idxs_unlabeled]
        unlabeled_labels = [self.pool_labels[idx] for idx in idxs_unlabeled]

        unlabeled_dataset = self.dataset(
            unlabeled_images,
            unlabeled_labels,
            augmentation=self.dataset_params["validation_augmentation"],
            preprocessing=self.dataset_params["preprocessing"],
            class_values=self.dataset_params["class_values"],
        )  # type: ignore[operator]

        if self.device == "cuda":
            unlabeled_loader = DataLoader(
                unlabeled_dataset, batch_size=10, shuffle=False, num_workers=2
            )
        else:
            unlabeled_loader = DataLoader(
                unlabeled_dataset, batch_size=10, shuffle=False
            )

        probs = []
        for batch_images, _ in unlabeled_loader:
            if self.device == "cuda":
                batch_images = batch_images.to(self.device)
            out = model.predict(batch_images)  # type: ignore[operator]
            batch_probs = F.softmax(out, dim=1)
            batch_probs = batch_probs.detach().cpu().numpy()
            probs.append(batch_probs)

        concat_probs = np.concatenate(probs, axis=0)

        return concat_probs
