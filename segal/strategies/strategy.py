from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from segmentation_models_pytorch import utils
from torch.utils.data import DataLoader


class Strategy:
    def __init__(
        self,
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
    ):
        self.pool_images = pool_images
        self.pool_labels = pool_labels
        self.val_images = val_images
        self.val_labels = val_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.idxs_lb = idxs_lb  # bool type
        self.dataset = dataset
        self.dataset_params = dataset_params  # dict
        self.val_dataset = None
        self.test_dataset = None
        self.n_pool = len(pool_images)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_params = model_params
        self.best_model = None
        self.train_logs = []
        self.val_logs = []
        self.test_logs = []

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def train(self, n_epoch=10, activation="softmax2d", base_path="output"):
        model = smp.__dict__[self.model_params["MODEL_NAME"]](
            encoder_name=self.model_params["ENCODER"],
            encoder_weights=self.model_params["ENCODER_WEIGHTS"],
            classes=self.model_params["NUM_CLASSES"],
        )
        model.to(self.device)

        if type(base_path) is str:
            base_path = Path(base_path)
        if not base_path.exists():
            base_path.mkdir(exist_ok=True, parents=True)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]

        train_images = [self.pool_images[idx] for idx in idxs_train]
        train_labels = [self.pool_labels[idx] for idx in idxs_train]

        train_dataset = self.dataset(
            train_images,
            train_labels,
            augmentation=self.dataset_params["training_augmentation"],
            preprocessing=self.dataset_params["preprocessing"],
            classes=self.dataset_params["classes"],
        )

        if self.val_dataset is None:
            valid_dataset = self.dataset(
                self.val_images,
                self.val_labels,
                augmentation=self.dataset_params["validation_augmentation"],
                preprocessing=self.dataset_params["preprocessing"],
                classes=self.dataset_params["classes"],
            )
            self.valid_dataset = valid_dataset
        else:
            valid_dataset = self.valid_dataset

        if self.device == "cuda":
            train_loader = DataLoader(
                train_dataset, batch_size=4, shuffle=True, num_workers=4
            )
            valid_loader = DataLoader(
                valid_dataset, batch_size=10, shuffle=False, num_workers=2
            )
        else:
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=10, shuffle=False)

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

        # Evaluate model on test data
        if self.test_dataset is None:
            test_dataset = self.dataset(
                self.test_images,
                self.test_labels,
                augmentation=self.dataset_params["validation_augmentation"],
                preprocessing=self.dataset_params["preprocessing"],
                classes=self.dataset_params["classes"],
            )
            self.test_dataset = test_dataset
        else:
            test_dataset = self.test_dataset

        print("Evaluate on test data")
        test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)
        test_epoch = utils.train.ValidEpoch(
            model=self.best_model,
            loss=loss,
            metrics=metrics,
            device=self.device,
        )

        round_test_log = test_epoch.run(test_dataloader)
        self.test_logs.append(round_test_log)

        return round_test_log

    def predict_prob(self, idxs_unlabeled):

        model = self.best_model
        unlabeled_images = [self.pool_images[idx] for idx in idxs_unlabeled]
        unlabeled_labels = [self.pool_labels[idx] for idx in idxs_unlabeled]

        unlabeled_dataset = self.dataset(
            unlabeled_images,
            unlabeled_labels,
            augmentation=self.dataset_params["validation_augmentation"],
            preprocessing=self.dataset_params["preprocessing"],
            classes=self.dataset_params["classes"],
        )

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
            out = model.predict(batch_images)
            batch_probs = F.softmax(out, dim=1)
            batch_probs = batch_probs.detach().cpu().numpy()
            probs.append(batch_probs)

        probs = np.concatenate(probs, axis=0)

        return probs
