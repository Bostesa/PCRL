"""UCI Human Activity Recognition dataset for PCRL.

Loads the real UCI HAR dataset:
- 10,299 samples (7352 train + 2947 test) with 561 features
- 6 activity classes: walking, walking_upstairs, walking_downstairs,
  sitting, standing, laying
- 30 subjects with distinct movement patterns

Since the official train/test split is subject-disjoint (different subjects
in each), we combine all data and re-split 70/15/15 randomly so that
subjects appear in all splits — required for meaningful privacy evaluation
(the PostHocAuditorSuite needs the same subject classes in train and test).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pcrl.data.base import DatasetInfo, PCRLDataset
from pcrl.purposes.spec import PurposeSpec

logger = logging.getLogger(__name__)

ACTIVITIES = [
    "walking",
    "walking_upstairs",
    "walking_downstairs",
    "sitting",
    "standing",
    "laying",
]

# Binary grouping: dynamic vs sedentary
ACTIVE_ACTIVITIES = {"walking", "walking_upstairs", "walking_downstairs"}

N_FEATURES = 561
N_SUBJECTS = 30
N_ACTIVITIES = 6


def _load_uci_har(root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the real UCI HAR dataset from disk.

    Combines official train and test splits into a single dataset.
    Activity labels are converted from 1-indexed to 0-indexed.
    Subject IDs are remapped to contiguous 0-indexed integers.

    Returns:
        (features, activity_labels, subject_labels, is_active_labels)
    """
    base = Path(root) / "UCI HAR Dataset"

    # Load train
    X_train = np.loadtxt(base / "train" / "X_train.txt")
    y_train = np.loadtxt(base / "train" / "y_train.txt", dtype=int)
    s_train = np.loadtxt(base / "train" / "subject_train.txt", dtype=int)

    # Load test
    X_test = np.loadtxt(base / "test" / "X_test.txt")
    y_test = np.loadtxt(base / "test" / "y_test.txt", dtype=int)
    s_test = np.loadtxt(base / "test" / "subject_test.txt", dtype=int)

    # Combine
    features = np.concatenate([X_train, X_test], axis=0).astype(np.float32)
    activity = np.concatenate([y_train, y_test]) - 1  # 1-indexed → 0-indexed
    subject = np.concatenate([s_train, s_test])

    # Remap subject IDs (1-30 with possible gaps) to contiguous 0-indexed
    unique_subjects = np.sort(np.unique(subject))
    subject_map = {old: new for new, old in enumerate(unique_subjects)}
    subject = np.array([subject_map[s] for s in subject], dtype=np.int64)

    # Normalize features to zero mean, unit variance
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    features = (features - mean) / std

    # Derive is_active: 1 for walking*, 0 for sitting/standing/laying
    is_active = np.array(
        [1 if ACTIVITIES[a] in ACTIVE_ACTIVITIES else 0 for a in activity],
        dtype=np.int64,
    )

    n_subjects = len(unique_subjects)
    logger.info(
        f"UCI HAR: {len(features)} samples, {features.shape[1]} features, "
        f"{len(np.unique(activity))} activities, {n_subjects} subjects"
    )

    return features, activity.astype(np.int64), subject, is_active


class HARDataset(PCRLDataset):
    """UCI Human Activity Recognition dataset.

    Loads real accelerometer/gyroscope sensor data where activity is the
    primary task but subject identity leaks through movement patterns —
    a realistic IoT privacy challenge.
    """

    def __init__(
        self,
        purposes: list[PurposeSpec],
        root: str = "data",
        split: str = "train",
        download: bool = True,
        transform: Any | None = None,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        super().__init__(purposes, root, split, download, transform)

    def _load_data(self) -> None:
        features, activity, subject, is_active = _load_uci_har(self.root)

        # Split: 70% train, 15% val, 15% test (random, NOT subject-disjoint)
        n = len(features)
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(n)

        train_end = int(0.70 * n)
        val_end = int(0.85 * n)

        if self.split == "train":
            idx = indices[:train_end]
        elif self.split == "val":
            idx = indices[train_end:val_end]
        else:
            idx = indices[val_end:]

        self.features = torch.tensor(features[idx], dtype=torch.float32)
        self.task_labels = {
            "activity": torch.tensor(activity[idx], dtype=torch.long),
            "is_active": torch.tensor(is_active[idx], dtype=torch.long),
        }
        self.sensitive_attrs = {
            "subject_id": torch.tensor(subject[idx], dtype=torch.long),
            "activity": torch.tensor(activity[idx], dtype=torch.long),
        }

        self.info = DatasetInfo(
            name="har",
            num_features=N_FEATURES,
            task_labels={"activity": N_ACTIVITIES, "is_active": 2},
            sensitive_attrs={"subject_id": N_SUBJECTS, "activity": N_ACTIVITIES},
            num_samples=len(self.features),
        )

        logger.info(
            f"HAR {self.split}: {len(self.features)} samples, "
            f"{N_FEATURES} features, {N_ACTIVITIES} activities, "
            f"{N_SUBJECTS} subjects"
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        features = self.features[idx]
        if self.transform is not None:
            features = self.transform(features)

        return {
            "features": features,
            "task_labels": {k: v[idx] for k, v in self.task_labels.items()},
            "sensitive_attrs": {k: v[idx] for k, v in self.sensitive_attrs.items()},
        }


def get_har_purposes() -> list[PurposeSpec]:
    """Define purposes for the HAR dataset."""
    return [
        PurposeSpec(
            name="activity_recognition",
            allowed_tasks=["activity"],
            disallowed_attrs=["subject_id"],
            task_type="classification",
            allowed_task_dims={"activity": N_ACTIVITIES},
            disallowed_attr_dims={"subject_id": N_SUBJECTS},
        ),
        PurposeSpec(
            name="health_monitoring",
            allowed_tasks=["is_active"],
            disallowed_attrs=["subject_id", "activity"],
            task_type="classification",
            allowed_task_dims={"is_active": 2},
            disallowed_attr_dims={"subject_id": N_SUBJECTS, "activity": N_ACTIVITIES},
        ),
    ]
