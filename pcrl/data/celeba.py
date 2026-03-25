"""CelebA dataset for PCRL.

Loads the CelebA face attribute dataset directly from extracted files:
- 202,599 aligned & cropped face images resized to 64x64
- 40 binary attributes per image
- Official train/val/test partition (0/1/2)

Purposes focus on facial attribute prediction while suppressing
demographic attributes (gender, age, attractiveness).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from pcrl.data.base import DatasetInfo
from pcrl.purposes.spec import PurposeSpec

logger = logging.getLogger(__name__)

# Attributes used in our purposes
# Some attributes serve as both tasks and sensitive attrs across different purposes
TASK_ATTRS = ["Smiling", "Young", "Mouth_Slightly_Open", "Male", "Attractive"]
SENSITIVE_ATTRS = ["Male", "Young", "Attractive", "Smiling"]
ALL_USED_ATTRS = sorted(set(TASK_ATTRS + SENSITIVE_ATTRS))


def _get_transform(split: str) -> transforms.Compose:
    """Get image transform for a split."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


class CelebADataset(Dataset):
    """CelebA face attribute dataset for PCRL.

    Loads images directly from extracted files and resizes to 64x64.
    Returns image tensors (3, 64, 64) instead of flattened features.
    Uses the official train/val/test partition.
    """

    def __init__(
        self,
        purposes: list[PurposeSpec],
        root: str = "data/celeba",
        split: str = "train",
        max_samples: int | None = None,
    ) -> None:
        self.purposes = purposes
        self.root = Path(root)
        self.split = split
        self.img_transform = _get_transform(split)

        self._load_data(max_samples)

    def _load_data(self, max_samples: int | None) -> None:
        # Load attributes and partition
        attr_df = pd.read_csv(self.root / "list_attr_celeba.csv")
        partition_df = pd.read_csv(self.root / "list_eval_partition.csv")
        df = attr_df.merge(partition_df, on="image_id")

        # Filter by split: 0=train, 1=val, 2=test
        split_map = {"train": 0, "val": 1, "test": 2}
        df = df[df["partition"] == split_map[self.split]].reset_index(drop=True)

        if max_samples is not None:
            df = df.head(max_samples)

        # Image paths
        img_dir = self.root / "img_align_celeba" / "img_align_celeba"
        self.image_paths = [str(img_dir / fname) for fname in df["image_id"]]

        # Convert -1/1 attributes to 0/1
        for attr in ALL_USED_ATTRS:
            df[attr] = ((df[attr] + 1) // 2).astype(np.int64)

        # Task labels
        self.task_labels = {
            attr: torch.tensor(df[attr].values, dtype=torch.long)
            for attr in TASK_ATTRS
        }

        # Sensitive attributes
        self.sensitive_attrs = {
            attr: torch.tensor(df[attr].values, dtype=torch.long)
            for attr in SENSITIVE_ATTRS
        }

        self.info = DatasetInfo(
            name="celeba",
            num_features=3 * 64 * 64,
            task_labels={attr: 2 for attr in TASK_ATTRS},
            sensitive_attrs={attr: 2 for attr in SENSITIVE_ATTRS},
            num_samples=len(self.image_paths),
        )

        logger.info(
            f"CelebA {self.split}: {len(self)} images, "
            f"tasks={list(self.task_labels.keys())}, "
            f"sensitive={list(self.sensitive_attrs.keys())}"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_tensor = self.img_transform(img)

        return {
            "features": img_tensor,
            "task_labels": {k: v[idx] for k, v in self.task_labels.items()},
            "sensitive_attrs": {k: v[idx] for k, v in self.sensitive_attrs.items()},
        }


def get_celeba_purposes() -> list[PurposeSpec]:
    """Define purposes for the CelebA dataset.

    The 5 purposes create conflicting constraints that expose LAFTR's
    limitation: Male, Attractive, and Smiling each appear as both allowed
    tasks and disallowed attributes across different purposes. A single
    representation cannot simultaneously preserve and suppress them.
    """
    return [
        PurposeSpec(
            name="smile_detection",
            allowed_tasks=["Smiling"],
            disallowed_attrs=["Male", "Young"],
            task_type="classification",
            allowed_task_dims={"Smiling": 2},
            disallowed_attr_dims={"Male": 2, "Young": 2},
        ),
        PurposeSpec(
            name="age_estimation",
            allowed_tasks=["Young"],
            disallowed_attrs=["Male", "Attractive"],
            task_type="classification",
            allowed_task_dims={"Young": 2},
            disallowed_attr_dims={"Male": 2, "Attractive": 2},
        ),
        PurposeSpec(
            name="expression_analysis",
            allowed_tasks=["Smiling", "Mouth_Slightly_Open"],
            disallowed_attrs=["Male", "Young", "Attractive"],
            task_type="classification",
            allowed_task_dims={"Smiling": 2, "Mouth_Slightly_Open": 2},
            disallowed_attr_dims={"Male": 2, "Young": 2, "Attractive": 2},
        ),
        PurposeSpec(
            name="attractiveness_prediction",
            allowed_tasks=["Attractive"],
            disallowed_attrs=["Male", "Young", "Smiling"],
            task_type="classification",
            allowed_task_dims={"Attractive": 2},
            disallowed_attr_dims={"Male": 2, "Young": 2, "Smiling": 2},
        ),
        PurposeSpec(
            name="gender_analysis",
            allowed_tasks=["Male"],
            disallowed_attrs=["Young", "Attractive", "Smiling"],
            task_type="classification",
            allowed_task_dims={"Male": 2},
            disallowed_attr_dims={"Young": 2, "Attractive": 2, "Smiling": 2},
        ),
    ]
