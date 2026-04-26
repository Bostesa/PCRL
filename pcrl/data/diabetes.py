"""UCI Diabetes 130-US hospitals dataset for PCRL.

Loads preprocessed .npz files from data/diabetes_processed/ (written by
experiments/preprocess_diabetes.py).  The preprocessing includes:
  - Dropping high-missing columns
  - 9-category ICD-9 grouping for primary diagnosis
  - 70/15/15 stratified split on readmission_outcome, seed=42.

Purposes (3 purposes, 6 disallowed pairs):
  billing_audit — predict primary_diagnosis_category; hide [race, gender]
  quality_research — predict readmission_outcome; hide [race, age_bucket]
  clinical_decision_support — predict medication_change_outcome; hide [race, gender]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pcrl.data.base import DatasetInfo, PCRLDataset
from pcrl.utils.config import PurposeSpec

logger = logging.getLogger(__name__)


class DiabetesDataset(PCRLDataset):
    """UCI Diabetes 130-US Hospitals (101,766 encounters, deidentified)."""

    def __init__(
        self,
        purposes: list[PurposeSpec],
        root: str = "data/diabetes_processed",
        split: str = "train",
        download: bool = False,
        transform: Any | None = None,
    ) -> None:
        super().__init__(purposes, root, split, download, transform)

    def _load_data(self) -> None:
        path = Path(self.root) / f"{self.split}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Preprocessed split {path} not found. "
                f"Run experiments/preprocess_diabetes.py first."
            )
        data = np.load(path)

        self.features = torch.tensor(data["features"], dtype=torch.float32)
        self.task_labels = {
            "primary_diagnosis_category": torch.tensor(data["primary_diagnosis_category"], dtype=torch.long),
            "readmission_outcome": torch.tensor(data["readmission_outcome"], dtype=torch.long),
            "medication_change_outcome": torch.tensor(data["medication_change_outcome"], dtype=torch.long),
        }
        self.sensitive_attrs = {
            "race": torch.tensor(data["race"], dtype=torch.long),
            "gender": torch.tensor(data["gender"], dtype=torch.long),
            "age_bucket": torch.tensor(data["age_bucket"], dtype=torch.long),
        }
        self.info = DatasetInfo(
            name="diabetes",
            num_features=int(self.features.shape[1]),
            task_labels={
                "primary_diagnosis_category": 9,
                "readmission_outcome": 2,
                "medication_change_outcome": 2,
            },
            sensitive_attrs={
                "race": 5,
                "gender": 2,
                "age_bucket": 10,
            },
            num_samples=int(self.features.shape[0]),
        )
        logger.info(
            f"Loaded Diabetes {self.split}: {self.info.num_samples} rows, "
            f"{self.info.num_features} features"
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


def get_diabetes_purposes() -> list[PurposeSpec]:
    """Return the three Diabetes purposes (6 disallowed-attr pairs total)."""
    return [
        PurposeSpec(
            name="billing_audit",
            allowed_tasks=["primary_diagnosis_category"],
            disallowed_attrs=["race", "gender"],
            task_type="classification",
            allowed_task_dims={"primary_diagnosis_category": 9},
            disallowed_attr_dims={"race": 5, "gender": 2},
        ),
        PurposeSpec(
            name="quality_research",
            allowed_tasks=["readmission_outcome"],
            disallowed_attrs=["race", "age_bucket"],
            task_type="classification",
            allowed_task_dims={"readmission_outcome": 2},
            disallowed_attr_dims={"race": 5, "age_bucket": 10},
        ),
        PurposeSpec(
            name="clinical_decision_support",
            allowed_tasks=["medication_change_outcome"],
            disallowed_attrs=["race", "gender"],
            task_type="classification",
            allowed_task_dims={"medication_change_outcome": 2},
            disallowed_attr_dims={"race": 5, "gender": 2},
        ),
    ]
