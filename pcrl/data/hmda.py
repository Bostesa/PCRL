"""HMDA 2023 California LAR dataset for PCRL.

Loads preprocessed arrays from ``data/hmda_processed/`` produced by
``experiments/prepare_hmda.py``. The upstream preprocessing handles CFPB
data-browser download, filtering to single-family first-lien home-purchase
applications with action_taken in {1, 3}, feature encoding, quintile-based
loan amount banding, tract-level denial rate thresholding, and the 70/15/15
train/val/test split.

Real regulated-ML setting: under ECOA, race and ethnicity are legally
disallowed inputs to lending decisions. HMDA exposes both as derived
fields, so we can evaluate PCRL's ability to meet the same statutory
bar the regulator expects.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pcrl.data.base import DatasetInfo, PCRLDataset
from pcrl.utils.config import PurposeSpec

logger = logging.getLogger(__name__)

HMDA_EXPECTED_SCHEMA = "hmda_2023_ca_v1"


class HMDADataset(PCRLDataset):
    """HMDA 2023 California LAR (home purchase, first lien, site-built).

    Expects the preprocessing script to have populated
    ``data/hmda_processed/{metadata.json, train.npz, val.npz, test.npz}``.

    Task labels:
        - loan_decision: Binary (1 = originated, 0 = denied).
        - loan_amount_band: 5-class (training-set quintiles of loan amount).
        - tract_denial_high: Binary (1 if applicant's census tract has a
          denial rate above the median tract denial rate on training data).

    Sensitive attributes:
        - race: 5-class aggregation of derived_race.
        - ethnicity: Binary (Hispanic/Latino vs not).
        - sex: Binary (Female=0, Male=1).
    """

    def __init__(
        self,
        purposes: list[PurposeSpec],
        root: str = "data",
        split: str = "train",
        download: bool = False,
        transform: Any | None = None,
    ) -> None:
        super().__init__(purposes, root, split, download, transform)

    def _load_data(self) -> None:
        root_path = Path(self.root) / "hmda_processed"
        meta_path = root_path / "metadata.json"
        data_path = root_path / f"{self.split}.npz"
        if not meta_path.exists() or not data_path.exists():
            raise FileNotFoundError(
                f"HMDA processed data not found at {root_path}. "
                f"Run experiments/prepare_hmda.py first."
            )

        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("schema_version") != HMDA_EXPECTED_SCHEMA:
            raise ValueError(
                f"HMDA schema mismatch: expected {HMDA_EXPECTED_SCHEMA}, "
                f"got {meta.get('schema_version')}"
            )

        npz = np.load(data_path)
        self.features = torch.from_numpy(npz["features"].astype(np.float32))

        task_labels: dict[str, torch.Tensor] = {}
        for name in meta["task_labels"].keys():
            task_labels[name] = torch.from_numpy(
                npz[f"task_{name}"].astype(np.int64)
            )
        self.task_labels = task_labels

        sensitive_attrs: dict[str, torch.Tensor] = {}
        for name in meta["sensitive_attrs"].keys():
            sensitive_attrs[name] = torch.from_numpy(
                npz[f"attr_{name}"].astype(np.int64)
            )
        self.sensitive_attrs = sensitive_attrs

        self.info = DatasetInfo(
            name="hmda",
            num_features=int(self.features.shape[1]),
            task_labels=meta["task_labels"],
            sensitive_attrs=meta["sensitive_attrs"],
            num_samples=int(self.features.shape[0]),
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


def get_hmda_purposes() -> list[PurposeSpec]:
    """Three HMDA purposes producing 6 compliance pairs.

    - underwriting: loan_decision; race, ethnicity disallowed (ECOA).
    - pricing_analysis: loan_amount_band; race, sex disallowed.
    - fair_lending_audit: tract_denial_high; race, sex disallowed.
    """
    return [
        PurposeSpec(
            name="underwriting",
            allowed_tasks=["loan_decision"],
            disallowed_attrs=["race", "ethnicity"],
            task_type="classification",
            allowed_task_dims={"loan_decision": 2},
            disallowed_attr_dims={"race": 5, "ethnicity": 2},
        ),
        PurposeSpec(
            name="pricing_analysis",
            allowed_tasks=["loan_amount_band"],
            disallowed_attrs=["race", "sex"],
            task_type="classification",
            allowed_task_dims={"loan_amount_band": 5},
            disallowed_attr_dims={"race": 5, "sex": 2},
        ),
        PurposeSpec(
            name="fair_lending_audit",
            allowed_tasks=["tract_denial_high"],
            disallowed_attrs=["race", "sex"],
            task_type="classification",
            allowed_task_dims={"tract_denial_high": 2},
            disallowed_attr_dims={"race": 5, "sex": 2},
        ),
    ]
