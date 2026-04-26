"""Folktables / ACSIncome dataset for PCRL.

Downloads the 2018 1-Year PUMS data for a single state (default CA) via the
`folktables` package, filters to employed adults (ACSIncome-standard filter),
and exposes a 3-purpose setup analogous to Adult.

Adaptation notes (see results/folktables/adaptation_notes.md):
  - Drop OCCP, POBP, RELP from ACSIncome's 10-feature spec to keep the
    one-hot feature vector small. All remaining ACSIncome features are used.
  - Education level, age group, hours-per-week band are derived by binning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pcrl.data.base import DatasetInfo, PCRLDataset
from pcrl.utils.config import PurposeSpec

logger = logging.getLogger(__name__)


# ── Feature bucketization maps ───────────────────────────────────────────

# SCHL (educational attainment, 1..24) → 4 levels matching Adult
SCHL_TO_EDU_LEVEL: dict[int, int] = {
    **{i: 0 for i in range(1, 16)},  # <HS
    16: 1, 17: 1,                     # HS-grad / GED
    18: 2, 19: 2, 20: 2,              # Some college / associate
    21: 3, 22: 3, 23: 3, 24: 3,       # Bachelors+
}

# MAR (1..5) → binary "married or not"
MAR_TO_BINARY: dict[int, int] = {1: 1, 2: 0, 3: 0, 4: 0, 5: 0}

AGE_BINS = [0, 25, 45, 65, 150]       # 4 bands
HOURS_BINS = [-1, 20, 35, 45, 200]    # 4 bands

# Fixed category sets for deterministic one-hot layout across splits
CATEGORY_VALUES: dict[str, list[int]] = {
    "COW":   [1, 2, 3, 4, 5, 6, 7, 8, 9],      # class of worker (1..9)
    "MAR":   [1, 2, 3, 4, 5],                   # marital status
    "SEX":   [1, 2],                             # 1=M, 2=F
    "RAC1P": [1, 2, 3, 4, 5, 6, 7, 8, 9],       # race (9 codes)
    "SCHL_group": [0, 1, 2, 3],
}

CATEGORICAL_COLUMNS = ["COW", "SCHL_group", "MAR", "SEX", "RAC1P"]
NUMERICAL_COLUMNS = ["AGEP", "WKHP"]


def _bucket_education(schl: pd.Series) -> pd.Series:
    return schl.map(SCHL_TO_EDU_LEVEL).fillna(0).astype(int)


def _bucket_age(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=AGE_BINS, labels=list(range(len(AGE_BINS) - 1)),
                  include_lowest=True).astype(int)


def _bucket_hours(hrs: pd.Series) -> pd.Series:
    return pd.cut(hrs, bins=HOURS_BINS, labels=list(range(len(HOURS_BINS) - 1)),
                  include_lowest=True).astype(int)


class FolktablesIncomeDataset(PCRLDataset):
    """ACSIncome (Folktables, 2018 1-Year PUMS, California by default).

    Task labels:
        - income:          binary, PINCP > 50000
        - hours_band:      4-class, binned from WKHP
        - education_level: 4-class, binned from SCHL

    Sensitive attributes:
        - sex:            binary (0=Female, 1=Male)
        - race:           9-class (RAC1P remapped to 0..8)
        - age_group:      4-class (binned from AGEP)
        - marital_status: binary (1 if MAR==Married-civ-spouse)
        - income:         binary (also used as sensitive for p3)
    """

    # Deterministic 80/10/10 split seeded off the full filtered DataFrame.
    SPLIT_SEED = 42
    TRAIN_FRAC = 0.80
    VAL_FRAC = 0.10  # test_frac = 0.10

    def __init__(
        self,
        purposes: list[PurposeSpec],
        root: str = "data",
        split: str = "train",
        download: bool = True,
        transform: Any | None = None,
        states: list[str] | None = None,
        survey_year: str = "2018",
        norm_stats: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.states = states or ["CA"]
        self.survey_year = survey_year
        self._norm_stats = norm_stats
        self.norm_stats: dict[str, tuple[float, float]] = {}
        super().__init__(purposes, root, split, download, transform)

    # ── Data loading ─────────────────────────────────────────────────────

    def _load_data(self) -> None:
        root_path = Path(self.root) / "folktables"
        root_path.mkdir(parents=True, exist_ok=True)

        cache = root_path / f"acs_{self.survey_year}_{'-'.join(self.states)}.parquet"
        if cache.exists():
            logger.info(f"Loading cached Folktables data from {cache}")
            df = pd.read_parquet(cache)
        else:
            df = self._download_and_filter(root_path)
            df.to_parquet(cache, index=False)
            logger.info(f"Cached filtered Folktables data to {cache} ({len(df)} rows)")

        # Fixed-seed 80/10/10 split
        rng = np.random.RandomState(self.SPLIT_SEED)
        perm = rng.permutation(len(df))
        n = len(df)
        n_train = int(self.TRAIN_FRAC * n)
        n_val = int(self.VAL_FRAC * n)
        if self.split == "train":
            idx = perm[:n_train]
        elif self.split == "val":
            idx = perm[n_train:n_train + n_val]
        else:  # test
            idx = perm[n_train + n_val:]
        df = df.iloc[idx].reset_index(drop=True)

        self._preprocess(df)

    def _download_and_filter(self, root_path: Path) -> pd.DataFrame:
        """Download PUMS via folktables and apply ACSIncome filter."""
        try:
            from folktables import ACSDataSource
        except ImportError as exc:
            raise ImportError(
                "folktables package is required for FolktablesIncomeDataset. "
                "Install with `pip install folktables`."
            ) from exc

        logger.info(f"Downloading ACS {self.survey_year} PUMS for states={self.states}")
        ds = ACSDataSource(
            survey_year=self.survey_year,
            horizon="1-Year",
            survey="person",
            root_dir=str(root_path),
        )
        raw = ds.get_data(states=self.states, download=True)

        # ACSIncome standard filter: employed adults, positive wages, observed hours
        mask = (
            (raw["AGEP"] > 16)
            & (raw["PINCP"] > 100)
            & (raw["WKHP"] > 0)
            & (raw["PWGTP"] >= 1)
        )
        needed = ["AGEP", "COW", "SCHL", "MAR", "WKHP", "SEX", "RAC1P", "PINCP"]
        df = raw.loc[mask, needed].dropna().reset_index(drop=True)
        # Clamp COW to valid 1..9 (rare spurious codes)
        df["COW"] = df["COW"].astype(int).clip(1, 9)
        df["SEX"] = df["SEX"].astype(int)
        df["RAC1P"] = df["RAC1P"].astype(int).clip(1, 9)
        df["MAR"] = df["MAR"].astype(int).clip(1, 5)
        df["SCHL"] = df["SCHL"].astype(int)
        return df

    # ── Preprocessing ────────────────────────────────────────────────────

    def _preprocess(self, df: pd.DataFrame) -> None:
        # Derived columns
        df = df.copy()
        df["SCHL_group"] = _bucket_education(df["SCHL"])
        age_group = _bucket_age(df["AGEP"])
        hours_band = _bucket_hours(df["WKHP"])
        marital_bin = df["MAR"].map(MAR_TO_BINARY).fillna(0).astype(int)

        # ── Feature encoding ─────────────────────────────────────────────
        encoded: list[np.ndarray] = []

        for col in NUMERICAL_COLUMNS:
            vals = df[col].values.astype(np.float32)
            if self._norm_stats and col in self._norm_stats:
                mean, std = self._norm_stats[col]
            else:
                mean = float(vals.mean())
                std = float(vals.std()) + 1e-8
            self.norm_stats[col] = (mean, std)
            vals = (vals - mean) / std
            encoded.append(vals.reshape(-1, 1))

        for col in CATEGORICAL_COLUMNS:
            categories = CATEGORY_VALUES[col]
            series = pd.Categorical(df[col], categories=categories)
            one_hot = pd.get_dummies(series, prefix=col)
            encoded.append(one_hot.values.astype(np.float32))

        self.features = torch.tensor(
            np.concatenate(encoded, axis=1), dtype=torch.float32,
        )

        # ── Task labels ──────────────────────────────────────────────────
        income_labels = torch.tensor(
            (df["PINCP"].values > 50000).astype(np.int64), dtype=torch.long,
        )
        hours_band_labels = torch.tensor(hours_band.values.astype(np.int64), dtype=torch.long)
        education_level_labels = torch.tensor(
            df["SCHL_group"].values.astype(np.int64), dtype=torch.long,
        )
        self.task_labels = {
            "income": income_labels,
            "hours_band": hours_band_labels,
            "education_level": education_level_labels,
        }

        # ── Sensitive attributes ─────────────────────────────────────────
        # SEX: 1=M, 2=F in ACS → remap to 1=M, 0=F to match Adult convention
        sex_labels = torch.tensor(
            (df["SEX"].values == 1).astype(np.int64), dtype=torch.long,
        )
        race_labels = torch.tensor(
            (df["RAC1P"].values - 1).astype(np.int64), dtype=torch.long,
        )
        age_labels = torch.tensor(age_group.values.astype(np.int64), dtype=torch.long)
        marital_labels = torch.tensor(marital_bin.values.astype(np.int64), dtype=torch.long)
        self.sensitive_attrs = {
            "sex": sex_labels,
            "race": race_labels,
            "age_group": age_labels,
            "marital_status": marital_labels,
            "income": income_labels,
        }

        # ── Dataset info ─────────────────────────────────────────────────
        self.info = DatasetInfo(
            name="folktables_acsincome",
            num_features=self.features.shape[1],
            task_labels={
                "income": 2,
                "hours_band": len(HOURS_BINS) - 1,
                "education_level": 4,
            },
            sensitive_attrs={
                "sex": 2,
                "race": 9,
                "age_group": len(AGE_BINS) - 1,
                "marital_status": 2,
                "income": 2,
            },
            num_samples=len(self.features),
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
            "sensitive_attrs": {
                k: v[idx] for k, v in self.sensitive_attrs.items()
            },
        }


def get_folktables_purposes() -> list[PurposeSpec]:
    """Three-purpose setup mirroring the Adult paper configuration."""
    return [
        PurposeSpec(
            name="income_prediction",
            allowed_tasks=["income"],
            disallowed_attrs=["race", "sex"],
            task_type="classification",
            allowed_task_dims={"income": 2},
            disallowed_attr_dims={"race": 9, "sex": 2},
        ),
        PurposeSpec(
            name="employment_analysis",
            allowed_tasks=["hours_band"],
            disallowed_attrs=["race", "age_group", "marital_status"],
            task_type="classification",
            allowed_task_dims={"hours_band": len(HOURS_BINS) - 1},
            disallowed_attr_dims={
                "race": 9,
                "age_group": len(AGE_BINS) - 1,
                "marital_status": 2,
            },
        ),
        PurposeSpec(
            name="education_assessment",
            allowed_tasks=["education_level"],
            disallowed_attrs=["sex", "race", "income"],
            task_type="classification",
            allowed_task_dims={"education_level": 4},
            disallowed_attr_dims={"sex": 2, "race": 9, "income": 2},
        ),
    ]
