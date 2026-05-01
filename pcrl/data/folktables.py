"""Folktables (ACS PUMS) dataset for PCRL — deep-research design.

Three-purpose unified dataset built on California 2018 1-Year ACS PUMS via
the ``folktables`` package (Ding et al., NeurIPS 2021). Targets and
sensitive-attribute conventions follow:

  - Han et al. ICLR 2024 (FFB): binary RAC1P (White vs non-White), binary SEX.
  - Jovanović et al. ICML 2023 (FARE): joint cardinality kept ≤ 12.
  - Defauw et al. 2024 (ABCFair): per-purpose disallowed attribute selection.

PCRL purposes (joint cardinality of disallowed attributes):
  P1 income_prediction       PINCP > 50000   sex(2) × race(2) = 4
  P2 employment_analysis     ESR == 1        sex(2) × race(2) × disability(2) = 8
  P3 public_coverage_assessment  PUBCOV     sex(2) × race(2) × age_group(3) = 12

The 12-cap stays below the 13-direction LEACE rank ceiling that broke
``diabetes/quality_research`` at LoRA rank 8 (see results/v2_fix1_audit.md).

Population filter: ACSEmployment-style (AGEP >= 16). PINCP/ESR/PUBCOV
fillna→0 so all three task labels are defined for every retained row.
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


# ── Configuration ───────────────────────────────────────────────────────

DEFAULT_STATE = "CA"
DEFAULT_YEAR = "2018"
SCHEMA_VERSION = "v2_ffb_binarized"
WHITE_RAC1P = 1                 # RAC1P==1 is "White alone"
SEX_MALE = 1                    # SEX==1 is Male in ACS
DIS_HAS = 1                     # DIS==1 is "with a disability"
ESR_EMPLOYED_CIVILIAN = 1       # ESR==1 is "civilian employed at work"

AGEP_BINS = [-1, 25, 60, 200]   # → categories {<=25, 26-60, >60}
N_AGE_GROUPS = len(AGEP_BINS) - 1   # 3

# Bucketing for occupation: OCCP // 100 truncates the 4-digit code to its
# 2-digit major group. NaN → -1 (own bucket). Capped to 0..99.
OCCP_NAN_BUCKET = -1
OCCP_BUCKETS = list(range(-1, 100))   # -1 (NaN), 0..99

# Place-of-birth bucketing: 1..56 = US states, 60..78 = US territories,
# 100+ = foreign. NaN → -1.
POBP_NAN_BUCKET = -1
POBP_US_STATE = 0
POBP_US_TERRITORY = 1
POBP_FOREIGN = 2
POBP_BUCKETS = [-1, POBP_US_STATE, POBP_US_TERRITORY, POBP_FOREIGN]

# Educational attainment SCHL → 4-bucket group (matches Adult convention)
SCHL_NAN_BUCKET = 0
SCHL_BUCKETS = [0, 1, 2, 3]

# Per-attribute one-hot category sets (NaN encoded as a dedicated bucket).
# These are FIXED across train/val/test so encoded feature dim is stable.
CATEGORY_VALUES: dict[str, list[int]] = {
    "COW":         [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9],            # class of worker
    "SCHL_group":  SCHL_BUCKETS,                                # 4 levels
    "MAR":         [1, 2, 3, 4, 5],
    "RELP":        [-1] + list(range(0, 18)),                   # 0..17 + NaN
    "ESP":         [-1, 1, 2, 3, 4, 5, 6, 7, 8],                # NaN ok
    "CIT":         [1, 2, 3, 4, 5],
    "MIG":         [-1, 1, 2, 3],
    "MIL":         [-1, 1, 2, 3, 4],
    "ANC":         [1, 2, 3, 4],
    "NATIVITY":    [1, 2],
    "DEAR":        [1, 2],
    "DEYE":        [1, 2],
    "DREM":        [-1, 1, 2],
    "FER":         [-1, 1, 2],
    "OCCP_2dig":   OCCP_BUCKETS,
    "POBP_bucket": POBP_BUCKETS,
    "SEX":         [1, 2],
    "RAC1P_bin":   [0, 1],
    "DIS":         [1, 2],
}

CATEGORICAL_COLUMNS = list(CATEGORY_VALUES.keys())
NUMERICAL_COLUMNS = ["AGEP", "WKHP"]

# ACS feature columns we read from the raw frame (union of the three task
# feature sets, minus the target columns ESR/PINCP/PUBCOV which would leak).
RAW_FEATURE_COLS = [
    "AGEP", "WKHP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP",
    "DIS", "ESP", "CIT", "MIG", "MIL", "ANC", "NATIVITY",
    "DEAR", "DEYE", "DREM", "FER", "SEX", "RAC1P",
]
RAW_TARGET_COLS = ["PINCP", "ESR", "PUBCOV"]


# ── Bucketing helpers ───────────────────────────────────────────────────


def _bucket_schl(schl: pd.Series) -> pd.Series:
    out = pd.Series(SCHL_NAN_BUCKET, index=schl.index, dtype=np.int64)
    s = pd.to_numeric(schl, errors="coerce")
    out[s.between(1, 15)] = 0     # < HS
    out[s.between(16, 17)] = 1    # HS-grad / GED
    out[s.between(18, 20)] = 2    # Some college / associate
    out[s.between(21, 24)] = 3    # Bachelors+
    return out


def _bucket_occp(occp: pd.Series) -> pd.Series:
    s = pd.to_numeric(occp, errors="coerce")
    bucket = (s // 100).clip(lower=0, upper=99)
    bucket = bucket.fillna(OCCP_NAN_BUCKET).astype(int)
    return bucket


def _bucket_pobp(pobp: pd.Series) -> pd.Series:
    s = pd.to_numeric(pobp, errors="coerce")
    out = pd.Series(POBP_NAN_BUCKET, index=pobp.index, dtype=np.int64)
    out[s.between(1, 56)] = POBP_US_STATE
    out[s.between(60, 78)] = POBP_US_TERRITORY
    out[s >= 100] = POBP_FOREIGN
    return out


def _agep_bin(agep: pd.Series) -> pd.Series:
    return pd.cut(
        agep, bins=AGEP_BINS,
        labels=list(range(N_AGE_GROUPS)), include_lowest=True,
    ).astype(int)


def _fillna_categorical(series: pd.Series) -> pd.Series:
    """NaN → -1 marker (matched against CATEGORY_VALUES)."""
    s = pd.to_numeric(series, errors="coerce")
    return s.fillna(-1).astype(int)


# ── Dataset class ───────────────────────────────────────────────────────


class FolktablesACSDataset(PCRLDataset):
    """Unified ACS PUMS dataset for PCRL with three task purposes.

    Task labels (all binary):
        - income:           PINCP > 50000   (target leakage column dropped from features)
        - employment:       ESR == 1        (civilian employed)
        - public_coverage:  PUBCOV == 1

    Sensitive attributes:
        - sex:        binary (1 if SEX==1 i.e. Male, else 0)
        - race:       binary (1 if RAC1P==1 i.e. White-alone, else 0)
        - disability: binary (1 if DIS==1 i.e. with a disability, else 0)
        - age_group:  3-class (<=25, 26-60, >60)

    Joint cardinality cap per purpose: 12 (= 2×2×3 on public_coverage).
    """

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
        survey_year: str = DEFAULT_YEAR,
        norm_stats: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.states = states or [DEFAULT_STATE]
        self.survey_year = survey_year
        self._norm_stats = norm_stats
        self.norm_stats: dict[str, tuple[float, float]] = {}
        super().__init__(purposes, root, split, download, transform)

    # ── Loading ─────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        root_path = Path(self.root) / "folktables"
        root_path.mkdir(parents=True, exist_ok=True)

        cache_name = (
            f"acs_{self.survey_year}_{'-'.join(self.states)}_{SCHEMA_VERSION}.parquet"
        )
        cache = root_path / cache_name
        if cache.exists():
            logger.info(f"Loading cached Folktables data from {cache}")
            df = pd.read_parquet(cache)
        else:
            df = self._download_and_filter(root_path)
            df.to_parquet(cache, index=False)
            logger.info(f"Cached filtered Folktables data to {cache} ({len(df)} rows)")

        # 80/10/10 deterministic split — same seed as legacy module so
        # train/val/test memberships are stable across runs.
        rng = np.random.RandomState(self.SPLIT_SEED)
        perm = rng.permutation(len(df))
        n = len(df)
        n_train = int(self.TRAIN_FRAC * n)
        n_val = int(self.VAL_FRAC * n)
        if self.split == "train":
            idx = perm[:n_train]
        elif self.split == "val":
            idx = perm[n_train:n_train + n_val]
        else:  # "test"
            idx = perm[n_train + n_val:]
        df = df.iloc[idx].reset_index(drop=True)

        self._preprocess(df)

    def _download_and_filter(self, root_path: Path) -> pd.DataFrame:
        try:
            from folktables import ACSDataSource
        except ImportError as exc:
            raise ImportError(
                "folktables package required. Install with `pip install folktables`."
            ) from exc

        logger.info(
            f"Downloading ACS {self.survey_year} 1-Year PUMS for states={self.states}"
        )
        ds = ACSDataSource(
            survey_year=self.survey_year, horizon="1-Year", survey="person",
            root_dir=str(root_path),
        )
        raw = ds.get_data(states=self.states, download=True)

        # Population filter: ACSEmployment-style (AGEP >= 16). Drop rows
        # with NaN demographic essentials so sex/race/age_group are
        # always defined for the audit metric.
        keep_cols = [c for c in (RAW_FEATURE_COLS + RAW_TARGET_COLS) if c in raw.columns]
        df = raw.loc[raw["AGEP"] >= 16, keep_cols].copy()
        df = df.dropna(subset=["AGEP", "SEX", "RAC1P"]).reset_index(drop=True)

        # Coerce demographics to int (raw PUMS sometimes stores as float).
        df["AGEP"] = df["AGEP"].astype(int)
        df["SEX"] = df["SEX"].astype(int)
        df["RAC1P"] = df["RAC1P"].astype(int)
        return df

    # ── Preprocessing ───────────────────────────────────────────────────

    def _preprocess(self, df: pd.DataFrame) -> None:
        df = df.copy()

        # ── Derived columns ─────────────────────────────────────────────
        df["SCHL_group"] = _bucket_schl(df["SCHL"])
        df["OCCP_2dig"] = _bucket_occp(df["OCCP"])
        df["POBP_bucket"] = _bucket_pobp(df["POBP"])

        # Binary RAC1P (White vs non-White) — used both as a feature input
        # and as the sensitive ``race`` attribute.
        df["RAC1P_bin"] = (df["RAC1P"].astype(int) == WHITE_RAC1P).astype(int)

        # Disability and remaining categoricals: NaN → -1 marker.
        for col in ["DIS", "ESP", "CIT", "MIG", "MIL", "ANC", "NATIVITY",
                    "DEAR", "DEYE", "DREM", "FER", "COW", "MAR", "RELP", "SEX"]:
            if col in df.columns:
                df[col] = _fillna_categorical(df[col])

        # WKHP: NaN means non-worker — treat as 0 hours.
        df["WKHP"] = pd.to_numeric(df["WKHP"], errors="coerce").fillna(0).astype(np.float32)
        df["AGEP"] = df["AGEP"].astype(np.float32)

        # ── Targets (binary) ────────────────────────────────────────────
        pincp = pd.to_numeric(df.get("PINCP", 0), errors="coerce").fillna(0)
        income_label = (pincp > 50000).astype(np.int64)

        esr = pd.to_numeric(df.get("ESR", 0), errors="coerce").fillna(0)
        employment_label = (esr == ESR_EMPLOYED_CIVILIAN).astype(np.int64)

        pubcov = pd.to_numeric(df.get("PUBCOV", 0), errors="coerce").fillna(0)
        public_coverage_label = (pubcov == 1).astype(np.int64)

        self.task_labels = {
            "income": torch.tensor(income_label.values, dtype=torch.long),
            "employment": torch.tensor(employment_label.values, dtype=torch.long),
            "public_coverage": torch.tensor(public_coverage_label.values, dtype=torch.long),
        }

        # ── Sensitive attributes ────────────────────────────────────────
        sex_bin = (df["SEX"].astype(int) == SEX_MALE).astype(np.int64)
        race_bin = df["RAC1P_bin"].astype(np.int64).values
        disability_bin = (df["DIS"].astype(int) == DIS_HAS).astype(np.int64)
        age_group = _agep_bin(df["AGEP"]).astype(np.int64)

        self.sensitive_attrs = {
            "sex": torch.tensor(sex_bin.values, dtype=torch.long),
            "race": torch.tensor(race_bin, dtype=torch.long),
            "disability": torch.tensor(disability_bin.values, dtype=torch.long),
            "age_group": torch.tensor(age_group.values, dtype=torch.long),
        }

        # ── Feature encoding ────────────────────────────────────────────
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

        # ── Dataset info ────────────────────────────────────────────────
        self.info = DatasetInfo(
            name="folktables_acs",
            num_features=self.features.shape[1],
            task_labels={
                "income": 2,
                "employment": 2,
                "public_coverage": 2,
            },
            sensitive_attrs={
                "sex": 2,
                "race": 2,
                "disability": 2,
                "age_group": N_AGE_GROUPS,
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


# ── Purpose registry ────────────────────────────────────────────────────


def get_folktables_purposes() -> list[PurposeSpec]:
    """Three-purpose Folktables setup with joint cardinality cap = 12.

    Joint cardinalities (= product of class counts of disallowed attrs):
        income_prediction:           sex(2) × race(2)              = 4
        employment_analysis:         sex(2) × race(2) × dis(2)     = 8
        public_coverage_assessment:  sex(2) × race(2) × age(3)     = 12
    """
    return [
        PurposeSpec(
            name="income_prediction",
            allowed_tasks=["income"],
            disallowed_attrs=["sex", "race"],
            task_type="classification",
            allowed_task_dims={"income": 2},
            disallowed_attr_dims={"sex": 2, "race": 2},
        ),
        PurposeSpec(
            name="employment_analysis",
            allowed_tasks=["employment"],
            disallowed_attrs=["sex", "race", "disability"],
            task_type="classification",
            allowed_task_dims={"employment": 2},
            disallowed_attr_dims={"sex": 2, "race": 2, "disability": 2},
        ),
        PurposeSpec(
            name="public_coverage_assessment",
            allowed_tasks=["public_coverage"],
            disallowed_attrs=["sex", "race", "age_group"],
            task_type="classification",
            allowed_task_dims={"public_coverage": 2},
            disallowed_attr_dims={"sex": 2, "race": 2, "age_group": N_AGE_GROUPS},
        ),
    ]


# Convenience constant for callers that want the (allowed_task, disallowed)
# tuple form rather than the PurposeSpec dataclass.
FOLKTABLES_PURPOSE_REGISTRY: dict[str, tuple[str, list[str]]] = {
    "income_prediction":          ("income",          ["sex", "race"]),
    "employment_analysis":        ("employment",      ["sex", "race", "disability"]),
    "public_coverage_assessment": ("public_coverage", ["sex", "race", "age_group"]),
}


def joint_cardinality(purpose_name: str) -> int:
    """Return ∏ |attr| for the disallowed attribute set of a purpose."""
    dims = {
        "sex": 2, "race": 2, "disability": 2, "age_group": N_AGE_GROUPS,
    }
    _, disallowed = FOLKTABLES_PURPOSE_REGISTRY[purpose_name]
    prod = 1
    for a in disallowed:
        prod *= dims[a]
    return prod


def assert_joint_cardinality_cap(cap: int = 12) -> None:
    """Raises if any purpose's joint cardinality exceeds the rank cap."""
    for name in FOLKTABLES_PURPOSE_REGISTRY:
        c = joint_cardinality(name)
        assert c <= cap, f"purpose {name} joint cardinality {c} exceeds cap {cap}"


# Run the cardinality check at import time so misconfiguration fails fast.
assert_joint_cardinality_cap()
