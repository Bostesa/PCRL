"""Adult/Census Income dataset for PCRL."""

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

# ── Grouping maps ────────────────────────────────────────────────────────

OCCUPATION_GROUPS: dict[str, int] = {
    "Prof-specialty": 0,
    "Tech-support": 0,
    "Exec-managerial": 1,
    "Adm-clerical": 1,
    "Other-service": 2,
    "Priv-house-serv": 2,
    "Protective-serv": 2,
    "Handlers-cleaners": 2,
    "Sales": 3,
    "Machine-op-inspct": 4,
    "Transport-moving": 4,
    "Craft-repair": 4,
    "Farming-fishing": 4,
    "Armed-Forces": 5,
}

EDUCATION_LEVELS: dict[str, int] = {
    "Preschool": 0,
    "1st-4th": 0,
    "5th-6th": 0,
    "7th-8th": 0,
    "9th": 0,
    "10th": 0,
    "11th": 0,
    "12th": 0,
    "HS-grad": 1,
    "Some-college": 2,
    "Assoc-voc": 2,
    "Assoc-acdm": 2,
    "Bachelors": 3,
    "Masters": 3,
    "Doctorate": 3,
    "Prof-school": 3,
}

MARITAL_BINARY: dict[str, int] = {
    "Never-married": 0,
    "Divorced": 0,
    "Separated": 0,
    "Widowed": 0,
    "Married-civ-spouse": 1,
    "Married-spouse-absent": 1,
    "Married-AF-spouse": 1,
}


class AdultDataset(PCRLDataset):
    """UCI Adult/Census Income dataset.

    Predicts whether income exceeds $50K/year based on census data.
    Sensitive attributes include race, sex, age group, and marital status.

    Task labels:
        - income: Binary (0 = <=50K, 1 = >50K)
        - occupation_group: 6-class (Professional, Executive, Service, Sales, Manual, Military)
        - education_level: 4-class (Less than HS, HS, Some College, College+)

    Sensitive attributes:
        - sex: Binary (0 = Female, 1 = Male)
        - race: 5-class
        - age_group: 4-class (binned)
        - marital_status: Binary (0 = Not married, 1 = Married)
        - income: Binary (also usable as sensitive attr)
    """

    DOWNLOAD_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    TRAIN_FILE = "adult.data"
    TEST_FILE = "adult.test"

    COLUMN_NAMES = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    CATEGORICAL_COLUMNS = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    NUMERICAL_COLUMNS = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    # Fixed category sets for consistent one-hot encoding across splits
    CATEGORY_VALUES: dict[str, list[str]] = {
        "workclass": [
            "Federal-gov", "Local-gov", "Never-worked", "Private",
            "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay",
        ],
        "education": sorted(EDUCATION_LEVELS.keys()),
        "marital-status": sorted(MARITAL_BINARY.keys()),
        "occupation": sorted(OCCUPATION_GROUPS.keys()),
        "relationship": [
            "Husband", "Not-in-family", "Other-relative",
            "Own-child", "Unmarried", "Wife",
        ],
        "race": [
            "Amer-Indian-Eskimo", "Asian-Pac-Islander",
            "Black", "Other", "White",
        ],
        "sex": ["Female", "Male"],
        "native-country": [
            "Cambodia", "Canada", "China", "Columbia", "Cuba",
            "Dominican-Republic", "Ecuador", "El-Salvador", "England",
            "France", "Germany", "Greece", "Guatemala", "Haiti",
            "Holand-Netherlands", "Honduras", "Hong", "Hungary",
            "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan",
            "Laos", "Mexico", "Nicaragua", "Outlying-US(Guam-USVI-etc)",
            "Peru", "Philippines", "Poland", "Portugal", "Puerto-Rico",
            "Scotland", "South", "Taiwan", "Thailand",
            "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia",
        ],
    }

    def __init__(
        self,
        purposes: list[PurposeSpec],
        root: str = "data",
        split: str = "train",
        download: bool = True,
        transform: Any | None = None,
        age_bins: list[int] | None = None,
        norm_stats: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.age_bins = age_bins or [0, 25, 45, 65, 100]
        self._encoders: dict[str, dict[str, int]] = {}
        self._norm_stats = norm_stats
        self.norm_stats: dict[str, tuple[float, float]] = {}
        super().__init__(purposes, root, split, download, transform)

    def _load_data(self) -> None:
        """Load and preprocess the Adult dataset (with synthetic fallback)."""
        root_path = Path(self.root) / "adult"
        root_path.mkdir(parents=True, exist_ok=True)

        train_path = root_path / self.TRAIN_FILE
        test_path = root_path / self.TEST_FILE

        # Download if needed
        if self.download and not train_path.exists():
            try:
                self._download_dataset(root_path)
            except Exception as e:
                logger.warning(f"Download failed ({e}). Using synthetic fallback.")
                self._generate_fallback(root_path)

        if not train_path.exists():
            logger.warning("Data files not found. Generating synthetic fallback.")
            self._generate_fallback(root_path)

        # Load appropriate split
        if self.split in ("train", "val"):
            df = self._load_csv(train_path)
            np.random.seed(42)
            indices = np.random.permutation(len(df))
            split_idx = int(0.8 * len(df))
            if self.split == "train":
                df = df.iloc[indices[:split_idx]].reset_index(drop=True)
            else:
                df = df.iloc[indices[split_idx:]].reset_index(drop=True)
        else:
            df = self._load_csv(test_path, skip_first=True)

        self._preprocess(df)

    def _download_dataset(self, root_path: Path) -> None:
        """Download the Adult dataset from UCI."""
        import urllib.request

        for filename in [self.TRAIN_FILE, self.TEST_FILE]:
            url = f"{self.DOWNLOAD_URL}{filename}"
            filepath = root_path / filename
            logger.info(f"Downloading {url}...")
            urllib.request.urlretrieve(url, filepath)

    def _generate_fallback(self, root_path: Path) -> None:
        """Generate a synthetic dataset with the same schema as Adult."""
        rng = np.random.RandomState(42)
        n_train = 32561
        n_test = 16281

        workclasses = [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
        ]
        educations = list(EDUCATION_LEVELS.keys())
        marital_statuses = list(MARITAL_BINARY.keys())
        occupations = list(OCCUPATION_GROUPS.keys())
        relationships = [
            "Wife",
            "Own-child",
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Unmarried",
        ]
        races = [
            "White",
            "Black",
            "Asian-Pac-Islander",
            "Amer-Indian-Eskimo",
            "Other",
        ]
        sexes = ["Female", "Male"]
        countries = ["United-States"]
        incomes = ["<=50K", ">50K"]

        def _make_df(n: int) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "age": rng.randint(17, 90, n),
                    "workclass": rng.choice(workclasses, n),
                    "fnlwgt": rng.randint(10000, 1500000, n),
                    "education": rng.choice(educations, n),
                    "education-num": rng.randint(1, 16, n),
                    "marital-status": rng.choice(marital_statuses, n),
                    "occupation": rng.choice(occupations, n),
                    "relationship": rng.choice(relationships, n),
                    "race": rng.choice(races, n),
                    "sex": rng.choice(sexes, n),
                    "capital-gain": rng.randint(0, 100000, n),
                    "capital-loss": rng.randint(0, 5000, n),
                    "hours-per-week": rng.randint(1, 99, n),
                    "native-country": rng.choice(countries, n),
                    "income": rng.choice(incomes, n),
                }
            )

        train_df = _make_df(n_train)
        train_df.to_csv(
            root_path / self.TRAIN_FILE,
            index=False,
            header=False,
        )

        # Test file has a header line that gets skipped
        test_df = _make_df(n_test)
        with open(root_path / self.TEST_FILE, "w") as f:
            f.write("|1x3 Cross validator\n")
        test_df.to_csv(
            root_path / self.TEST_FILE,
            index=False,
            header=False,
            mode="a",
        )
        logger.info(f"Generated synthetic Adult data at {root_path}")

    def _load_csv(self, path: Path, skip_first: bool = False) -> pd.DataFrame:
        """Load a CSV file into a DataFrame."""
        skiprows = 1 if skip_first else 0
        df = pd.read_csv(
            path,
            names=self.COLUMN_NAMES,
            sep=r",\s*",
            engine="python",
            na_values="?",
            skiprows=skiprows,
        )
        return df

    def _preprocess(self, df: pd.DataFrame) -> None:
        """Preprocess the dataframe into tensors."""
        df = df.dropna().reset_index(drop=True)
        # Keep a copy of the post-dropna, post-split dataframe so callers
        # can derive integer-coded labels for any of the 14 input columns
        # (used by the 91-pair composition benchmark).
        self.raw_df = df.copy()

        # ── Feature encoding ─────────────────────────────────────────────
        encoded_features: list[np.ndarray] = []

        for col in self.NUMERICAL_COLUMNS:
            values = df[col].values.astype(np.float32)
            if self._norm_stats and col in self._norm_stats:
                mean, std = self._norm_stats[col]
            else:
                mean, std = float(values.mean()), float(values.std()) + 1e-8
            self.norm_stats[col] = (mean, std)
            values = (values - mean) / std
            encoded_features.append(values.reshape(-1, 1))

        # One-hot encode categoricals with fixed category sets for consistency
        for col in self.CATEGORICAL_COLUMNS:
            categories = self.CATEGORY_VALUES.get(col)
            if categories is not None:
                cat_series = pd.Categorical(df[col], categories=categories)
                one_hot = pd.get_dummies(cat_series, prefix=col)
            else:
                one_hot = pd.get_dummies(df[col], prefix=col)
            encoded_features.append(one_hot.values.astype(np.float32))

        self.features = torch.tensor(
            np.concatenate(encoded_features, axis=1),
            dtype=torch.float32,
        )

        # ── Task labels ──────────────────────────────────────────────────
        # Income (binary)
        income_map = {"<=50K": 0, "<=50K.": 0, ">50K": 1, ">50K.": 1}
        income_labels = torch.tensor(
            df["income"].map(income_map).fillna(0).values.astype(np.int64),
            dtype=torch.long,
        )

        # Occupation group (6 classes)
        occ_mapped = df["occupation"].map(OCCUPATION_GROUPS).fillna(2)
        occupation_group_labels = torch.tensor(
            occ_mapped.values.astype(np.int64), dtype=torch.long
        )

        # Education level (4 classes)
        edu_mapped = df["education"].map(EDUCATION_LEVELS).fillna(0)
        education_level_labels = torch.tensor(
            edu_mapped.values.astype(np.int64), dtype=torch.long
        )

        self.task_labels = {
            "income": income_labels,
            "occupation_group": occupation_group_labels,
            "education_level": education_level_labels,
        }

        # ── Sensitive attributes ─────────────────────────────────────────
        # Sex (binary)
        sex_map = {"Female": 0, "Male": 1}
        sex_labels = torch.tensor(
            df["sex"].map(sex_map).fillna(0).values.astype(np.int64),
            dtype=torch.long,
        )

        # Race (multi-class)
        race_categories = sorted(df["race"].unique().tolist())
        race_map = {cat: i for i, cat in enumerate(race_categories)}
        race_labels = torch.tensor(
            df["race"].map(race_map).values.astype(np.int64),
            dtype=torch.long,
        )
        n_race = len(race_categories)

        # Age group (binned)
        age_binned = pd.cut(
            df["age"],
            bins=self.age_bins,
            labels=list(range(len(self.age_bins) - 1)),
        )
        age_labels = torch.tensor(
            age_binned.values.astype(int), dtype=torch.long
        )

        # Marital status (binary)
        marital_mapped = df["marital-status"].map(MARITAL_BINARY).fillna(0)
        marital_labels = torch.tensor(
            marital_mapped.values.astype(np.int64), dtype=torch.long
        )

        self.sensitive_attrs = {
            "sex": sex_labels,
            "race": race_labels,
            "age_group": age_labels,
            "marital_status": marital_labels,
            "income": income_labels,
        }

        # ── Dataset info ─────────────────────────────────────────────────
        self.info = DatasetInfo(
            name="adult",
            num_features=self.features.shape[1],
            task_labels={
                "income": 2,
                "occupation_group": 6,
                "education_level": 4,
            },
            sensitive_attrs={
                "sex": 2,
                "race": n_race,
                "age_group": len(self.age_bins) - 1,
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


def get_adult_purposes() -> list[PurposeSpec]:
    """Get standard purpose specifications for the Adult dataset.

    Returns three purposes as described in the PCRL paper:
    1. income_prediction: Predict income, hide race and sex
    2. employment_analysis: Predict occupation group, hide race, age, marital status
    3. education_assessment: Predict education level, hide sex, race, income
    """
    return [
        PurposeSpec(
            name="income_prediction",
            allowed_tasks=["income"],
            disallowed_attrs=["race", "sex"],
            task_type="classification",
            allowed_task_dims={"income": 2},
            disallowed_attr_dims={"race": 5, "sex": 2},
        ),
        PurposeSpec(
            name="employment_analysis",
            allowed_tasks=["occupation_group"],
            disallowed_attrs=["race", "age_group", "marital_status"],
            task_type="classification",
            allowed_task_dims={"occupation_group": 6},
            disallowed_attr_dims={
                "race": 5,
                "age_group": 4,
                "marital_status": 2,
            },
        ),
        PurposeSpec(
            name="education_assessment",
            allowed_tasks=["education_level"],
            disallowed_attrs=["sex", "race", "income"],
            task_type="classification",
            allowed_task_dims={"education_level": 4},
            disallowed_attr_dims={"sex": 2, "race": 5, "income": 2},
        ),
    ]
