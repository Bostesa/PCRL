"""Adult/Census Income dataset for PCRL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pcrl.data.base import DatasetInfo, PCRLDataset
from pcrl.utils.config import PurposeSpec


class AdultDataset(PCRLDataset):
    """UCI Adult/Census Income dataset.

    Predicts whether income exceeds $50K/year based on census data.
    Sensitive attributes include race, sex, and age.

    Features:
        - age, workclass, education, education-num, marital-status,
        - occupation, relationship, race, sex, capital-gain,
        - capital-loss, hours-per-week, native-country

    Task labels:
        - income: Binary (0 = <=50K, 1 = >50K)

    Sensitive attributes:
        - sex: Binary (0 = Female, 1 = Male)
        - race: Multi-class (White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other)
        - age: Binned into groups
    """

    DOWNLOAD_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    TRAIN_FILE = "adult.data"
    TEST_FILE = "adult.test"

    COLUMN_NAMES = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    CATEGORICAL_COLUMNS = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]

    NUMERICAL_COLUMNS = [
        "age", "fnlwgt", "education-num", "capital-gain",
        "capital-loss", "hours-per-week"
    ]

    def __init__(
        self,
        purposes: list[PurposeSpec],
        root: str = "data",
        split: str = "train",
        download: bool = True,
        transform: Any | None = None,
        age_bins: list[int] | None = None,
    ) -> None:
        """Initialize the Adult dataset.

        Args:
            purposes: List of purpose specifications.
            root: Root directory for dataset storage.
            split: Dataset split ("train", "val", or "test").
            download: Whether to download the dataset if not present.
            transform: Optional transform to apply to features.
            age_bins: Bin edges for age discretization. Defaults to [0, 25, 45, 65, 100].
        """
        self.age_bins = age_bins or [0, 25, 45, 65, 100]
        self._encoders: dict[str, dict[str, int]] = {}
        super().__init__(purposes, root, split, download, transform)

    def _load_data(self) -> None:
        """Load and preprocess the Adult dataset."""
        root_path = Path(self.root) / "adult"
        root_path.mkdir(parents=True, exist_ok=True)

        train_path = root_path / self.TRAIN_FILE
        test_path = root_path / self.TEST_FILE

        # Download if needed
        if self.download and not train_path.exists():
            self._download_dataset(root_path)

        # Load appropriate split
        if self.split in ("train", "val"):
            df = self._load_csv(train_path)
            # Split train into train/val (80/20)
            np.random.seed(42)
            indices = np.random.permutation(len(df))
            split_idx = int(0.8 * len(df))
            if self.split == "train":
                df = df.iloc[indices[:split_idx]].reset_index(drop=True)
            else:
                df = df.iloc[indices[split_idx:]].reset_index(drop=True)
        else:
            df = self._load_csv(test_path, skip_first=True)

        # Preprocess
        self._preprocess(df)

    def _download_dataset(self, root_path: Path) -> None:
        """Download the Adult dataset."""
        import urllib.request

        for filename in [self.TRAIN_FILE, self.TEST_FILE]:
            url = f"{self.DOWNLOAD_URL}{filename}"
            filepath = root_path / filename
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, filepath)

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
        # Drop rows with missing values
        df = df.dropna().reset_index(drop=True)

        # Store category mappings for task labels
        self._workclass_categories = sorted(df["workclass"].unique().tolist())
        self._education_categories = sorted(df["education"].unique().tolist())

        # Encode categorical columns (excluding those used as tasks/sensitive)
        encoded_features = []
        excluded_categorical = {"sex", "race", "workclass", "education"}

        for col in self.NUMERICAL_COLUMNS:
            values = df[col].values.astype(np.float32)
            # Standardize numerical features
            mean, std = values.mean(), values.std() + 1e-8
            values = (values - mean) / std
            encoded_features.append(values.reshape(-1, 1))

        for col in self.CATEGORICAL_COLUMNS:
            if col in excluded_categorical:
                continue
            # One-hot encode
            one_hot = pd.get_dummies(df[col], prefix=col)
            encoded_features.append(one_hot.values.astype(np.float32))

        # Concatenate all features
        self.features = torch.tensor(
            np.concatenate(encoded_features, axis=1),
            dtype=torch.float32,
        )

        # Process task labels
        # Income (binary)
        income_map = {"<=50K": 0, "<=50K.": 0, ">50K": 1, ">50K.": 1}
        income_labels = torch.tensor(
            df["income"].map(income_map).values,
            dtype=torch.long,
        )

        # Workclass (multi-class)
        workclass_map = {cat: i for i, cat in enumerate(self._workclass_categories)}
        workclass_labels = torch.tensor(
            df["workclass"].map(workclass_map).values,
            dtype=torch.long,
        )

        # Education (multi-class)
        education_map = {cat: i for i, cat in enumerate(self._education_categories)}
        education_labels = torch.tensor(
            df["education"].map(education_map).values,
            dtype=torch.long,
        )

        self.task_labels = {
            "income": income_labels,
            "workclass": workclass_labels,
            "education": education_labels,
        }

        # Process sensitive attributes
        # Sex
        sex_map = {"Female": 0, "Male": 1}
        sex_labels = torch.tensor(
            df["sex"].map(sex_map).values,
            dtype=torch.long,
        )

        # Race
        race_categories = sorted(df["race"].unique().tolist())
        race_map = {cat: i for i, cat in enumerate(race_categories)}
        race_labels = torch.tensor(
            df["race"].map(race_map).values,
            dtype=torch.long,
        )
        self._race_categories = race_categories

        # Age (binned)
        age_binned = pd.cut(
            df["age"],
            bins=self.age_bins,
            labels=list(range(len(self.age_bins) - 1)),
        )
        age_labels = torch.tensor(age_binned.values.astype(int), dtype=torch.long)

        self.sensitive_attrs = {
            "sex": sex_labels,
            "race": race_labels,
            "age": age_labels,
            "income": income_labels,  # Also treat income as sensitive for some purposes
        }

        # Set dataset info
        self.info = DatasetInfo(
            name="adult",
            num_features=self.features.shape[1],
            task_labels={
                "income": 2,
                "workclass": len(self._workclass_categories),
                "education": len(self._education_categories),
            },
            sensitive_attrs={
                "sex": 2,
                "race": len(race_categories),
                "age": len(self.age_bins) - 1,
                "income": 2,
            },
            num_samples=len(self.features),
        )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single sample."""
        features = self.features[idx]

        if self.transform is not None:
            features = self.transform(features)

        return {
            "features": features,
            "task_labels": {k: v[idx] for k, v in self.task_labels.items()},
            "sensitive_attrs": {k: v[idx] for k, v in self.sensitive_attrs.items()},
        }


def get_adult_purposes() -> list[PurposeSpec]:
    """Get standard purpose specifications for the Adult dataset.

    Returns three purposes demonstrating different fairness constraints:
    1. income_prediction: Predict income while hiding sex and race
    2. workclass_prediction: Predict workclass while hiding sex, race, and income
    3. education_prediction: Predict education level while hiding sex, race, and income
    """
    return [
        PurposeSpec(
            name="income_prediction",
            allowed_tasks=["income"],
            disallowed_attrs=["sex", "race"],
            task_type="classification",
        ),
        PurposeSpec(
            name="workclass_prediction",
            allowed_tasks=["workclass"],
            disallowed_attrs=["sex", "race", "income"],
            task_type="classification",
        ),
        PurposeSpec(
            name="education_prediction",
            allowed_tasks=["education"],
            disallowed_attrs=["sex", "race", "income"],
            task_type="classification",
        ),
    ]
