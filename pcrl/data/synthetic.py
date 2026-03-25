"""Synthetic data generator for testing PCRL end-to-end.

Generates a dataset with known structure:
- x[0:5] correlated with task label y_A
- x[5:10] correlated with task label y_B
- x[10:15] correlated with sensitive attribute z_1
- x[15:20] correlated with sensitive attribute z_2
- Cross-correlations between groups make attribute removal non-trivial

Two purposes:
  task_A: allowed=[y_A], disallowed=[z_1, z_2]
  task_B: allowed=[y_B], disallowed=[z_1]
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from pcrl.purposes.spec import PurposeSpec


class SyntheticPCRLDataset(Dataset):
    """Synthetic dataset for testing PCRL.

    Features are grouped into 4 blocks of 5 dimensions each (20 total).
    Cross-correlations between task features and sensitive features ensure
    that attribute removal is non-trivial.

    Attributes:
        features: (n_samples, 20) float tensor.
        task_labels: Dict of (n_samples,) long tensors for y_A, y_B.
        sensitive_attrs: Dict of (n_samples,) long tensors for z_1, z_2.
    """

    def __init__(
        self,
        n_samples: int = 5000,
        n_features: int = 20,
        cross_correlation: float = 0.3,
        noise_scale: float = 0.5,
        seed: int = 42,
    ) -> None:
        """Initialize the synthetic dataset.

        Args:
            n_samples: Number of samples to generate.
            n_features: Number of features (must be >= 20).
            cross_correlation: Strength of cross-correlation between
                task features and sensitive features.
            noise_scale: Scale of Gaussian noise added to labels.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.n_samples = n_samples
        rng = np.random.RandomState(seed)

        # Generate base features
        X = rng.randn(n_samples, n_features).astype(np.float32)

        # Generate task labels from feature blocks
        # y_A depends on x[0:5]
        y_A_signal = X[:, 0:5].sum(axis=1)
        y_A = (y_A_signal + noise_scale * rng.randn(n_samples) > 0).astype(
            np.int64
        )

        # y_B depends on x[5:10]
        y_B_signal = X[:, 5:10].sum(axis=1)
        y_B = (y_B_signal + noise_scale * rng.randn(n_samples) > 0).astype(
            np.int64
        )

        # Generate sensitive attributes with cross-correlations
        # z_1 depends on x[10:15] + cross-correlation with x[0:5]
        z_1_signal = (
            X[:, 10:15].sum(axis=1)
            + cross_correlation * X[:, 0:5].sum(axis=1)
        )
        z_1 = (z_1_signal + noise_scale * rng.randn(n_samples) > 0).astype(
            np.int64
        )

        # z_2 depends on x[15:20] + cross-correlation with x[5:10]
        z_2_signal = (
            X[:, 15:20].sum(axis=1)
            + cross_correlation * X[:, 5:10].sum(axis=1)
        )
        z_2 = (z_2_signal + noise_scale * rng.randn(n_samples) > 0).astype(
            np.int64
        )

        # Store as tensors
        self.features = torch.from_numpy(X)
        self.task_labels = {
            "y_A": torch.from_numpy(y_A),
            "y_B": torch.from_numpy(y_B),
        }
        self.sensitive_attrs = {
            "z_1": torch.from_numpy(z_1),
            "z_2": torch.from_numpy(z_2),
        }

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "task_labels": {k: v[idx] for k, v in self.task_labels.items()},
            "sensitive_attrs": {
                k: v[idx] for k, v in self.sensitive_attrs.items()
            },
        }


def get_synthetic_purposes() -> list[PurposeSpec]:
    """Return purpose specs for the synthetic dataset.

    Returns:
        List of two PurposeSpec objects:
        - task_A: allowed=[y_A], disallowed=[z_1, z_2]
        - task_B: allowed=[y_B], disallowed=[z_1]
    """
    return [
        PurposeSpec(
            name="task_A",
            allowed_tasks=["y_A"],
            disallowed_attrs=["z_1", "z_2"],
            task_type="classification",
            allowed_task_dims={"y_A": 2},
            disallowed_attr_dims={"z_1": 2, "z_2": 2},
        ),
        PurposeSpec(
            name="task_B",
            allowed_tasks=["y_B"],
            disallowed_attrs=["z_1"],
            task_type="classification",
            allowed_task_dims={"y_B": 2},
            disallowed_attr_dims={"z_1": 2},
        ),
    ]
