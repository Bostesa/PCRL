"""Base dataset class with purpose specifications."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from pcrl.utils.config import PurposeSpec


@dataclass
class DatasetInfo:
    """Metadata about a dataset.

    Attributes:
        name: Dataset identifier.
        num_features: Number of input features.
        task_labels: Mapping from task name to number of classes (or 1 for regression).
        sensitive_attrs: Mapping from attribute name to number of classes.
        num_samples: Total number of samples.
    """
    name: str
    num_features: int
    task_labels: dict[str, int]
    sensitive_attrs: dict[str, int]
    num_samples: int


class PCRLDataset(Dataset, ABC):
    """Abstract base class for PCRL datasets.

    Subclasses must implement:
        - _load_data(): Load and preprocess the dataset.
        - __len__(): Return the number of samples.
        - __getitem__(): Return a single sample.

    Attributes:
        purposes: List of purpose specifications for this dataset.
        info: Dataset metadata.
    """

    def __init__(
        self,
        purposes: list[PurposeSpec],
        root: str = "data",
        split: str = "train",
        download: bool = True,
        transform: Any | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            purposes: List of purpose specifications.
            root: Root directory for dataset storage.
            split: Dataset split ("train", "val", or "test").
            download: Whether to download the dataset if not present.
            transform: Optional transform to apply to features.
        """
        super().__init__()
        self.purposes = purposes
        self.root = root
        self.split = split
        self.download = download
        self.transform = transform

        self._validate_split(split)
        self._load_data()
        self._validate_purposes()

    def _validate_split(self, split: str) -> None:
        """Validate that the split is valid."""
        valid_splits = {"train", "val", "test"}
        if split not in valid_splits:
            raise ValueError(f"split must be one of {valid_splits}, got {split}")

    def _validate_purposes(self) -> None:
        """Validate that purposes are compatible with this dataset."""
        for purpose in self.purposes:
            # Check that allowed tasks exist
            for task in purpose.allowed_tasks:
                if task not in self.info.task_labels:
                    raise ValueError(
                        f"Purpose '{purpose.name}' specifies task '{task}' "
                        f"but dataset only has tasks: {list(self.info.task_labels.keys())}"
                    )
            # Check that disallowed attrs exist
            for attr in purpose.disallowed_attrs:
                if attr not in self.info.sensitive_attrs:
                    raise ValueError(
                        f"Purpose '{purpose.name}' specifies disallowed attr '{attr}' "
                        f"but dataset only has attrs: {list(self.info.sensitive_attrs.keys())}"
                    )

    @abstractmethod
    def _load_data(self) -> None:
        """Load and preprocess the dataset. Must set self.info."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single sample as a dictionary.

        Returns:
            Dictionary with keys:
                - "features": Input features tensor.
                - "task_labels": Dict mapping task name to label tensor.
                - "sensitive_attrs": Dict mapping attr name to label tensor.
        """
        pass

    def get_purpose_batch(
        self,
        batch: dict[str, Any],
        purpose: PurposeSpec,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Extract relevant labels for a specific purpose.

        Args:
            batch: Batch dictionary from the dataloader.
            purpose: The purpose specification.

        Returns:
            Tuple of (features, task_labels, sensitive_attrs) where:
                - features: Input features.
                - task_labels: Only labels for allowed tasks.
                - sensitive_attrs: Only labels for disallowed attributes.
        """
        features = batch["features"]

        task_labels = {
            task: batch["task_labels"][task]
            for task in purpose.allowed_tasks
            if task in batch["task_labels"]
        }

        sensitive_attrs = {
            attr: batch["sensitive_attrs"][attr]
            for attr in purpose.disallowed_attrs
            if attr in batch["sensitive_attrs"]
        }

        return features, task_labels, sensitive_attrs


def collate_pcrl_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function for PCRL datasets.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Collated batch dictionary.
    """
    features = torch.stack([sample["features"] for sample in batch])

    # Collate task labels
    task_keys = batch[0]["task_labels"].keys()
    task_labels = {
        key: torch.stack([sample["task_labels"][key] for sample in batch])
        for key in task_keys
    }

    # Collate sensitive attrs
    attr_keys = batch[0]["sensitive_attrs"].keys()
    sensitive_attrs = {
        key: torch.stack([sample["sensitive_attrs"][key] for sample in batch])
        for key in attr_keys
    }

    return {
        "features": features,
        "task_labels": task_labels,
        "sensitive_attrs": sensitive_attrs,
    }
