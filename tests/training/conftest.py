# tests/training/conftest.py
"""Shared fixtures for LAFTR-proxy trainer tests."""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from pcrl.data.base import collate_pcrl_batch
from pcrl.purposes.spec import PurposeRegistry, PurposeSpec


class ToyPCRLDataset(Dataset):
    def __init__(self, n: int = 256, d: int = 16, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, d, generator=g)
        # task labels correlated with first 4 dims
        self.y_task = (self.x[:, :4].sum(dim=1) > 0).long()
        # sensitive attrs correlated with last 4 dims (so they're erasable)
        self.attr_a = (self.x[:, -4:].sum(dim=1) > 0).long()
        self.attr_b = (self.x[:, -2:].sum(dim=1) > 0).long()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        return {
            "features": self.x[i],
            "task_labels": {"toy_task": self.y_task[i]},
            "sensitive_attrs": {"attr_a": self.attr_a[i], "attr_b": self.attr_b[i]},
        }


@pytest.fixture
def toy_purposes() -> list[PurposeSpec]:
    return [
        PurposeSpec(
            name="purpose_0",
            task_type="classification",
            allowed_tasks=["toy_task"],
            allowed_task_dims={"toy_task": 2},
            disallowed_attrs=["attr_a"],
            disallowed_attr_dims={"attr_a": 2},
        ),
        PurposeSpec(
            name="purpose_1",
            task_type="classification",
            allowed_tasks=["toy_task"],
            allowed_task_dims={"toy_task": 2},
            disallowed_attrs=["attr_b"],
            disallowed_attr_dims={"attr_b": 2},
        ),
    ]


@pytest.fixture
def toy_loaders():
    train = DataLoader(ToyPCRLDataset(n=256, seed=0), batch_size=32, shuffle=True,
                       collate_fn=collate_pcrl_batch)
    val = DataLoader(ToyPCRLDataset(n=64, seed=1), batch_size=32, shuffle=False,
                     collate_fn=collate_pcrl_batch)
    return train, val
