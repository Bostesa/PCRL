"""Unchanged latent diagnostic F with a newly trained task-only head.

Unlike 10-bit D arms, F intentionally transmits the entire scaled latent.
Its complete view is attacked as a continuous feature plus task output.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.model import FixedScaler


class ContinuousHead(nn.Module):
    """The same 64-unit task head opportunity as the hard-code arms."""

    def __init__(self, latent_dim: int, classes: int):
        super().__init__()
        if latent_dim < 1 or classes < 2:
            raise ValueError("positive latent width and at least two classes required")
        self.layer1 = nn.Linear(latent_dim, 64)
        self.layer2 = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.gelu(self.layer1(x)))


class UnchangedLatentRelease:
    """F wire: unchanged fixed-scaled latent and its new head output."""

    def __init__(self, heads: tuple[ContinuousHead, ContinuousHead], scaler: FixedScaler,
                 version: str, source_sha256: str):
        if (len(heads) != 2 or not all(isinstance(h, ContinuousHead) for h in heads)
                or not version or len(source_sha256) != 64
                or any(head.layer1.in_features != scaler.dim for head in heads)):
            raise ValueError("F head, scaler, version or source pin invalid")
        self.heads = tuple(copy.deepcopy(head).eval() for head in heads)
        for head in self.heads:
            for parameter in head.parameters():
                parameter.requires_grad_(False)
        self.scaler = scaler
        self.version = version
        self.source_sha256 = source_sha256

    def recipient(self, purpose: int, latent: np.ndarray) -> dict:
        features = self.scaler.transform(purpose, latent)
        with torch.no_grad():
            probability = torch.softmax(
                self.heads[purpose](torch.from_numpy(features)), dim=1).numpy().copy()
        return {
            "features": features,
            "probabilities": probability,
            "decision": np.argmax(probability, axis=1).astype(np.int64),
            "metadata": {"schema": "latent-unchanged-v1", "purpose": purpose,
                         "latent_dimensions": self.scaler.dim, "version": self.version,
                         "source_sha256": self.source_sha256},
        }
