"""Unchanged latent diagnostic F with a newly trained task-only head.

Unlike 10-bit D arms, F intentionally transmits the entire scaled latent.
Its complete view is attacked as a continuous feature plus task output.
"""

from __future__ import annotations

import copy
import hashlib
import json

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
                 version: str, source_sha256: str,
                 calibrators: tuple[dict, dict] | None = None):
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
        if calibrators is not None:
            if len(calibrators) != 2 or any(
                record.get("schema") != "pcrl-portfolio-global-temp-v1"
                or record.get("task") != ("income", "occupation")[purpose]
                or not 0.25 <= float(record.get("alpha", float("nan"))) <= 4.0
                for purpose, record in enumerate(calibrators)
            ):
                raise ValueError("F requires one compatible fixed calibrator per task")
            self.calibrators = tuple(dict(record) for record in calibrators)
        else:
            self.calibrators = None

    def uncalibrated_probabilities_from_features(self, purpose: int,
                                                 features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if (purpose not in (0, 1) or x.ndim != 2 or x.shape[1] != self.scaler.dim
                or not np.isfinite(x).all()):
            raise ValueError("F head requires finite unchanged scaled latent")
        with torch.no_grad():
            logits = self.heads[purpose](torch.from_numpy(x)).numpy().astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        values = np.exp(logits)
        return values / values.sum(axis=1, keepdims=True)

    def recipient(self, purpose: int, latent: np.ndarray) -> dict:
        features = self.scaler.transform(purpose, latent)
        probability = self.uncalibrated_probabilities_from_features(purpose, features)
        if self.calibrators is not None:
            alpha = float(self.calibrators[purpose]["alpha"])
            clipped = np.maximum(probability, 1e-12)
            clipped /= clipped.sum(axis=1, keepdims=True)
            values = clipped ** alpha
            probability = values / values.sum(axis=1, keepdims=True)
        metadata = {"schema": "latent-unchanged-v1", "purpose": purpose,
                    "latent_dimensions": self.scaler.dim, "version": self.version,
                    "source_sha256": self.source_sha256}
        if self.calibrators is not None:
            payload = (json.dumps(self.calibrators[purpose], sort_keys=True,
                                  separators=(",", ":"), allow_nan=False) + "\n").encode()
            metadata["calibrator_sha256"] = hashlib.sha256(payload).hexdigest()
            metadata["calibration_family"] = "H-GLOBAL-TEMP"
        return {
            "features": features,
            "probabilities": probability,
            "decision": np.argmax(probability, axis=1).astype(np.int64),
            "metadata": metadata,
        }
