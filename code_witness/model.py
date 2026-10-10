"""Recipient-local hard-bit service at the two predeclared capacities."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.model import FixedScaler


HIDDEN = 64
ALLOWED_BITS = frozenset({10, 16})
WIRE_SCHEMA = "code-utility-hardbits-v1"


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        array = np.ascontiguousarray(state[name].detach().cpu().numpy())
        digest.update(name.encode() + b"\0")
        digest.update(str(array.dtype).encode() + b"\0")
        digest.update(str(array.shape).encode() + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


class _Mapper(nn.Module):
    def __init__(self, latent_dim: int, bits: int):
        super().__init__()
        self.layer1 = nn.Linear(latent_dim, HIDDEN)
        self.layer2 = nn.Linear(HIDDEN, bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.layer2(F.relu(self.layer1(x))))


class _CodeHead(nn.Module):
    def __init__(self, bits: int, classes: int):
        super().__init__()
        self.layer1 = nn.Linear(bits, HIDDEN)
        self.layer2 = nn.Linear(HIDDEN, classes)

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.gelu(self.layer1(code)))


class HardCodeModel(nn.Module):
    """Hard values in both fitting and inference; straight-through gradients in fitting."""

    def __init__(self, latent_dim: int, classes: tuple[int, int], bits: int):
        super().__init__()
        if latent_dim < 1 or tuple(classes) != (2, 6) or bits not in ALLOWED_BITS:
            raise ValueError("requires 16-D source-compatible tasks and 10 or 16 bits")
        self.latent_dim = int(latent_dim)
        self.classes = tuple(int(k) for k in classes)
        self.bits_count = int(bits)
        self.mappers = nn.ModuleList([_Mapper(latent_dim, bits) for _ in (0, 1)])
        self.heads = nn.ModuleList([_CodeHead(bits, k) for k in classes])

    def bits(self, purpose: int, x: torch.Tensor, straight_through: bool = True) -> torch.Tensor:
        if purpose not in (0, 1) or x.ndim != 2 or x.shape[1] != self.latent_dim:
            raise ValueError("wrong purpose or scaled latent shape")
        p = self.mappers[purpose](x)
        hard = (p >= 0.5).to(p.dtype)
        return p + (hard - p).detach() if straight_through else hard

    def logits_from_code(self, purpose: int, code: torch.Tensor) -> torch.Tensor:
        if purpose not in (0, 1) or code.ndim != 2 or code.shape[1] != self.bits_count:
            raise ValueError("wrong purpose or hard binary code shape")
        if not bool(torch.all((code == 0) | (code == 1))):
            raise ValueError("head accepts only hard binary code")
        return self.heads[purpose](code.float())

    def forward(self, purpose: int, x: torch.Tensor,
                straight_through: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        code = self.bits(purpose, x, straight_through)
        return code, self.heads[purpose](code)


class HardCodeRelease:
    """The complete recipient wire is code, code-derived probability and decision."""

    def __init__(self, model: HardCodeModel, scaler: FixedScaler,
                 version: str, source_sha256: str,
                 calibrators: tuple[dict, dict] | None = None):
        if scaler.dim != model.latent_dim or not version or len(source_sha256) != 64:
            raise ValueError("release model/scaler/version/source pin invalid")
        self.model = HardCodeModel(model.latent_dim, model.classes, model.bits_count)
        self.model.load_state_dict(model.state_dict(), strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.scaler = scaler
        self.version = version
        self.source_sha256 = source_sha256
        self.scaler_sha256 = state_sha256({
            "scaler.center": torch.from_numpy(scaler.center),
            "scaler.divisor": torch.from_numpy(scaler.divisor),
        })
        state = self.model.state_dict()
        self.model_sha256 = state_sha256(state)
        self.head_sha256 = tuple(state_sha256({k: v for k, v in state.items()
                                               if k.startswith(f"heads.{i}.")})
                                 for i in (0, 1))
        if calibrators is not None:
            if len(calibrators) != 2:
                raise ValueError("one fixed global calibrator per task required")
            for i, record in enumerate(calibrators):
                if (record.get("schema") != "pcrl-portfolio-global-temp-v1"
                        or record.get("task") != ("income", "occupation")[i]
                        or not 0.25 <= float(record.get("alpha", float("nan"))) <= 4.0):
                    raise ValueError("incompatible global temperature")
            self.calibrators = tuple(dict(item) for item in calibrators)
        else:
            self.calibrators = None

    def uncalibrated_probabilities_from_code(self, purpose: int,
                                             code: np.ndarray) -> np.ndarray:
        value = np.asarray(code)
        if (value.ndim != 2 or value.shape[1] != self.model.bits_count
                or not np.isin(value, (0, 1)).all()):
            raise ValueError("only hard binary code reaches task head")
        with torch.no_grad():
            t = torch.from_numpy(value.astype(np.float32, copy=False))
            logits = self.model.logits_from_code(purpose, t).numpy().astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def probabilities_from_code(self, purpose: int, code: np.ndarray) -> np.ndarray:
        prob = self.uncalibrated_probabilities_from_code(purpose, code)
        if self.calibrators is None:
            return prob
        alpha = float(self.calibrators[purpose]["alpha"])
        if alpha == 1.0:
            return prob
        clipped = np.maximum(prob, 1e-12)
        clipped /= clipped.sum(axis=1, keepdims=True)
        logs = alpha * np.log(clipped)
        logs -= logs.max(axis=1, keepdims=True)
        exp = np.exp(logs)
        return exp / exp.sum(axis=1, keepdims=True)

    def recipient(self, purpose: int, latent: np.ndarray) -> dict[str, Any]:
        x = self.scaler.transform(purpose, latent)
        with torch.no_grad():
            code = self.model.bits(purpose, torch.from_numpy(x), False).to(torch.uint8).numpy().copy()
        probability = self.probabilities_from_code(purpose, code)
        metadata = {
            "schema": WIRE_SCHEMA, "purpose": purpose, "bits": self.model.bits_count,
            "bit_order": "c0_least_significant",
            "capacity_states": 1 << self.model.bits_count,
            "version": self.version, "source_sha256": self.source_sha256,
            "model_sha256": self.model_sha256,
            "scaler_sha256": self.scaler_sha256,
            "head_sha256": self.head_sha256[purpose],
        }
        if self.calibrators is not None:
            canonical = (json.dumps(self.calibrators[purpose], sort_keys=True,
                                    separators=(",", ":"), allow_nan=False) + "\n").encode()
            metadata["calibrator_sha256"] = hashlib.sha256(canonical).hexdigest()
            metadata["calibration_family"] = "H-GLOBAL-TEMP"
        return {"code": code, "probabilities": probability,
                "decision": probability.argmax(axis=1).astype(np.int64),
                "metadata": metadata}

    def recipient_from_raw(self, purpose: int, raw: np.ndarray,
                           columns: tuple[str, ...], encoder: Any) -> dict[str, Any]:
        if encoder.model_sha256 != self.source_sha256:
            raise ValueError("frozen source checkpoint differs from release pin")
        latent = encoder.encode_one(purpose, raw, columns, role="DEPLOYMENT")
        return self.recipient(purpose, latent)
