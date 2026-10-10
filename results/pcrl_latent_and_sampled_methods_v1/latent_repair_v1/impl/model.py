"""Exact ten-bit service transport for the frozen-latent D experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


BITS = 10
HIDDEN = 64
WIRE_SCHEMA = "latent-repair-hardbits-v1"


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        array = np.ascontiguousarray(state[name].detach().cpu().numpy())
        digest.update(name.encode() + b"\0")
        digest.update(str(array.dtype).encode() + b"\0")
        digest.update(str(array.shape).encode() + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _latent(value: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(value)
    if x.ndim != 2 or x.shape[1] != dim or not np.isfinite(x).all():
        raise ValueError(f"expected finite (n,{dim}) latent array")
    return np.asarray(x, dtype=np.float32)


@dataclass(frozen=True)
class FixedScaler:
    """Fit-role centering and attenuation, shared unchanged by every arm."""

    center: np.ndarray
    divisor: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float32)
        divisor = np.asarray(self.divisor, dtype=np.float32)
        if (center.ndim != 2 or center.shape[0] != 2 or center.shape != divisor.shape
                or not np.isfinite(center).all() or not np.isfinite(divisor).all()
                or np.any(divisor < 1.0)):
            raise ValueError("two purpose-wise finite centers and divisors >= 1 required")
        object.__setattr__(self, "center", center.copy())
        object.__setattr__(self, "divisor", divisor.copy())

    @classmethod
    def fit(cls, h0: np.ndarray, h1: np.ndarray) -> "FixedScaler":
        arrays = (np.asarray(h0, dtype=np.float64), np.asarray(h1, dtype=np.float64))
        if (any(x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all() for x in arrays)
                or arrays[0].shape[1] != arrays[1].shape[1]):
            raise ValueError("two finite nonempty purpose latent arrays required")
        center = np.stack([x.mean(axis=0) for x in arrays]).astype(np.float32)
        # std below one is preserved. This cannot amplify tiny-variance axes.
        divisor = np.stack([np.maximum(x.std(axis=0), 1.0) for x in arrays]).astype(np.float32)
        return cls(center, divisor)

    @property
    def dim(self) -> int:
        return int(self.center.shape[1])

    def transform(self, purpose: int, latent: np.ndarray) -> np.ndarray:
        if purpose not in (0, 1):
            raise ValueError("purpose must be 0 or 1")
        x = _latent(latent, self.dim)
        return ((x - self.center[purpose]) / self.divisor[purpose]).astype(np.float32)


class _Mapper(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layer1 = nn.Linear(dim, HIDDEN)
        self.layer2 = nn.Linear(HIDDEN, BITS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.layer2(F.relu(self.layer1(x))))


class _CodeHead(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.layer1 = nn.Linear(BITS, HIDDEN)
        self.layer2 = nn.Linear(HIDDEN, classes)

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.gelu(self.layer1(code)))


class HardCodeModel(nn.Module):
    """Purpose-local mapper and code-only head, with hard forward training."""

    def __init__(self, latent_dim: int, classes: tuple[int, int]):
        super().__init__()
        if latent_dim < 1 or len(classes) != 2 or min(classes) < 2:
            raise ValueError("invalid latent dimension or task class counts")
        self.latent_dim = int(latent_dim)
        self.classes = tuple(int(x) for x in classes)
        self.mappers = nn.ModuleList([_Mapper(latent_dim) for _ in range(2)])
        self.heads = nn.ModuleList([_CodeHead(k) for k in classes])

    def bits(self, purpose: int, x: torch.Tensor, straight_through: bool = True) -> torch.Tensor:
        if purpose not in (0, 1) or x.ndim != 2 or x.shape[1] != self.latent_dim:
            raise ValueError("wrong purpose or scaled latent shape")
        p = self.mappers[purpose](x)
        hard = (p >= 0.5).to(p.dtype)
        return p + (hard - p).detach() if straight_through else hard

    def logits_from_code(self, purpose: int, code: torch.Tensor) -> torch.Tensor:
        if (purpose not in (0, 1) or code.ndim != 2 or code.shape[1] != BITS
                or not bool(torch.all((code == 0) | (code == 1)))):
            raise ValueError("head accepts only ten hard bits")
        return self.heads[purpose](code.float())

    def forward(self, purpose: int, x: torch.Tensor,
                straight_through: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        code = self.bits(purpose, x, straight_through)
        # Numerical values are hard even when gradients use the ST surrogate.
        return code, self.heads[purpose](code)


class HardCodeRelease:
    """Service release; returned fields are exactly the public per-row view."""

    def __init__(self, model: HardCodeModel, scaler: FixedScaler,
                 version: str, source_sha256: str,
                 calibrators: tuple[dict, dict] | None = None):
        if scaler.dim != model.latent_dim or not version or len(source_sha256) != 64:
            raise ValueError("release model, scaler, version or source pin invalid")
        self.model = HardCodeModel(model.latent_dim, model.classes)
        self.model.load_state_dict(model.state_dict(), strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.scaler = scaler
        self.version = version
        self.source_sha256 = source_sha256
        state = self.model.state_dict()
        self.model_sha256 = _state_sha256(state)
        self.head_sha256 = tuple(_state_sha256({k: v for k, v in state.items()
                                                if k.startswith(f"heads.{purpose}.")})
                                 for purpose in (0, 1))
        if calibrators is not None:
            if len(calibrators) != 2:
                raise ValueError("one globally fixed calibrator per purpose required")
            for purpose, record in enumerate(calibrators):
                if (record.get("schema") != "pcrl-portfolio-global-temp-v1"
                        or record.get("task") != ("income", "occupation")[purpose]
                        or not 0.25 <= float(record.get("alpha", float("nan"))) <= 4.0):
                    raise ValueError("incompatible global temperature record")
            self.calibrators = tuple(dict(x) for x in calibrators)
        else:
            self.calibrators = None

    def uncalibrated_probabilities_from_code(self, purpose: int,
                                             code: np.ndarray) -> np.ndarray:
        """Float32 head logits followed by normalized float64 softmax."""
        bits = np.asarray(code)
        if (bits.ndim != 2 or bits.shape[1] != BITS
                or not np.isin(bits, (0, 1)).all()):
            raise ValueError("only ten hard binary coordinates may reach task head")
        with torch.no_grad():
            t = torch.from_numpy(bits.astype(np.float32, copy=False))
            logits = self.model.logits_from_code(purpose, t).numpy().astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        values = np.exp(logits)
        return values / values.sum(axis=1, keepdims=True)

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
        values = np.exp(logs)
        return values / values.sum(axis=1, keepdims=True)

    def recipient(self, purpose: int, latent: np.ndarray) -> dict[str, Any]:
        x = self.scaler.transform(purpose, latent)
        with torch.no_grad():
            bits = self.model.bits(purpose, torch.from_numpy(x), False).to(torch.uint8).numpy().copy()
        probability = self.probabilities_from_code(purpose, bits)
        metadata = {
            "schema": WIRE_SCHEMA, "purpose": purpose, "bits": BITS,
            "bit_order": "c0_least_significant", "capacity_states": 1 << BITS,
            "version": self.version, "source_sha256": self.source_sha256,
            "model_sha256": self.model_sha256,
            "head_sha256": self.head_sha256[purpose],
        }
        if self.calibrators is not None:
            canonical = json.dumps(self.calibrators[purpose], sort_keys=True,
                                   separators=(",", ":"), allow_nan=False).encode() + b"\n"
            metadata["calibrator_sha256"] = hashlib.sha256(canonical).hexdigest()
            metadata["calibration_family"] = "H-GLOBAL-TEMP"
        return {
            "code": bits,
            "probabilities": probability,
            "decision": np.argmax(probability, axis=1).astype(np.int64),
            "metadata": metadata,
        }

    def recipient_from_raw(self, purpose: int, raw: np.ndarray,
                           columns: tuple[str, ...], encoder: Any) -> dict[str, Any]:
        """Pinned raw schema -> purpose's frozen encoder -> actual wire release."""
        if encoder.model_sha256 != self.source_sha256:
            raise ValueError("frozen source checkpoint differs from release pin")
        latent = encoder.encode_one(purpose, raw, columns, role="DEPLOYMENT")
        return self.recipient(purpose, latent)
