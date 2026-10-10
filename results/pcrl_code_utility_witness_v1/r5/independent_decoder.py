"""Independent hard-code model forward from serialized tensor values.

No candidate model or deployment module is imported. PyTorch supplies ordinary
float32 linear/GELU primitives so this replay matches serialized-model math.
The caller verifies model/scaler/calibrator file hashes before passing values.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch.nn import functional as F


def _weight(state: Mapping, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    if name not in state:
        raise ValueError(f"missing serialized tensor {name}")
    value = torch.as_tensor(state[name]).detach().cpu()
    if value.dtype != torch.float32 or tuple(value.shape) != shape or not torch.isfinite(value).all():
        raise ValueError(f"invalid serialized tensor {name}")
    return value


def hard_bits_from_latent(state: Mapping, latent: np.ndarray,
                          center: np.ndarray, divisor: np.ndarray,
                          purpose: int, bits: int) -> np.ndarray:
    """Reconstruct hard mapper output from admitted frozen purpose latent."""
    if purpose not in (0, 1) or bits not in (10, 16):
        raise ValueError("purpose or sealed bit width invalid")
    h = np.asarray(latent, dtype=np.float32)
    c = np.asarray(center, dtype=np.float32)
    d = np.asarray(divisor, dtype=np.float32)
    if (h.ndim != 2 or h.shape[1] != 16 or c.shape != (2, 16)
            or d.shape != (2, 16) or not np.isfinite(h).all()
            or not np.isfinite(c).all() or not np.isfinite(d).all()
            or np.any(d < 1)):
        raise ValueError("latent/scaler shape or non-amplifying contract differs")
    scaled = torch.from_numpy(((h - c[purpose]) / d[purpose]).astype(np.float32))
    prefix = f"mappers.{purpose}."
    with torch.no_grad():
        hidden = F.relu(F.linear(scaled,
                                 _weight(state, prefix + "layer1.weight", (64, 16)),
                                 _weight(state, prefix + "layer1.bias", (64,))))
        prebit = F.linear(hidden,
                          _weight(state, prefix + "layer2.weight", (bits, 64)),
                          _weight(state, prefix + "layer2.bias", (bits,)))
        code = (torch.sigmoid(prebit) >= .5).to(torch.uint8)
    return code.numpy().copy()


def probabilities_from_hard_bits(state: Mapping, public_bits: np.ndarray,
                                 purpose: int, bits: int,
                                 alpha: float | None = None) -> np.ndarray:
    """Decode only the public hard bits; global alpha is an admitted fixed CAL fit."""
    if purpose not in (0, 1) or bits not in (10, 16):
        raise ValueError("purpose or sealed bit width invalid")
    code = np.asarray(public_bits)
    if (code.ndim != 2 or code.shape[1] != bits
            or not np.isin(code, (0, 1)).all()):
        raise ValueError("only hard public bits can reach the head")
    classes = 2 if purpose == 0 else 6
    prefix = f"heads.{purpose}."
    with torch.no_grad():
        h = F.gelu(F.linear(torch.from_numpy(code.astype(np.float32, copy=False)),
                            _weight(state, prefix + "layer1.weight", (64, bits)),
                            _weight(state, prefix + "layer1.bias", (64,))))
        logits = F.linear(h,
                          _weight(state, prefix + "layer2.weight", (classes, 64)),
                          _weight(state, prefix + "layer2.bias", (classes,)))
    raw = logits.numpy().astype(np.float64)
    raw -= raw.max(axis=1, keepdims=True)
    value = np.exp(raw)
    p = value / value.sum(axis=1, keepdims=True)
    if alpha is None or alpha == 1.0:
        return p
    if not np.isfinite(alpha) or not .25 <= alpha <= 4.0:
        raise ValueError("global temperature alpha invalid")
    clipped = np.maximum(p, 1e-12)
    clipped /= clipped.sum(axis=1, keepdims=True)
    logp = alpha * np.log(clipped)
    logp -= logp.max(axis=1, keepdims=True)
    powered = np.exp(logp)
    return powered / powered.sum(axis=1, keepdims=True)


def public_head_table(state: Mapping, purpose: int, bits: int,
                      alpha: float | None = None, batch: int = 4096) -> np.ndarray:
    """Build every output in the fixed public code table from serialized head."""
    if bits not in (10, 16) or batch < 1:
        raise ValueError("invalid public table capacity or batch")
    outputs = []
    powers = np.arange(bits, dtype=np.uint64)
    for first in range(0, 1 << bits, batch):
        ids = np.arange(first, min(first + batch, 1 << bits), dtype=np.uint64)
        code = ((ids[:, None] >> powers[None, :]) & 1).astype(np.uint8)
        outputs.append(probabilities_from_hard_bits(state, code, purpose, bits, alpha))
    return np.concatenate(outputs, axis=0)
