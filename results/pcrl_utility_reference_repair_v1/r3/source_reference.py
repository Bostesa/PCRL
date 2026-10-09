"""Independent U0/Ucal forward path from pinned source artifacts.

This module does not import historical or candidate inference, scoring or
selection code. Its caller authenticates role-only data and the queue seal.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import joblib
import numpy as np
import torch


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class SourceNetwork(torch.nn.Module):
    """U source topology: separate 83-64-64-16 purpose encoders and heads."""

    def __init__(self):
        super().__init__()
        self.enc = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(83, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 16),
            ) for _ in range(2)
        ])
        self.head = torch.nn.ModuleList([torch.nn.Linear(16, 2),
                                         torch.nn.Linear(16, 6)])


def restore_source(model_path: Path, head_paths: tuple[Path, Path],
                   expected_shas: tuple[str, str, str]):
    paths = (model_path, *head_paths)
    for path, expected in zip(paths, expected_shas):
        if file_sha(path) != expected:
            raise ValueError("source model or deployed head SHA differs")
    torch.set_num_threads(1)
    model = SourceNetwork()
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    heads = tuple(joblib.load(path) for path in head_paths)
    return model, heads


def forward_u0(model: SourceNetwork, heads: tuple, features: np.ndarray,
               batch_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features)
    if (x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 83
            or not np.isfinite(x).all()):
        raise ValueError("U0 forward needs finite admitted 83-column float32 features")
    if len(heads) != 2 or (batch_size is not None and batch_size < 1):
        raise ValueError("U0 requires two deployed heads and positive batch size")
    # Whole-role forward is the primary parity convention. Float32 matrix
    # kernels can change at a different chunk size by about 1e-7 in a
    # synthetic fixture, so chunked results are diagnostic only.
    if batch_size is None:
        batch_size = max(1, len(x))
    batches = [[], []]
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            tensor = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size]))
            for purpose in range(2):
                representation = model.enc[purpose](tensor).to(torch.float64).numpy()
                batches[purpose].append(np.asarray(
                    heads[purpose].predict_proba(representation), dtype=np.float64))
    output = tuple(np.concatenate(batches[purpose], axis=0) for purpose in range(2))
    if output[0].shape != (len(x), 2) or output[1].shape != (len(x), 6):
        raise ValueError("source U0 head class order or shape differs")
    return output


def _ordered_rowsum(matrix: np.ndarray) -> np.ndarray:
    total = np.zeros(len(matrix), dtype=np.float64)
    for column in range(matrix.shape[1]):
        total += matrix[:, column]
    return total


def _log_inputs(prob: np.ndarray) -> np.ndarray:
    p = np.asarray(prob, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] < 2 or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > 1)
            or np.max(np.abs(_ordered_rowsum(p) - 1), initial=0.0) > 1e-9):
        raise ValueError("invalid source U probabilities")
    floor = np.maximum(p, 1e-12)
    return np.log(floor / _ordered_rowsum(floor)[:, None])


def _derivative(log_prob: np.ndarray, labels: np.ndarray, alpha: float) -> float:
    z = alpha * log_prob
    e = np.exp(z - z.max(axis=1, keepdims=True))
    scaled = e / _ordered_rowsum(e)[:, None]
    expected = _ordered_rowsum(scaled * log_prob)
    return math.fsum((expected - log_prob[np.arange(len(labels)), labels]).tolist()) / len(labels)


def fit_inverse_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Independent bounded derivative bisection for hcal's global family."""
    log_prob = _log_inputs(probabilities)
    y = np.asarray(labels)
    if (y.dtype.kind not in "iu" or y.shape != (len(log_prob),)
            or len(y) == 0 or np.any(y < 0) or np.any(y >= log_prob.shape[1])):
        raise ValueError("invalid calibration labels")
    low, high = 0.25, 4.0
    g_low = _derivative(log_prob, y, low)
    g_high = _derivative(log_prob, y, high)
    if g_low >= 0:
        return low
    if g_high <= 0:
        return high
    for _ in range(100):
        mid = (low + high) / 2
        if not low < mid < high:
            break
        g_mid = _derivative(log_prob, y, mid)
        if g_mid <= 0:
            low, g_low = mid, g_mid
        else:
            high, g_high = mid, g_mid
    return high if abs(g_high) < abs(g_low) else low


def apply_inverse_temperature(probabilities: np.ndarray, alpha: float) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if not 0.25 <= alpha <= 4.0:
        raise ValueError("inverse temperature outside pinned bounds")
    if alpha == 1.0:
        return p.copy()
    z = alpha * _log_inputs(p)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    output = e / _ordered_rowsum(e)[:, None]
    if not np.array_equal(output.argmax(axis=1), p.argmax(axis=1)):
        raise ValueError("global temperature changed a source decision")
    return output
