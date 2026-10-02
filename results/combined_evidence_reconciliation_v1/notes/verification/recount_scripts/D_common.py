"""Shared helpers for the D (baseline fairness) recounts.

Standalone: reads committed files via `git show` and extracted drive members;
never imports PCRL repository code. System python3 (numpy) only.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = "/Users/nathansamson/PCRL"
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-nathansamson-PCRL/"
    "f1ff337a-0f10-4ee1-bd1d-5817210be5ea/scratchpad/recon/verification/D_drive"
)
OUT = Path(__file__).resolve().parent.parent / "recount_outputs"
OUT.mkdir(exist_ok=True)

_INPUTS: list[dict] = []


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git_bytes(ref: str, path: str) -> bytes:
    b = subprocess.run(["git", "-C", REPO, "show", f"{ref}:{path}"],
                       capture_output=True, check=True).stdout
    _INPUTS.append({"location": f"Bostesa/PCRL@{ref}:{path}", "sha256": sha(b)})
    return b


def git_json(ref: str, path: str):
    return json.loads(git_bytes(ref, path))


def file_bytes(p: Path | str, label: str | None = None) -> bytes:
    b = Path(p).read_bytes()
    _INPUTS.append({"location": label or str(p), "sha256": sha(b)})
    return b


def inputs() -> list[dict]:
    seen, out = set(), []
    for i in _INPUTS:
        k = (i["location"], i["sha256"])
        if k not in seen:
            seen.add(k)
            out.append(i)
    return out


def r2_onehot(H: np.ndarray, y: np.ndarray, reg: float = 1e-6) -> float:
    """Pooled one-hot ridge R^2, fit and scored on the same rows, float64.

    Written from the definition (centre H and one-hot Y, ridge solve with
    Tikhonov reg*I, R^2 = 1 - SS_res/SS_tot pooled over the K columns,
    clamp at 0) -- the same estimator the paper calls R^2_onehot.
    """
    H = H.astype(np.float64)
    K = int(y.max()) + 1
    Y = np.eye(K)[y.astype(int)]
    Hc = H - H.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    G = Hc.T @ Hc + reg * np.eye(H.shape[1])
    W = np.linalg.solve(G, Hc.T @ Yc)
    res = Yc - Hc @ W
    r2 = 1.0 - (res ** 2).sum() / max((Yc ** 2).sum(), 1e-12)
    return float(max(0.0, r2))


def per_dim_std_and_eff_rank(H: np.ndarray) -> tuple[float, float]:
    """Mean per-dimension std and exp-entropy effective rank of singular values."""
    H = H.astype(np.float64)
    std = float(H.std(0).mean())
    s = np.linalg.svd(H - H.mean(0), compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return std, float(np.exp(-(p * np.log(p)).sum()))


def dump(name: str, obj) -> Path:
    p = OUT / name
    p.write_text(json.dumps(obj, indent=1, default=float))
    return p
