"""Pinned frozen U encoder on an already admitted 83-column role matrix.

The source model is loaded from its model.pt bytes. This module imports the
historical model architecture, not its data loaders or all-role archives.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from jcv.train import Model


ALLOWED_ROLES = frozenset({"OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT",
                           "ATTACK_FIT_NEW", "INNER_SELECTION"})


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenUEncoder:
    def __init__(self, model: Model, model_sha256: str, columns: tuple[str, ...],
                 schema_sha256: str, seed: int):
        self.model = model.eval()
        self.model_sha256 = model_sha256
        self.columns = columns
        self.schema_sha256 = schema_sha256
        self.seed = seed

    @classmethod
    def load(cls, model_pt: str | Path, expected_sha: str,
             ordered_names: tuple[str, ...] | list[str] | np.ndarray,
             schema_sha: str, seed: int) -> "FrozenUEncoder":
        path = Path(model_pt).resolve(strict=True)
        names = tuple(str(name) for name in ordered_names)
        if (seed not in (0, 1) or len(names) != len(set(names)) or len(names) != 83
                or hashlib.sha256("\n".join(names).encode()).hexdigest() != schema_sha
                or len(expected_sha) != 64 or sha_file(path) != expected_sha):
            raise ValueError("source seed, ordered schema or model byte hash differs")
        if path.name != "model.pt" or path.parent.name != f"rel__s{seed}__U":
            raise ValueError("source path is not the admitted U0 unit")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("source checkpoint is not a state dictionary")
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            model = Model(83, [2, 6], seed, masks=None)
        expected_shapes = {name: tuple(parameter.shape) for name, parameter in model.state_dict().items()}
        if (set(state) != set(expected_shapes)
                or any(not isinstance(state[name], torch.Tensor)
                       or tuple(state[name].shape) != shape
                       or state[name].dtype != torch.float32
                       or not bool(torch.isfinite(state[name]).all())
                       for name, shape in expected_shapes.items())):
            raise ValueError("source checkpoint architecture or value differs")
        model.load_state_dict(state, strict=True)
        return cls(model, expected_sha, names, schema_sha, seed)

    def _input(self, X: np.ndarray, names, role: str) -> torch.Tensor:
        if role not in ALLOWED_ROLES and role != "DEPLOYMENT":
            raise ValueError("role is not admitted to frozen U")
        if tuple(str(name) for name in names) != self.columns:
            raise ValueError("predictor order differs")
        value = np.asarray(X)
        if (value.dtype != np.float32 or value.ndim != 2 or value.shape[1] != 83
                or not np.isfinite(value).all()):
            raise ValueError("frozen U requires finite float32 (n,83)")
        return torch.from_numpy(np.ascontiguousarray(value))

    def encode_one(self, purpose: int, X: np.ndarray, names, role: str) -> np.ndarray:
        if purpose not in (0, 1):
            raise ValueError("purpose must be 0 income or 1 occupation")
        value = self._input(X, names, role)
        with torch.no_grad():
            result = self.model.encode(purpose, value).numpy().copy()
        if result.dtype != np.float32 or result.shape != (len(value), 16):
            raise ValueError("source latent shape differs")
        return result

    def encode_pair(self, X: np.ndarray, names, role: str) -> tuple[np.ndarray, np.ndarray]:
        if role == "DEPLOYMENT":
            raise ValueError("deployment must call recipient-local encode_one")
        value = self._input(X, names, role)
        with torch.no_grad():
            result = tuple(self.model.encode(purpose, value).numpy().copy()
                           for purpose in (0, 1))
        if any(array.dtype != np.float32 or array.shape != (len(value), 16)
               for array in result):
            raise ValueError("source latent shape differs")
        return result
