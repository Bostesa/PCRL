"""Exact preprivacy U encoder for S2-admitted role-only 83-column arrays.

This module has no data loader. Its caller supplies an already admitted `X`
array and column names. It never opens all-role source storage or labels.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ALLOWED_ROLES = frozenset({"OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT",
                           "ATTACK_FIT_NEW", "INNER_SELECTION"})
DEPLOYMENT_ROLE = "DEPLOYMENT"


def hash_file(path: Path) -> str:
    """Streaming SHA-256 of the checkpoint bytes (ordinary cached read)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def schema_hash(columns: tuple[str, ...]) -> str:
    """Source convention: sha256 of newline-joined ordered feature names."""
    if (len(columns) != 83 or not all(isinstance(x, str) and x for x in columns)
            or len(set(columns)) != 83):
        raise ValueError("exactly 83 distinct permitted columns are required")
    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()


def _expected_state_shapes() -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for purpose in (0, 1):
        for layer, input_dim, output_dim in ((0, 83, 64), (2, 64, 64), (4, 64, 16)):
            result[f"enc.{purpose}.{layer}.weight"] = (output_dim, input_dim)
            result[f"enc.{purpose}.{layer}.bias"] = (output_dim,)
        result[f"head.{purpose}.weight"] = ((2, 6)[purpose], 16)
        result[f"head.{purpose}.bias"] = ((2, 6)[purpose],)
    return result


class FrozenUEncoder:
    """The source U `Model.encode(i, X)`, before maps or source heads."""

    def __init__(self, state: dict[str, torch.Tensor], model_sha256: str,
                 columns: tuple[str, ...], schema_sha256: str, seed: int):
        self._state = {name: value.detach().contiguous().cpu().clone()
                       for name, value in state.items()}
        self.model_sha256 = model_sha256
        self.columns = columns
        self.schema_sha256 = schema_sha256
        self.seed = int(seed)

    @classmethod
    def load(cls, model_path: Path, expected_model_sha256: str,
             expected_columns: tuple[str, ...], expected_schema_sha256: str,
             seed: int) -> "FrozenUEncoder":
        columns = tuple(expected_columns)
        if schema_hash(columns) != expected_schema_sha256:
            raise ValueError("83-column schema SHA does not match the source pin")
        if (len(expected_model_sha256) != 64 or
                hash_file(Path(model_path)) != expected_model_sha256):
            raise ValueError("source checkpoint SHA does not match the admitted pin")
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        shapes = _expected_state_shapes()
        if (not isinstance(state, dict) or set(state) != set(shapes)
                or any(not isinstance(state[name], torch.Tensor)
                       or tuple(state[name].shape) != shape
                       or state[name].dtype != torch.float32
                       or not bool(torch.isfinite(state[name]).all())
                       for name, shape in shapes.items())):
            raise ValueError("checkpoint does not have the exact U architecture")
        return cls(state, expected_model_sha256, columns,
                   expected_schema_sha256, seed)

    def _check_input(self, x: np.ndarray, columns: tuple[str, ...]) -> torch.Tensor:
        if tuple(columns) != self.columns:
            raise ValueError("83-column order differs from the admitted schema")
        array = np.asarray(x)
        if array.ndim != 2 or array.shape[1] != 83:
            raise ValueError("X must have exactly 83 permitted columns")
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("X must be finite float32 from the admitted preprocessing")
        return torch.from_numpy(np.ascontiguousarray(array))

    def _forward_one(self, purpose: int, x: torch.Tensor) -> torch.Tensor:
        if purpose not in (0, 1):
            raise ValueError("purpose must be 0 or 1")
        with torch.no_grad():
            h = x
            for layer in (0, 2, 4):
                h = F.linear(h, self._state[f"enc.{purpose}.{layer}.weight"],
                             self._state[f"enc.{purpose}.{layer}.bias"])
                if layer != 4:
                    h = F.relu(h)
        return h

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_one(0, x), self._forward_one(1, x)

    def encode_one(self, purpose: int, x: np.ndarray,
                   columns: tuple[str, ...], role: str) -> np.ndarray:
        """Single-purpose path; deployment reads only this purpose's encoder."""
        if role not in ALLOWED_ROLES and role != DEPLOYMENT_ROLE:
            raise ValueError("forbidden role for Team D latent forward")
        return self._forward_one(purpose, self._check_input(x, columns)).numpy().copy()

    def encode_pair(self, x: np.ndarray,
                    columns: tuple[str, ...], role: str) -> tuple[np.ndarray, np.ndarray]:
        """Trusted S2 offline path only; deployment must call encode_one."""
        if role not in ALLOWED_ROLES:
            raise ValueError("forbidden role for paired Team D latent forward")
        h0, h1 = self._forward(self._check_input(x, columns))
        return h0.numpy().copy(), h1.numpy().copy()

    def encode_with_receipt(self, x: np.ndarray, columns: tuple[str, ...],
                            role: str) -> tuple[np.ndarray, np.ndarray, dict]:
        if role not in ALLOWED_ROLES:
            raise ValueError("forbidden role for Team D latent forward")
        xt = self._check_input(x, columns)
        h0, h1 = self._forward(xt)
        # Independent parity with the pinned source implementation. Importing
        # jcv.train does not import or open any mixed-role data loader.
        from jcv.train import Model

        with torch.random.fork_rng(devices=[]):
            source = Model(83, [2, 6], self.seed, masks=None).eval()
        source.load_state_dict(self._state, strict=True)
        with torch.no_grad():
            ref0, ref1 = source.encode(0, xt), source.encode(1, xt)
        parity = bool(torch.equal(h0, ref0) and torch.equal(h1, ref1))
        if not parity:
            raise ValueError("preprivacy encoder differs from source Model.encode")
        receipt = {
            "schema": "team-d-frozen-u-encoder-parity-v1",
            "role": role,
            "rows": int(len(xt)),
            "input_columns": 83,
            "purpose_output_shapes": [list(h0.shape), list(h1.shape)],
            "tensor": "Model.encode(i, X) before maps/head",
            "source_model_sha256": self.model_sha256,
            "feature_names_sha256": self.schema_sha256,
            "source_seed": self.seed,
            "bitwise_source_parity": parity,
            "private_arrays_written": False,
        }
        return h0.numpy().copy(), h1.numpy().copy(), receipt
