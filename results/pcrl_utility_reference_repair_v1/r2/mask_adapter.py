"""Selected-input adapter for an 83-column frozen task architecture.

The adapter receives only selected per-person values. Omitted positions are
filled from one public vector estimated on permitted defense-fitting features.
No runtime path can obtain omitted per-person values through this API.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from results.pcrl_method_portfolio_v1.teams.c.prototype import AdultGroupedSchema


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def fit_public_fill_constants(fit_features: np.ndarray) -> np.ndarray:
    """Freeze column means using only the admitted 83-column FIT matrix."""
    x = np.asarray(fit_features)
    if (x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 83
            or len(x) == 0 or not np.isfinite(x).all()):
        raise ValueError("fill constants require finite float32 FIT features")
    result = np.asarray(x.mean(axis=0, dtype=np.float64), dtype=np.float32)
    result.setflags(write=False)
    return result


def reconstruct_selected(
    selected: np.ndarray,
    source_indices: tuple[int, ...],
    fill_constants: np.ndarray,
) -> np.ndarray:
    """Produce a full-layout matrix without reading omitted person features."""
    indices = tuple(source_indices)
    constants = np.asarray(fill_constants)
    values = np.asarray(selected)
    if (not indices or indices != tuple(sorted(set(indices)))
            or any(type(index) is not int or index < 0 or index >= 83
                   for index in indices)):
        raise ValueError("selected source indices are invalid")
    if (constants.dtype != np.float32 or constants.shape != (83,)
            or not np.isfinite(constants).all()):
        raise ValueError("fill constants must be finite float32 83-vector")
    if (values.dtype != np.float32 or values.ndim != 2
            or values.shape[1] != len(indices) or not np.isfinite(values).all()):
        raise ValueError("head requires finite selected float32 columns")
    output = np.broadcast_to(constants, (len(values), 83)).copy()
    output[:, indices] = values
    return np.ascontiguousarray(output)


@dataclass(frozen=True)
class FixedFillSelectedHead:
    """Expose a full-input model as a selected-column PurposeRelease head."""

    purpose: str
    selected_atoms: tuple[str, ...]
    source_indices: tuple[int, ...]
    full_input_head: Any = field(repr=False)
    base_model_id: str
    base_model_sha256: str
    fill_constants: np.ndarray = field(repr=False)
    source_schema_sha256: str
    selected_columns: int = field(init=False)
    model_id: str = field(init=False)
    fill_constants_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        schema = AdultGroupedSchema()
        expected_schema = hashlib.sha256(
            "\n".join(schema.feature_names).encode()
        ).hexdigest()
        atoms = schema.normalize_atoms(tuple(self.selected_atoms))
        columns = schema.column_indices(atoms)
        if (self.purpose not in ("income", "occupation_group")
                or not self.base_model_id or not hasattr(self.full_input_head, "predict_proba")
                or len(self.base_model_sha256) != 64
                or any(ch not in "0123456789abcdef" for ch in self.base_model_sha256)
                or self.source_schema_sha256 != expected_schema):
            raise ValueError("fixed-fill head purpose, model or source schema is invalid")
        if tuple(self.source_indices) != columns:
            raise ValueError("mask columns differ from pinned source groups")
        constants = np.asarray(self.fill_constants)
        if (constants.dtype != np.float32 or constants.shape != (83,)
                or not np.isfinite(constants).all()):
            raise ValueError("fill constants must be finite float32 83-vector")
        constants = constants.copy()
        constants.setflags(write=False)
        const_sha = _array_sha(constants)
        identity = {
            "purpose": self.purpose,
            "selected_atoms": atoms,
            "source_indices": columns,
            "base_model_id": self.base_model_id,
            "base_model_sha256": self.base_model_sha256,
            "source_schema_sha256": expected_schema,
            "fill_constants_sha256": const_sha,
        }
        identity_sha = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        object.__setattr__(self, "selected_atoms", atoms)
        object.__setattr__(self, "source_indices", columns)
        object.__setattr__(self, "fill_constants", constants)
        object.__setattr__(self, "selected_columns", len(columns))
        object.__setattr__(self, "fill_constants_sha256", const_sha)
        object.__setattr__(self, "model_id",
                           f"{self.base_model_id}:FIXED-FILL:{identity_sha[:24]}")

    def predict_proba(self, selected: np.ndarray) -> np.ndarray:
        full = reconstruct_selected(selected, self.source_indices,
                                    self.fill_constants)
        probability = np.asarray(self.full_input_head.predict_proba(full),
                                 dtype=np.float64)
        classes = 2 if self.purpose == "income" else 6
        if (probability.shape != (len(full), classes)
                or not np.isfinite(probability).all()
                or np.any(probability < 0) or np.any(probability > 1)
                or not np.allclose(probability.sum(axis=1), 1, rtol=0, atol=1e-8)):
            raise ValueError("full-input model returned invalid probabilities")
        return probability
