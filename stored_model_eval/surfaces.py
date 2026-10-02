"""Release surfaces: what a recipient sees. Kept separate from attackers and metrics.

  rep          : the released representation H (n x d)
  outputs      : the released task outputs (logits; probabilities are converted to log-probabilities)
  rep+outputs  : the column concatenation of both (the combined release)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SURFACES = ("rep", "outputs", "rep+outputs")


@dataclass(frozen=True)
class SurfaceRecord:
    name: str
    components: tuple
    n_columns: int
    output_transform: str

    def to_json(self):
        return {"surface": self.name, "components": list(self.components), "n_columns": self.n_columns,
                "output_transform": self.output_transform}


def _outputs_matrix(outputs: np.ndarray, kind: str) -> tuple[np.ndarray, str]:
    O = np.asarray(outputs, dtype=np.float64)
    if O.ndim == 1:
        O = O[:, None]
    if kind == "probabilities":
        return np.log(np.clip(O, 1e-12, 1.0)), "log_probabilities"
    if kind == "logits":
        return O, "identity_logits"
    raise ValueError("outputs kind must be 'logits' or 'probabilities'")


def build_surface(surface: str, rep: np.ndarray | None = None, outputs: np.ndarray | None = None,
                  outputs_kind: str = "logits") -> tuple[np.ndarray, SurfaceRecord]:
    if surface not in SURFACES:
        raise ValueError(f"unknown surface {surface}")
    parts, comps, tr = [], [], "none"
    if surface in ("rep", "rep+outputs"):
        if rep is None:
            raise ValueError(f"surface {surface} needs representations")
        parts.append(np.asarray(rep, dtype=np.float64))
        comps.append("representation")
    if surface in ("outputs", "rep+outputs"):
        if outputs is None:
            raise ValueError(f"surface {surface} needs outputs")
        O, tr = _outputs_matrix(outputs, outputs_kind)
        parts.append(O)
        comps.append("task_outputs")
    X = np.concatenate(parts, axis=1)
    return X, SurfaceRecord(surface, tuple(comps), X.shape[1], tr)
