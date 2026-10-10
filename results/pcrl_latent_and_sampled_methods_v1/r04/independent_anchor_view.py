"""Rebuild each U0/Ucal score-and-decision complete view independently."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


FIELDS = {"row_id", "unit", "U0_1", "U0_2", "Ucal_1", "Ucal_2", "d_1", "d_2"}


def _recipient(record: Mapping, anchor: str, purpose: int) -> tuple[np.ndarray, np.ndarray]:
    if set(record) != FIELDS or anchor not in ("U0", "Ucal") or purpose not in (1, 2):
        raise ValueError("anchor source or recipient differs")
    row = np.asarray(record["row_id"])
    unit = np.asarray(record["unit"])
    q = np.asarray(record[f"{anchor}_{purpose}"])
    decision = np.asarray(record[f"d_{purpose}"])
    width = (2, 6)[purpose - 1]
    if (row.dtype != np.int64 or unit.dtype != np.int64
            or row.ndim != 1 or not len(row) or unit.shape != row.shape
            or not np.all(np.diff(row) > 0) or q.dtype != np.float64
            or q.shape != (len(row), width) or not np.isfinite(q).all()
            or np.any(q < 0) or np.max(np.abs(q.sum(axis=1) - 1)) > 1e-8
            or decision.dtype != np.int64 or decision.shape != row.shape
            or not np.array_equal(decision, np.argmax(q, axis=1))):
        raise ValueError("actual anchor score/decision wire invalid")
    return q, decision.astype(np.float64)


def build_anchor_view(att: Mapping, inner: Mapping, anchor: str,
                      view: str) -> tuple[dict, dict]:
    """Return ATT and INNER x and defense-aware x without R02's adapter."""
    if view not in ("r1", "r2", "pair"):
        raise ValueError("unknown complete anchor view")
    purposes = (1, 2) if view == "pair" else ((1,) if view == "r1" else (2,))

    def encode(record: Mapping) -> dict:
        values = [_recipient(record, anchor, purpose) for purpose in purposes]
        x = np.concatenate([component for q, decision in values
                            for component in (q, decision[:, None])], axis=1)
        derived = [np.log(np.maximum(q, 1e-12)) for q, _ in values]
        return {"x": x, "x_defense_aware": np.concatenate([x, *derived], axis=1)}

    return encode(att), encode(inner)
