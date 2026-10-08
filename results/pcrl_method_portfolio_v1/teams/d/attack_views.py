"""Complete Team D feature views and direct finite-code readers.

Fit only on S2's ATTACK_FIT_NEW internal training split. The caller owns the
group-aware split, fixed common reader slate and INNER scoring. Public model
parameters may be passed to defense-aware readers, but no private per-person
input is inferred from their existence. Code IDs are categorical, not ordered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import hashlib
import re

import numpy as np


ALPHAS = (0.5, 1.0, 5.0)
PUBLIC_METADATA_FIELDS = frozenset(("purpose", "code_length_bits", "block_count",
                                    "fallback", "stochastic", "kind", "calibration",
                                    "calibrator_sha256"))


def validate_recipient_release(release: Mapping, n: int, purpose: int) -> dict:
    """Fail closed on an unknown transmitted field or inconsistent head output."""
    required = {"code_ids", "decoded", "probabilities", "decision", "metadata"}
    if set(release) != required:
        raise ValueError("release must disclose exactly the registered wire fields")
    ids = np.asarray(release["code_ids"])
    feature = np.asarray(release["decoded"], dtype=np.float64)
    probability = np.asarray(release["probabilities"], dtype=np.float64)
    decision = np.asarray(release["decision"])
    meta = release["metadata"]
    if not isinstance(meta, Mapping) or not set(meta) <= PUBLIC_METADATA_FIELDS:
        raise ValueError("unknown or malformed public metadata must be registered")
    if any(isinstance(value, (np.ndarray, list, tuple, dict)) for value in meta.values()):
        raise ValueError("this D wire admits only arm-constant scalar metadata")
    if ("kind" in meta and (not isinstance(meta["kind"], str) or not meta["kind"])):
        raise ValueError("public kind must be a registered scalar string")
    if (not isinstance(meta.get("calibration"), str)
            or type(meta.get("purpose")) is not int
            or type(meta.get("code_length_bits")) is not int
            or type(meta.get("block_count")) is not int):
        raise ValueError("public release metadata has non-scalar or wrong-typed fields")
    if meta.get("purpose") != purpose or not all(k in meta for k in
                                                  ("fallback", "stochastic", "code_length_bits",
                                                   "block_count", "calibration")):
        raise ValueError("purpose or required release metadata disagree")
    if meta["calibration"] not in ("UNFITTED", "H-GLOBAL-TEMP"):
        raise ValueError("unknown calibration marker")
    if meta["calibration"] == "H-GLOBAL-TEMP" and "calibrator_sha256" not in meta:
        raise ValueError("scored calibration must bind its public calibrator SHA-256")
    if "calibrator_sha256" in meta and (not isinstance(meta["calibrator_sha256"], str)
                                      or re.fullmatch(r"[0-9a-f]{64}", meta["calibrator_sha256"]) is None):
        raise ValueError("calibrator hash must be lowercase SHA-256")
    if (ids.ndim != 2 or ids.shape[0] != n or ids.shape[1] not in (0, 1)
            or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0)
            or feature.ndim != 2 or feature.shape[0] != n
            or probability.ndim != 2 or probability.shape[0] != n or probability.shape[1] < 2
            or decision.shape != (n,) or not np.issubdtype(decision.dtype, np.integer)
            or not np.isfinite(feature).all() or not np.isfinite(probability).all()
            or np.any(probability < 0) or not np.allclose(probability.sum(1), 1, rtol=0, atol=1e-8)
            or not np.array_equal(decision, probability.argmax(1))):
        raise ValueError("invalid complete D release arrays")
    if ids.shape[1] == 1 and (int(meta["code_length_bits"]) != 10
                                  or int(meta["block_count"]) != 1 or np.any(ids >= 1024)):
        raise ValueError("finite D code arm must transmit one valid 10-bit code")
    if ids.shape[1] == 0 and (int(meta["code_length_bits"]) != 0
                                  or int(meta["block_count"]) != 0):
        raise ValueError("continuous arm must have no code ID or block")
    if meta["fallback"] is not False or meta["stochastic"] is not False:
        raise ValueError("this fixed D contract has no fallback or stochastic draw")
    return {"code_ids": ids.copy(), "decoded": feature.copy(),
            "probabilities": probability.copy(), "decision": decision.copy(),
            "metadata": dict(meta)}


@dataclass(frozen=True)
class AlignedDViews:
    row_id: np.ndarray
    unit_id: np.ndarray
    recipient0_row_id: np.ndarray
    recipient1_row_id: np.ndarray
    recipient0: dict
    recipient1: dict
    row_order_sha256: str
    require_calibrated: bool = True

    def __post_init__(self) -> None:
        rows = np.asarray(self.row_id)
        units = np.asarray(self.unit_id)
        recipient_rows = (np.asarray(self.recipient0_row_id),
                          np.asarray(self.recipient1_row_id))
        if rows.ndim != 1 or units.shape != rows.shape or len(rows) == 0:
            raise ValueError("aligned row and resampling-unit identities are required")
        if any(not np.array_equal(r, rows) for r in recipient_rows):
            raise ValueError("recipient releases must be bound to the same row order")
        if (not isinstance(self.row_order_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.row_order_sha256) is None):
            raise ValueError("S2 admitted row/order SHA-256 is required")
        r0 = validate_recipient_release(self.recipient0, len(rows), 0)
        r1 = validate_recipient_release(self.recipient1, len(rows), 1)
        if self.require_calibrated and any(r["metadata"]["calibration"] != "H-GLOBAL-TEMP"
                                           for r in (r0, r1)):
            raise ValueError("scored D release must use the fixed H-GLOBAL-TEMP marker")
        object.__setattr__(self, "row_id", rows.copy())
        object.__setattr__(self, "unit_id", units.copy())
        object.__setattr__(self, "recipient0_row_id", recipient_rows[0].copy())
        object.__setattr__(self, "recipient1_row_id", recipient_rows[1].copy())
        object.__setattr__(self, "recipient0", r0)
        object.__setattr__(self, "recipient1", r1)

    def codes(self, view: str) -> np.ndarray:
        if view == "recipient0":
            return self.recipient0["code_ids"].copy()
        if view == "recipient1":
            return self.recipient1["code_ids"].copy()
        if view == "pair":
            if not (self.recipient0["code_ids"].shape[1] == 1
                    and self.recipient1["code_ids"].shape[1] == 1):
                raise ValueError("exact pair-code reader requires both finite-code releases")
            return np.concatenate((self.recipient0["code_ids"],
                                   self.recipient1["code_ids"]), axis=1)
        raise ValueError("unknown view")


def view_matrix(views: AlignedDViews, view: str) -> np.ndarray:
    """Dense reader input; exact ID cells and string metadata stay available.

    String markers are arm-constant and also held exactly in the release/view
    manifest. Their SHA-256 bytes are included as fixed numeric columns so this
    matrix does not silently drop a transmitted public metadata component.
    """
    if view == "recipient0":
        receivers = (views.recipient0,)
    elif view == "recipient1":
        receivers = (views.recipient1,)
    elif view == "pair":
        receivers = (views.recipient0, views.recipient1)
    else:
        raise ValueError("unknown view")
    parts = []
    for r in receivers:
        ids = r["code_ids"]
        if ids.shape[1] == 1:
            if np.any(ids >= 1024):
                raise ValueError("code IDs outside fixed 1024-cell alphabet")
            parts.append(np.eye(1024, dtype=np.float32)[ids[:, 0]])
        parts.append(np.asarray(r["decoded"], dtype=np.float32))
        # The calibrated wire carries float64 confidence; do not round it in
        # a fresh reader's complete view.
        parts.append(np.asarray(r["probabilities"], dtype=np.float64))
        parts.append(np.eye(r["probabilities"].shape[1], dtype=np.float32)[r["decision"]])
        meta = r["metadata"]
        parts.append(np.tile(np.asarray([float(meta["purpose"]), float(meta["code_length_bits"]),
                                         float(meta["block_count"]), float(meta["fallback"]),
                                         float(meta["stochastic"])], dtype=np.float32),
                             (len(views.row_id), 1)))
        for field in ("kind", "calibration", "calibrator_sha256"):
            value = str(meta.get(field, "ABSENT"))
            digest = np.frombuffer(hashlib.sha256(value.encode()).digest(), dtype=np.uint8)
            parts.append(np.tile(digest.astype(np.float32)[None, :] / 255.0,
                                 (len(views.row_id), 1)))
    return np.concatenate(parts, axis=1)


def _tuples(codes: np.ndarray, columns: int) -> list[tuple[int, ...]]:
    ids = np.asarray(codes)
    if (ids.ndim != 2 or ids.shape[1] != columns
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(ids >= 1024)):
        raise ValueError("finite 1024-cell code IDs with registered width required")
    return [tuple(map(int, row)) for row in ids]


@dataclass(frozen=True)
class CellReader:
    """Smoothed empirical P(SEX=1 | hard code), with fixed pair fallback."""

    alpha: float
    columns: int
    prior: float
    counts: dict[tuple[int, ...], tuple[int, int]]
    marginal_counts: tuple[dict[int, tuple[int, int]], ...]

    @classmethod
    def fit(cls, codes: np.ndarray, sex: np.ndarray, alpha: float) -> "CellReader":
        ids = np.asarray(codes)
        if ids.ndim != 2 or ids.shape[1] not in (1, 2):
            raise ValueError("one local code or an aligned pair is required")
        keys = _tuples(ids, ids.shape[1])
        y = np.asarray(sex)
        if y.shape != (len(keys),) or not np.isin(y, (0, 1)).all() or len(keys) == 0:
            raise ValueError("binary fitting SEX and aligned codes are required")
        if alpha not in ALPHAS:
            raise ValueError("alpha must be predeclared")
        counts: dict[tuple[int, ...], list[int]] = {}
        marginals: list[dict[int, list[int]]] = [dict() for _ in range(ids.shape[1])]
        for key, label in zip(keys, y):
            pair = counts.setdefault(key, [0, 0])
            pair[int(label)] += 1
            for j, code in enumerate(key):
                marginal = marginals[j].setdefault(code, [0, 0])
                marginal[int(label)] += 1
        return cls(float(alpha), ids.shape[1], float(np.mean(y)),
                   {key: tuple(value) for key, value in counts.items()},
                   tuple({key: tuple(value) for key, value in d.items()} for d in marginals))

    def _posterior(self, counts: tuple[int, int] | None) -> float:
        if counts is None:
            return self.prior
        n0, n1 = counts
        return (n1 + self.alpha * self.prior) / (n0 + n1 + self.alpha)

    def predict_proba(self, codes: np.ndarray) -> np.ndarray:
        keys = _tuples(codes, self.columns)
        out = np.empty(len(keys), dtype=np.float64)
        for i, key in enumerate(keys):
            seen = self.counts.get(key)
            if seen is not None:
                out[i] = self._posterior(seen)
            elif self.columns == 2:
                # Fixed structured fallback for unseen joint cells. This is
                # not tuned on INNER labels and keeps local clues available.
                out[i] = np.mean([self._posterior(self.marginal_counts[j].get(code))
                                  for j, code in enumerate(key)])
            else:
                out[i] = self.prior
        return np.column_stack((1.0 - out, out))


def fixed_cell_bank(codes: np.ndarray, sex: np.ndarray) -> dict[float, CellReader]:
    """All three declared alpha readers are fitted; none is dropped post outcome."""
    return {alpha: CellReader.fit(codes, sex, alpha) for alpha in ALPHAS}
