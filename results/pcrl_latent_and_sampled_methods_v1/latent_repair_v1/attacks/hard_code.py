"""Fail-closed D hard-code wire checks and exact categorical keys.

This module opens no role files or labels. The caller passes the actual public
release returned by the D deployment adapter and the admitted row order.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

import numpy as np


BITS = 10
ALPHABET = 1 << BITS
WIRE_KEYS = frozenset({"code", "probabilities", "decision", "metadata"})


def _scalar(value: object) -> bool:
    if value is None or type(value) in (str, bool, int):
        return True
    return type(value) is float and math.isfinite(value)


def code_ids(bits: np.ndarray) -> np.ndarray:
    """Pack public bits with column zero least significant; IDs stay categorical."""
    arr = np.asarray(bits)
    if (arr.ndim != 2 or arr.shape[1] != BITS or arr.dtype != np.uint8
            or not np.isin(arr, (0, 1)).all()):
        raise ValueError("D code must be an (n,10) uint8 array of hard bits")
    return np.sum(arr.astype(np.uint16) << np.arange(BITS, dtype=np.uint16),
                  axis=1, dtype=np.uint16)


@dataclass(frozen=True)
class PublicCode:
    row_id: np.ndarray
    code: np.ndarray
    code_id: np.ndarray
    probabilities: np.ndarray
    decision: np.ndarray
    metadata: dict[str, object]

    @classmethod
    def from_release(cls, row_id: np.ndarray, release: Mapping) -> "PublicCode":
        if not isinstance(release, Mapping) or set(release) != WIRE_KEYS:
            raise ValueError("D release must contain exactly the four public wire fields")
        rows = np.asarray(row_id)
        if (rows.dtype != np.int64 or rows.ndim != 1 or len(rows) == 0
                or len(np.unique(rows)) != len(rows)):
            raise ValueError("admitted unique int64 row IDs in exact release order are required")
        bits = np.asarray(release["code"])
        ids = code_ids(bits)
        q = np.asarray(release["probabilities"])
        decision = np.asarray(release["decision"])
        metadata = release["metadata"]
        if (not isinstance(metadata, Mapping) or not metadata
                or any(not isinstance(k, str) or not k or not _scalar(v)
                       for k, v in metadata.items())):
            raise ValueError("every transmitted metadata item must be a named scalar constant")
        if (q.ndim != 2 or q.shape[0] != len(rows) or q.shape[1] < 2
                or q.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or not np.isfinite(q).all() or np.any(q < 0)
                or not np.allclose(q.sum(axis=1, dtype=np.float64), 1.0, rtol=0, atol=1e-6)):
            raise ValueError("released probabilities must be finite, nonnegative, normalized rows")
        if (bits.shape[0] != len(rows) or decision.shape != (len(rows),)
                or not np.issubdtype(decision.dtype, np.integer)
                or np.any(decision < 0) or np.any(decision >= q.shape[1])
                or not np.array_equal(decision, q.argmax(axis=1))):
            raise ValueError("released rows, code, probabilities, and decisions disagree")
        return cls(rows.copy(), bits.copy(), ids, q.copy(), decision.copy(), dict(metadata))

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(self.metadata, sort_keys=True, separators=(",", ":"),
                            allow_nan=False).encode())
        for value in (self.row_id, self.code, self.probabilities, self.decision):
            array = np.ascontiguousarray(value)
            h.update(str(array.dtype).encode())
            h.update(str(array.shape).encode())
            h.update(array.tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class AlignedPair:
    first: PublicCode
    second: PublicCode

    def __post_init__(self) -> None:
        if not np.array_equal(self.first.row_id, self.second.row_id):
            raise ValueError("coalition release requires exactly aligned admitted row IDs")

    @property
    def pair_ids(self) -> np.ndarray:
        return np.column_stack((self.first.code_id, self.second.code_id))


def coverage(fit_ids: np.ndarray, screen_ids: np.ndarray) -> dict[str, object]:
    """Finite-code support diagnostics, including unseen-person mass."""
    fit = np.asarray(fit_ids)
    screen = np.asarray(screen_ids)
    if fit.ndim not in (1, 2) or screen.ndim != fit.ndim:
        raise ValueError("fit/screen ID rank disagrees")
    if fit.ndim == 2 and (fit.shape[1] != screen.shape[1] or fit.shape[1] != 2):
        raise ValueError("pair ID matrix must have two columns")
    if (not np.issubdtype(fit.dtype, np.integer) or not np.issubdtype(screen.dtype, np.integer)
            or np.any(fit < 0) or np.any(fit >= ALPHABET)
            or np.any(screen < 0) or np.any(screen >= ALPHABET)):
        raise ValueError("IDs must be in the public 10-bit alphabet")
    if len(fit) == 0 or len(screen) == 0:
        raise ValueError("fit/screen support cannot be empty")
    # Packed pair IDs need more than uint16 (the alphabet has 2**20 cells).
    f = fit if fit.ndim == 1 else fit[:, 0].astype(np.int64) * ALPHABET + fit[:, 1]
    s = screen if screen.ndim == 1 else screen[:, 0].astype(np.int64) * ALPHABET + screen[:, 1]
    unique, counts = np.unique(f, return_counts=True)
    p = counts.astype(np.float64) / len(f)
    unseen = ~np.isin(s, unique)
    return {
        "alphabet": ALPHABET if fit.ndim == 1 else ALPHABET ** 2,
        "fit_occupied": int(len(unique)),
        "screen_occupied": int(len(np.unique(s))),
        "fit_entropy_nats": float(-np.dot(p, np.log(p))),
        "fit_singleton_cells": int(np.sum(counts == 1)),
        "fit_singleton_person_fraction": float(np.sum(counts[counts == 1]) / len(f)),
        "screen_unseen_person_fraction": float(np.mean(unseen)),
        "screen_unseen_people": int(np.sum(unseen)),
    }
