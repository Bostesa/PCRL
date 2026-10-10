"""Independent reconstruction of the frozen bank's original-ID action map.

This audit intentionally has no imports from candidate, source action-builder,
utility scorer, or checkpoint selection implementations.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    a = np.maximum(left, 1e-12)
    b = np.maximum(right, 1e-12)
    a = a / np.sum(a)
    b = b / np.sum(b)
    return float((np.sum(a * np.log(a / b)) + np.sum(b * np.log(b / a))) / 2)


def original_id_actions(table: np.ndarray, classes: np.ndarray,
                        cell_token: np.ndarray, cell_class: np.ndarray):
    """Return source-tie-rule actions expressed in original QPC table IDs."""
    p = np.asarray(table, dtype=np.float64)
    c = np.asarray(classes, dtype=np.int64)
    token = np.asarray(cell_token, dtype=np.int64)
    target = np.asarray(cell_class, dtype=np.int64)
    if (p.ndim != 2 or c.shape != (len(p),) or token.shape != target.shape
            or token.ndim != 1 or np.any(token < 0) or np.any(token >= len(p))
            or not np.array_equal(c, p.argmax(axis=1))
            or not np.array_equal(target, c[token])):
        raise ValueError("bank or original fine-token map differs")
    keys = [(int(c[i]), p[i].astype('<f8', copy=False).tobytes())
            for i in range(len(p))]
    unique = sorted(set(keys))
    if len(unique) != len(p):
        raise ValueError("duplicate bank state prevents original-ID bijection")
    position = {value: index for index, value in enumerate(unique)}
    old_to_sorted = np.asarray([position[value] for value in keys], dtype=np.int64)
    sorted_to_old = np.empty(len(p), dtype=np.int64)
    sorted_to_old[old_to_sorted] = np.arange(len(p), dtype=np.int64)
    ordered = p[sorted_to_old]
    ordered_class = c[sorted_to_old]
    medoid = {}
    for cls in sorted(set(target.tolist())):
        choices = np.flatnonzero(ordered_class == cls)
        totals = [sum(_distance(ordered[i], ordered[j]) for j in choices)
                  for i in choices]
        medoid[cls] = int(min(zip(totals, choices), key=lambda pair: (pair[0], pair[1]))[1])
    actions = []
    for old, cls in zip(token, target):
        first = int(old_to_sorted[old])
        choices = [int(i) for i in np.flatnonzero(ordered_class == cls) if i != first]
        choices.sort(key=lambda i: (_distance(ordered[first], ordered[i]), i))
        sorted_ids = (first, *choices[:2], medoid[int(cls)])
        row = tuple(dict.fromkeys(int(sorted_to_old[i]) for i in sorted_ids))
        if row[0] != int(old) or any(c[i] != cls for i in row):
            raise ValueError("original identity or class preservation differs")
        actions.append(row)
    return tuple(actions), old_to_sorted


def action_digest(actions) -> str:
    normalized = [[int(code) for code in row] for row in actions]
    payload = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(b"SCV1-action-map-v1\n" + payload).hexdigest()
