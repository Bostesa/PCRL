"""Frozen FIT-only group splits shared by full-input and masked recipes."""

from __future__ import annotations

import hashlib
import math

import numpy as np


OUTER_FRACTION = 0.25
OUTER_SEED_OFFSET = 26000
INNER_FRACTION = 0.15
INNER_SEED_OFFSET = 17001


def array_sha(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(array.dtype).encode() + str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def _split(unit: np.ndarray, labels: dict[str, np.ndarray], seed: int,
           fraction: float, offset: int, schema: str):
    u = np.asarray(unit)
    if (u.dtype != np.int64 or u.ndim != 1 or len(u) < 2 or
            set(labels) != {"income", "occupation_group"}):
        raise ValueError("FIT group/label split contract differs")
    for name, cap in (("income", 2), ("occupation_group", 6)):
        y = np.asarray(labels[name])
        if (y.dtype != np.int64 or y.shape != u.shape or
                not np.array_equal(np.unique(y), np.arange(cap))):
            raise ValueError("FIT task class support differs")
    groups = np.unique(u)
    selected_count = min(len(groups) - 1, max(1, math.ceil(len(groups) * fraction)))
    perm = np.random.default_rng(seed + offset).permutation(len(groups))
    selected_groups = groups[perm[:selected_count]]
    in_fit = ~np.isin(u, selected_groups)
    moved = []
    for name, cap in (("income", 2), ("occupation_group", 6)):
        y = labels[name]
        for cls in range(cap):
            if np.any(y[in_fit] == cls):
                continue
            eligible = np.unique(u[(~in_fit) & (y == cls)])
            if len(eligible) == 0:
                raise ValueError("FIT class support cannot be repaired")
            chosen = min((int(value) for value in eligible),
                         key=lambda value: (int(np.count_nonzero(u == value)), value))
            in_fit[u == chosen] = True
            moved.append(chosen)
    fit = np.flatnonzero(in_fit)
    validation = np.flatnonzero(~in_fit)
    if (len(validation) == 0 or
            len(np.intersect1d(u[fit], u[validation]))):
        raise ValueError("FIT split lost validation or group separation")
    receipt = {
        "schema": schema, "seed": seed, "split_seed_offset": offset,
        "selection_fraction": fraction,
        "fit_rows": int(len(fit)), "validation_rows": int(len(validation)),
        "fit_group_count": int(len(np.unique(u[fit]))),
        "validation_group_count": int(len(np.unique(u[validation]))),
        "fit_row_index_sha256": array_sha(fit.astype(np.int64)),
        "validation_row_index_sha256": array_sha(validation.astype(np.int64)),
        "fit_group_sha256": array_sha(np.unique(u[fit])),
        "validation_group_sha256": array_sha(np.unique(u[validation])),
        "task_class_repair_moved_count": len(moved),
        "task_class_repair_moved_group_sha256": array_sha(np.asarray(moved, dtype=np.int64)),
    }
    return fit, validation, receipt


def outer_split(unit: np.ndarray, income: np.ndarray,
                occupation: np.ndarray, seed: int):
    """Same 25% grouped split and repair as the preceding focused study."""
    return _split(unit, {"income": income, "occupation_group": occupation},
                  seed, OUTER_FRACTION, OUTER_SEED_OFFSET,
                  "pcrl-focused-task-only-fit-split-v1")


def inner_split(unit: np.ndarray, income: np.ndarray,
                occupation: np.ndarray, seed: int):
    """15% grouped FIT fold for B C selection, unseen by its encoder."""
    return _split(unit, {"income": income, "occupation_group": occupation},
                  seed, INNER_FRACTION, INNER_SEED_OFFSET,
                  "pcrl-utility-inner-head-selection-v1")
