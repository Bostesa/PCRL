"""Independent numerical checks of the task-only FIT validation stage.

This module reconstructs the group split and metric margins without importing
the fitting pipeline or its selector. It has no protected-label parameter.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import numpy as np


CAPS = {"accuracy": 0.010, "nll": 0.010, "brier": 0.005}
REFERENCES = ("U0", "Ucal", "F")


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def task_only_split(unit: np.ndarray, income: np.ndarray,
                    occupation: np.ndarray, seed: int):
    """Rebuild the group-disjoint 75/25 FIT split and task-only repair."""
    groups = np.asarray(unit)
    first, second = np.asarray(income), np.asarray(occupation)
    if (type(seed) is not int or seed not in (0, 1, 2)
            or groups.dtype != np.int64 or groups.ndim != 1 or len(groups) < 2
            or first.dtype != np.int64 or second.dtype != np.int64
            or first.shape != groups.shape or second.shape != groups.shape):
        raise ValueError("FIT task-only split source invalid")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("FIT group split requires distinct groups")
    permutation = np.random.default_rng(seed + 26_000).permutation(len(unique))
    size = min(len(unique) - 1, max(1, math.ceil(len(unique) * 0.25)))
    in_fit = ~np.isin(groups, unique[permutation[:size]])
    moved = []
    for name, labels, width in (("income", first, 2),
                                ("occupation", second, 6)):
        if not np.array_equal(np.unique(labels), np.arange(width)):
            raise ValueError(f"FIT {name} class support incomplete")
        for category in range(width):
            if np.any(labels[in_fit] == category):
                continue
            eligible = np.unique(groups[~in_fit & (labels == category)])
            if not len(eligible):
                raise ValueError(f"FIT {name} class support cannot be repaired")
            chosen = min(map(int, eligible),
                         key=lambda item: (int(np.count_nonzero(groups == item)), item))
            in_fit[groups == chosen] = True
            moved.append(chosen)
    fit = np.flatnonzero(in_fit)
    val = np.flatnonzero(~in_fit)
    if not len(val) or len(np.intersect1d(groups[fit], groups[val])):
        raise ValueError("FIT task split lost validation or group separation")
    receipt = {
        "schema": "pcrl-focused-task-only-fit-split-v1",
        "seed": seed, "split_seed_offset": 26_000,
        "selection_fraction": 0.25,
        "fit_rows": int(len(fit)), "validation_rows": int(len(val)),
        "fit_group_count": int(len(np.unique(groups[fit]))),
        "validation_group_count": int(len(np.unique(groups[val]))),
        "fit_row_index_sha256": array_sha256(fit.astype(np.int64)),
        "validation_row_index_sha256": array_sha256(val.astype(np.int64)),
        "fit_group_sha256": array_sha256(np.unique(groups[fit])),
        "validation_group_sha256": array_sha256(np.unique(groups[val])),
        "task_class_repair_moved_count": len(moved),
        "task_class_repair_moved_group_sha256": array_sha256(
            np.asarray(moved, dtype=np.int64)),
    }
    return fit, val, receipt


def compare_task_row(saved: Mapping, observed: Mapping,
                     references: Mapping) -> bool:
    """Check all saved FIT validation metrics and every U0/Ucal/F margin."""
    if set(observed) != set(CAPS) or set(saved.get("metrics", {})) != set(CAPS):
        raise ValueError("task metric set changed")
    for metric in CAPS:
        if not math.isclose(float(saved["metrics"][metric]), float(observed[metric]),
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError("task validation metric differs from package replay")
    margins = saved.get("utility_margins_by_ref_metric")
    if not isinstance(margins, Mapping) or set(margins) != set(REFERENCES) or \
            set(references) != set(REFERENCES):
        raise ValueError("task reference margin coverage incomplete")
    feasible = True
    for ref in REFERENCES:
        if set(references[ref]) != set(CAPS) or set(margins[ref]) != set(CAPS):
            raise ValueError("task reference metric set changed")
        expected = {
            "accuracy": observed["accuracy"] -
                        (references[ref]["accuracy"] - CAPS["accuracy"]),
            "nll": references[ref]["nll"] + CAPS["nll"] - observed["nll"],
            "brier": references[ref]["brier"] + CAPS["brier"] - observed["brier"],
        }
        for metric, value in expected.items():
            if not math.isfinite(value) or not math.isclose(
                    float(margins[ref][metric]), float(value),
                    rel_tol=0, abs_tol=1e-12):
                raise ValueError("task reference margin differs")
            feasible &= value >= 0
    if type(saved.get("utility_feasible")) is not bool or \
            saved["utility_feasible"] != feasible:
        raise ValueError("task utility flag differs from numeric margins")
    return True
