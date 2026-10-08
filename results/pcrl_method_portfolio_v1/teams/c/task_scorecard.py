"""Task-only grouped-mask scorecard for the C-SIMPLE utility control.

This fitting boundary accepts features, task labels and group IDs only. It
cannot inspect SEX, and its group repair depends only on the two task labels.
The privacy scorecard used by the other simple arms is fitted separately.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from prototype import ATOM_ORDER, AdultGroupedSchema
from selector import TaskDelta


@dataclass(frozen=True)
class TaskScorecardFitResult:
    optional_atoms: tuple[str, ...]
    task_deltas: Mapping[str, Mapping[str, TaskDelta]]
    mask_metrics: tuple[dict[str, Any], ...]
    split_receipt: dict[str, Any]
    fit_counts: dict[str, int]


def _hash_groups(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def group_disjoint_split(
    unit: np.ndarray, *, seed: int, selection_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the fixed group split without importing the privacy builder."""
    unit = np.asarray(unit)
    if unit.ndim != 1 or len(unit) < 2 or not 0 < selection_fraction < 1:
        raise ValueError("invalid task-only group split input")
    groups = np.unique(unit)
    if len(groups) < 2:
        raise ValueError("task-only split needs at least two groups")
    permutation = np.random.default_rng(seed).permutation(len(groups))
    selection_count = min(len(groups) - 1, max(1, math.ceil(len(groups) * selection_fraction)))
    chosen = groups[permutation[:selection_count]]
    is_selection = np.isin(unit, chosen)
    return np.flatnonzero(~is_selection), np.flatnonzero(is_selection)


def preserve_fit_class_support(
    unit: np.ndarray, fit: np.ndarray, selection: np.ndarray,
    labels: Mapping[str, np.ndarray], vocabularies: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Repair only income and occupation fit classes with whole groups."""
    if (set(labels) != {"income", "occupation_group"}
            or dict(vocabularies) != {"income": 2, "occupation_group": 6}):
        raise ValueError("task-only repair accepts exactly the two task labels")
    unit = np.asarray(unit)
    if unit.ndim != 1 or unit.dtype.kind not in "iu":
        raise ValueError("numeric group IDs are required")
    in_fit = np.zeros(len(unit), dtype=bool)
    in_fit[fit] = True
    if len(fit) + len(selection) != len(unit) or np.any(in_fit[selection]):
        raise ValueError("task-only split does not partition rows")
    moved = []
    for task in sorted(labels):
        y = np.asarray(labels[task])
        classes = vocabularies[task]
        if y.shape != unit.shape or not np.array_equal(np.unique(y), np.arange(classes)):
            raise ValueError(f"{task} fitting role lacks declared classes")
        for missing in range(classes):
            if missing in set(np.unique(y[in_fit])):
                continue
            eligible = np.unique(unit[(~in_fit) & (y == missing)])
            if len(eligible) == 0:
                raise ValueError(f"cannot retain {task} class {missing} in fit")
            chosen = min(
                (int(candidate) for candidate in eligible),
                key=lambda candidate: (int(np.count_nonzero(unit == candidate)), candidate),
            )
            in_fit[unit == chosen] = True
            moved.append(chosen)
    new_fit, new_selection = np.flatnonzero(in_fit), np.flatnonzero(~in_fit)
    if len(new_selection) == 0:
        raise ValueError("task-only class repair leaves no validation rows")
    return new_fit, new_selection, tuple(moved)


def build_task_scorecard(
    X: np.ndarray,
    feature_names: tuple[str, ...],
    income: np.ndarray,
    occupation: np.ndarray,
    unit: np.ndarray,
    *,
    seed: int,
    selection_fraction: float = 0.25,
) -> TaskScorecardFitResult:
    """Fit 10 FULL-base masks with two task GLMs per mask, without SEX input."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    schema = AdultGroupedSchema()
    schema.validate_feature_names(feature_names)
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 83 or not np.isfinite(X).all():
        raise ValueError("X must be a finite 83-column matrix")
    unit = np.asarray(unit)
    if unit.shape != (len(X),) or unit.dtype.kind not in "iu":
        raise ValueError("group IDs must be aligned integers")
    labels = {}
    for task, value, classes in (
        ("income", income, 2), ("occupation_group", occupation, 6),
    ):
        value = np.asarray(value)
        if (value.shape != (len(X),) or value.dtype.kind not in "iu"
                or not np.array_equal(np.unique(value), np.arange(classes))):
            raise ValueError(f"{task} labels lack the declared full-role classes")
        labels[task] = value.astype(np.int64, copy=False)

    fit, selection = group_disjoint_split(
        unit, seed=seed, selection_fraction=selection_fraction,
    )
    fit, selection, moved = preserve_fit_class_support(
        unit, fit, selection, labels,
        {"income": 2, "occupation_group": 6},
    )
    for task, classes in (("income", 2), ("occupation_group", 6)):
        if not np.array_equal(np.unique(labels[task][fit]), np.arange(classes)):
            raise ValueError(f"internal fit lacks a {task} class")

    masks = [("base", ATOM_ORDER)] + [
        (f"without:{atom}", tuple(a for a in ATOM_ORDER if a != atom))
        for atom in ATOM_ORDER
    ]
    metrics = []
    for mask_id, kept in masks:
        indices = schema.column_indices(kept)
        x_fit = X[np.ix_(fit, indices)]
        x_selection = X[np.ix_(selection, indices)]
        task_scores = {}
        for task, classes in (("income", 2), ("occupation_group", 6)):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=seed),
            )
            model.fit(x_fit, labels[task][fit])
            probabilities = model.predict_proba(x_selection)
            if not np.array_equal(model[-1].classes_, np.arange(classes)):
                raise ValueError(f"{task} proxy changed class order")
            y = labels[task][selection]
            nll = float(-np.log(np.clip(
                probabilities[np.arange(len(y)), y], 1e-12, 1.0,
            )).mean())
            one_hot = np.eye(classes, dtype=float)[y]
            brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())
            accuracy = float((probabilities.argmax(axis=1) == y).mean())
            task_scores[task] = {
                "nll": nll, "brier": brier, "accuracy": accuracy,
            }
        metrics.append({
            "mask_id": mask_id,
            "selected_atoms": kept,
            "selected_columns": len(indices),
            "task": task_scores,
        })

    base = metrics[0]
    by_removed = {row["mask_id"].split(":", 1)[1]: row for row in metrics[1:]}
    task_deltas = {
        task: {
            atom: TaskDelta(
                nll=by_removed[atom]["task"][task]["nll"] - base["task"][task]["nll"],
                brier=by_removed[atom]["task"][task]["brier"] - base["task"][task]["brier"],
                accuracy=by_removed[atom]["task"][task]["accuracy"] - base["task"][task]["accuracy"],
            )
            for atom in ATOM_ORDER
        }
        for task in ("income", "occupation_group")
    }
    fit_groups = np.unique(unit[fit]).astype("<i8", copy=False)
    selection_groups = np.unique(unit[selection]).astype("<i8", copy=False)
    receipt = {
        "seed": int(seed),
        "selection_fraction": float(selection_fraction),
        "fit_rows": len(fit), "selection_rows": len(selection),
        "fit_groups": len(fit_groups), "selection_groups": len(selection_groups),
        "overlap_groups": len(np.intersect1d(fit_groups, selection_groups)),
        "task_only_moved_groups": len(moved),
        "task_only_moved_group_sha256": _hash_groups(np.asarray(moved)),
        "fit_group_sha256": _hash_groups(fit_groups),
        "selection_group_sha256": _hash_groups(selection_groups),
        "sensitive_label_input": False,
    }
    return TaskScorecardFitResult(
        optional_atoms=tuple(sorted(ATOM_ORDER)),
        task_deltas=task_deltas,
        mask_metrics=tuple(metrics),
        split_receipt=receipt,
        fit_counts={"task_proxy": 20, "sensitive_proxy": 0, "total": 20},
    )
