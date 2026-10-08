"""In-memory Team C grouped-mask scorecard fitting.

The caller supplies arrays from S2's admitted OSF_DEFENSE_FIT role. This module
has no filesystem paths, source-loader import, calibration role, or access to
HEAD_VALIDATION/assessment data. It fits fixed cheap proxies solely for mask
selection; final task and privacy evaluation use separate fitted models.
"""

from __future__ import annotations

import hashlib
import math
import copy
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np

from prototype import AdultGroupedSchema
from selector import Scorecard, TaskDelta


@dataclass(frozen=True)
class ScorecardFitResult:
    card: Scorecard
    mask_metrics: tuple[dict[str, Any], ...]
    split_receipt: dict[str, Any]
    fit_counts: dict[str, int]
    alias_masks: tuple[str, ...]


@dataclass
class ScorecardProxyCache:
    """Exact within-seed fit cache for full and relationship-removed cards."""

    fingerprint: str | None = None
    mask_metrics: dict[tuple[str, ...], dict[str, Any]] = field(default_factory=dict)


def _fit_fingerprint(
    X: np.ndarray,
    income: np.ndarray,
    occupation: np.ndarray,
    sex: np.ndarray,
    unit: np.ndarray,
    feature_names: tuple[str, ...],
    seed: int,
    selection_fraction: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"team-c-scorecard-proxy-v2|class-safe|lr-C1-iter300|tree-depth2-leaf10")
    digest.update("\n".join(feature_names).encode())
    digest.update(f"|seed={seed}|fraction={selection_fraction}".encode())
    for value in (X, income, occupation, sex, unit):
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def group_disjoint_split(
    unit: np.ndarray, *, seed: int, selection_fraction: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Make a deterministic group-disjoint split without reading any labels."""
    unit = np.asarray(unit)
    if unit.ndim != 1 or len(unit) < 2:
        raise ValueError("unit must be a one-dimensional array with at least two rows")
    if not 0 < selection_fraction < 1:
        raise ValueError("selection_fraction must lie strictly between zero and one")
    groups = np.unique(unit)
    if len(groups) < 2:
        raise ValueError("at least two distinct groups are needed")
    permutation = np.random.default_rng(seed).permutation(len(groups))
    selection_count = min(len(groups) - 1, max(1, math.ceil(len(groups) * selection_fraction)))
    selected = groups[permutation[:selection_count]]
    is_selection = np.isin(unit, selected)
    return np.flatnonzero(~is_selection), np.flatnonzero(is_selection)


def preserve_fit_class_support(
    unit: np.ndarray,
    fit: np.ndarray,
    selection: np.ndarray,
    labels: dict[str, np.ndarray],
    vocabularies: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Move the smallest whole validation group for each missing fit class.

    This fixed repair uses only the caller's already-authorized fitting-role
    labels. It never replenishes validation or resplits per mask. Moved IDs
    stay private; public receipts contain only their count and hash.
    """
    unit = np.asarray(unit)
    if unit.ndim != 1 or unit.dtype.kind not in "iu":
        raise ValueError("numeric group IDs are required")
    if set(labels) != set(vocabularies):
        raise ValueError("labels and vocabularies differ")
    in_fit = np.zeros(len(unit), dtype=bool)
    in_fit[fit] = True
    if len(fit) + len(selection) != len(unit) or np.any(in_fit[selection]):
        raise ValueError("input split does not partition rows")
    moved = []
    for name in sorted(labels):
        y = np.asarray(labels[name])
        classes = vocabularies[name]
        if y.shape != unit.shape or not np.array_equal(np.unique(y), np.arange(classes)):
            raise ValueError(f"{name} full fitting role lacks declared classes")
        for missing in range(classes):
            if missing in set(np.unique(y[in_fit])):
                continue
            eligible = np.unique(unit[(~in_fit) & (y == missing)])
            if len(eligible) == 0:
                raise ValueError(f"cannot retain {name} class {missing} in fit")
            chosen = min(
                (int(candidate) for candidate in eligible),
                key=lambda candidate: (int(np.count_nonzero(unit == candidate)), candidate),
            )
            in_fit[unit == chosen] = True
            moved.append(chosen)
    new_fit, new_selection = np.flatnonzero(in_fit), np.flatnonzero(~in_fit)
    if len(new_selection) == 0:
        raise ValueError("class-safe split leaves no validation rows")
    return new_fit, new_selection, tuple(moved)


def _validate_inputs(
    X: np.ndarray,
    feature_names: tuple[str, ...],
    income: np.ndarray,
    occupation: np.ndarray,
    sex: np.ndarray,
    unit: np.ndarray,
    schema: AdultGroupedSchema,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    schema.validate_feature_names(feature_names)
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 83 or not np.isfinite(X).all():
        raise ValueError("X must be a finite 83-column matrix")
    labels = []
    for name, value, class_count in (
        ("income", income, 2),
        ("occupation", occupation, 6),
        ("sex", sex, 2),
    ):
        value = np.asarray(value)
        if (value.shape != (len(X),) or value.dtype.kind not in "iu"
                or np.any(value < 0) or np.any(value >= class_count)):
            raise ValueError(f"{name} must be an aligned integer class vector")
        labels.append(value.astype(np.int64, copy=False))
    unit = np.asarray(unit)
    if unit.shape != (len(X),):
        raise ValueError("unit IDs must align with X")
    return X, *labels, unit


def build_scorecard(
    X: np.ndarray,
    feature_names: tuple[str, ...],
    income: np.ndarray,
    occupation: np.ndarray,
    sex: np.ndarray,
    unit: np.ndarray,
    *,
    base_atoms: tuple[str, ...],
    optional_atoms: tuple[str, ...],
    seed: int,
    selection_fraction: float = 0.25,
    cache: ScorecardProxyCache | None = None,
) -> ScorecardFitResult:
    """Fit base and one-atom-removal proxies on one defense-fit internal split.

    Per mask: two task logistic proxies, one SEX logistic proxy, one fixed
    depth-two SEX tree. SEX recovery is max(AUC, 1-AUC) for each proxy, then
    max across proxies; this rule is fixed for every mask and arm.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    schema = AdultGroupedSchema()
    X, income, occupation, sex, unit = _validate_inputs(
        X, feature_names, income, occupation, sex, unit, schema
    )
    if cache is None:
        cache = ScorecardProxyCache()
    fingerprint = _fit_fingerprint(
        X, income, occupation, sex, unit, schema.feature_names,
        seed, selection_fraction,
    )
    if cache.fingerprint is None:
        cache.fingerprint = fingerprint
    elif cache.fingerprint != fingerprint:
        raise ValueError("scorecard cache identity differs from current inputs or proxy settings")
    base_atoms = schema.normalize_atoms(base_atoms)
    optional_atoms = schema.normalize_atoms(optional_atoms)
    if not optional_atoms or not set(optional_atoms) <= set(base_atoms):
        raise ValueError("optional atoms must be a nonempty subset of base atoms")
    if len(base_atoms) - 1 < 1:
        raise ValueError("single-atom removals must leave at least one selected atom")
    fit, selection = group_disjoint_split(
        unit, seed=seed, selection_fraction=selection_fraction
    )
    fit, selection, moved_groups = preserve_fit_class_support(
        unit, fit, selection,
        {"income": income, "occupation": occupation, "sex": sex},
        {"income": 2, "occupation": 6, "sex": 2},
    )
    for name, labels, class_count in (
        ("income", income, 2), ("occupation", occupation, 6), ("sex", sex, 2)
    ):
        if not np.array_equal(np.unique(labels[fit]), np.arange(class_count)):
            raise ValueError(f"internal fit lacks a {name} class")
    if not np.array_equal(np.unique(sex[selection]), np.arange(2)):
        raise ValueError("internal selection lacks a SEX class for AUC")

    metrics = []
    labels_by_task = {"income": income, "occupation_group": occupation}
    classes_by_task = {"income": 2, "occupation_group": 6}
    masks = [("base", base_atoms)] + [
        (f"without:{atom}", tuple(a for a in base_atoms if a != atom))
        for atom in optional_atoms
    ]
    alias_masks = []
    new_mask_count = 0
    for mask_id, kept in masks:
        if kept in cache.mask_metrics:
            reused = copy.deepcopy(cache.mask_metrics[kept])
            reused["mask_id"] = mask_id
            metrics.append(reused)
            alias_masks.append(mask_id)
            continue
        new_mask_count += 1
        indices = schema.column_indices(kept)
        x_fit = X[np.ix_(fit, indices)]
        x_selection = X[np.ix_(selection, indices)]
        task_scores = {}
        for task, task_labels in labels_by_task.items():
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=seed),
            )
            model.fit(x_fit, task_labels[fit])
            probabilities = model.predict_proba(x_selection)
            y = task_labels[selection]
            k = classes_by_task[task]
            if not np.array_equal(model[-1].classes_, np.arange(k)):
                raise ValueError(f"{task} proxy changed class order")
            nll = float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-12, 1.0)).mean())
            one_hot = np.eye(k, dtype=float)[y]
            brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())
            accuracy = float((probabilities.argmax(axis=1) == y).mean())
            task_scores[task] = {"nll": nll, "brier": brier, "accuracy": accuracy}

        sex_lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=seed),
        )
        sex_tree = DecisionTreeClassifier(
            max_depth=2, min_samples_leaf=10, random_state=seed
        )
        sex_lr.fit(x_fit, sex[fit])
        sex_tree.fit(x_fit, sex[fit])
        reader_scores = {}
        for name, model in (("logistic", sex_lr), ("tree_depth2", sex_tree)):
            probability = model.predict_proba(x_selection)[:, 1]
            raw_auc = float(roc_auc_score(sex[selection], probability))
            reader_scores[name] = {
                "oriented_auc": raw_auc,
                "reversible_auc": max(raw_auc, 1.0 - raw_auc),
            }
        recovery = max(value["reversible_auc"] for value in reader_scores.values())
        metrics.append({
            "mask_id": mask_id,
            "selected_atoms": kept,
            "selected_columns": len(indices),
            "task": task_scores,
            "sex_readers": reader_scores,
            "sex_proxy_recovery_auc": recovery,
        })
        cache.mask_metrics[kept] = copy.deepcopy(metrics[-1])

    base = metrics[0]
    by_removed = {row["mask_id"].split(":", 1)[1]: row for row in metrics[1:]}
    task_deltas = {}
    for task in labels_by_task:
        task_deltas[task] = {
            atom: TaskDelta(
                nll=by_removed[atom]["task"][task]["nll"] - base["task"][task]["nll"],
                brier=by_removed[atom]["task"][task]["brier"] - base["task"][task]["brier"],
                accuracy=by_removed[atom]["task"][task]["accuracy"] - base["task"][task]["accuracy"],
            )
            for atom in optional_atoms
        }
    auc_drop = {
        atom: base["sex_proxy_recovery_auc"] - by_removed[atom]["sex_proxy_recovery_auc"]
        for atom in optional_atoms
    }
    fit_groups = np.unique(unit[fit]).astype("<i8", copy=False)
    selection_groups = np.unique(unit[selection]).astype("<i8", copy=False)
    receipt = {
        "seed": int(seed),
        "selection_fraction": float(selection_fraction),
        "fit_rows": len(fit),
        "selection_rows": len(selection),
        "fit_groups": len(fit_groups),
        "selection_groups": len(selection_groups),
        "overlap_groups": len(np.intersect1d(fit_groups, selection_groups)),
        "class_safe_moved_groups": len(moved_groups),
        "class_safe_moved_group_sha256": hashlib.sha256(
            np.asarray(moved_groups, dtype="<i8").tobytes()
        ).hexdigest(),
        "fit_group_sha256": hashlib.sha256(fit_groups.tobytes()).hexdigest(),
        "selection_group_sha256": hashlib.sha256(selection_groups.tobytes()).hexdigest(),
    }
    return ScorecardFitResult(
        card=Scorecard(optional_atoms, auc_drop, task_deltas),
        mask_metrics=tuple(metrics),
        split_receipt=receipt,
        fit_counts={
            "task_proxy": 2 * new_mask_count,
            "sex_logistic_proxy": new_mask_count,
            "sex_tree_proxy": new_mask_count,
            "total": 4 * new_mask_count,
        },
        alias_masks=tuple(alias_masks),
    )
