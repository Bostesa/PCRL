"""Bounded shared-bank FIT computation with an explicit pre-SEX T boundary.

All arrays are supplied by the guarded worker. This module opens no role file.
Its callbacks can only fit the five frozen masks or the shared full-F mask.
The task-only table and T receipt must publish successfully before `read_sex`
is invoked. Search critics see raw selected values only; final complete-view
attack fitting belongs to M4 on disjoint ATTACK_FIT_NEW rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from bank import ATOM_ORDER, SharedBank, _sha
from table import TaskMetrics, build_pair_table, PairTable


SPLIT_SEED_OFFSET = 26_000
SELECTION_FRACTION = 0.25
CAPS = {"accuracy": 0.010, "nll": 0.010, "brier": 0.005}


def _array_sha(value) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode() + str(arr.shape).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def task_only_split(unit: np.ndarray, income: np.ndarray,
                    occupation: np.ndarray, seed: int):
    """One group split; missing task classes move smallest groups, never SEX."""
    unit = np.asarray(unit)
    if (unit.ndim != 1 or unit.dtype != np.int64 or len(unit) < 2
            or income.shape != unit.shape or occupation.shape != unit.shape):
        raise ValueError("task-only FIT split input geometry is invalid")
    groups = np.unique(unit)
    if len(groups) < 2:
        raise ValueError("task-only FIT split requires distinct groups")
    permutation = np.random.default_rng(seed + SPLIT_SEED_OFFSET).permutation(len(groups))
    selected_groups = groups[permutation[:min(len(groups)-1, max(1, math.ceil(len(groups)*SELECTION_FRACTION)))]]
    in_fit = ~np.isin(unit, selected_groups)
    moved = []
    for name, labels, width in (("income", income, 2), ("occupation", occupation, 6)):
        labels = np.asarray(labels)
        if (labels.dtype != np.int64 or labels.shape != unit.shape or
                not np.array_equal(np.unique(labels), np.arange(width))):
            raise ValueError(f"full FIT {name} class support is incomplete")
        for category in range(width):
            if np.any(labels[in_fit] == category):
                continue
            eligible = np.unique(unit[~in_fit & (labels == category)])
            if len(eligible) == 0:
                raise ValueError(f"cannot preserve FIT {name} class support")
            chosen = min((int(item) for item in eligible),
                         key=lambda item: (int(np.count_nonzero(unit == item)), item))
            in_fit[unit == chosen] = True
            moved.append(chosen)
    fit = np.flatnonzero(in_fit)
    selection = np.flatnonzero(~in_fit)
    if not len(selection) or len(np.intersect1d(unit[fit], unit[selection])):
        raise ValueError("task-only split lost validation or group separation")
    receipt = {
        "schema": "pcrl-focused-task-only-fit-split-v1",
        "seed": seed, "split_seed_offset": SPLIT_SEED_OFFSET,
        "selection_fraction": SELECTION_FRACTION,
        "fit_rows": int(len(fit)), "validation_rows": int(len(selection)),
        "fit_group_count": int(len(np.unique(unit[fit]))),
        "validation_group_count": int(len(np.unique(unit[selection]))),
        "fit_row_index_sha256": _array_sha(fit.astype(np.int64)),
        "validation_row_index_sha256": _array_sha(selection.astype(np.int64)),
        "fit_group_sha256": _array_sha(np.unique(unit[fit])),
        "validation_group_sha256": _array_sha(np.unique(unit[selection])),
        "task_class_repair_moved_count": len(moved),
        "task_class_repair_moved_group_sha256": _array_sha(np.asarray(moved, dtype=np.int64)),
    }
    return fit, selection, receipt


def _task_metrics(prob, labels) -> TaskMetrics:
    p = np.asarray(prob, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if (p.ndim != 2 or p.shape[0] != len(y) or y.ndim != 1
            or not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-8)
            or np.any(y < 0) or np.any(y >= p.shape[1])):
        raise ValueError("task metric probabilities or labels are invalid")
    onehot = np.eye(p.shape[1], dtype=np.float64)[y]
    return TaskMetrics(
        accuracy=float(np.mean(p.argmax(axis=1) == y)),
        nll=float(np.mean(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)))),
        brier=float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
    )


def _task_row(mask, package, metrics: TaskMetrics,
              references: Mapping[str, TaskMetrics]) -> dict:
    margins = {
        name: {
            "accuracy": metrics.accuracy - (ref.accuracy - CAPS["accuracy"]),
            "nll": ref.nll + CAPS["nll"] - metrics.nll,
            "brier": ref.brier + CAPS["brier"] - metrics.brier,
        }
        for name, ref in references.items()
    }
    return {
        "mask_id": mask.mask_id, "mask_sha256": mask.mask_sha256,
        "head_id": package.model_id,
        "metrics": dict(vars(metrics)),
        "utility_margins_by_ref_metric": margins,
        "utility_feasible": all(value >= 0 for group in margins.values()
                                for value in group.values()),
    }


def _full_mask(bank: SharedBank) -> dict:
    columns = tuple(sorted({column for mask in bank.masks
                            for column in mask.column_indices}))
    if columns != tuple(range(83)):
        raise ValueError("shared bank does not cover the full F source schema")
    identity = {"source_schema_sha256": bank.source_schema_sha256,
                "atoms": ATOM_ORDER, "column_indices": columns}
    return {**identity, "mask_id": "F_FULL83", "mask_sha256": _sha(identity)}


def _package_record(package) -> dict:
    return {name: getattr(package, name) for name in (
        "cache_key", "model_id", "head_sha256", "calibrator_sha256",
        "package_sha256", "receipt_sha256", "relative_package_path",
    )}


def _validation_package_record(package, purpose: str, mask_id: str) -> dict:
    record = _package_record(package)
    receipt_path = Path(record["relative_package_path"]).parent / "HEAD_POOL_RECEIPT.json"
    record["head_pool_receipt_sha256"] = record["receipt_sha256"]
    record["head_pool_receipt_relative_path"] = receipt_path.as_posix()
    return {"stage": "validation", "purpose": purpose, "mask_id": mask_id,
            **record}


@dataclass(frozen=True)
class FitPipelineResult:
    task_table: dict
    task_only_choice: dict
    validation_packages: dict[str, dict[str, dict]]
    pair_table: PairTable | None
    final_packages: dict[str, dict[str, dict]]
    full_f_packages: dict[str, dict]
    split_receipt: dict
    raw_search_receipt: dict


def run_fit_pipeline(*, bank: SharedBank, seed: int,
                     fit_x: np.ndarray, fit_row_id: np.ndarray,
                     fit_unit: np.ndarray, fit_income: np.ndarray,
                     fit_occupation: np.ndarray,
                     fit_references: Mapping[str, np.ndarray],
                     feature_names: tuple[str, ...],
                     fit_package: Callable,
                     read_sex: Callable[[], np.ndarray],
                     fit_raw_search_auc: Callable,
                     select_task_only: Callable,
                     on_task_complete: Callable,
                     on_pair_complete: Callable | None = None) -> FitPipelineResult:
    """Run the predeclared 5×5 FIT search without reading any screen role."""
    x = np.asarray(fit_x)
    row = np.asarray(fit_row_id)
    unit = np.asarray(fit_unit)
    y = {"income": np.asarray(fit_income),
         "occupation_group": np.asarray(fit_occupation)}
    if (type(seed) is not int or seed < 0 or x.dtype != np.float32
            or x.ndim != 2 or x.shape[1] != 83
            or row.dtype != np.int64 or row.shape != (len(x),)
            or unit.dtype != np.int64 or unit.shape != row.shape
            or len(np.unique(row)) != len(row)
            or len(feature_names) != 83 or not np.isfinite(x).all()):
        raise ValueError("admitted FIT feature/identity schema is invalid")
    if set(fit_references) != {"U0_1", "U0_2", "Ucal_1", "Ucal_2"}:
        raise ValueError("frozen U0/Ucal reference fields are incomplete")
    fit_rows, val_rows, split_receipt = task_only_split(
        unit, y["income"], y["occupation_group"], seed)
    full = _full_mask(bank)
    refs: dict[str, dict[str, TaskMetrics]] = {}
    f_validation = {}
    validation_packages: dict[str, dict[str, dict]] = {
        "income": {}, "occupation_group": {},
    }
    for purpose, index, width in (("income", 1, 2),
                                   ("occupation_group", 2, 6)):
        for kind in ("U0", "Ucal"):
            p = np.asarray(fit_references[f"{kind}_{index}"])
            if p.shape != (len(x), width):
                raise ValueError("FIT reference probability shape differs from source")
            refs.setdefault(purpose, {})[kind] = _task_metrics(p[val_rows], y[purpose][val_rows])
        f_package = fit_package(purpose, full, "validation", fit_rows)
        f_validation[purpose] = f_package
        validation_packages[purpose]["F_FULL83"] = _validation_package_record(
            f_package, purpose, "F_FULL83")
        f_q = f_package.restored_release.view(x[val_rows], feature_names)["probabilities"]
        refs[purpose]["F"] = _task_metrics(f_q, y[purpose][val_rows])
    task_rows = {"income": [], "occupation_group": []}
    evaluations: dict[str, dict[str, dict]] = {"income": {}, "occupation_group": {}}
    for purpose in ("income", "occupation_group"):
        for mask in bank.masks:
            package = fit_package(purpose, mask.public_record(), "validation", fit_rows)
            validation_packages[purpose][mask.mask_id] = _validation_package_record(
                package, purpose, mask.mask_id)
            q = package.restored_release.view(x[val_rows], feature_names)["probabilities"]
            metrics = _task_metrics(q, y[purpose][val_rows])
            row_score = _task_row(mask, package, metrics, refs[purpose])
            task_rows[purpose].append(row_score)
            evaluations[purpose][mask.mask_id] = {
                "head_id": package.model_id, "metrics": metrics,
            }
    content = {
        "schema": "pcrl-focused-task-table-v1",
        "bank_sha256": bank.bank_sha256,
        "split_receipt": split_receipt,
        "income_masks": task_rows["income"],
        "occupation_masks": task_rows["occupation_group"],
    }
    task_table = {**content, "task_table_sha256": _sha(content)}
    try:
        choice = select_task_only(task_table, bank)
    except ValueError as exc:
        if not str(exc).startswith("no utility-feasible task-only "):
            raise
        choice = {
            "status": "NO_UTILITY_FEASIBLE_T",
            "pair_id": None,
            "reason": str(exc),
            "task_table_sha256": task_table["task_table_sha256"],
            "bank_sha256": bank.bank_sha256,
        }
    if (not isinstance(choice, Mapping)
            or (choice.get("status") == "NO_UTILITY_FEASIBLE_T"
                and choice.get("pair_id") is not None)
            or (choice.get("status") != "NO_UTILITY_FEASIBLE_T"
                and choice.get("pair_id") not in
                {pair.pair_id for pair in bank.pairs})):
        raise ValueError("pre-sensitive T did not yield a frozen pair or explicit no-go")
    on_task_complete(task_table, dict(choice), validation_packages)

    if choice.get("status") == "NO_UTILITY_FEASIBLE_T":
        # The task premise has failed. No protected-label read or privacy
        # search is justified, and no final release package is produced.
        return FitPipelineResult(task_table, dict(choice), validation_packages,
                                 None, {}, {},
                                 split_receipt, {"status": "NOT_RUN_TASK_NO_GO"})

    # This is the first point at which a protected FIT label may be requested.
    sex = np.asarray(read_sex())
    if (sex.dtype != np.int64 or sex.shape != (len(x),)
            or not np.array_equal(np.unique(sex[fit_rows]), [0, 1])
            or not np.array_equal(np.unique(sex[val_rows]), [0, 1])):
        raise ValueError("FIT SEX search split lacks two classes or row alignment")
    local_auc = {}
    for mask in bank.masks:
        local_auc[mask.mask_id] = float(fit_raw_search_auc(
            mask.column_indices, fit_rows, val_rows, sex))
        if not 0.5 <= local_auc[mask.mask_id] <= 1:
            raise ValueError("raw selected local search AUC is not oriented")
    union_auc = {}
    for pair in bank.pairs:
        if pair.union_sha256 not in union_auc:
            union_auc[pair.union_sha256] = float(fit_raw_search_auc(
                pair.union_columns, fit_rows, val_rows, sex))
        if not 0.5 <= union_auc[pair.union_sha256] <= 1:
            raise ValueError("raw union coalition search AUC is not oriented")
    for purpose in evaluations:
        for mask_id, record in evaluations[purpose].items():
            record["local_search_auc"] = local_auc[mask_id]
    pair_table = build_pair_table(
        bank, evaluations, refs, union_auc,
        critic_input_kind="RAW_UNION_ONLY",
    )
    if on_pair_complete is not None:
        on_pair_complete(pair_table)
    final_packages = {"income": {}, "occupation_group": {}}
    full_f = {}
    all_rows = np.arange(len(x), dtype=np.int64)
    for purpose in final_packages:
        for mask in bank.masks:
            final_packages[purpose][mask.mask_id] = _package_record(
                fit_package(purpose, mask.public_record(), "final", all_rows))
        full_f[purpose] = _package_record(
            fit_package(purpose, full, "final", all_rows))
    search_receipt = {
        "schema": "pcrl-focused-raw-search-proxy-v1",
        "local_critic_input_kind": "RAW_SELECTED_ONLY",
        "coalition_critic_input_kind": "RAW_UNION_ONLY",
        "unique_local_masks": len(local_auc),
        "unique_raw_unions": len(union_auc),
        "orientation": "max(AUC,1-AUC) per fitted reader then max readers",
        "sex_fit_class_counts": np.bincount(sex[fit_rows], minlength=2).tolist(),
        "sex_validation_class_counts": np.bincount(sex[val_rows], minlength=2).tolist(),
    }
    return FitPipelineResult(task_table, dict(choice), validation_packages,
                             pair_table,
                             final_packages, full_f, split_receipt,
                             search_receipt)
