"""Task-only T selection before the fitting worker opens protected labels.

This module imports no score/critic code and accepts a strict task-only table.
The worker must save the resulting receipt before reading FIT SEX.
"""

from __future__ import annotations

import hashlib
import json
from math import isfinite


_REFS = ("U0", "Ucal", "F")
_METRICS = ("accuracy", "nll", "brier")
_MASK_KEYS = {"mask_id", "mask_sha256", "head_id", "metrics",
              "utility_margins_by_ref_metric", "utility_feasible"}


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _full_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _masks(rows, purpose):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"task-only {purpose} mask list is empty")
    if len({row.get("mask_id") for row in rows}) != len(rows):
        raise ValueError("task-only mask IDs are duplicate")
    for row in rows:
        if not isinstance(row, dict) or set(row) != _MASK_KEYS:
            raise ValueError("task-only mask fields differ from strict contract")
        if (not isinstance(row["mask_id"], str) or not row["mask_id"]
                or "__" in row["mask_id"] or not _full_sha(row["mask_sha256"])
                or not isinstance(row["head_id"], str) or not row["head_id"]):
            raise ValueError("task-only mask/head identity is invalid")
        metrics = row["metrics"]
        if (not isinstance(metrics, dict) or set(metrics) != set(_METRICS)
                or any(type(value) not in (float, int) or not isfinite(value)
                       for value in metrics.values())
                or not 0 <= metrics["accuracy"] <= 1 or metrics["nll"] < 0
                or not 0 <= metrics["brier"] <= 2):
            raise ValueError("task-only validation metrics are invalid")
        margins = row["utility_margins_by_ref_metric"]
        if not isinstance(margins, dict) or set(margins) != set(_REFS):
            raise ValueError("task-only reference margins are incomplete")
        values = []
        for reference in _REFS:
            entry = margins[reference]
            if (not isinstance(entry, dict) or set(entry) != set(_METRICS)
                    or any(type(value) not in (float, int) or not isfinite(value)
                           for value in entry.values())):
                raise ValueError("task-only utility margins are invalid")
            values.extend(entry.values())
        if (type(row["utility_feasible"]) is not bool or
                row["utility_feasible"] != all(value >= 0 for value in values)):
            raise ValueError("task-only utility status differs from margins")
    feasible = [row for row in rows if row["utility_feasible"]]
    if not feasible:
        raise ValueError(f"no utility-feasible task-only {purpose} mask")
    return min(feasible, key=lambda row: (row["metrics"]["nll"], row["mask_sha256"]))


def select_task_only_pre_sensitive(task_table):
    """Choose T from task metrics only; fail on every unregistered field."""
    if (not isinstance(task_table, dict) or set(task_table) != {
        "schema", "bank_sha256", "split_receipt", "income_masks",
        "occupation_masks", "task_table_sha256",
    } or task_table["schema"] != "pcrl-focused-task-table-v1"
            or not _full_sha(task_table["bank_sha256"])
            or not isinstance(task_table["split_receipt"], dict)
            or not task_table["split_receipt"]):
        raise ValueError("task-only table fields differ from strict contract")
    content = {key: value for key, value in task_table.items()
               if key != "task_table_sha256"}
    if not _full_sha(task_table["task_table_sha256"]) or \
            task_table["task_table_sha256"] != _sha(content):
        raise ValueError("task-only table SHA differs from contents")
    first = _masks(task_table["income_masks"], "income")
    second = _masks(task_table["occupation_masks"], "occupation")
    selection = {
        "schema": "pcrl-focused-pre-sensitive-T-v1",
        "task_table_sha256": task_table["task_table_sha256"],
        "bank_sha256": task_table["bank_sha256"],
        "pair_id": f"{first['mask_id']}__{second['mask_id']}",
        "income_mask_id": first["mask_id"],
        "occupation_mask_id": second["mask_id"],
        "income_mask_sha256": first["mask_sha256"],
        "occupation_mask_sha256": second["mask_sha256"],
        "head_ids": {"income": first["head_id"],
                     "occupation_group": second["head_id"]},
    }
    return {**selection, "selection_receipt_sha256": _sha(selection)}
