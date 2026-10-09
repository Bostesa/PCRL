"""Strict task-only five-mask table and T choice, with no SEX input or import."""

from __future__ import annotations

import hashlib
import json

from results.pcrl_capacity_matched_feature_selection_v1.m3.task_only import (
    select_task_only_pre_sensitive,
)


TASKS = ("income", "occupation_group")
REFERENCES = ("U0", "Ucal", "F")
METRICS = ("accuracy", "nll", "brier")
CAPS = {"accuracy": 0.010, "nll": 0.010, "brier": 0.005}
GATE_KEYS = {
    "schema", "seed", "mask_id", "mask_sha256", "selected_recipe",
    "full_input_gate_sha256", "conditional_lock_sha256",
    "mask_fit_receipt_sha256", "R1_gate_receipt_sha256",
    "common_role", "common_row_unit_pair_sha256",
    "prediction_relative_path", "prediction_sha256", "metrics",
    "point_eligibility", "all_task_reference_points_eligible",
    "development_only",
}


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _three(record: dict) -> dict[str, float]:
    if not isinstance(record, dict) or not all(key in record for key in METRICS):
        raise ValueError("task metric record lacks a common endpoint")
    return {key: float(record[key]) for key in METRICS}


def build_pre_sensitive_task_table(bank, seed: int, common_row_sha256: str,
                                   gates_by_mask: dict, head_ids: dict,
                                   gate_receipt_shas: dict, *,
                                   full_input_gate_sha256: str,
                                   selected_recipe: str,
                                   r1_gate_receipt_sha256: str,
                                   mask_lock_sha256_by_id: dict) -> tuple[dict, dict | None]:
    """Build the source strict T input from admitted INNER task aggregates.

    The caller supplies only five task-gate aggregate receipts and public head
    IDs. No protected outcome is present in this function's contract.
    """
    mask_ids = [mask.mask_id for mask in bank.masks]
    if (seed not in (0, 1, 2) or len(common_row_sha256) != 64
            or set(gates_by_mask) != set(mask_ids)
            or set(head_ids) != set(mask_ids)
            or set(gate_receipt_shas) != set(mask_ids)
            or set(mask_lock_sha256_by_id) != set(mask_ids)
            or selected_recipe not in ("A", "B")):
        raise ValueError("five-mask task-only table inputs are incomplete")
    reference_metrics = {}
    rows = {task: [] for task in TASKS}
    for mask in bank.masks:
        record = gates_by_mask[mask.mask_id]
        if (set(record) != GATE_KEYS
                or record.get("schema") != "pcrl-utility-masked-common-task-gate-v1"
                or record.get("mask_id") != mask.mask_id
                or record.get("mask_sha256") != mask.mask_sha256
                or record.get("seed") != seed
                or record.get("common_row_unit_pair_sha256") != common_row_sha256
                or record.get("full_input_gate_sha256") != full_input_gate_sha256
                or record.get("selected_recipe") != selected_recipe
                or record.get("R1_gate_receipt_sha256") != r1_gate_receipt_sha256
                or record.get("conditional_lock_sha256") !=
                mask_lock_sha256_by_id[mask.mask_id]
                or record.get("common_role") != "INNER_SELECTION"
                or not record.get("development_only")):
            raise ValueError("mask task gate identity differs")
        for task in TASKS:
            if set(head_ids[mask.mask_id]) != set(TASKS):
                raise ValueError("task head identities are incomplete")
            candidate = _three(record["metrics"]["mask"][task])
            margins = {}
            for reference in REFERENCES:
                baseline = _three(record["metrics"][reference][task])
                if task not in reference_metrics:
                    reference_metrics[task] = {}
                previous = reference_metrics[task].get(reference)
                if previous is None:
                    reference_metrics[task][reference] = baseline
                elif baseline != previous:
                    raise ValueError("reference task metrics drifted across mask gates")
                margin = {
                    "accuracy": CAPS["accuracy"] - (
                        baseline["accuracy"] - candidate["accuracy"]),
                    "nll": CAPS["nll"] - (
                        candidate["nll"] - baseline["nll"]),
                    "brier": CAPS["brier"] - (
                        candidate["brier"] - baseline["brier"]),
                }
                margins[reference] = margin
                declared = record["point_eligibility"][task]["references"][reference]
                if bool(declared["point_eligible"]) != all(
                        value >= 0 for value in margin.values()):
                    raise ValueError("gate point eligibility differs from task margins")
            rows[task].append({
                "mask_id": mask.mask_id,
                "mask_sha256": mask.mask_sha256,
                "head_id": head_ids[mask.mask_id][task],
                "metrics": candidate,
                "utility_margins_by_ref_metric": margins,
                "utility_feasible": all(
                    value >= 0 for reference in REFERENCES
                    for value in margins[reference].values()),
            })
    content = {
        "schema": "pcrl-focused-task-table-v1",
        "bank_sha256": bank.bank_sha256,
        "split_receipt": {
            "schema": "pcrl-utility-common-task-table-input-v1",
            "seed": seed, "role": "INNER_SELECTION",
            "common_row_unit_pair_sha256": common_row_sha256,
            "mask_gate_receipt_sha256_by_mask": gate_receipt_shas,
            "mask_lock_sha256_by_id": mask_lock_sha256_by_id,
            "full_input_gate_sha256": full_input_gate_sha256,
            "selected_recipe": selected_recipe,
            "R1_gate_receipt_sha256": r1_gate_receipt_sha256,
        },
        "income_masks": rows["income"],
        "occupation_masks": rows["occupation_group"],
    }
    table = {**content, "task_table_sha256": _sha(content)}
    try:
        choice = select_task_only_pre_sensitive(table)
    except ValueError as exc:
        if not str(exc).startswith("no utility-feasible task-only "):
            raise
        choice = None
    return table, choice
