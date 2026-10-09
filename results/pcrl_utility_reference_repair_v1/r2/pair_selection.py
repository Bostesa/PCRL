"""Frozen J/T/I/sequential/shared controls from one five-mask score table.

The task-only table and T receipt exist before this module may receive FIT
SEX-derived search AUCs. This module opens no role data and fits no reader.
"""

from __future__ import annotations

import sys
from pathlib import Path

from results.pcrl_capacity_matched_feature_selection_v1.m3.control_rules import (
    build_control_plan, select_task_only,
)
from results.pcrl_capacity_matched_feature_selection_v1.m3.task_only import (
    select_task_only_pre_sensitive,
)


REPO = Path(__file__).resolve().parents[3]
M2_DIR = REPO / "results/pcrl_capacity_matched_feature_selection_v1/m2"
if str(M2_DIR) not in sys.path:
    sys.path.insert(0, str(M2_DIR))
from joint import select_joint  # noqa: E402
from table import build_pair_table, validate_pair_table  # noqa: E402


TASKS = ("income", "occupation_group")
REFERENCES = ("U0", "Ucal", "F")


def build_matched_selection(bank, task_table: dict, t_receipt: dict,
                            reference_metrics: dict,
                            local_search_auc_by_mask: dict,
                            coalition_search_auc_by_union: dict):
    """Run the source's strict selectors on one task/critic bank."""
    source_t = select_task_only_pre_sensitive(task_table)
    if (t_receipt != source_t or source_t["bank_sha256"] != bank.bank_sha256
            or set(local_search_auc_by_mask) !=
            {mask.mask_id for mask in bank.masks}
            or set(coalition_search_auc_by_union) !=
            {pair.union_sha256 for pair in bank.pairs}
            or set(reference_metrics) != set(TASKS)):
        raise ValueError("task-only T or raw search coverage differs")
    evaluation = {}
    references = {}
    for task, rows_key in (("income", "income_masks"),
                           ("occupation_group", "occupation_masks")):
        rows = task_table[rows_key]
        if {row["mask_id"] for row in rows} != {
                mask.mask_id for mask in bank.masks}:
            raise ValueError("task-only mask rows are incomplete")
        evaluation[task] = {
            row["mask_id"]: {
                "head_id": row["head_id"],
                "metrics": row["metrics"],
                "local_search_auc": local_search_auc_by_mask[row["mask_id"]],
            } for row in rows
        }
        if set(reference_metrics[task]) != set(REFERENCES):
            raise ValueError("task reference metric coverage differs")
        references[task] = {
            reference: {key: float(reference_metrics[task][reference][key])
                        for key in ("accuracy", "nll", "brier")}
            for reference in REFERENCES
        }
    table = build_pair_table(
        bank, evaluation, references, coalition_search_auc_by_union,
        critic_input_kind="RAW_UNION_ONLY")
    validate_pair_table(table, bank)
    if select_task_only(table).pair_id != source_t["pair_id"]:
        raise ValueError("privacy score contaminated frozen task-only T")
    for task, source_rows in (("income", table.income_masks),
                              ("occupation_group", table.occupation_masks)):
        declared = {row["mask_id"]: row["utility_feasible"]
                    for row in task_table[
                        "income_masks" if task == "income" else
                        "occupation_masks"]}
        if any(row.utility_feasible != declared[row.mask_id]
               for row in source_rows):
            raise ValueError("shared pair-table feasibility differs from task gate")
    joint = select_joint(table, t_pair_id=source_t["pair_id"])
    plan = build_control_plan(table, bank, joint.pair_id)
    if (plan["task_only_pair_id"] != source_t["pair_id"]
            or plan["joint_pair_id"] != joint.pair_id):
        raise ValueError("matched-control plan changed source selections")
    return table, joint, plan
