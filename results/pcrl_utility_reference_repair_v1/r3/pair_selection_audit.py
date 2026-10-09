"""Independent JSON-only replay of the frozen five-mask pair selection.

This module imports no candidate, source selector, runner, or scoring code.
It never opens a role export or person-level label/prediction file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MASKS = ("M_DROP_NATIVE_COUNTRY", "M_DROP_MARITAL_STATUS",
         "M_DROP_EDUCATION_FACT", "M_DROP_RELATIONSHIP", "M_DROP_WORKCLASS")
RULES = ("J", "T", "I", "S12", "S21", "SHARED", "T_U", "I_U",
         "S12_U", "S21_U")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _same_margins(first: dict, second: dict) -> bool:
    return (set(first) == set(second) == {"U0", "Ucal", "F"} and
            all(set(first[reference]) == set(second[reference]) ==
                {"accuracy", "nll", "brier"} and
                all(abs(first[reference][metric] - second[reference][metric])
                    <= 1e-12 for metric in ("accuracy", "nll", "brier"))
                for reference in ("U0", "Ucal", "F")))


def _local_key(row: dict) -> tuple:
    return row["local_search_auc"], row["task_validation_nll"], row["mask_sha256"]


def _pair_key(row: dict) -> tuple:
    return row["task_loss"], row["income_mask_sha256"], row["occupation_mask_sha256"]


def _select_controls(table: dict, t_id: str) -> tuple[dict, dict]:
    rows = {row["pair_id"]: row for row in table["pairs"]}
    left = {row["mask_id"]: row for row in table["income_masks"]}
    right = {row["mask_id"]: row for row in table["occupation_masks"]}
    t = rows[t_id]
    legal = [row for row in rows.values() if
             row["pair_feasible"] and
             row["income_local_search_auc"] <= t["income_local_search_auc"] + .010 and
             row["occupation_local_search_auc"] <= t["occupation_local_search_auc"] + .010]
    _require(bool(legal), "empty legal pair bank")
    j = min(legal, key=lambda row: (row["coalition_search_auc"],
                                   *_pair_key(row), row["pair_id"]))
    quota = (j["union_group_count"], j["union_encoded_columns"])

    def local_first(pool: list[dict], purpose: str) -> dict:
        if purpose == "income":
            allowed = {row["income_mask_id"] for row in pool}
            chosen = min((left[name] for name in allowed), key=_local_key)["mask_id"]
            return min((row for row in pool if row["income_mask_id"] == chosen),
                       key=lambda row: (row["coalition_search_auc"], *_pair_key(row)))
        allowed = {row["occupation_mask_id"] for row in pool}
        chosen = min((right[name] for name in allowed), key=_local_key)["mask_id"]
        return min((row for row in pool if row["occupation_mask_id"] == chosen),
                   key=lambda row: (row["coalition_search_auc"], *_pair_key(row)))

    i_left = min((left[name] for name in {row["income_mask_id"] for row in legal}),
                 key=_local_key)["mask_id"]
    i_right = min((right[name] for name in {row["occupation_mask_id"] for row in legal}),
                  key=_local_key)["mask_id"]
    matched = [row for row in legal if
               (row["union_group_count"], row["union_encoded_columns"]) == quota]
    chosen = {
        "J": j, "T": t, "I": rows[f"{i_left}__{i_right}"],
        "S12": local_first(legal, "income"),
        "S21": local_first(legal, "occupation"),
        "SHARED": min((row for row in legal if row["income_mask_id"] ==
                       row["occupation_mask_id"]),
                      key=lambda row: (row["coalition_search_auc"], *_pair_key(row)),
                      default=None),
        "T_U": min(matched, key=_pair_key, default=None),
        "I_U": min(matched, key=lambda row: (
            row["income_local_search_auc"] + row["occupation_local_search_auc"],
            *_pair_key(row)), default=None),
        "S12_U": local_first(matched, "income") if matched else None,
        "S21_U": local_first(matched, "occupation") if matched else None,
    }
    return chosen, {"utility_feasible": sum(row["pair_feasible"] for row in rows.values()),
                    "legal": len(legal), "quota": quota}


def audit_seed(study: Path, bank: dict, audit: dict, seed: int,
               pair_lock_sha: str) -> dict:
    private = study / ".private"
    key = str(seed)
    folder = private / "pair_select" / f"seed{seed}"
    receipt_path = folder / "PAIR_SELECT_RECEIPT.json"
    receipt = _read(receipt_path)
    table = _read(folder / "PAIR_TABLE.json")
    joint = _read(folder / "JOINT_SELECTION.json")
    control = _read(folder / "CONTROL_PLAN.json")
    search = _read(folder / "RAW_SEARCH_RECEIPT.json")
    task_folder = private / "task_table" / f"seed{seed}"
    task_table = _read(task_folder / "TASK_ONLY_TABLE.json")
    t_selection = _read(task_folder / "T_SELECTION.json")
    task_receipt_path = task_folder / "TASK_TABLE_RECEIPT.json"
    _require(_sha(task_receipt_path) == audit["by_seed"][key]["receipt_sha256"],
             "task-only receipt changed")
    _require(_sha(task_folder / "TASK_ONLY_TABLE.json") ==
             audit["by_seed"][key]["table_sha256"], "task-only table changed")
    _require(_sha(task_folder / "T_SELECTION.json") ==
             audit["by_seed"][key]["T_selection_sha256"], "T selection changed")
    _require(receipt["schema"] == "pcrl-utility-pair-selection-receipt-v1" and
             receipt["seed"] == seed and
             receipt["conditional_lock_sha256"] == pair_lock_sha and
             receipt["task_table_receipt_sha256"] == _sha(task_receipt_path) and
             receipt["task_only_T_pair_id"] == t_selection["pair_id"] and
             receipt["fitted_raw_search_readers"] == 22 and
             receipt["sex_roles_read"] == ["OSF_DEFENSE_FIT"] and
             receipt["no_ATT_or_INNER_SEX_read"] is True,
             "pair-selection receipt lineage/count differs")
    for name in ("PAIR_TABLE.json", "JOINT_SELECTION.json",
                 "CONTROL_PLAN.json", "RAW_SEARCH_RECEIPT.json"):
        _require(_sha(folder / name) == receipt["files_sha256"][name],
                 f"pair-selection output changed: {name}")
    _require(table["schema"] == "pcrl-focused-pair-table-v1" and
             table["bank_sha256"] == task_table["bank_sha256"] and
             table["critic_input_kind"] == "RAW_UNION_ONLY" and
             len(table["pairs"]) == 25 and
             search["fitted_reader_count"] == 22 and
             search["local_feature_sets"] == 5 and
             search["unique_union_feature_sets"] == 6 and
             search["reader_ids"] == ["LR_C1", "HGB_15x64"] and
             search["task_table_receipt_sha256"] == _sha(task_receipt_path),
             "pair table or raw search geometry differs")
    canonical = {name: table[name] for name in (
        "critic_input_kind", "bank_sha256", "income_masks", "occupation_masks", "pairs")}
    _require(_content_sha(canonical) == table["table_sha256"],
             "pair-table content digest differs")
    masks = {row["mask_id"]: row for row in bank["masks"]}
    bank_pairs = {row["pair_id"]: row for row in bank["pairs"]}
    _require(set(masks) == set(MASKS) and len(bank_pairs) == 25,
             "bank masks/pairs differ")
    locals_by_task = {}
    for task, key_name in (("income", "income_masks"),
                           ("occupation_group", "occupation_masks")):
        by_mask = {row["mask_id"]: row for row in table[key_name]}
        task_by_mask = {row["mask_id"]: row for row in task_table[key_name]}
        _require(set(by_mask) == set(MASKS), f"{task} mask coverage differs")
        locals_by_task[task] = by_mask
        for mask_id, row in by_mask.items():
            original = task_by_mask[mask_id]
            local = search["local"][mask_id]
            _require(row["mask_sha256"] == masks[mask_id]["mask_sha256"] and
                     row["head_id"] == original["head_id"] and
                     row["task_validation_accuracy"] == original["metrics"]["accuracy"] and
                     row["task_validation_nll"] == original["metrics"]["nll"] and
                     row["task_validation_brier"] == original["metrics"]["brier"] and
                     row["utility_feasible"] == original["utility_feasible"] and
                     _same_margins(row["utility_margins_by_ref_metric"],
                                   original["utility_margins_by_ref_metric"]) and
                     row["local_search_auc"] == local["selected_auc"] and
                     local["fit_count"] == 2 and
                     set(local["readers"]) == {"LR_C1", "HGB_15x64"},
                     f"{task} mask score differs: {mask_id}")
    rows = {row["pair_id"]: row for row in table["pairs"]}
    _require(set(rows) == set(bank_pairs), "pair bank incomplete")
    for pair_id, row in rows.items():
        first = locals_by_task["income"][row["income_mask_id"]]
        second = locals_by_task["occupation_group"][row["occupation_mask_id"]]
        bank_row = bank_pairs[pair_id]
        _require(row["income_mask_sha256"] == first["mask_sha256"] and
                 row["occupation_mask_sha256"] == second["mask_sha256"] and
                 row["income_local_search_auc"] == first["local_search_auc"] and
                 row["occupation_local_search_auc"] == second["local_search_auc"] and
                 row["coalition_search_auc"] ==
                 search["union"][row["union_sha256"]]["selected_auc"] and
                 row["task_loss"] == first["task_validation_nll"] +
                 second["task_validation_nll"] and
                 row["pair_feasible"] ==
                 (first["utility_feasible"] and second["utility_feasible"]) and
                 row["union_sha256"] == bank_row["union_sha256"] and
                 row["union_columns"] == bank_row["union_column_indices"] and
                 row["union_group_count"] == bank_row["union_group_count"] and
                 row["union_encoded_columns"] == bank_row["union_encoded_columns"],
                 f"pair score or geometry differs: {pair_id}")
    _require(t_selection["pair_id"] == audit["by_seed"][key]["T_pair_id"],
             "pre-sensitive T changed")
    chosen, counts = _select_controls(table, t_selection["pair_id"])
    _require(joint["pair_id"] == chosen["J"]["pair_id"] and
             joint["t_pair_id"] == chosen["T"]["pair_id"] and
             joint["selected_row"] == chosen["J"] and
             joint["evaluated_pair_count"] == 25 and
             joint["utility_feasible_pair_count"] == counts["utility_feasible"] and
             joint["feasible_pair_count"] == counts["legal"] and
             joint["alias_of_t"] == (chosen["J"]["pair_id"] ==
                                     chosen["T"]["pair_id"]) and
             receipt["joint_pair_id"] == joint["pair_id"],
             "joint selection replay differs")
    _require(control["schema"] == "pcrl-focused-m3-control-plan-v1" and
             control["joint_pair_id"] == joint["pair_id"] and
             control["task_only_pair_id"] == t_selection["pair_id"] and
             control["table_sha256"] == table["table_sha256"] and
             set(control["rules"]) == set(RULES),
             "matched control plan lineage differs")
    selected = {}
    for name, row in chosen.items():
        item = control["rules"][name]
        pair_id = None if row is None else row["pair_id"]
        _require(item["pair_id"] == pair_id and
                 item["status"] == ("SELECTED" if row else
                                     "INVALID_NO_ELIGIBLE_PAIR"),
                 f"matched rule replay differs: {name}")
        selected[name] = pair_id
    _require(receipt["selected_rule_count"] == sum(
        pair_id is not None for pair_id in selected.values()),
        "selected-rule count differs")
    source_plan = dict(control)
    source_plan.pop("selection_receipt_sha256")
    _require(_content_sha(source_plan) == control["selection_receipt_sha256"],
             "control-plan content digest differs")
    return {"pair_receipt_sha256": _sha(receipt_path),
            "pair_table_sha256": _sha(folder / "PAIR_TABLE.json"),
            "control_plan_sha256": _sha(folder / "CONTROL_PLAN.json"),
            "joint_selection_sha256": _sha(folder / "JOINT_SELECTION.json"),
            "raw_search_receipt_sha256": _sha(folder / "RAW_SEARCH_RECEIPT.json"),
            "task_only_T_pair_id": t_selection["pair_id"],
            "selected_rule_pair_ids": selected,
            "distinct_selected_pairs": len({value for value in selected.values()
                                             if value is not None}),
            "raw_search_reader_fits": search["fitted_reader_count"],
            "joint_eligible_pair_count": counts["legal"]}


def audit(study: Path, bank_path: Path, task_audit_path: Path,
          pair_lock_path: Path) -> dict:
    bank, task_audit, lock = (_read(bank_path), _read(task_audit_path),
                              _read(pair_lock_path))
    queue_path = study / ".private/queue_g13/QUEUE.json"
    queue = _read(queue_path)
    _require(task_audit["status"] == "PASS_AGGREGATE_SELECTION_REPLAY" and
             task_audit["mask_bank_sha256"] == _sha(bank_path) and
             lock["stage"] == "fixed_five_mask_pair_search" and
             lock["task_table_selection_audit_sha256"] == _sha(task_audit_path) and
             lock["mask_bank_sha256"] == _sha(bank_path) and
             set(queue["units"]) == {"R2_PAIR_SELECT_S0", "R2_PAIR_SELECT_S1"} and
             all(item["status"] == "DONE" for item in queue["units"].values()),
             "frozen task/bank/selection lock differs")
    lock_sha = _sha(pair_lock_path)
    by_seed = {str(seed): audit_seed(study, bank, task_audit, seed, lock_sha)
               for seed in (0, 1)}
    return {"schema": "pcrl-utility-r3-pair-selection-audit-v1",
            "status": "PASS_AGGREGATE_SELECTION_REPLAY",
            "scope": "sealed JSON aggregates only; no role arrays or SEX labels",
            "development_only": True,
            "pair_search_lock_sha256": lock_sha,
            "pair_search_queue_state_sha256": _sha(queue_path),
            "task_table_selection_audit_sha256": _sha(task_audit_path),
            "mask_bank_sha256": _sha(bank_path),
            "by_seed": by_seed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--task-audit", required=True, type=Path)
    parser.add_argument("--pair-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.study, args.bank, args.task_audit, args.pair_lock)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
