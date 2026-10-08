"""Data-free J/control-plan publication against the exact common table."""

import json

from test_bank_joint import _table, _records
from bank import build_shared_bank, _sha
from fit_worker import _publish_pair_selection, _publish_pre_sensitive_no_go


def test_pair_sink_writes_joint_and_all_ten_control_rules(tmp_path):
    bank = build_shared_bank(_records())
    table = _table()
    from results.pcrl_capacity_matched_feature_selection_v1.m3.control_rules import select_task_only
    t = select_task_only(table)
    out = _publish_pair_selection(tmp_path, bank, table, {"pair_id": t.pair_id})
    assert (tmp_path / "PAIR_TABLE.json").is_file()
    assert (tmp_path / "JOINT_SELECTION.json").is_file()
    assert (tmp_path / "CONTROL_PLAN.json").is_file()
    plan = json.loads((tmp_path / "CONTROL_PLAN.json").read_text())
    assert plan["task_only_pair_id"] == t.pair_id
    assert plan["joint_pair_id"] == out["joint_pair_id"]
    assert set(plan["rules"]) == {
        "J", "T", "I", "S12", "S21", "SHARED", "T_U", "I_U", "S12_U", "S21_U"
    }
    assert all(len(out[key]) == 64 for key in (
        "pair_table_sha256", "joint_selection_sha256", "control_plan_sha256"
    ))


def test_pre_sensitive_no_go_publishes_all_invalid_statuses_without_pair_table(tmp_path):
    bank = build_shared_bank(_records())
    rows = []
    for mask in bank.masks:
        rows.append({
            "mask_id": mask.mask_id, "mask_sha256": mask.mask_sha256,
            "head_id": f"head:{mask.mask_id}",
            "metrics": {"accuracy": 0.7, "nll": 0.7, "brier": 0.3},
            "utility_margins_by_ref_metric": {
                ref: {"accuracy": -0.1, "nll": -0.1, "brier": -0.1}
                for ref in ("U0", "Ucal", "F")},
            "utility_feasible": False,
        })
    content = {"schema": "pcrl-focused-task-table-v1",
               "bank_sha256": bank.bank_sha256,
               "split_receipt": {"fit_rows": 10},
               "income_masks": rows, "occupation_masks": rows}
    table = {**content, "task_table_sha256": _sha(content)}
    choice = {"status": "NO_UTILITY_FEASIBLE_T", "pair_id": None,
              "reason": "no utility-feasible task-only income mask",
              "task_table_sha256": table["task_table_sha256"],
              "bank_sha256": bank.bank_sha256}
    out = _publish_pre_sensitive_no_go(tmp_path, bank, table, choice)
    plan = json.loads((tmp_path / "CONTROL_PLAN.json").read_text())
    assert out["status"] == "NO_FEASIBLE_J"
    assert plan["status"] == "NO_FEASIBLE_J"
    assert plan["joint_pair_id"] is None
    assert not (tmp_path / "PAIR_TABLE.json").exists()
    assert set(plan["rules"]) == {
        "J", "T", "I", "S12", "S21", "SHARED", "T_U", "I_U", "S12_U", "S21_U"
    }
    assert all(rule["status"] == "INVALID_NO_UTILITY_FEASIBLE_PAIR"
               for rule in plan["rules"].values())
