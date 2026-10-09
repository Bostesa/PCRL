"""Data-free tests for independent pre-sensitive no-go replay."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from results.pcrl_capacity_matched_feature_selection_v1.m5.replay_task_no_go import (
    _historical_fit_receipt, _sha, audit_no_go_table, digest_json,
)
from results.pcrl_capacity_matched_feature_selection_v1.m5 import replay_task_no_go


RULES = ("J", "T", "I", "S12", "S21", "SHARED", "T_U", "I_U", "S12_U", "S21_U")


def _fixture():
    masks = tuple(SimpleNamespace(mask_id=f"M_{i}", mask_sha256=f"{i + 1:064x}")
                  for i in range(5))
    bank = SimpleNamespace(bank_sha256="a" * 64, masks=masks)
    observed = {purpose: {mask.mask_id: {"accuracy": 0.70, "nll": 0.80,
                                         "brier": 0.40} for mask in masks}
                for purpose in ("income", "occupation_group")}
    refs = {purpose: {ref: {"accuracy": 0.90, "nll": 0.30, "brier": 0.20}
                      for ref in ("U0", "Ucal", "F")}
            for purpose in observed}
    def row(mask):
        return {"mask_id": mask.mask_id, "mask_sha256": mask.mask_sha256,
                "head_id": f"head:{mask.mask_id}",
                "metrics": observed["income"][mask.mask_id],
                "utility_margins_by_ref_metric": {ref: {
                    "accuracy": -0.19, "nll": -0.49, "brier": -0.195,
                } for ref in ("U0", "Ucal", "F")},
                "utility_feasible": False}
    table = {"schema": "pcrl-focused-task-table-v1", "bank_sha256": "a" * 64,
             "split_receipt": {"seed": 0},
             "income_masks": [row(mask) for mask in masks],
             "occupation_masks": [row(mask) for mask in masks]}
    table["task_table_sha256"] = digest_json(table)
    selection = {"status": "NO_UTILITY_FEASIBLE_T", "pair_id": None,
                 "reason": "no utility-feasible task-only income mask",
                 "task_table_sha256": table["task_table_sha256"],
                 "bank_sha256": bank.bank_sha256}
    task_receipt = {"selection": selection, "bank_sha256": bank.bank_sha256,
                    "task_table_sha256": "placeholder"}
    plan = {"schema": "pcrl-focused-m3-control-plan-v1",
            "status": "NO_FEASIBLE_J", "reason": "NO_UTILITY_FEASIBLE_TASK_MASK",
            "bank_sha256": bank.bank_sha256,
            "task_table_sha256": table["task_table_sha256"],
            "joint_pair_id": None, "task_only_pair_id": None,
            "table_sha256": None,
            "joint_union_quota": None,
            "release_alias_status": "NOT_APPLICABLE_NO_SELECTED_RELEASE",
            "dropped_invalid_rules": list(RULES),
            "main_control_order": list(RULES[1:]),
            "attribution_control_order": list(RULES[6:]) + list(RULES[1:6]),
            "rules": {rule: {"status": "INVALID_NO_UTILITY_FEASIBLE_PAIR",
                             "reason": "NO_UTILITY_FEASIBLE_TASK_MASK",
                             "pair_id": None} for rule in RULES}}
    plan["selection_receipt_sha256"] = digest_json(plan)
    return bank, table, task_receipt, plan, observed, refs


def test_audit_no_go_table_accepts_complete_infeasible_five_mask_bank():
    bank, table, task_receipt, plan, observed, refs = _fixture()
    result = audit_no_go_table(bank, table, task_receipt, plan, observed, refs)
    assert result["status"] == "INDEPENDENT_TASK_NO_GO_PASS"
    assert result["feasible_mask_count_by_task"] == {"income": 0,
                                                      "occupation_group": 0}
    assert result["mask_row_count"] == 10


def test_audit_no_go_table_rejects_favorable_margin_or_valid_rule():
    bank, table, task_receipt, plan, observed, refs = _fixture()
    table["income_masks"][0]["utility_margins_by_ref_metric"]["U0"]["nll"] = 0.0
    table["task_table_sha256"] = digest_json({k: v for k, v in table.items()
                                               if k != "task_table_sha256"})
    with pytest.raises(ValueError, match="margin"):
        audit_no_go_table(bank, table, task_receipt, plan, observed, refs)
    bank, table, task_receipt, plan, observed, refs = _fixture()
    plan["rules"]["J"]["status"] = "SELECTED"
    plan["selection_receipt_sha256"] = digest_json({k: v for k, v in plan.items()
                                                      if k != "selection_receipt_sha256"})
    with pytest.raises(ValueError, match="invalid"):
        audit_no_go_table(bank, table, task_receipt, plan, observed, refs)


def test_historical_fit_receipt_requires_archived_done_spec_and_seal(tmp_path, monkeypatch):
    study = tmp_path / "study"
    archive = study / ".private/queue_science_no_go"
    spec = archive / "specs/M2-FIT-S0.json"
    fit = study / ".private/m2/fit/seed_0/FIT_RECEIPT.json"
    lock = study / "amendments/CAMPAIGN_LOCK_SCIENCE_NOGO.json"
    seal = study / "amendments/LOCAL_SEAL_RECEIPT_SCIENCE_NOGO.json"
    for path in (spec, fit, lock, seal):
        path.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("{}\n")
    seal.write_text(json.dumps({"status": "PASS", "lock_sha256": _sha(lock)}))
    fit.write_text('{"status":"NO_FEASIBLE_J"}\n')
    spec.write_text(json.dumps({"unit_id": "M2-FIT-S0", "stage": "fit_bank",
                                "seal_sha256": _sha(seal)}))
    queue_path = archive / "QUEUE.json"
    queue_path.write_text(json.dumps({"units": {"M2-FIT-S0": {
        "status": "DONE", "exit_code": 0, "spec_sha256": _sha(spec),
        "output_sha256": {str(fit): _sha(fit)}}}}))
    source = {"archived_queue_relative_path": ".private/queue_science_no_go/QUEUE.json",
              "archived_queue_sha256": _sha(queue_path),
              "archived_lock_relative_path": "amendments/CAMPAIGN_LOCK_SCIENCE_NOGO.json",
              "archived_seal_relative_path": "amendments/LOCAL_SEAL_RECEIPT_SCIENCE_NOGO.json",
              "lock_sha256": _sha(lock), "local_seal_sha256": _sha(seal),
              "by_seed": {"0": {"spec_sha256": _sha(spec),
                                "fit_receipt_sha256": _sha(fit)}}}
    monkeypatch.setattr(replay_task_no_go, "STUDY", study)
    assert _historical_fit_receipt(source, 0) == _sha(fit)
    spec.write_text(spec.read_text() + "\n")
    with pytest.raises(PermissionError, match="spec changed"):
        _historical_fit_receipt(source, 0)
