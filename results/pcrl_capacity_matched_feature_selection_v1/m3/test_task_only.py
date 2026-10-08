"""Synthetic pre-sensitive-data selection boundary for T."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace


def _sha(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _mask(mask_id, nll, feasible=True):
    return {
        "mask_id": mask_id, "mask_sha256": {"A": "a" * 64, "B": "b" * 64}[mask_id],
        "head_id": f"head:{mask_id}",
        "metrics": {"accuracy": 0.8, "nll": nll, "brier": 0.2},
        "utility_margins_by_ref_metric": {
            ref: {"accuracy": 0.01 if feasible else -0.001,
                  "nll": 0.01, "brier": 0.005}
            for ref in ("U0", "Ucal", "F")
        },
        "utility_feasible": feasible,
    }


def _table():
    payload = {
        "schema": "pcrl-focused-task-table-v1", "bank_sha256": "1" * 64,
        "split_receipt": {"role": "OSF_DEFENSE_FIT", "fold_sha256": "2" * 64},
        "income_masks": [_mask("B", 0.31), _mask("A", 0.30)],
        "occupation_masks": [_mask("A", 0.40), _mask("B", 0.40)],
    }
    return {**payload, "task_table_sha256": _sha(payload)}


def test_pre_sensitive_t_uses_only_task_table_and_ties_by_mask_sha():
    from task_only import select_task_only_pre_sensitive

    selected = select_task_only_pre_sensitive(_table())
    assert selected["pair_id"] == "A__A"
    assert selected["income_mask_id"] == "A"
    assert selected["occupation_mask_id"] == "A"
    assert len(selected["selection_receipt_sha256"]) == 64


def test_pre_sensitive_t_rejects_critic_field_and_tampered_table_hash():
    from task_only import select_task_only_pre_sensitive

    with_critic = _table()
    with_critic["income_masks"][0]["local_search_auc"] = 0.5
    with_critic["task_table_sha256"] = _sha({key: value for key, value in with_critic.items()
                                             if key != "task_table_sha256"})
    try:
        select_task_only_pre_sensitive(with_critic)
    except ValueError as error:
        assert "task-only" in str(error)
    else:
        raise AssertionError("a search critic entered pre-SEX T selection")
    tampered = _table()
    tampered["income_masks"][0]["metrics"]["nll"] = 0.01
    try:
        select_task_only_pre_sensitive(tampered)
    except ValueError as error:
        assert "SHA" in str(error)
    else:
        raise AssertionError("changed task loss escaped table custody")


def test_pre_sensitive_t_skips_ineligible_mask_and_refuses_empty_task():
    from task_only import select_task_only_pre_sensitive

    table = _table()
    table["income_masks"][1] = _mask("A", 0.30, feasible=False)
    table["task_table_sha256"] = _sha({key: value for key, value in table.items()
                                      if key != "task_table_sha256"})
    assert select_task_only_pre_sensitive(table)["income_mask_id"] == "B"
    table["income_masks"][0] = _mask("B", 0.31, feasible=False)
    table["task_table_sha256"] = _sha({key: value for key, value in table.items()
                                      if key != "task_table_sha256"})
    try:
        select_task_only_pre_sensitive(table)
    except ValueError as error:
        assert "no utility-feasible" in str(error)
    else:
        raise AssertionError("T selected an ineligible task mask")


def test_pre_sensitive_no_go_has_all_rule_slots_without_critic_or_sex_input():
    from task_only import build_pre_sensitive_no_go_control_plan

    table = _table()
    table["income_masks"] = [_mask("A", 0.30, feasible=False),
                             _mask("B", 0.31, feasible=False)]
    table["task_table_sha256"] = _sha({key: value for key, value in table.items()
                                      if key != "task_table_sha256"})
    bank = SimpleNamespace(bank_sha256=table["bank_sha256"], masks=(
        SimpleNamespace(mask_id="A", mask_sha256="a" * 64),
        SimpleNamespace(mask_id="B", mask_sha256="b" * 64),
    ))
    plan = build_pre_sensitive_no_go_control_plan(
        table, bank, reason="NO_UTILITY_FEASIBLE_TASK_MASK")
    assert plan["status"] == "NO_FEASIBLE_J"
    assert plan["joint_pair_id"] is None and plan["task_only_pair_id"] is None
    assert plan["table_sha256"] is None
    assert plan["task_table_sha256"] == table["task_table_sha256"]
    assert len(plan["rules"]) == 10
    assert all(rule["status"] == "INVALID_NO_UTILITY_FEASIBLE_PAIR"
               for rule in plan["rules"].values())
    assert len(plan["selection_receipt_sha256"]) == 64
    try:
        build_pre_sensitive_no_go_control_plan(
            _table(), bank, reason="NO_UTILITY_FEASIBLE_TASK_MASK")
    except ValueError as error:
        assert "feasible" in str(error)
    else:
        raise AssertionError("a feasible T mask was treated as no-go")
