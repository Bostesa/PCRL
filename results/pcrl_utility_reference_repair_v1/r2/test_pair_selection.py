"""Data-free control-plan breadth and pre-sensitive T immutability."""

from __future__ import annotations

import pytest

from results.pcrl_utility_reference_repair_v1.r2.pair_selection import (
    build_matched_selection,
)
from results.pcrl_utility_reference_repair_v1.r2.task_table import (
    build_pre_sensitive_task_table,
)
from results.pcrl_utility_reference_repair_v1.r2.test_task_table import _fixture


def _inputs():
    bank, gates, heads, shas = _fixture()
    table, t = build_pre_sensitive_task_table(
        bank, 0, "e" * 64, gates, heads, shas,
        full_input_gate_sha256="a" * 64, selected_recipe="A",
        r1_gate_receipt_sha256="d" * 64,
        mask_lock_sha256_by_id={mask.mask_id: "b" * 64 for mask in bank.masks})
    references = {
        task: {reference: gates[bank.masks[0].mask_id]["metrics"][reference][task]
               for reference in ("U0", "Ucal", "F")}
        for task in ("income", "occupation_group")}
    local = {mask.mask_id: 0.70 + 0.001 * index
             for index, mask in enumerate(bank.masks)}
    unions = {pair.union_sha256: 0.74 for pair in bank.pairs}
    return bank, table, t, references, local, unions


def test_matched_plan_scores_all_pairs_and_keeps_t_frozen():
    bank, task_table, t, references, local, unions = _inputs()
    table, joint, plan = build_matched_selection(
        bank, task_table, t, references, local, unions)
    assert len(table.pairs) == 25
    assert len({pair.union_sha256 for pair in table.pairs}) == 6
    assert plan["task_only_pair_id"] == t["pair_id"]
    assert joint.t_pair_id == t["pair_id"]
    assert set(plan["rules"]) == {
        "J", "T", "I", "S12", "S21", "SHARED",
        "T_U", "I_U", "S12_U", "S21_U"}


def test_pair_selection_refuses_changed_task_only_identity():
    bank, task_table, t, references, local, unions = _inputs()
    t = dict(t)
    t["pair_id"] = "changed"
    with pytest.raises(ValueError, match="task-only T"):
        build_matched_selection(bank, task_table, t, references, local, unions)
