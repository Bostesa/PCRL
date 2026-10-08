"""Data-free behavior checks for M3's shared-bank control selectors."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace


M2 = Path(__file__).resolve().parents[1] / "m2"
sys.path.insert(0, str(M2))

from table import MaskScore, PairRow, PairTable  # noqa: E402


def _mask(mask_id: str, task: str, nll: float, auc: float) -> MaskScore:
    return MaskScore(
        mask_id=mask_id,
        mask_sha256={"A": "a" * 64, "B": "b" * 64, "C": "c" * 64}[mask_id],
        recipient_capacity_signature=(4, 4, 77),
        head_id=f"{task}:{mask_id}",
        task_validation_accuracy=0.8,
        task_validation_nll=nll,
        task_validation_brier=0.2,
        utility_margins_by_ref_metric=tuple(
            (ref, (0.01, 0.01, 0.005)) for ref in ("U0", "Ucal", "F")
        ),
        local_search_auc=auc,
        utility_feasible=True,
    )


def _table() -> PairTable:
    income = (
        _mask("A", "income", 0.30, 0.80),
        _mask("B", "income", 0.32, 0.75),
        _mask("C", "income", 0.33, 0.78),
    )
    occupation = (
        _mask("A", "occupation", 0.31, 0.80),
        _mask("B", "occupation", 0.34, 0.79),
        _mask("C", "occupation", 0.35, 0.74),
    )
    pair_auc = {
        ("A", "A"): 0.88, ("A", "B"): 0.82, ("A", "C"): 0.82,
        ("B", "A"): 0.82, ("B", "B"): 0.69, ("B", "C"): 0.82,
        ("C", "A"): 0.82, ("C", "B"): 0.82, ("C", "C"): 0.68,
    }
    widths = {"A": 77, "B": 76, "C": 42}
    rows = []
    for first in income:
        for second in occupation:
            same = first.mask_id == second.mask_id
            rows.append(PairRow(
                pair_id=f"{first.mask_id}__{second.mask_id}",
                income_mask_id=first.mask_id,
                occupation_mask_id=second.mask_id,
                income_mask_sha256=first.mask_sha256,
                occupation_mask_sha256=second.mask_sha256,
                union_atoms=("atom",) if same else ("atom", "other"),
                union_columns=tuple(range(widths[first.mask_id] if same else 83)),
                union_group_count=8 if same else 9,
                union_encoded_columns=widths[first.mask_id] if same else 83,
                union_sha256=(first.mask_sha256 if same else "f" * 64),
                coalition_search_auc=pair_auc[(first.mask_id, second.mask_id)],
                income_local_search_auc=first.local_search_auc,
                occupation_local_search_auc=second.local_search_auc,
                task_loss=first.task_validation_nll + second.task_validation_nll,
                pair_feasible=True,
            ))
    return PairTable(income, occupation, tuple(rows), "1" * 64, "2" * 64,
                     "RAW_UNION_ONLY")


def test_t_uses_only_task_loss_even_if_privacy_scores_reverse():
    from control_rules import select_task_only

    table = _table()
    assert select_task_only(table).pair_id == "A__A"
    reversed_privacy = replace(
        table,
        income_masks=tuple(replace(mask, local_search_auc=1.0 - (mask.local_search_auc - 0.5))
                           for mask in table.income_masks),
        occupation_masks=tuple(replace(mask, local_search_auc=1.0 - (mask.local_search_auc - 0.5))
                               for mask in table.occupation_masks),
        pairs=tuple(replace(row, coalition_search_auc=1.0 - (row.coalition_search_auc - 0.5),
                            income_local_search_auc=1.0 - (row.income_local_search_auc - 0.5),
                            occupation_local_search_auc=1.0 - (row.occupation_local_search_auc - 0.5))
                    for row in table.pairs),
    )
    assert select_task_only(reversed_privacy).pair_id == "A__A"


def test_local_independent_and_both_sequential_orders_freeze_first_mask():
    from control_rules import select_independent, select_sequential, select_shared

    table = _table()
    t = table.pair("A__A")
    assert select_independent(table, t).pair_id == "B__C"
    assert select_sequential(table, t, first="income").pair_id == "B__B"
    assert select_sequential(table, t, first="occupation").pair_id == "C__C"
    assert select_shared(table, t).pair_id == "C__C"


def test_union_quota_matches_size_not_identity_and_keeps_exact_ties():
    from control_rules import (
        select_union_task_only, select_union_independent,
        select_sequential, select_shared,
    )

    table = _table()
    t = table.pair("A__A")
    distinct_quota = table.pair("B__C").union_size_signature
    assert select_union_task_only(table, t, distinct_quota).pair_id == "B__A"
    assert select_union_independent(table, t, distinct_quota).pair_id == "B__C"
    assert select_sequential(table, t, first="income", union_quota=distinct_quota).pair_id == "B__A"
    assert select_sequential(table, t, first="occupation", union_quota=distinct_quota).pair_id == "A__C"
    assert select_shared(table, t, union_quota=distinct_quota) is None
    diagonal_quota = table.pair("A__A").union_size_signature
    assert select_union_task_only(table, t, diagonal_quota).pair_id == "A__A"
    assert select_union_independent(table, t, diagonal_quota).pair_id == "A__A"
    assert select_sequential(table, t, first="income", union_quota=diagonal_quota).pair_id == "A__A"


def test_local_guard_is_inclusive_and_utility_infeasible_masks_are_removed():
    from control_rules import select_independent

    table = _table()
    t = table.pair("A__A")
    guarded = replace(
        table,
        income_masks=tuple(replace(m, local_search_auc=0.81) if m.mask_id == "B"
                           else replace(m, utility_feasible=False) if m.mask_id == "C" else m
                           for m in table.income_masks),
        pairs=tuple(replace(r, income_local_search_auc=0.81) if r.income_mask_id == "B"
                    else replace(r, pair_feasible=False) if r.income_mask_id in ("A", "C") else r
                    for r in table.pairs),
    )
    assert select_independent(guarded, t).income_mask_id == "B"
    barred = replace(
        guarded,
        income_masks=tuple(replace(m, local_search_auc=0.810001) if m.mask_id == "B" else m
                           for m in guarded.income_masks),
        pairs=tuple(replace(r, income_local_search_auc=0.810001) if r.income_mask_id == "B" else r
                    for r in guarded.pairs),
    )
    assert select_independent(barred, t) is None
    utility_barred = replace(
        table,
        income_masks=tuple(replace(m, utility_feasible=False) if m.mask_id == "B" else m
                           for m in table.income_masks),
        pairs=tuple(replace(r, pair_feasible=False) if r.income_mask_id == "B" else r
                    for r in table.pairs),
    )
    assert select_independent(utility_barred, t).income_mask_id == "C"


def test_complete_control_plan_retains_joint_mask_pair_ties_and_actual_widths():
    from control_rules import build_control_plan

    table = _table()
    widths = {"A": 77, "B": 76, "C": 42}
    bank = SimpleNamespace(
        bank_sha256=table.bank_sha256,
        mask=lambda mask_id: SimpleNamespace(
            source_group_count=8, encoded_columns=widths[mask_id],
            capacity_signature=(4, 4, 77),
        ),
    )
    plan = build_control_plan(table, bank, "C__C")
    assert set(plan["rules"]) == {
        "J", "T", "I", "S12", "S21", "SHARED", "T_U", "I_U", "S12_U", "S21_U",
    }
    assert plan["rules"]["S21"]["pair_id"] == "C__C"
    assert plan["rules"]["SHARED"]["pair_id"] == "C__C"
    assert plan["rules"]["T_U"]["pair_id"] == "C__C"
    assert plan["rules"]["S21"]["matches_joint_mask_pair"] is True
    assert plan["rules"]["S21"]["recipient_encoded_columns"] == {
        "income": 42, "occupation_group": 42,
    }
    assert plan["rules"]["T"]["recipient_encoded_columns"] == {
        "income": 77, "occupation_group": 77,
    }
    assert plan["rules"]["T_U"]["union_capacity_signature"] == {
        "source_groups": 8, "encoded_columns": 42,
    }
    assert "release_alias" not in plan["rules"]["S21"]
    assert len(plan["selection_receipt_sha256"]) == 64


def test_joint_outside_t_guard_is_refused_before_plan_building():
    from control_rules import build_control_plan

    table = _table()
    bank = SimpleNamespace(bank_sha256=table.bank_sha256)
    changed = replace(table, pairs=tuple(
        replace(row, occupation_local_search_auc=0.82) if row.pair_id == "C__C" else row
        for row in table.pairs
    ))
    try:
        build_control_plan(changed, bank, "C__C")
    except ValueError as error:
        assert "local guard" in str(error)
    else:
        raise AssertionError("an invalid J pair entered the control registry")


def test_plan_rejects_union_cache_with_unregistered_critic_inputs():
    from control_rules import build_control_plan

    table = replace(_table(), critic_input_kind="COMPLETE_HEAD_OUTPUTS")
    bank = SimpleNamespace(bank_sha256=table.bank_sha256)
    try:
        build_control_plan(table, bank, "C__C")
    except ValueError as error:
        assert "critic input" in str(error)
    else:
        raise AssertionError("a head-dependent critic was cached by raw union")


def _endpoint(selection, auc: float, *, seed: int, rule_id: str,
              eligible: bool = True) -> dict:
    return {
        "seed": seed, "rule_id": rule_id,
        "pair_id": selection["pair_id"],
        "income_mask_id": selection["income_mask_id"],
        "occupation_mask_id": selection["occupation_mask_id"],
        "union_sha256": selection["union_sha256"],
        "recipient_capacity_signatures": selection["recipient_capacity_signatures"],
        "union_capacity_signature": selection["union_capacity_signature"],
        "task": {purpose: {"references": {
            ref: {"point_eligible": eligible} for ref in ("U0", "Ucal", "F")
        }} for purpose in ("income", "occupation_group")},
        "recovery": {"coalition": {"primary_envelope_auc": auc}},
    }


def test_strongest_registry_keeps_alias_and_uses_distinct_main_and_union_pools():
    from control_rules import build_control_plan, strongest_controls

    table = _table()
    widths = {"A": 77, "B": 76, "C": 42}
    bank = SimpleNamespace(
        bank_sha256=table.bank_sha256,
        mask=lambda mask_id: SimpleNamespace(
            source_group_count=8, encoded_columns=widths[mask_id],
            capacity_signature=(4, 4, 77),
        ),
    )
    plans = {seed: build_control_plan(table, bank, "B__C") for seed in (0, 1)}
    # Same fitted pair means the same final recovery for J, I and I_U.
    by_pair = {
        "A__A": 0.60, "B__C": 0.70, "B__A": 0.71,
        "A__C": 0.72, "B__B": 0.75, "C__C": 0.74,
    }
    endpoints = {}
    for seed, plan in plans.items():
        endpoints[seed] = {}
        for rule_id, selection in plan["rules"].items():
            if selection["status"] != "SELECTED":
                continue
            endpoints[seed][rule_id] = _endpoint(
                selection, by_pair[selection["pair_id"]], seed=seed, rule_id=rule_id,
            )
    selected = strongest_controls(plans, endpoints)
    assert selected["main"]["rule_id"] == "T"
    assert selected["attribution"]["rule_id"] == "I_U"
    assert selected["attribution"]["pair_ids"] == ["B__C", "B__C"]
    assert selected["attribution"]["ties_joint_mask_pair"] is True
    assert "T" not in selected["attribution"]["eligible_rule_ids"]


def test_strongest_registry_requires_all_seed_utility_and_fails_closed_on_missing_endpoint():
    from control_rules import build_control_plan, strongest_controls

    table = _table()
    widths = {"A": 77, "B": 76, "C": 42}
    bank = SimpleNamespace(
        bank_sha256=table.bank_sha256,
        mask=lambda mask_id: SimpleNamespace(
            source_group_count=8, encoded_columns=widths[mask_id],
            capacity_signature=(4, 4, 77),
        ),
    )
    plans = {seed: build_control_plan(table, bank, "B__C") for seed in (0, 1)}
    endpoints = {}
    for seed, plan in plans.items():
        endpoints[seed] = {}
        for rule_id, selection in plan["rules"].items():
            if selection["status"] == "SELECTED":
                endpoints[seed][rule_id] = _endpoint(
                    selection, 0.70 if rule_id in ("I", "I_U") else 0.75,
                    seed=seed, rule_id=rule_id,
                    eligible=not (seed == 1 and rule_id in ("I", "I_U")),
                )
    selected = strongest_controls(plans, endpoints)
    assert "I" not in selected["main"]["eligible_rule_ids"]
    assert "I_U" not in selected["attribution"]["eligible_rule_ids"]
    del endpoints[1]["S12"]
    try:
        strongest_controls(plans, endpoints)
    except ValueError as error:
        assert "missing endpoint" in str(error)
    else:
        raise AssertionError("a missing scored arm silently left the control pool")


def test_strongest_registry_rejects_a_scored_view_with_wrong_union_identity():
    from control_rules import build_control_plan, strongest_controls

    table = _table()
    widths = {"A": 77, "B": 76, "C": 42}
    bank = SimpleNamespace(
        bank_sha256=table.bank_sha256,
        mask=lambda mask_id: SimpleNamespace(
            source_group_count=8, encoded_columns=widths[mask_id],
            capacity_signature=(4, 4, 77),
        ),
    )
    plan = build_control_plan(table, bank, "B__C")
    endpoints = {rule_id: _endpoint(selection, 0.75, seed=0, rule_id=rule_id)
                 for rule_id, selection in plan["rules"].items()
                 if selection["status"] == "SELECTED"}
    endpoints["I"]["union_sha256"] = "0" * 64
    try:
        strongest_controls({0: plan}, {0: endpoints})
    except ValueError as error:
        assert "union" in str(error)
    else:
        raise AssertionError("misbound final view entered the strongest-control registry")
