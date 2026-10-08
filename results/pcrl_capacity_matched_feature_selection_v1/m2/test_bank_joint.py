"""Synthetic, data-free checks for the focused shared bank and J rule."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from bank import _sha, build_shared_bank
from table import TaskMetrics, build_pair_table, validate_pair_table
from joint import select_joint


ATOMS = ("age", "education_fact", "capital-gain", "capital-loss",
         "hours-per-week", "workclass", "marital-status", "relationship",
         "native-country")
CATEGORICAL = ("education_fact", "workclass", "marital-status",
               "relationship", "native-country")
WIDTHS = {"education_fact": 17, "workclass": 8, "marital-status": 7,
          "relationship": 6, "native-country": 41}


def _records():
    # Synthetic column sets preserve the real width geometry (83 total).
    columns = {}
    cursor = 0
    for atom in ATOMS:
        width = WIDTHS.get(atom, 1)
        columns[atom] = tuple(range(cursor, cursor + width))
        cursor += width
    assert cursor == 83
    result = []
    for dropped in CATEGORICAL:
        kept = tuple(atom for atom in ATOMS if atom != dropped)
        selected = tuple(i for atom in kept for i in columns[atom])
        identity = {
            "source_schema_sha256": "a" * 64,
            "atoms": kept, "column_indices": selected,
        }
        source_fields = set(kept) - {"education_fact"}
        if "education_fact" in kept:
            source_fields.update(("education", "education-num"))
        result.append({
            "mask_id": f"drop-{dropped}",
            "atoms": kept,
            "column_indices": selected,
            "numeric_group_count": 4,
            "categorical_group_count": 4,
            "source_group_count": 8,
            "encoded_columns": len(selected),
            "source_fields": sorted(source_fields),
            "source_schema_sha256": "a" * 64,
            "capacity_signature": {
                "numeric_groups": 4, "categorical_groups": 4,
                "encoded_columns_max": 77,
            },
            "mask_sha256": _sha(identity),
            "historical_exposure": {
                "fit_proxy_search": True,
                "final_screen_arms": (["C-SIMPLE-TASK-SHARED1"]
                                      if dropped == "native-country" else []),
            },
        })
    return result


def _evaluations(bank):
    # Same synthetic fitting estimates for both tasks; privacy scores are
    # deliberately arranged so utility-only and J choose different pairs.
    utility = {}
    for purpose in ("income", "occupation_group"):
        utility[purpose] = {}
        for i, mask in enumerate(bank.masks):
            utility[purpose][mask.mask_id] = {
                "head_id": f"{purpose}:{mask.mask_id}",
                "metrics": TaskMetrics(0.80, 0.50 + i * 0.0001, 0.20),
                "local_search_auc": 0.74 - i * 0.01,
            }
    references = {
        purpose: {name: TaskMetrics(0.80, 0.50, 0.20)
                  for name in ("U0", "Ucal", "F")}
        for purpose in ("income", "occupation_group")
    }
    return utility, references


def _table():
    bank = build_shared_bank(_records(), max_per_purpose=8)
    utility, refs = _evaluations(bank)
    # Five diagonal unions plus the full union for all distinct-drop pairs.
    union_auc = {}
    for pair in bank.pairs:
        union_auc[pair.union_sha256] = (
            0.65 if pair.income_mask_id == pair.occupation_mask_id
            else 0.90
        )
    return build_pair_table(bank, utility, refs, union_auc)


def test_bank_contains_all_five_masks_and_25_pairs_without_outcome_search():
    bank = build_shared_bank(_records(), max_per_purpose=8)
    assert len(bank.masks) == 5
    assert len(bank.pairs) == 25
    assert len({pair.union_sha256 for pair in bank.pairs}) == 6
    assert all(pair.union_group_count == 9 and pair.union_encoded_columns == 83
               for pair in bank.pairs
               if pair.income_mask_id != pair.occupation_mask_id)
    assert all(pair.union_group_count == 8
               for pair in bank.pairs
               if pair.income_mask_id == pair.occupation_mask_id)
    assert bank.bank_sha256 == build_shared_bank(
        tuple(reversed(_records())), max_per_purpose=8
    ).bank_sha256


def test_bank_rejects_missing_feasible_mask_and_false_width():
    records = _records()
    with pytest.raises(ValueError, match="five feasible"):
        build_shared_bank(records[:-1], max_per_purpose=8,
                          expected_feasible_count=5)
    records = copy.deepcopy(records)
    records[0]["encoded_columns"] += 1
    with pytest.raises(ValueError, match="encoded column count"):
        build_shared_bank(records, max_per_purpose=8)


def test_joint_uses_utility_then_t_local_guard_then_coalition_and_ties():
    table = _table()
    # T is the first mask for both purposes (lowest task NLL). J prefers a
    # same-drop pair for coalition privacy; local guard limits each recipient.
    t_id = table.pair_id("drop-education_fact", "drop-education_fact")
    chosen = select_joint(table, t_pair_id=t_id, local_auc_cap_excess=0.010)
    assert chosen.pair_id in {row.pair_id for row in table.pairs}
    assert chosen.row.coalition_search_auc == 0.65
    assert chosen.row.income_mask_id == chosen.row.occupation_mask_id
    assert chosen.feasible_pair_count > 0
    assert chosen.evaluated_pair_count == 25


def test_joint_keeps_exact_alias_and_fails_when_no_utility_feasible_pair():
    table = _table()
    t_id = table.pair_id("drop-education_fact", "drop-education_fact")
    selected = select_joint(table, t_pair_id=t_id)
    assert selected.alias_of_t is (selected.pair_id == t_id)
    poor = copy.deepcopy(_records())
    bank = build_shared_bank(poor)
    utility, refs = _evaluations(bank)
    for purpose in utility:
        for score in utility[purpose].values():
            score["metrics"] = TaskMetrics(0.70, 0.60, 0.30)
    union_auc = {pair.union_sha256: 0.75 for pair in bank.pairs}
    blocked = build_pair_table(bank, utility, refs, union_auc)
    with pytest.raises(ValueError, match="no utility-feasible"):
        select_joint(blocked, t_pair_id=blocked.pairs[0].pair_id)


def test_task_utility_margin_boundary_is_inclusive():
    bank = build_shared_bank(_records())
    utility, refs = _evaluations(bank)
    for purpose in utility:
        for score in utility[purpose].values():
            score["metrics"] = TaskMetrics(0.80 - 0.010, 0.50 + 0.010,
                                           0.20 + 0.005)
    table = build_pair_table(
        bank, utility, refs,
        {pair.union_sha256: 0.75 for pair in bank.pairs},
    )
    assert all(mask.utility_feasible for mask in table.income_masks)
    assert all(pair.pair_feasible for pair in table.pairs)


def test_union_critic_cache_cannot_claim_complete_pair_view():
    bank = build_shared_bank(_records())
    utility, refs = _evaluations(bank)
    scores = {pair.union_sha256: 0.75 for pair in bank.pairs}
    with pytest.raises(ValueError, match="RAW_UNION_ONLY"):
        build_pair_table(bank, utility, refs, scores,
                         critic_input_kind="COMPLETE_PAIR_VIEW")
    table = build_pair_table(bank, utility, refs, scores)
    assert table.critic_input_kind == "RAW_UNION_ONLY"
    assert table.to_payload()["critic_input_kind"] == "RAW_UNION_ONLY"


def test_persisted_pair_table_rejects_local_or_union_tamper():
    table = _table()
    bank = build_shared_bank(_records())
    validate_pair_table(table, bank)
    changed_local = replace(table.pairs[0], income_local_search_auc=0.99)
    with pytest.raises(ValueError, match="pair row"):
        validate_pair_table(replace(table, pairs=(changed_local,) + table.pairs[1:]), bank)
    changed_union = replace(table.pairs[0], union_encoded_columns=1)
    with pytest.raises(ValueError, match="union geometry"):
        validate_pair_table(replace(table, pairs=(changed_union,) + table.pairs[1:]), bank)
