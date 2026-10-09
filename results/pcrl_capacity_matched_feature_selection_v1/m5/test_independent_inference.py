"""Tiny deterministic, role-free checks for the full independent family."""

import numpy as np

from results.pcrl_capacity_matched_feature_selection_v1.m5.independent_inference import (
    CLAUSE_NAMES, infer_from_saved_values,
)


RULES = ("J", "T", "I", "S12", "S21", "SHARED", "T_U", "I_U", "S12_U", "S21_U")
LOCAL = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
         "RAW_LR_C1", "RAW_HGB_15x200")


def _values(offset):
    n = 8
    result = {
        "truth__row_id": np.arange(n, dtype=np.int64),
        "truth__unit": np.arange(n, dtype=np.int64),
        "truth__income": np.array([0, 1] * 4, dtype=np.int64),
        "truth__occupation": np.arange(n, dtype=np.int64) % 6,
        "truth__sex": np.array([0, 1] * 4, dtype=np.int64),
    }
    for task, width in (("income", 2), ("occupation_group", 6)):
        p = np.full((n, width), .1 / (width - 1), dtype=np.float64)
        p[np.arange(n), result["truth__income" if task == "income"
                              else "truth__occupation"]] = .9
        for ref in ("U0", "Ucal", "F"):
            result[f"reference__{task}__{ref}"] = p.copy()
        result[f"candidate__{task}"] = p.copy()
    for view in ("income", "occupation_group", "coalition"):
        readers = LOCAL if view != "coalition" else LOCAL + tuple(
            alias + reader for alias in ("IGNORE_R1__", "IGNORE_R2__")
            for reader in LOCAL)
        for reader in readers:
            result[f"reader__{view}__{reader}"] = (
                np.linspace(.1, .9, n, dtype=np.float64) + offset)
    return result


def test_full_207_family_and_private_seed_bits():
    assert len(CLAUSE_NAMES) == 207
    data = {0: {"P": _values(0)}, 1: {"P": _values(0)}}
    plans = {seed: {"schema": "pcrl-focused-m3-control-plan-v1",
                    "rules": {rule: {"status": "SELECTED", "pair_id": "P"}
                              for rule in RULES}} for seed in (0, 1)}
    seed_a = bytes(range(32))
    seed_b = seed_a[:16] + bytes(reversed(seed_a[16:]))
    first = infer_from_saved_values(data, plans, seed_a, expected_seeds=(0, 1),
                                    expected_rows=8, expected_groups=8, replicates=8)
    second = infer_from_saved_values(data, plans, seed_b, expected_seeds=(0, 1),
                                     expected_rows=8, expected_groups=8, replicates=8)
    assert first["registered_clause_count"] == 207
    assert first["active_clause_count"] == 207
    assert [c["name"] for c in first["clauses"]] == list(CLAUSE_NAMES)
    assert first["private_seed_sha256"] != second["private_seed_sha256"]


def test_invalid_rule_keeps_registered_unresolved_slots():
    data = {0: {"P": _values(0)}, 1: {"P": _values(0)}}
    plans = {seed: {"schema": "pcrl-focused-m3-control-plan-v1",
                    "rules": {rule: {"status": "SELECTED", "pair_id": "P"}
                              for rule in RULES}} for seed in (0, 1)}
    for seed in (0, 1):
        plans[seed]["rules"]["T"] = {"status": "INVALID", "pair_id": None}
    result = infer_from_saved_values(data, plans, bytes(range(32)),
                                     expected_seeds=(0, 1), expected_rows=8,
                                     expected_groups=8, replicates=8)
    by_name = {row["name"]: row for row in result["clauses"]}
    assert by_name["T|income|U0|accuracy"]["status"] == "UNRESOLVED_INVALID_FIT_RULE"
    assert by_name["J|vs|T|coalition_benefit"]["status"] == "UNRESOLVED_INVALID_FIT_RULE"
