"""Data-free checks of the focused paired-group simultaneous family."""

from __future__ import annotations

import numpy as np


RULES = ("J", "T", "I", "S12", "S21", "SHARED", "T_U", "I_U", "S12_U", "S21_U")


def test_pcg_seed_uses_all_32_locked_bytes():
    from inference import _pcg_seed_integer

    first = bytes.fromhex("01" * 16 + "00" * 16)
    second = bytes.fromhex("01" * 16 + "02" * 16)
    assert _pcg_seed_integer(first) == int.from_bytes(first, "big")
    assert _pcg_seed_integer(second) == int.from_bytes(second, "big")
    assert _pcg_seed_integer(first) != _pcg_seed_integer(second)


def _inputs(*, invalid=()):
    n = 40
    row = np.arange(n, dtype=np.int64)
    groups = np.repeat(np.arange(20, dtype=np.int64), 2)
    sex = np.repeat(np.arange(20, dtype=np.int64) % 2, 2)
    income = row % 2
    occupation = row % 6
    prob_income = np.column_stack((np.where(income == 0, 0.8, 0.2),
                                   np.where(income == 1, 0.8, 0.2)))
    prob_occupation = np.full((n, 6), 0.04)
    prob_occupation[np.arange(n), occupation] = 0.8
    refs = {
        "income": {ref: prob_income.copy() for ref in ("U0", "Ucal", "F")},
        "occupation_group": {ref: prob_occupation.copy() for ref in ("U0", "Ucal", "F")},
    }
    inputs, plans = {}, {}
    for seed in (0, 1):
        selected = {}
        rules = {}
        for index, rule_id in enumerate(RULES):
            status = "INVALID_NO_ELIGIBLE_PAIR" if rule_id in invalid else "SELECTED"
            selected[rule_id] = {
                "status": status, "pair_id": rule_id if status == "SELECTED" else None,
                "income_mask_id": rule_id, "occupation_mask_id": rule_id,
                "union_sha256": f"{index + 1:064x}",
                "recipient_capacity_signatures": {
                    purpose: {"numeric_groups": 4, "categorical_groups": 4,
                              "encoded_columns_max": 77, "actual_source_groups": 8,
                              "actual_encoded_columns": 77}
                    for purpose in ("income", "occupation_group")
                },
                "union_capacity_signature": {"source_groups": 8, "encoded_columns": 77},
            }
            if status != "SELECTED":
                rules[rule_id] = {"status": status, "pair_id": None}
                continue
            # The second fixed reader has a perfect SEX ranking. It must enter
            # every replicate's envelope even though the first is constant.
            scores = {"constant": np.full(n, 0.5),
                      "informative": sex.astype(np.float64)}
            rules[rule_id] = {
                "status": status, "pair_id": rule_id,
                "prob_income": prob_income.copy(),
                "prob_occupation": prob_occupation.copy(),
                "reader_predictions": {
                    view: {name: values.copy() for name, values in scores.items()}
                    for view in ("income", "occupation_group", "coalition")
                },
            }
        plans[seed] = {"schema": "pcrl-focused-m3-control-plan-v1", "rules": selected}
        inputs[seed] = {
            "truth": {"row_id": row.copy(), "unit": groups.copy(),
                      "income": income.copy(), "occupation": occupation.copy(),
                      "sex": sex.copy()},
            "references": {task: {ref: value.copy() for ref, value in refs[task].items()}
                           for task in refs},
            "rules": rules,
        }
    return inputs, plans


def test_all_fixed_clauses_and_reader_envelopes_are_in_one_paired_family():
    from inference import prepare, evaluate

    inputs, plans = _inputs()
    prepared = prepare(inputs, plans, expected_rows=40, expected_groups=20)
    assert len(prepared.names) == 10 * 2 * 3 * 3 + 9 * 3
    point = evaluate(prepared, np.arange(40, dtype=np.int64))
    by_name = dict(zip(prepared.names, point))
    assert by_name["J|income|U0|accuracy"] == 0.010
    assert abs(by_name["J|income|U0|nll"] - 0.010) < 1e-12
    assert by_name["J|vs|T|coalition_benefit"] == -0.020
    assert abs(by_name["J|vs|T|income_local_cap"] - 0.010) < 1e-12
    assert prepared.reader_count_by_view == {
        "income": 2, "occupation_group": 2, "coalition": 2,
    }


def test_2048_style_bootstrap_preserves_invalid_clauses_as_incomplete():
    from inference import prepare, _bootstrap

    inputs, plans = _inputs(invalid=("SHARED",))
    prepared = prepare(inputs, plans, expected_rows=40, expected_groups=20)
    output = _bootstrap(prepared, b"k" * 32, replicates=16)
    assert len(output["clauses"]) == 207
    by_name = {row["name"]: row for row in output["clauses"]}
    assert by_name["SHARED|income|U0|nll"]["status"] == "UNRESOLVED_INVALID_FIT_RULE"
    assert by_name["J|vs|SHARED|coalition_benefit"]["status"] == "UNRESOLVED_INVALID_FIT_RULE"
    assert by_name["J|vs|T|coalition_benefit"]["status"] == "VIOLATION"
    assert output["bootstrap_replicates"] == 16
    assert output["inner_groups"] == 20


def test_prepare_rejects_row_permutation_and_missing_reader():
    from inference import prepare

    inputs, plans = _inputs()
    inputs[1]["truth"]["row_id"] = inputs[1]["truth"]["row_id"][::-1].copy()
    try:
        prepare(inputs, plans, expected_rows=40, expected_groups=20)
    except ValueError as error:
        assert "aligned" in str(error)
    else:
        raise AssertionError("paired seeds were not aligned")
    inputs, plans = _inputs()
    del inputs[0]["rules"]["T"]["reader_predictions"]["coalition"]["informative"]
    try:
        prepare(inputs, plans, expected_rows=40, expected_groups=20)
    except ValueError as error:
        assert "reader" in str(error)
    else:
        raise AssertionError("the frozen reader slate narrowed")
    inputs, plans = _inputs()
    inputs[1]["truth"]["sex"][0] ^= 1
    try:
        prepare(inputs, plans, expected_rows=40, expected_groups=20)
    except ValueError as error:
        assert "paired labels" in str(error)
    else:
        raise AssertionError("paired seeds used different protected labels")


def test_prepare_rejects_numeric_casts_of_identity_labels_and_wire_scores():
    from inference import prepare

    for field in ("row_id", "unit", "income", "occupation", "sex"):
        inputs, plans = _inputs()
        inputs[0]["truth"][field] = inputs[0]["truth"][field].astype(np.float64)
        try:
            prepare(inputs, plans, expected_rows=40, expected_groups=20)
        except ValueError as error:
            assert "dtype" in str(error)
        else:
            raise AssertionError(f"float {field} was silently cast to int64")
    inputs, plans = _inputs()
    inputs[0]["rules"]["T"]["prob_income"] = inputs[0]["rules"]["T"]["prob_income"].astype(np.float32)
    try:
        prepare(inputs, plans, expected_rows=40, expected_groups=20)
    except ValueError as error:
        assert "dtype" in str(error)
    else:
        raise AssertionError("float32 probability was silently cast to float64")


def test_point_selected_strongest_control_maps_into_predeclared_family():
    from inference import prepare, _bootstrap, selected_control_clauses

    inputs, plans = _inputs()
    intervals = _bootstrap(prepare(inputs, plans, expected_rows=40, expected_groups=20),
                           b"k" * 32, replicates=16)
    registry = {
        "schema": "pcrl-focused-m3-strongest-controls-v1",
        "main": {"status": "SELECTED", "rule_id": "I"},
        "attribution": {"status": "SELECTED", "rule_id": "T_U"},
    }
    chosen = selected_control_clauses(intervals, registry)
    assert set(chosen["main"]) == {"coalition_benefit", "income_local_cap",
                                   "occupation_local_cap"}
    assert chosen["main"]["coalition_benefit"]["name"] == "J|vs|I|coalition_benefit"
    assert chosen["attribution"]["coalition_benefit"]["name"] == "J|vs|T_U|coalition_benefit"
    assert chosen["selection_adjustment"] == "FULL_207_CLAUSE_SIMULTANEOUS_FAMILY"


def test_independent_point_endpoints_preserve_all_three_references_and_envelopes():
    from inference import prepare, point_endpoints

    inputs, plans = _inputs()
    endpoints = point_endpoints(prepare(inputs, plans, expected_rows=40, expected_groups=20),
                                plans)
    assert set(endpoints) == {0, 1}
    first = endpoints[0]["J"]
    assert set(first["task"]["income"]["references"]) == {"U0", "Ucal", "F"}
    assert first["task"]["income"]["references"]["F"]["point_eligible"] is True
    assert first["recovery"]["coalition"]["primary_envelope_auc"] == 1.0
    assert first["union_sha256"] == plans[0]["rules"]["J"]["union_sha256"]
