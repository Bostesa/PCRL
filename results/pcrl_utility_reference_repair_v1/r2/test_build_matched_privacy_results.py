"""Aggregate-only checks for the compact matched comparison."""

import csv

import pytest

from results.pcrl_utility_reference_repair_v1.r2.build_matched_privacy_results import (
    STUDY, _utility_map, build,
)


def test_compact_table_covers_each_frozen_rule_seed_view():
    rows, receipt = build()
    assert len(rows) == 60
    assert {(row["seed"], row["rule_id"], row["view"]) for row in rows} == {
        (seed, rule, view) for seed in (0, 1)
        for rule in ("J", "T", "I", "S12", "S21", "SHARED", "T_U",
                     "I_U", "S12_U", "S21_U")
        for view in ("income", "occupation_group", "coalition")}
    assert sum(row["rule_status"] != "SELECTED" for row in rows) == 3
    assert all(row["point_utility_all_refs"] is True for row in rows
               if row["rule_status"] == "SELECTED")
    joint = [row for row in rows if row["rule_id"] == "J" and
             row["view"] == "coalition"]
    assert abs(sum(row["main_control_minus_rule_auc"] for row in joint) / 2 -
               receipt["control_minus_joint_coalition_auc"]) < 1e-12
    assert receipt["main_strongest_control_rule_id"] == "S21"


@pytest.mark.parametrize("field,altered", [
    ("gate_receipt_sha256", "0" * 64),
    ("point_eligible", "TOGGLE"),
    ("mask_id", "M_UNREGISTERED"),
])
def test_utility_source_rejects_mutated_receipt_point_or_key(
        tmp_path, field, altered):
    source = STUDY / "MASK_UTILITY.csv"
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if altered == "TOGGLE":
        altered = "False" if rows[0][field] == "True" else "True"
    assert rows[0][field] != altered
    rows[0][field] = altered
    changed = tmp_path / "MASK_UTILITY.csv"
    with changed.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(PermissionError):
        _utility_map(changed)
