"""Pure task-only T selection on five synthetic mask aggregates."""

from __future__ import annotations

from pathlib import Path

import pytest

from results.pcrl_utility_reference_repair_v1.r2.frozen_bank import load_frozen_bank
from results.pcrl_utility_reference_repair_v1.r2.task_table import (
    build_pre_sensitive_task_table,
)
from results.pcrl_utility_reference_repair_v1.r2.task_table_worker import (
    _private_record_file, resolve_mask_lock_lineage,
)


ROOT = Path(__file__).resolve().parents[3]
MASK_BANK = ROOT / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
MASK_SHA = "a52750191a1e16051367cade59537bc65ed797d69efcb375e0b1cc2e9364722c"


def _fixture():
    bank = load_frozen_bank(MASK_BANK, expected_sha256=MASK_SHA)
    gates, heads, shas = {}, {}, {}
    for index, mask in enumerate(bank.masks):
        mask_metrics = {
            "income": {"accuracy": 0.79, "nll": 0.40 + index * 0.001,
                       "brier": 0.32},
            "occupation_group": {"accuracy": 0.72,
                                 "nll": 1.40 + index * 0.001,
                                 "brier": 0.59},
        }
        ref_metrics = {
            "income": {"accuracy": 0.79, "nll": 0.40, "brier": 0.32},
            "occupation_group": {"accuracy": 0.72, "nll": 1.40,
                                 "brier": 0.59},
        }
        gates[mask.mask_id] = {
            "schema": "pcrl-utility-masked-common-task-gate-v1",
            "seed": 0, "mask_id": mask.mask_id,
            "mask_sha256": mask.mask_sha256,
            "selected_recipe": "A", "full_input_gate_sha256": "a" * 64,
            "conditional_lock_sha256": "b" * 64,
            "mask_fit_source_lock_sha256": "c" * 64,
            "mask_fit_receipt_sha256": "c" * 64,
            "R1_gate_receipt_sha256": "d" * 64,
            "common_role": "INNER_SELECTION",
            "common_row_unit_pair_sha256": "e" * 64,
            "prediction_relative_path": "private/synthetic",
            "prediction_sha256": "f" * 64,
            "metrics": {"mask": mask_metrics,
                        **{name: ref_metrics for name in ("U0", "Ucal", "F")}},
            "point_eligibility": {
                task: {"references": {name: {"point_eligible": True}
                                      for name in ("U0", "Ucal", "F")}}
                for task in ("income", "occupation_group")},
            "all_task_reference_points_eligible": True,
            "development_only": True,
        }
        heads[mask.mask_id] = {"income": f"head-in-{index}",
                               "occupation_group": f"head-oc-{index}"}
        shas[mask.mask_id] = str(index) * 64
    return bank, gates, heads, shas


def test_task_only_table_picks_lowest_nll_mask_without_privacy_fields():
    bank, gates, heads, shas = _fixture()
    table, t = build_pre_sensitive_task_table(
        bank, 0, "e" * 64, gates, heads, shas,
        full_input_gate_sha256="a" * 64, selected_recipe="A",
        r1_gate_receipt_sha256="d" * 64,
        mask_lock_sha256_by_id={mask.mask_id: "b" * 64 for mask in bank.masks})
    assert t is not None
    assert t["income_mask_id"] == bank.masks[0].mask_id
    assert t["occupation_mask_id"] == bank.masks[0].mask_id
    assert set(table["income_masks"][0]) == {
        "mask_id", "mask_sha256", "head_id", "metrics",
        "utility_margins_by_ref_metric", "utility_feasible"}
    assert "sex" not in repr(table).lower()


def test_task_only_table_rejects_protected_field_and_reference_drift():
    bank, gates, heads, shas = _fixture()
    gates[bank.masks[0].mask_id]["sex"] = [0, 1]
    with pytest.raises(ValueError, match="mask task gate identity"):
        build_pre_sensitive_task_table(
            bank, 0, "e" * 64, gates, heads, shas,
            full_input_gate_sha256="a" * 64, selected_recipe="A",
            r1_gate_receipt_sha256="d" * 64,
            mask_lock_sha256_by_id={mask.mask_id: "b" * 64 for mask in bank.masks})
    del gates[bank.masks[0].mask_id]["sex"]
    gates[bank.masks[1].mask_id]["metrics"]["U0"] = {
        **gates[bank.masks[1].mask_id]["metrics"]["U0"],
        "income": {"accuracy": 0.78, "nll": 0.40, "brier": 0.32},
    }
    with pytest.raises(ValueError, match="drifted"):
        build_pre_sensitive_task_table(
            bank, 0, "e" * 64, gates, heads, shas,
            full_input_gate_sha256="a" * 64, selected_recipe="A",
            r1_gate_receipt_sha256="d" * 64,
            mask_lock_sha256_by_id={mask.mask_id: "b" * 64 for mask in bank.masks})


def test_bank_lock_lineage_accepts_archived_bank_gate_after_new_table_seal():
    bank, _, _, _ = _fixture()
    leads = {"M_DROP_NATIVE_COUNTRY", "M_DROP_MARITAL_STATUS"}
    mapping = {mask.mask_id: ("a" * 64 if mask.mask_id in leads else
                              "CURRENT_LOCK") for mask in bank.masks}
    resolved = resolve_mask_lock_lineage(
        {"mask_lock_sha256_by_id": mapping}, bank, "b" * 64)
    assert {resolved[mask_id] for mask_id in leads} == {"a" * 64}
    assert {resolved[mask.mask_id] for mask in bank.masks
            if mask.mask_id not in leads} == {"b" * 64}
    archived = {mask.mask_id: ("a" * 64 if mask.mask_id in leads else
                               "c" * 64) for mask in bank.masks}
    resolved = resolve_mask_lock_lineage(
        {"mask_lock_sha256_by_id": archived}, bank, "b" * 64)
    assert {resolved[mask_id] for mask_id in leads} == {"a" * 64}
    assert {resolved[mask.mask_id] for mask in bank.masks
            if mask.mask_id not in leads} == {"c" * 64}
    mapping["M_DROP_NATIVE_COUNTRY"] = "CURRENT_LOCK"
    with pytest.raises(PermissionError, match="lineage"):
        resolve_mask_lock_lineage(
            {"mask_lock_sha256_by_id": mapping}, bank, "b" * 64)
    archived["M_DROP_EDUCATION_FACT"] = "bad"
    with pytest.raises(PermissionError, match="lineage"):
        resolve_mask_lock_lineage(
            {"mask_lock_sha256_by_id": archived}, bank, "b" * 64)


def test_task_table_rejects_private_path_escape(tmp_path):
    private = tmp_path / ".private"
    private.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    with pytest.raises(PermissionError, match="escapes private root"):
        _private_record_file(private, {
            "relative_path": "../outside.json", "sha256": "a" * 64,
        }, "task-only head")
