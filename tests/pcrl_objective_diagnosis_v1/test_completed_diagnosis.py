"""Cross-artifact checks with independent original receipts and fixed facts."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/"results/pcrl_objective_diagnosis_v1"


def read(name):
    return json.loads((RESULTS/name).read_text())


def test_primary_aggregate_is_unchanged_and_secondary_direction_is_explicit():
    x = read("PROSPECTIVE_AGGREGATE_REPLAY.json")
    for name in ("Q", "D17"):
        assert x["primary"][name]["decision"] == "fail"
        assert (x["primary"][name]["task_passed"], x["primary"][name]["task_total"]) == (0, 2)
        assert (x["primary"][name]["privacy_passed"], x["primary"][name]["privacy_total"]) == (8, 8)
    for name in ("D17", "D33"):
        task = x["secondary"][name]["task"]
        assert len(task) == 2 and task[0]["weighting"] == "unweighted"
        assert task[0]["lower"] > 0 and task[0]["excludes_zero"]
        assert task[1]["lower"] < 0 < task[1]["upper"]
        assert x["secondary"][name]["sensitive_positive_excludes_zero"] == 0


def test_archived_inputs_and_fixed_objective_certificate():
    index = read("REUSABLE_INPUTS.json")
    slacks = read("CONSTRAINT_SLACKS.json")
    certs = read("SOLVER_CERTIFICATE.json")
    for anchor in "012":
        for key in ("prepared", "encoder", "historical_fineC_tables"):
            item = index["anchors"][anchor][key]
            assert item["sha256"] == item["archive"]["member_sha256"]
            assert item["archive"]["archive_part_sha256"]
        row = slacks[anchor]
        assert abs(row["D17_minus_rowwise_lower_bound"]) < 1e-12
        assert row["D17_rows_differing_from_first_cost_argmin"] == 0
        assert row["Q_minus_D17_fixed_objective"] > 0
        assert all(v["D17_excess"] > 0 for v in row["local_constraints"].values())
        assert all(v["Q_slack"] >= -1e-7 for v in row["local_constraints"].values())
        certificate = certs[anchor]
        assert certificate["status"] == "dual_affine_bound"
        assert 0 <= certificate["guarded_primal_gap"] < 2e-7
        assert certificate["conservative_lower_bound"] < certificate["primal_objective"]
        assert certificate["lp_primal_residual"] < 1e-8


def test_2018_decoder_rank_reversal_is_recorded_without_refit():
    record = read("DECODER_COMPARISON_2018.json")
    contrast = record["Q_minus_D17"]
    assert contrast["fixed_decoder"]["unweighted"] > 0
    assert contrast["independent_probe"]["unweighted"] < 0
    assert contrast["selected_deployment"]["unweighted"] > 0
    assert record["scope"].startswith("historical supervised 2018")


def test_fine_conditioning_uses_frozen_partition_and_reports_sparsity():
    profile = read("RANDOMIZATION_PROFILE.json")
    for anchor in "012":
        result = profile[anchor]
        assert result["state_count"] == 32 and result["unsupported_state_count"] == 0
        assert result["information_radius_upper_bound_nats"] < result["D17_information_radius_upper_bound_nats"]
        pools = result["conditional_laws"]["pools"]
        for pool in ("representation_fit:mechanism", "attacker_validation"):
            coarse = pools[pool]["coarse2"]
            fine = pools[pool]["fine4"]
            assert coarse["context_cells_A"] == 2 and fine["context_cells_A"] == 4
            assert fine["support"]["AB/RAC1P"]["empty_context_class_cells"] > 0
        assert "unidentified" in result["conditional_laws"]["full_H"]
