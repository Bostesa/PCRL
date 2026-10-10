"""Data-free Q queue-spec fixtures; no role arrays or Adult FITs are opened."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from results.pcrl_latent_and_sampled_methods_v1.r17 import q_specs


CAMPAIGN = Path(__file__).resolve().parents[1]
AMENDMENT = CAMPAIGN / "AMENDMENT_20261010_A_PRODUCER_STAGE_FIT_RETRY_001.json"
SEAL = CAMPAIGN / "CAMPAIGN_LOCK.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_q_specs_bind_public_maps_and_complete_worker_contract(tmp_path):
    """A fixed Q spec must use the admitted public map, never a role token."""
    private = tmp_path / ".private"
    private.mkdir(mode=0o700)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"units": {}}))
    with mock.patch.object(q_specs, "_latest_a_amendment"), \
         mock.patch.object(q_specs, "compile_graph"):
        for arm, seed in (("T_STAR", 0), ("A_IDENTITY", 0),
                          ("A_CLASS_ONLY", 1), ("D0_SEQ21", 1)):
            unit_id = f"A-Q-PREPROCESS-{arm}-S{seed}"
            spec = q_specs.q_spec(
                unit_id=unit_id, amendment_path=AMENDMENT,
                amendment_sha256=_sha(AMENDMENT), seal_path=SEAL,
                seal_sha256=_sha(SEAL), queue_state=state,
                private_parent=private)
            assert spec["unit_id"] == unit_id
            assert spec["deps"] == []
            assert spec["cpu_reserve_s"] == spec["max_wall_s"] == 30
            assert spec["zero_support_rule"] == "all_zero_features_with_zero_support_v1"
            assert len(spec["source_descriptor_sha256"]) == 64
            assert len(spec["outputs"]) == 2
            assert spec["outputs"][0].endswith("Q_TABLE_RECEIPT.json")
            assert spec["outputs"][1].endswith("q_table.npz")
            assert any(item["path"].endswith(f"A_FIXED_MAP_{'T_STAR' if arm.startswith('A_') else arm}_S{seed}.npz")
                       for item in spec["sources"])


def test_selected_q_fields_refuse_no_policy_and_bind_retry_receipt(tmp_path):
    """A completed but infeasible FIT cannot supply a selected Q law."""
    parent = tmp_path / ".private"
    parent.mkdir(mode=0o700)
    package = parent / "A-FIT-A_JOINT_STOCH-S0-R1"
    package.mkdir(mode=0o700)
    fit_id = package.name
    plan = {"fit_unit_id": fit_id, "arm_id": "A_JOINT_STOCH", "seed": 0}
    amendment_sha = _sha(AMENDMENT)
    shared = {"arm_id": "A_JOINT_STOCH", "source_seed": 0,
              "source_binding_sha256": "b" * 64,
              "row_unit_pair_sha256": "c" * 64,
              "bank_sha256_by_recipient": {"r1": "d" * 64, "r2": "e" * 64},
              "action_sha256_by_recipient": {"r1": "f" * 64, "r2": "0" * 64}}
    fit = {**shared, "schema": "SCV1-private-stochastic-fit-v1",
           "status": "COMPLETE_NO_FEASIBLE_POLICY", "unit_id": fit_id,
           "a_amendment_sha256": amendment_sha,
           "selected": None}
    fit_path = package / "FIT_RECEIPT.json"
    policy_path = package / "POLICY_RECEIPT.json"
    parameter_path = package / "policy.npz"
    fit_path.write_text(json.dumps(fit))
    parameter_path.write_bytes(b"synthetic policy parameters only")
    policy = {**shared, "status": "SELECTED_FIT_POLICY", "fit_unit_id": fit_id,
              "fit_receipt_sha256": _sha(fit_path),
              "selected_start": "identity_biased", "selected_update": 25,
              "release_id": "A_JOINT_STOCH-s0-v1",
              "mechanism_sha256": "1" * 64}
    policy_path.write_text(json.dumps(policy))
    state = {"units": {fit_id: {"status": "done", "spec": {
        "unit_id": fit_id, "stage": "a", "amendment_sha256": amendment_sha,
        "arm_id": "A_JOINT_STOCH", "seed": 0,
        "argv": ["python", "--private-output", str(package)],
        "outputs": [str(fit_path)]},
        "output_sha256": {str(fit_path): _sha(fit_path)}}}}
    with pytest.raises(ValueError, match="no complete selected policy"):
        q_specs._selected_fields(plan=plan, state=state,
                                 amendment_sha256=amendment_sha, sources={})
    fit.update(status="SELECTED_FIT_POLICY", selected={"start": 0, "update": 25})
    fit_path.write_text(json.dumps(fit))
    policy["fit_receipt_sha256"] = _sha(fit_path)
    policy_path.write_text(json.dumps(policy))
    state["units"][fit_id]["output_sha256"][str(fit_path)] = _sha(fit_path)
    sources = {}
    fields = q_specs._selected_fields(plan=plan, state=state,
                                     amendment_sha256=amendment_sha, sources=sources)
    assert fields["fit_receipt_sha256"] == _sha(fit_path)
    assert fields["policy_receipt_sha256"] == _sha(policy_path)
    assert fields["source_binding_sha256"] == "b" * 64
    assert len(sources) == 3
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    with mock.patch.object(q_specs, "_latest_a_amendment"), \
         mock.patch.object(q_specs, "compile_graph"):
        spec = q_specs.q_spec(
            unit_id="A-Q-PREPROCESS-A_JOINT_STOCH-S0",
            amendment_path=AMENDMENT, amendment_sha256=amendment_sha,
            seal_path=SEAL, seal_sha256=_sha(SEAL),
            queue_state=state_path, private_parent=parent)
    assert spec["fit_unit_id"] == fit_id
    assert spec["deps"] == [fit_id]
    assert spec["source_descriptor_sha256"] == fields["source_descriptor_sha256"]
    assert any(item["path"] == str(policy_path) for item in spec["sources"])
