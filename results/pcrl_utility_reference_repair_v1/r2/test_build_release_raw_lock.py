"""q16 lock planning may use role manifest metadata, never role bytes."""

import hashlib

import pytest

from results.pcrl_utility_reference_repair_v1.r2 import build_release_raw_lock


def test_materialize_q16_inputs_keeps_role_file_unopened(tmp_path):
    repo = tmp_path
    study = repo / "results/pcrl_utility_reference_repair_v1"
    study.mkdir(parents=True)
    frozen = study / "FULL_INPUT_GATE.json"
    frozen.write_bytes(b"sealed")
    digest = hashlib.sha256(b"sealed").hexdigest()
    role = study / ".private/role_export_v1/ATTACK_FIT_NEW/sex.npz"
    role_key = "ATTACK_FIT_NEW:sex"
    manifest = {
        "role_file_relative_path": {
            role_key: ".private/role_export_v1/ATTACK_FIT_NEW/sex.npz"},
        "role_file_sha256": {role_key: "a" * 64},
    }
    proposal = {
        "results/pcrl_utility_reference_repair_v1/FULL_INPUT_GATE.json": {
            "sha256": digest, "source": "FROZEN_EXISTING"},
        "results/pcrl_utility_reference_repair_v1/.private/role_export_v1/ATTACK_FIT_NEW/sex.npz": {
            "sha256": "a" * 64,
            "source": "FROZEN_MANIFEST_ROLE_METADATA"},
    }
    inputs, declared = build_release_raw_lock.materialize_unit_inputs(
        proposal, repo, study, manifest, (role_key,))
    assert inputs == {str(frozen): digest, str(role): "a" * 64}
    assert declared == {role_key: {"path": str(role), "sha256": "a" * 64}}


def test_q16_rejects_role_metadata_mismatch(tmp_path):
    repo = tmp_path
    study = repo / "results/pcrl_utility_reference_repair_v1"
    study.mkdir(parents=True)
    key = "ATTACK_FIT_NEW:sex"
    manifest = {"role_file_relative_path": {
        key: ".private/role_export_v1/ATTACK_FIT_NEW/sex.npz"},
        "role_file_sha256": {key: "b" * 64}}
    proposal = {
        "results/pcrl_utility_reference_repair_v1/.private/role_export_v1/ATTACK_FIT_NEW/sex.npz": {
            "sha256": "a" * 64,
            "source": "FROZEN_MANIFEST_ROLE_METADATA"},
    }
    with pytest.raises(PermissionError, match="manifest"):
        build_release_raw_lock.materialize_unit_inputs(
            proposal, repo, study, manifest, (key,))


def test_q16_requires_completed_independent_interval_carry(tmp_path):
    replay_path = tmp_path / "UTILITY_INTERVAL_REPLAY.json"
    q14 = {"units": {"R1_UTILITY_INTERVALS": {
        "status": "DONE", "charged_cpu_s": 4.0}},
        "carry_forward": {"charged_cpu_s": 100.0}, "used_cpu_s": 104.0}
    q15 = {"units": {"R3_UTILITY_INTERVAL_REPLAY": {
        "status": "DONE", "charged_cpu_s": 3.0,
        "output_sha256": {str(replay_path): "c" * 64}}},
        "carry_forward": {"archived_queue_relative": "queue_g14",
                          "archived_queue_sha256": "a" * 64,
                          "charged_cpu_s": 104.0, "unit_count": 1},
        "used_cpu_s": 107.0}
    replay = {"schema": "pcrl-utility-r3-independent-interval-replay-v1",
              "status": "REPLAYED", "family_size": 24,
              "replicates": 2048, "all_clause_statuses_match": True}
    carry = build_release_raw_lock.validate_interval_predecessors(
        q14, q15, "a" * 64, "b" * 64, replay, "c" * 64, replay_path)
    assert carry == {"archived_queue_relative": "queue_g15",
                     "archived_queue_sha256": "b" * 64,
                     "charged_cpu_s": 107.0, "unit_count": 1}
    q15["units"]["R3_UTILITY_INTERVAL_REPLAY"]["status"] = "FAILED"
    with pytest.raises(PermissionError, match="incomplete"):
        build_release_raw_lock.validate_interval_predecessors(
            q14, q15, "a" * 64, "b" * 64, replay, "c" * 64, replay_path)
