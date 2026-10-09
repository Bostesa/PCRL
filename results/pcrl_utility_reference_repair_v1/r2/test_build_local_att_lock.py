"""q18 must bind completed q16 producers and avoid preseal role-byte reads."""

import hashlib

import pytest

from results.pcrl_utility_reference_repair_v1.r2 import build_local_att_lock


def test_resolve_q18_inputs_requires_completed_q16_and_does_not_open_role(tmp_path):
    repo = tmp_path
    study = repo / "results/pcrl_utility_reference_repair_v1"
    study.mkdir(parents=True)
    output = study / ".private/releases/seed0/pair/RELEASE_RECEIPT.json"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"complete release")
    output_sha = hashlib.sha256(b"complete release").hexdigest()
    role = study / ".private/role_export_v1/ATTACK_FIT_NEW/sex.npz"
    role_key = "ATTACK_FIT_NEW:sex"
    manifest = {
        "role_file_relative_path": {role_key:
            ".private/role_export_v1/ATTACK_FIT_NEW/sex.npz"},
        "role_file_sha256": {role_key: "a" * 64},
    }
    proposal = {
        str(output.relative_to(repo)): {"sha256": "FROM_PRODUCER_UNIT",
                                        "producer_unit": "R2_RELEASE_S0_PAIR"},
        str(role.relative_to(repo)): {"sha256": "a" * 64,
                                      "source": "FROZEN_MANIFEST_ROLE_METADATA"},
    }
    q16 = {"units": {"R2_RELEASE_S0_PAIR": {
        "status": "DONE", "output_sha256": {str(output): output_sha}}}}
    inputs, declared = build_local_att_lock.resolve_att_inputs(
        proposal, repo, study, manifest, (role_key,), q16,
        {"R2_RELEASE_S0_PAIR"})
    assert inputs == {str(output): output_sha, str(role): "a" * 64}
    assert declared == {role_key: {"path": str(role), "sha256": "a" * 64}}
    q16["units"]["R2_RELEASE_S0_PAIR"]["status"] = "FAILED"
    with pytest.raises(PermissionError, match="producer"):
        build_local_att_lock.resolve_att_inputs(
            proposal, repo, study, manifest, (role_key,), q16,
            {"R2_RELEASE_S0_PAIR"})


def test_q18_requires_independent_release_replay_pass(tmp_path):
    output = tmp_path / "RELEASE_REPLAY.json"
    q16 = {"units": {
        **{f"R2_RELEASE_S0_PAIR{i}": {"status": "DONE", "charged_cpu_s": 1.0}
           for i in range(4)},
        **{f"R2_RAW_UNION_S0_UNION{i}": {"status": "DONE", "charged_cpu_s": 1.0}
           for i in range(7)}},
        "carry_forward": {"charged_cpu_s": 100.0}, "used_cpu_s": 111.0}
    q17 = {"units": {"R3_RELEASE_REPLAY": {
        "status": "DONE", "charged_cpu_s": 2.0,
        "output_sha256": {str(output): "c" * 64}}},
        "carry_forward": {"archived_queue_relative": "queue_g16",
                          "archived_queue_sha256": "a" * 64,
                          "charged_cpu_s": 111.0, "unit_count": 11},
        "used_cpu_s": 113.0}
    selected = {"0": ["PAIR0"], "1": ["PAIR1"]}
    replay = {"schema": "pcrl-utility-r3-conditional-release-replay-v1",
              "status": "PASS_RELEASE_REPLAY",
              "source_release_lock_sha256": "d" * 64,
              "selected_pairs_by_seed": selected,
              "role_access": ["ATTACK_FIT_NEW:features",
                              "INNER_SELECTION:features"],
              "max_abs_probability_difference": 1e-8,
              "max_abs_complete_view_difference": 1e-8}
    carry = build_local_att_lock.validate_prior_stage(
        q16, q17, "a" * 64, "b" * 64, replay, "c" * 64,
        output, "d" * 64, selected)
    assert carry == {"archived_queue_relative": "queue_g17",
                     "archived_queue_sha256": "b" * 64,
                     "charged_cpu_s": 113.0, "unit_count": 1}
    replay["max_abs_complete_view_difference"] = 3e-7
    with pytest.raises(PermissionError, match="exceeds cap"):
        build_local_att_lock.validate_prior_stage(
            q16, q17, "a" * 64, "b" * 64, replay, "c" * 64,
            output, "d" * 64, selected)
