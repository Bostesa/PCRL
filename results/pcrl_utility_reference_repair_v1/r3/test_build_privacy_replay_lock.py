"""Data-free q21 builder checks and a denied preseal SEX archive read."""

from pathlib import Path

import pytest

from results.pcrl_utility_reference_repair_v1.r3 import (
    build_privacy_replay_lock as q21,
)


def test_sex_roles_are_bound_from_manifest_and_prior_specs_without_archive_open(
        tmp_path, monkeypatch):
    study = tmp_path / "results/pcrl_utility_reference_repair_v1"
    study.mkdir(parents=True)
    paths = {
        "ATTACK_FIT_NEW:sex":
        ".private/role_export_v1/ATTACK_FIT_NEW/sex.npz",
        "INNER_SELECTION:sex":
        ".private/role_export_v1/INNER_SELECTION/sex.npz",
    }
    digests = {"ATTACK_FIT_NEW:sex": "a" * 64,
               "INNER_SELECTION:sex": "b" * 64}
    manifest = {"role_file_relative_path": paths, "role_file_sha256": digests}
    specs = {"q18": {}, "q20": {}}
    for stage, key in (("q18", "ATTACK_FIT_NEW:sex"),
                       ("q20", "INNER_SELECTION:sex")):
        path = str(study / paths[key])
        specs[stage][stage] = {
            "role_inputs": {key: {"path": path, "sha256": digests[key]}},
            "inputs": {path: digests[key]},
        }
    old_open = Path.open

    def denied_sex_open(path, *args, **kwargs):
        if str(path).endswith("/sex.npz"):
            raise AssertionError("SEX archive opened before q21 seal")
        return old_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_sex_open)
    bound = q21.bind_sex_roles(study, manifest, specs)
    assert set(bound) == set(paths)
    assert bound["INNER_SELECTION:sex"]["sha256"] == "b" * 64
    specs["q18"]["q18"]["inputs"][str(study / paths["ATTACK_FIT_NEW:sex"])] = "0" * 64
    with pytest.raises(PermissionError, match="predecessor"):
        q21.bind_sex_roles(study, manifest, specs)


def test_completed_score_carry_requires_both_seed_units():
    state = {
        "units": {name: {"status": "DONE", "charged_cpu_s": 2.0}
                  for name in ("R2_PRIVACY_SCORE_S0", "R2_PRIVACY_SCORE_S1")},
        "carry_forward": {"archived_queue_relative": "queue_g19",
                          "archived_queue_sha256": "a" * 64,
                          "charged_cpu_s": 170.0, "unit_count": 4},
        "used_cpu_s": 174.0,
    }
    assert q21.completed_score_carry(state, "b" * 64, "a" * 64) == {
        "archived_queue_relative": "queue_g20",
        "archived_queue_sha256": "b" * 64,
        "charged_cpu_s": 174.0,
        "unit_count": 2,
    }
    state["units"]["R2_PRIVACY_SCORE_S1"]["status"] = "FAILED"
    with pytest.raises(PermissionError, match="two completed"):
        q21.completed_score_carry(state, "b" * 64, "a" * 64)


def test_selected_pairs_require_two_per_seed():
    audit = {"by_seed": {str(seed): {"selected_rule_pair_ids": {
        "J": "PAIR_A", "T": "PAIR_A", "I": "PAIR_B"}}
        for seed in (0, 1)}}
    assert q21._selected(audit) == {"0": ["PAIR_A", "PAIR_B"],
                                    "1": ["PAIR_A", "PAIR_B"]}
    audit["by_seed"]["1"]["selected_rule_pair_ids"]["I"] = "PAIR_A"
    with pytest.raises(PermissionError, match="pair set"):
        q21._selected(audit)
