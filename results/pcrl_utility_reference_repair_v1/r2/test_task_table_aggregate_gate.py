"""Synthetic proof that derived task table admission never opens NPZ arrays."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from results.pcrl_utility_reference_repair_v1.r2 import task_table_worker as worker


def _save(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return worker.sha_file(path)


def _fixture(tmp_path, monkeypatch):
    study = tmp_path / "study"
    private = study / ".private"
    private.mkdir(parents=True)
    monkeypatch.setattr(worker, "STUDY", study)
    source = private / "stage_seals/gate/CAMPAIGN_LOCK.json"
    source_sha = _save(source, {"schema": "synthetic-source-lock"})
    artifact = study / "F_ARTIFACT_LOCK.json"
    artifact_sha = _save(artifact, {
        "schema": "pcrl-utility-f-artifact-lock-v1", "status": "FROZEN",
        "selected_recipe": "A", "common_role": "INNER_SELECTION",
        "recipe_selection_sha256": "1" * 64,
        "f_package_sha256_by_seed_task": {
            seed: {task: "2" * 64 for task in worker.TASKS}
            for seed in ("0", "1")},
    })
    replay_path = private / "r3/TASK_GATE_REPLAY.json"
    replay_sha = _save(replay_path, {
        "schema": "pcrl-utility-r3-task-gate-replay-v1",
        "selected_recipe": "A", "F_artifact_lock_sha256": artifact_sha,
        "conditional_admission_eligible": True,
        "point_eligible_all_seed_task_reference_clauses": True,
    })
    numerical_path = study / "r3/FULL_INPUT_TASK_NUMERICAL_VERIFICATION.json"
    numerical_sha = _save(numerical_path, {
        "schema": "pcrl-utility-r3-full-input-task-numerical-verification-v1",
        "status": "PASS_POINT_GATE_REPLAY",
        "private_replay_output_sha256": replay_sha,
        "full_input": {"all_seed_task_reference_point_clauses_pass": True},
    })
    r1 = private / "r1/gate/seed_0/GATE_RECEIPT.json"
    r1_sha = _save(r1, {
        "schema": "pcrl-utility-full-input-gate-result-v1", "seed": 0,
        "selected_recipe": "A", "inner_row_unit_sha256": "3" * 64,
        "F_artifact_lock_sha256": artifact_sha,
        "predictions_sha256": "4" * 64,
    })
    gate_path = study / "FULL_INPUT_GATE.json"
    gate_sha = _save(gate_path, {
        "schema": "pcrl-utility-full-input-gate-v1",
        "status": "PASS_FULL_INPUT_POINT_GATE",
        "selected_recipe": "A", "common_role": "INNER_SELECTION",
        "source_science_lock_sha256": source_sha,
        "common_row_unit_pair_sha256": "3" * 64,
        "per_seed_task": {
            seed: {task: {"U0": True, "Ucal": True}
                   for task in worker.TASKS} for seed in ("0", "1")},
        "F_artifact_lock_sha256": artifact_sha,
        "recipe_selection_sha256": "1" * 64,
        "f_packages": {
            seed: {task: {"sha256": "2" * 64}
                   for task in worker.TASKS} for seed in ("0", "1")},
        "r3_numerical_verification_sha256": numerical_sha,
        "r3_numerical_replay_sha256": replay_sha,
        "r1_gate_receipt_sha256_by_seed": {"0": r1_sha, "1": "5" * 64},
        "r1_gate_prediction_sha256_by_seed": {"0": "4" * 64,
                                              "1": "6" * 64},
    })
    lock = {
        "full_input_gate_sha256": gate_sha,
        "source_science_lock_sha256": source_sha,
        "common_row_unit_pair_sha256": "3" * 64,
        "selected_recipe": "A",
        "f_artifact_lock_sha256": artifact_sha,
        "r3_full_input_numerical_verification_sha256": numerical_sha,
        "r3_full_input_replay_sha256": replay_sha,
    }
    inputs = {str(path): worker.sha_file(path)
              for path in (source, artifact, replay_path, numerical_path,
                           r1, gate_path)}
    return gate_path, private, lock, {"inputs": inputs}


def test_aggregate_gate_admits_json_without_opening_any_npz(tmp_path,
                                                              monkeypatch):
    gate, private, lock, spec = _fixture(tmp_path, monkeypatch)
    original = Path.open

    def no_npz(self, *args, **kwargs):
        if self.suffix == ".npz":
            raise AssertionError("derived aggregate stage opened NPZ")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_npz)
    admitted = worker.admit_aggregate_task_gate(gate, private, lock, spec, 0)
    assert admitted["selected_recipe"] == "A"
    assert admitted["r1_gate_receipt_sha256"] == spec["inputs"][
        str(private / "r1/gate/seed_0/GATE_RECEIPT.json")]


def test_model_identity_uses_only_fit_and_package_json(tmp_path, monkeypatch):
    mask = SimpleNamespace(mask_id="M_SYNTH", atoms=("age",),
                           column_indices=(0,))
    model = {
        "relative_path": "masks/M_SYNTH/seed0/final_income_head.npz",
        "sha256": "a" * 64, "model_id": "synthetic-head",
        "training_receipt": {"model_id": "synthetic-head",
                             "purpose": "income", "seed": 0,
                             "selected_atoms": ["age"],
                             "selected_columns": 1},
    }
    fit = {"model_stages": {"final": {"income": model}}}
    package = {"model": model, "model_id": "synthetic-head"}
    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError("opened model")))
    assert worker.model_identity_from_json(fit, package, mask, "income", 0) == \
        "synthetic-head:SHA256:" + "a" * 64
    package["model"] = {**model, "sha256": "b" * 64}
    with pytest.raises(PermissionError, match="model JSON identity"):
        worker.model_identity_from_json(fit, package, mask, "income", 0)


def test_derived_package_reader_rejects_npz_path_without_open(tmp_path,
                                                                monkeypatch):
    private = tmp_path / ".private"
    private.mkdir()
    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError("opened NPZ")))
    with pytest.raises(PermissionError, match="JSON-only"):
        worker._private_record_file(private, {
            "relative_path": "masks/seed0/head.npz", "sha256": "a" * 64,
        }, "derived package")
