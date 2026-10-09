"""Data-free checks for a charged, new-identity q22 replay retry."""

import hashlib
from pathlib import Path

import pytest

from results.pcrl_utility_reference_repair_v1.r3 import (
    build_privacy_replay_retry_lock as q22,
)


def test_failed_q21_carry_includes_one_charged_output_free_unit(tmp_path):
    old_output = tmp_path / "PRIVACY_REPLAY.json"
    old_spec = {"unit_id": "R3_PRIVACY_REPLAY",
                "stage": "r3_privacy_replay", "outputs": [str(old_output)]}
    prior = {"archived_queue_relative": "queue_g20",
             "archived_queue_sha256": "a" * 64,
             "charged_cpu_s": 174.85992329311557, "unit_count": 2}
    state = {"units": {"R3_PRIVACY_REPLAY": {
        "status": "FAILED", "exit_code": 1, "output_sha256": {},
        "charged_cpu_s": q22.FAILED_CPU_S}},
        "carry_forward": prior,
        "used_cpu_s": prior["charged_cpu_s"] + q22.FAILED_CPU_S,
        "gate": {"seal_sha256": "b" * 64}}
    carry = q22.failed_replay_carry(state, old_spec, "c" * 64,
                                     "b" * 64, "a" * 64)
    assert carry["charged_cpu_s"] == state["used_cpu_s"]
    assert carry["archived_queue_relative"] == "queue_g21"
    old_output.write_text("invalid previous output")
    with pytest.raises(PermissionError, match="preserved failed"):
        q22.failed_replay_carry(state, old_spec, "c" * 64,
                                "b" * 64, "a" * 64)


def test_inherited_inputs_rehash_every_nonrole_except_repaired_worker(
        tmp_path, monkeypatch):
    worker = tmp_path / "privacy_replay_worker.py"
    worker.write_text("corrected source\n")
    static = tmp_path / "frozen.json"
    static.write_text("frozen evidence\n")
    role = tmp_path / "role_export_v1/INNER_SELECTION/sex.npz"
    digest = "a" * 64
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    old_inputs = {str(worker): "0" * 64,
                  str(static): sha(static), str(role): digest}
    roles = {"ATTACK_FIT_NEW:sex": {"path": str(tmp_path / "att.npz"),
                                     "sha256": "b" * 64},
             "INNER_SELECTION:sex": {"path": str(role), "sha256": digest}}
    old_inputs[roles["ATTACK_FIT_NEW:sex"]["path"]] = "b" * 64
    old_open = Path.open

    def denied_role_open(path, *args, **kwargs):
        if str(path).endswith(".npz"):
            raise AssertionError("SEX role opened before retry seal")
        return old_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_role_open)
    inherited = q22.inherit_inputs(old_inputs, roles, worker)
    assert inherited[str(worker)] == sha(worker)
    assert inherited[str(role)] == digest
    static.write_text("mutated evidence\n")
    with pytest.raises(PermissionError, match="nonrole input"):
        q22.inherit_inputs(old_inputs, roles, worker)
