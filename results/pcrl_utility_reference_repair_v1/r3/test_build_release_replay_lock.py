"""Data-free q17 lock checks: no feature-role archive is opened preseal."""

import hashlib
from pathlib import Path

import pytest

from results.pcrl_utility_reference_repair_v1.r3 import build_release_replay_lock as q17


def test_manifest_feature_bind_does_not_open_role_archive(tmp_path, monkeypatch):
    study = tmp_path / "results/pcrl_utility_reference_repair_v1"
    study.mkdir(parents=True)
    key = "ATTACK_FIT_NEW:features"
    role = study / ".private/role_export_v1/ATTACK_FIT_NEW/features.npz"
    digest = "a" * 64
    manifest = {"role_file_relative_path": {
        key: ".private/role_export_v1/ATTACK_FIT_NEW/features.npz"},
        "role_file_sha256": {key: digest}}
    old_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if Path(path) == role:
            raise AssertionError("feature archive opened before q17 seal")
        return old_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    declared = q17.bind_feature_roles(study, manifest, {
        str(role): digest}, (key,))
    assert declared == {key: {"path": str(role), "sha256": digest}}


def test_completed_q16_release_outputs_bind_queue_hashes(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    receipt = root / "RELEASE_RECEIPT.json"
    att = root / "ATTACK_FIT_NEW.npz"
    inner = root / "INNER_SELECTION.npz"
    for path, content in ((receipt, b"receipt"), (att, b"att"),
                          (inner, b"inner")):
        path.write_bytes(content)
    outputs = [str(receipt), str(att), str(inner)]
    digests = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
               for name in outputs}
    record = {"status": "DONE", "output_sha256": digests,
              "charged_cpu_s": 1.0}
    assert q17.bind_completed_release_outputs(record, outputs) == digests
    record["output_sha256"][str(att)] = "0" * 64
    with pytest.raises(PermissionError, match="release output"):
        q17.bind_completed_release_outputs(record, outputs)


def test_q16_carry_requires_all_eleven_done():
    units = {f"R{i}": {"status": "DONE", "charged_cpu_s": 2.0}
             for i in range(11)}
    state = {"units": units, "used_cpu_s": 122.0,
             "carry_forward": {"archived_queue_relative": "queue_g15",
                               "archived_queue_sha256": "b" * 64,
                               "charged_cpu_s": 100.0, "unit_count": 1}}
    result = q17.q16_carry(state, "c" * 64, "b" * 64)
    assert result == {"archived_queue_relative": "queue_g16",
                      "archived_queue_sha256": "c" * 64,
                      "charged_cpu_s": 122.0, "unit_count": 11}
    units["R3"]["status"] = "FAILED"
    with pytest.raises(PermissionError, match="q16"):
        q17.q16_carry(state, "c" * 64, "b" * 64)
