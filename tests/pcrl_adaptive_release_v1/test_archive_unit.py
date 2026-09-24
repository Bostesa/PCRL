import hashlib
import json

import pytest

from experiments.pcrl_adaptive_release_v1.archive_unit import inventory


def test_inventory_checks_completed_bytes_and_refuses_symlink(tmp_path):
    unit = tmp_path / "unit"
    unit.mkdir()
    artifact = unit / "candidate.npz"
    artifact.write_bytes(b"fitted-object-fixture")
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (unit / "COMPLETE.json").write_text(json.dumps({"artifacts": {"candidate.npz": expected}}))
    assert inventory(unit)["candidate.npz"] == expected
    artifact.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="artifact mismatch"):
        inventory(unit)
    artifact.write_bytes(b"fitted-object-fixture")
    (unit / "outside").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlinks"):
        inventory(unit)


def test_inventory_accepts_new_fitter_receipt_key(tmp_path):
    unit = tmp_path / "unit"
    unit.mkdir()
    (unit / "INPUTS.json").write_text("{}")
    expected = hashlib.sha256(b"{}").hexdigest()
    (unit / "COMPLETE.json").write_text(
        json.dumps({"artifact_sha256": {"INPUTS.json": expected}}))
    assert inventory(unit)["INPUTS.json"] == expected
