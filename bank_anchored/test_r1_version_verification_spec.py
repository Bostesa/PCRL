"""Data-free successor spec preserves the statically audited source spec."""

import hashlib
import json
import os

import pytest

from bank_anchored.r1_version_verification_spec import version


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_version_binds_amendment_without_mutating_original(tmp_path):
    root = tmp_path
    r1 = root / "results/pcrl_bank_anchored_release_v1/r1"
    private = r1 / ".private"
    private.mkdir(parents=True, mode=0o700)
    unit = "R1_DEPLOY_S0_B0"
    config = private / f"{unit}_CONFIG.json"
    config_sha = _write(config, {"unit_id": unit})
    original_path = private / f"{unit}_QUEUE_SPEC.json"
    original_sha = _write(original_path,
                          {"unit_id": unit, "stage": "verification",
                           "argv": ["python", "-m", "worker", "--config", str(config)],
                           "sources": [{"path": str(config), "sha256": config_sha}]})
    _write(r1 / f"{unit}_SPEC_INDEX.json",
           {"unit_id": unit, "private_queue_spec_sha256": original_sha,
            "private_config_sha256": config_sha})
    amendment = root / "results/pcrl_bank_anchored_release_v1/r0/VERIFY_AMENDMENT.json"
    amendment_sha = _write(amendment, {"stages": ["verification"]})
    new_path, index = version(root, arm_id="B0", seed=0,
                              amendment=amendment,
                              amendment_sha256=amendment_sha)
    successor = json.loads(new_path.read_text())
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == original_sha
    assert successor["unit_id"] == unit
    assert successor["amendment_sha256"] == amendment_sha
    assert successor["sources"][-1] == {"path": str(amendment),
                                        "sha256": amendment_sha}
    assert index["private_queue_spec_sha256"] == hashlib.sha256(new_path.read_bytes()).hexdigest()
    assert os.stat(new_path).st_mode & 0o077 == 0


def test_version_refuses_wrong_amendment_hash(tmp_path):
    root = tmp_path
    private = root / "results/pcrl_bank_anchored_release_v1/r1/.private"
    private.mkdir(parents=True, mode=0o700)
    amendment = root / "results/pcrl_bank_anchored_release_v1/r0/VERIFY_AMENDMENT.json"
    _write(amendment, {"stages": ["verification"]})
    with pytest.raises(ValueError, match="amendment"):
        version(root, arm_id="B0", seed=0, amendment=amendment,
                amendment_sha256="0" * 64)
