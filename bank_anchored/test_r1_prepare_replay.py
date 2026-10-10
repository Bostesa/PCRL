"""Data-free construction of immutable verification specs after FIT packages."""

import hashlib
import json
import os
from pathlib import Path
import sys

from bank_anchored.r1_prepare_replay import prepare


def _write(path: Path, payload: bytes = b"source") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path, obj: dict) -> str:
    return _write(path, (json.dumps(obj, sort_keys=True) + "\n").encode())


def test_prepare_pins_exact_b0_package_and_raw_sample_without_labels(tmp_path):
    root = tmp_path
    pkg = root / "results/pcrl_bank_anchored_release_v1"
    r1 = pkg / "r1"
    private = r1 / ".private"
    private.mkdir(parents=True, mode=0o700)
    prior = private / "R1_BANK_S0_FIT_SAMPLE32"
    prior.mkdir(mode=0o700)
    sample_hash = _write(prior / "sample_input.npz", b"permitted sample")
    for name in ("schema.json", "recipient0.npz", "recipient1.npz"):
        _write(prior / name)
    source_files = {}
    for name in ("public_table", "role_features", "model"):
        path = root / f"sources/{name}.bin"
        _write(path)
        source_files[name] = str(path)
    _json(private / "R1_BANK_S0_FIT_SAMPLE32_CONFIG.json",
          {"seed": 0, "unit_id": "R1_BANK_S0_FIT_SAMPLE32", "pins": source_files})
    _json(r1 / "R1_BANK_S0_FIT_RAW_RECEIPT.json",
          {"status": "PASS", "seed": 0, "sample_input_sha256": sample_hash})
    lock_hash = _json(pkg / "CAMPAIGN_LOCK.json", {"selected_bank": "W_TSTAR"})
    fit_id = "R2_B0_S0_FIT"
    fit_root = pkg / "r2/.private/units" / fit_id
    policy = fit_root / "policy"
    policy.mkdir(parents=True, mode=0o700)
    policy_hash = _write(policy / "policy.npz", b"policy arrays")
    fit_hash = _json(pkg / "r2" / f"{fit_id}_FIT_RECEIPT.json",
                     {"unit_id": fit_id, "status": "COMPLETE_IDENTITY_CONTROL",
                      "source_binding_sha256": "a" * 64})
    receipt_hash = _json(policy / "POLICY_RECEIPT.json",
                         {"arm_id": "B0", "source_seed": 0,
                          "source_binding_sha256": "a" * 64,
                          "fit_receipt_sha256": fit_hash,
                          "policy_file_sha256": policy_hash})
    release = fit_root / "OSF_DEFENSE_FIT_R0"
    release.mkdir(mode=0o700)
    release_hash = _write(release / "release_codes.npz")
    _json(release / "RELEASE_RECEIPT.json",
          {"arm_id": "B0", "source_seed": 0, "role": "OSF_DEFENSE_FIT",
           "realization": 0, "release_codes_sha256": release_hash,
           "policy_receipt_sha256": receipt_hash})
    _json(pkg / "r2" / f"{fit_id}_UNIT_RECEIPT.json",
          {"unit_id": fit_id, "arm_id": "B0", "source_seed": 0,
           "status": "COMPLETE_IDENTITY_CONTROL",
           "source_binding_sha256": "a" * 64,
           "fit_receipt_sha256": fit_hash,
           "policy_receipt_sha256": receipt_hash})
    for name in ("r1_deploy.py", "r1_runtime.py", "r1_release.py",
                 "r1_deployment_replay.py"):
        _write(root / "bank_anchored" / name)
    for name in ("r2_laws.py", "r2_package.py"):
        _write(pkg / "bank_anchored" / name)
    _write(pkg / "r0/compute_queue.py")
    _write(root / "code_witness/bank.py")
    _write(root / "code_witness/bank_deploy.py")
    _write(root / "dpc/deploy.py")
    _write(root / "dpc/partition.py")
    _write(root / "results/pcrl_latent_and_sampled_methods_v1/r01/frozen_u.py")
    _write(root / "results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/impl/release.py")
    spec_path, index = prepare(root, arm_id="B0", seed=0, fit_unit_id=fit_id,
                               python=sys.executable)
    spec = json.loads(spec_path.read_text())
    config = json.loads((private / "R1_DEPLOY_S0_B0_CONFIG.json").read_text())
    assert spec["unit_id"] == "R1_DEPLOY_S0_B0"
    assert spec["deps"] == [fit_id]
    assert spec["stage"] == "verification"
    assert config["pins"]["raw_sample"] == str(prior / "sample_input.npz")
    assert "stream_roots" not in config["pins"]
    assert all("sex" not in path.lower() and "income.npz" not in path.lower()
               for path in config["pins"].values())
    assert index["source_lock_sha256"] == lock_hash
    assert index["policy_receipt_sha256"] == receipt_hash
    assert os.stat(spec_path).st_mode & 0o077 == 0
