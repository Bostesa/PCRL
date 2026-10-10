"""Data-free restore inventory tests for two paired seed identities."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from bank_anchored.r1_snapshot_manifest import build


def _write(path: Path, data: bytes = b"source") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _json(path: Path, obj: dict) -> str:
    return _write(path, (json.dumps(obj, sort_keys=True) + "\n").encode())


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path
    pkg = root / "results/pcrl_bank_anchored_release_v1"
    private = pkg / "r1/.private"
    private.mkdir(parents=True, mode=0o700)
    for seed in (0, 1):
        hcal = root / f"hcal{seed}"
        qpc = root / f"qpc{seed}"
        admitted = hcal / f"admitted/rel__s{seed}__U"
        pins = {}
        for alias, name in (("complete", "COMPLETE.json"),
                            ("model", "model.pt"), ("head0", "head_0.joblib"),
                            ("head1", "head_1.joblib"),
                            ("record", "record.json")):
            path = admitted / name
            _write(path)
            pins[alias] = str(path)
        decoder = hcal / "package" / f"decoder__s{seed}.json"
        route = qpc / "run/units" / f"pol__s{seed}__U_FINE-TASK_i8o64/policy.json"
        table = root / f"table{seed}.json"
        for alias, path in (("decoder", decoder), ("policy", route),
                            ("public_table", table)):
            _write(path)
            pins[alias] = str(path)
        cfg = private / f"R1_BANK_S{seed}_FIT_SAMPLE32_CONFIG.json"
        _json(cfg, {"seed": seed, "source_root": str(hcal),
                    "qpc_root": str(qpc), "pins": pins})
        sample = private / f"R1_BANK_S{seed}_FIT_SAMPLE32"
        sample_hash = _write(sample / "sample_input.npz", b"X only")
        _write(sample / "schema.json", b"83 permitted names")
        _json(pkg / "r1" / f"R1_BANK_S{seed}_FIT_RAW_RECEIPT.json",
              {"seed": seed, "status": "PASS",
               "sample_input_sha256": sample_hash})
        fit_id = f"R2_B0_S{seed}_FIT_F1"
        unit = pkg / "r2/.private/units" / fit_id
        policy = unit / "policy"
        policy_hash = _write(policy / "policy.npz", b"public bank and action arrays")
        fit_sha = _json(pkg / "r2" / f"{fit_id}_FIT_RECEIPT.json",
                        {"unit_id": fit_id, "status": "COMPLETE_IDENTITY_CONTROL"})
        policy_sha = _json(policy / "POLICY_RECEIPT.json",
                           {"arm_id": "B0", "source_seed": seed,
                            "policy_file_sha256": policy_hash,
                            "fit_receipt_sha256": fit_sha})
        _json(pkg / "r2" / f"{fit_id}_UNIT_RECEIPT.json",
              {"unit_id": fit_id, "status": "COMPLETE_IDENTITY_CONTROL",
               "policy_receipt_sha256": policy_sha,
               "fit_receipt_sha256": fit_sha})
        cached = unit / "OSF_DEFENSE_FIT_R0"
        codes_sha = _write(cached / "release_codes.npz", b"codes only")
        _json(cached / "RELEASE_RECEIPT.json",
              {"policy_receipt_sha256": policy_sha,
               "release_codes_sha256": codes_sha})
    return root, private


def test_selected_inventory_restores_loader_paths_without_labels(tmp_path):
    root, private = _fixture(tmp_path)
    output = private / "snapshot.json"
    inventory = build(root, output, selected_arms=("B0",),
                      fit_unit_ids={("B0", seed): f"R2_B0_S{seed}_FIT_F1"
                                    for seed in (0, 1)})
    assert inventory["status"] == "SELECTED_POLICY_BOUND"
    names = {entry["archive_name"] for entry in inventory["entries"]}
    assert "seed0/source_root/admitted/rel__s0__U/model.pt" in names
    assert "seed1/qpc_root/run/units/pol__s1__U_FINE-TASK_i8o64/policy.json" in names
    assert "seed0/selected/B0/policy/policy.npz" in names
    assert all("sex" not in entry["archive_name"].lower()
               and "income.npz" not in entry["archive_name"].lower()
               for entry in inventory["entries"])
    assert os.stat(output).st_mode & 0o077 == 0


def test_selected_inventory_refuses_changed_policy(tmp_path):
    root, private = _fixture(tmp_path)
    path = root / "results/pcrl_bank_anchored_release_v1/r2/.private/units/R2_B0_S1_FIT_F1/policy/policy.npz"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="selected policy"):
        build(root, private / "snapshot.json", selected_arms=("B0",),
              fit_unit_ids={("B0", seed): f"R2_B0_S{seed}_FIT_F1"
                            for seed in (0, 1)})
