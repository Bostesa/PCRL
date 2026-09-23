"""Package already-fitted PCRL channel arrays without altering source artifacts.

This is a one-off operational script outside the study source tree. It does not
fit or score any ACS row. It aborts on unexpected source bytes or package drift.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from experiments.pcrl_task_aligned_cuts_v1.release import ChannelArtifact, sha256_file


ROOT = Path("results/pcrl_task_aligned_cuts_v1")
PRIVATE = ROOT / "private"
ENCODER_SHA = "997ccade4f8f86e2f6a49d1531ac4b8191a0c86847daf332684cfaf7b2140f2d"
ORIGINAL_CENTER_MANIFEST_SHA = "d80050e2b921aa2e1e54cf13daad8c7ab3e04fed559bcf0b712113b5ec35d8d1"
SPECS = (
    ("exchange_candidate", "exchange_r01/channel/Q.npz",
     "93c4ea8d6eb2384496a98a917cb2ce47cd005dd3cac76a97dda11b3964fab294",
     "a0_u1p1_z000_exchange_r01_324_cut_bank", "EXPERIMENTAL_NO_ADVANTAGE"),
    ("round_zero_center", "run/a0_u1p1_z000/channel/Q.npz",
     "1ff5f34ca098b3c12adf3ce8365312ee4d8ce077732659f08e7821870d0b1ed3",
     "a0_u1p1_z000_initial_312_cut_bank_center", "EXPERIMENTAL_NO_ADVANTAGE"),
    ("baseline_D_U1", "controls/a0_simple/D_U1.npz",
     "b30d3d94456d9235b85898aee65d40e3f6bdceec1868efe0901b19daf3bcf3e2",
     "D_U1_unprotected_task_optimizer", "BASELINE_CONTROL"),
    ("baseline_a0_MILP", "controls/a0_u1p1_z000_initial_bank/MILP_Q.npz",
     "de7a8ec304a4855fc4b8c5ac61c712a749bcc76982a7c0660e0d9cc6d112685c",
     "a0_u1p1_z000_initial_bank_deterministic_MILP", "BASELINE_CONTROL"),
    ("baseline_exchanged_MILP", "exchange_r01_deterministic/MILP_Q.npz",
     "23a10c01701200f2bca5ef29c96ad2755962f7890e7064b11b08ba72bad2251c",
     "a0_u1p1_z000_exchange_r01_324_cut_bank_deterministic_MILP", "BASELINE_CONTROL"),
)


def array_digest(q):
    return hashlib.sha256(np.ascontiguousarray(q, dtype=np.float64).tobytes()).hexdigest()


def one(spec):
    name, source_relative, source_expected_sha, source_name, status = spec
    source = PRIVATE / source_relative
    if sha256_file(source) != source_expected_sha:
        raise ValueError(f"source archive drift: {source}")
    with np.load(source, allow_pickle=False) as archive:
        if list(archive.files) != ["Q"]:
            raise ValueError(f"source does not contain exactly Q: {source}")
        q = archive["Q"].copy()
    if q.shape != (32, 17) or q.dtype != np.float64:
        raise ValueError(f"source channel schema drift: {source}")
    dest = PRIVATE / "packages" / name / "channel"
    if not dest.exists():
        ChannelArtifact(q, ENCODER_SHA, source_name, status).save(dest)
    loaded = ChannelArtifact.load(dest)
    if (loaded.model_status != status or loaded.source != source_name or
            loaded.encoder_sha256 != ENCODER_SHA or
            loaded.Q.dtype != q.dtype or loaded.Q.shape != q.shape or
            loaded.Q.tobytes() != q.tobytes()):
        raise ValueError(f"packaged channel differs from source or status: {dest}")
    return {
        "name": name, "status": status,
        "source_relative_path": str(source.relative_to(ROOT)),
        "source_file_sha256": source_expected_sha,
        "source_array_sha256": array_digest(q),
        "package_channel_relative_path": str(dest.relative_to(ROOT)),
        "package_Q_file_sha256": sha256_file(dest / "Q.npz"),
        "package_manifest_sha256": sha256_file(dest / "manifest.json"),
        "package_array_sha256": array_digest(loaded.Q),
        "array_byte_identical": True,
        "shape": [32, 17],
        "encoder_sha256": ENCODER_SHA,
    }


def verified_audit(name, audit_relative, sidecar_name, unit_id, q_relative,
                   expected_receipt_sha):
    audit_dir = PRIVATE / audit_relative
    receipt_path = audit_dir / "COMPLETE.json"
    if sha256_file(receipt_path) != expected_receipt_sha:
        raise ValueError(f"{name} audit receipt differs from the requested pin")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("unit_id") != unit_id or
            receipt.get("queue_sha256") != sha256_file(
                ROOT / "agents/release_math" / sidecar_name)):
        raise ValueError(f"{name} audit is not the registered unit")
    for relative, digest in receipt["artifacts"].items():
        path = (audit_dir / relative).resolve()
        if not path.is_relative_to(audit_dir.resolve()) or sha256_file(path) != digest:
            raise ValueError(f"{name} audit artifact changed: {relative}")
    report = json.loads((audit_dir / "INNER_PANEL.json").read_text())
    q_path = PRIVATE / q_relative
    with np.load(q_path, allow_pickle=False) as source:
        q = source["Q"].copy()
    if (report.get("channel_sha256") != array_digest(q) or
            report.get("slate") != "standard" or
            report.get("outer_pool_opened") is not False or
            set(report.get("roles", {})) != {
                "attack:A/SEX", "attack:A/RAC1P",
                "attack:AB/SEX", "attack:AB/RAC1P", "utility:A/same_residence"}):
        raise ValueError(f"{name} audit report scope or channel differs")
    return {"unit_id": unit_id, "receipt_sha256": expected_receipt_sha,
            "inner_panel_sha256": sha256_file(audit_dir / "INNER_PANEL.json"),
            "artifact_count": len(receipt["artifacts"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange-audit-complete-sha256", required=True)
    parser.add_argument("--matched-milp-audit-complete-sha256", required=True)
    args = parser.parse_args()
    exchange_audit = verified_audit(
        "exchange candidate", "exchange_r01_audit", "EXCHANGE_AUDIT_SIDECAR.json",
        "a0_u1p1_z000_exchange_r01_audit", "exchange_r01/channel/Q.npz",
        args.exchange_audit_complete_sha256)
    matched_milp_audit = verified_audit(
        "matched MILP", "exchange_r01_deterministic_audit",
        "EXCHANGE_MILP_AUDIT_SIDECAR.json",
        "a0_u1p1_z000_exchange_r01_deterministic_audit",
        "exchange_r01_deterministic/MILP_Q.npz",
        args.matched_milp_audit_complete_sha256)
    original_manifest = ROOT / "private/run/a0_u1p1_z000/channel/manifest.json"
    original_before = sha256_file(original_manifest)
    if original_before != ORIGINAL_CENTER_MANIFEST_SHA:
        raise ValueError("historical center manifest drift before packaging")
    records = [one(spec) for spec in SPECS]
    if sha256_file(original_manifest) != original_before:
        raise AssertionError("historical center manifest changed during packaging")
    receipt = {
        "schema": "pcrl-packaged-channels-v1",
        "study": "pcrl_task_aligned_cuts_v1",
        "scope": "2018 used development; no new fit or person-level score",
        "package_script_sha256": sha256_file(Path(__file__)),
        "exchange_audit": exchange_audit,
        "matched_milp_audit": matched_milp_audit,
        "original_center_manifest_sha256": original_before,
        "packages": records,
        "warning": "Exchange and round-zero center are separate experimental checkpoints. D_U1 is unprotected; initial MILP feasibility is for 312 cuts and exchanged MILP for the same 324 cuts as the exchange candidate, all on anchor 0. No independent algorithm advantage or population privacy guarantee is implied.",
    }
    dest = PRIVATE / "packages/PACKAGE_RECEIPT.json"
    if dest.exists():
        if json.loads(dest.read_text()) != receipt:
            raise ValueError("existing package receipt differs")
    else:
        payload = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with tempfile.NamedTemporaryFile("w", dir=dest.parent, prefix=".package-receipt-",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, dest)
    print(json.dumps({"receipt": str(dest), "receipt_sha256": sha256_file(dest),
                      "packages": [r["name"] for r in records]}, sort_keys=True))


if __name__ == "__main__":
    main()
