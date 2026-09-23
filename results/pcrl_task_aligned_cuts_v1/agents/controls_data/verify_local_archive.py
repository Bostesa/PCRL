"""Restore and replay one frozen scientific unit from the private local archive.

The complete archive member inventory is checked. The original anchor-0 bank,
center, exchanged bank, matched controls, fitted weights, and five packaged
channels are extracted into a fresh temporary directory. No original or
archived artifact is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from experiments.pcrl_task_aligned_cuts_v1 import archive, controls, solver
from experiments.pcrl_task_aligned_cuts_v1.release import ChannelArtifact
from experiments.pcrl_task_directed_release_v1.audits import load_candidate


BASE = "results/pcrl_task_aligned_cuts_v1/private"
PREFIXES = (
    f"{BASE}/run/a0_reference_bank",
    f"{BASE}/run/a0_u1p1_z000",
    f"{BASE}/controls/a0_u1p1_z000_initial_bank",
    f"{BASE}/controls/a0_simple",
    f"{BASE}/exchange_r01",
    f"{BASE}/exchange_r01_deterministic",
    f"{BASE}/packages",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(staging: Path) -> dict:
    staging = staging.resolve()
    manifest_path = staging / "MANIFEST.private.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("study") != archive.STUDY or not manifest.get("originals_retained"):
        raise ValueError("wrong local archive manifest")
    files = manifest["files"]
    known = {item["path"] for item in files}
    if len(known) != len(files):
        raise ValueError("duplicate archive member")
    listed = [name for part in manifest["parts"] for name in part["members"]]
    if set(listed) != known or len(listed) != len(known):
        raise ValueError("archive parts do not cover each manifest member exactly once")
    restored = []
    with tempfile.TemporaryDirectory(prefix="pcrl_archive_replay_") as tmp:
        root = Path(tmp)
        for part in manifest["parts"]:
            compressed = staging / part["name"]
            if sha(compressed) != part["sha256"]:
                raise ValueError("archive part hash changed")
            members = [item for item in files if item["path"] in set(part["members"])]
            if len(members) != part["files"]:
                raise ValueError("archive member count differs from manifest")
            check = archive.verify_restore(compressed, members, root, prefixes=PREFIXES)
            if check["verified_files"] != len(members):
                raise ValueError("archive readback incomplete")
            restored.extend(check["restored_paths"])
        if not any(path.endswith("candidate.joblib") for path in restored):
            raise ValueError("no fitted attack weights restored")
        bank_dir = root / PREFIXES[0]
        arm_dir = root / PREFIXES[1]
        control_dir = root / PREFIXES[2]
        cost, cuts, bank_hash = controls.load_saved_bank(bank_dir, cost_dir=arm_dir)
        arm = json.loads((arm_dir / "FIT_P1_ARM.json").read_text())
        with np.load(arm_dir / "channel/Q.npz", allow_pickle=False) as stored:
            q = stored["Q"]
        replay = solver.replay_p1(q, cost, cuts)
        if (bank_hash != arm["adjusted_bank_sha256"]
                or replay["bank_sha256"] != bank_hash
                or abs(replay["objective"] - arm["solution"]["objective"]) > 1e-10
                or replay["maximum_cut_violation"] > 1e-7
                or replay["simplex_residual"] > 1e-10):
            raise ValueError("restored center scientific replay differs from frozen report")
        controls_report = json.loads((control_dir / "CONTROLS.json").read_text())
        with np.load(control_dir / "MILP_Q.npz", allow_pickle=False) as stored:
            milp_q = stored["Q"]
        milp = solver.replay_p1(milp_q, cost, cuts)
        if (controls_report["bank_sha256"] != bank_hash
                or abs(milp["objective"] - controls_report["milp"]["replay"]["objective"]) > 1e-10
                or milp["maximum_cut_violation"] > 1e-7):
            raise ValueError("restored matched deterministic control differs from frozen report")
        exchange_dir = root / BASE / "exchange_r01"
        exchange_cost, exchange_cuts, exchange_bank_hash = controls.load_saved_bank(exchange_dir)
        exchange_report = json.loads((exchange_dir / "EXCHANGE_ROUND.json").read_text())
        with np.load(exchange_dir / "channel/Q.npz", allow_pickle=False) as stored:
            exchange_q = stored["Q"]
        exchange_replay = solver.replay_p1(exchange_q, exchange_cost, exchange_cuts)
        if (exchange_bank_hash != exchange_report["bank_sha256"]
                or exchange_replay["bank_sha256"] != exchange_bank_hash
                or abs(exchange_replay["objective"] - exchange_report["solution"]["objective"]) > 1e-10
                or exchange_replay["maximum_cut_violation"] > 1e-7):
            raise ValueError("restored exchanged channel differs from frozen bank/report")
        exchanged_milp_dir = root / BASE / "exchange_r01_deterministic"
        with np.load(exchanged_milp_dir / "MILP_Q.npz", allow_pickle=False) as stored:
            exchanged_milp_q = stored["Q"]
        exchanged_milp = solver.replay_p1(exchanged_milp_q, exchange_cost, exchange_cuts)
        if exchanged_milp["maximum_cut_violation"] > 1e-7:
            raise ValueError("restored exchanged-bank deterministic control is infeasible")
        results_root = root / "results/pcrl_task_aligned_cuts_v1"
        packages_dir = results_root / "private/packages"
        package_receipt_path = packages_dir / "PACKAGE_RECEIPT.json"
        package_receipt = json.loads(package_receipt_path.read_text())
        if (package_receipt.get("schema") != "pcrl-packaged-channels-v1"
                or len(package_receipt.get("packages", [])) != 5):
            raise ValueError("restored five-mode package receipt is malformed")
        package_names = []
        for item in package_receipt["packages"]:
            source = results_root / item["source_relative_path"]
            channel_dir = results_root / item["package_channel_relative_path"]
            packaged_q = channel_dir / "Q.npz"
            packaged_manifest = channel_dir / "manifest.json"
            if (sha(source) != item["source_file_sha256"]
                    or sha(packaged_q) != item["package_Q_file_sha256"]
                    or sha(packaged_manifest) != item["package_manifest_sha256"]):
                raise ValueError(f"restored package file hash mismatch: {item['name']}")
            channel = ChannelArtifact.load(channel_dir)
            with np.load(source, allow_pickle=False) as stored:
                source_q = stored["Q"]
            digest = hashlib.sha256(np.ascontiguousarray(channel.Q, dtype=np.float64).tobytes()).hexdigest()
            if (not np.array_equal(channel.Q, source_q)
                    or digest != item["package_array_sha256"]
                    or digest != item["source_array_sha256"]
                    or list(channel.Q.shape) != item["shape"]
                    or channel.encoder_sha256 != item["encoder_sha256"]
                    or channel.model_status != item["status"]
                    or item["array_byte_identical"] is not True):
                raise ValueError(f"restored package channel differs from original: {item['name']}")
            package_names.append(item["name"])
        smoke_receipt_path = packages_dir / "SMOKE_RECEIPT.json"
        smoke = json.loads(smoke_receipt_path.read_text())
        if smoke.get("package_receipt_sha256") != sha(package_receipt_path):
            raise ValueError("restored package smoke is not tied to package receipt")
        model_directory = bank_dir / "attack_bank/A_SEX/H/models/logistic"
        model_path = model_directory / "candidate.joblib"
        model = load_candidate(model_directory)
        synthetic = np.zeros((2, len(model.mean)), dtype=np.float64)
        predictions = model.predict_token_proba(synthetic, model.n_tokens)
        if (predictions.shape != (2, model.n_tokens, model.n_classes)
                or not np.isfinite(predictions).all()
                or not np.allclose(predictions.sum(axis=2), 1., atol=1e-12, rtol=0)):
            raise ValueError("restored fitted attacker fails prediction smoke")
        return {
            "schema": "pcrl-local-archive-restore-replay-v1",
            "manifest_sha256": sha(manifest_path),
            "verified_archive_files": len(files),
            "restored_files": len(restored),
            "fitted_model_restored_sha256": sha(model_path),
            "fitted_model_prediction_smoke": True,
            "adjusted_bank_sha256": bank_hash,
            "center_objective": replay["objective"],
            "center_max_cut_violation": replay["maximum_cut_violation"],
            "milp_objective": milp["objective"],
            "milp_max_cut_violation": milp["maximum_cut_violation"],
            "exchanged_bank_sha256": exchange_bank_hash,
            "exchanged_objective": exchange_replay["objective"],
            "exchanged_max_cut_violation": exchange_replay["maximum_cut_violation"],
            "exchanged_milp_objective": exchanged_milp["objective"],
            "exchanged_milp_max_cut_violation": exchanged_milp["maximum_cut_violation"],
            "package_receipt_sha256": sha(package_receipt_path),
            "package_smoke_receipt_sha256": sha(smoke_receipt_path),
            "restored_package_names": package_names,
            "all_originals_retained": True,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.staging), sort_keys=True))
