"""Data-free spec builder for queued selected bank deployment replay.

Run only after the corresponding R2 protection unit has produced a complete
hash-bound policy. This builder opens JSON receipts and streams file hashes;
it does not deserialize FIT arrays or read any label archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess


ARMS = frozenset({"B0", "T", "D", "P", "L", "J", "S12", "S21"})


def _interpreter_entrypoint(python: str, root: Path) -> str:
    """Keep the venv symlink path and smoke the exact queued executable."""
    path = Path(os.path.abspath(python))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("queued interpreter entrypoint is unavailable")
    check = subprocess.run(
        [str(path), "-c", "import numpy, scipy, torch, sklearn, joblib"],
        cwd=root, capture_output=True, text=True, timeout=30, check=False,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
             "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"})
    if check.returncode != 0:
        raise ValueError("queued interpreter failed scientific import smoke")
    return str(path)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _new_json(path: Path, value: dict, *, private: bool) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                 0o600 if private else 0o644)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())


def prepare(root: Path, *, arm_id: str, seed: int, fit_unit_id: str,
            python: str) -> tuple[Path, dict]:
    """Write immutable private config/spec and public hash-only source index."""
    root = Path(root).resolve(strict=True)
    pkg = root / "results/pcrl_bank_anchored_release_v1"
    r1 = pkg / "r1"
    private = r1 / ".private"
    if (arm_id not in ARMS or seed not in (0, 1) or
            re.fullmatch(rf"R2_{arm_id}_S{seed}_FIT(?:_F[1-9][0-9]*)?",
                         fit_unit_id) is None or
            stat.S_IMODE(private.stat().st_mode) & 0o077):
        raise ValueError("selected FIT unit, interpreter or private root differs")
    python_entrypoint = _interpreter_entrypoint(python, root)
    unit = f"R1_DEPLOY_S{seed}_{arm_id}"
    source_config = private / f"R1_BANK_S{seed}_FIT_SAMPLE32_CONFIG.json"
    source = json.loads(source_config.read_text())
    source_receipt = r1 / f"R1_BANK_S{seed}_FIT_RAW_RECEIPT.json"
    witness = json.loads(source_receipt.read_text())
    witness_dir = private / f"R1_BANK_S{seed}_FIT_SAMPLE32"
    raw_sample = witness_dir / "sample_input.npz"
    table = Path(source["pins"]["public_table"])
    if (source.get("seed") != seed or source.get("unit_id") !=
            f"R1_BANK_S{seed}_FIT_SAMPLE32" or
            witness.get("status") != "PASS" or witness.get("seed") != seed or
            witness.get("sample_input_sha256") != _sha(raw_sample)):
        raise ValueError("selected raw sample lacks admitted witness")
    fit_root = pkg / "r2/.private/units" / fit_unit_id
    policy_dir = fit_root / "policy"
    policy_file = policy_dir / "policy.npz"
    policy_receipt = policy_dir / "POLICY_RECEIPT.json"
    package = json.loads(policy_receipt.read_text())
    fit_unit_receipt = pkg / "r2" / f"{fit_unit_id}_UNIT_RECEIPT.json"
    unit_receipt = json.loads(fit_unit_receipt.read_text())
    fit_receipt = pkg / "r2" / f"{fit_unit_id}_FIT_RECEIPT.json"
    fit = json.loads(fit_receipt.read_text())
    cached_root = fit_root / "OSF_DEFENSE_FIT_R0"
    cached_release = cached_root / "release_codes.npz"
    cached_receipt = cached_root / "RELEASE_RECEIPT.json"
    cached = json.loads(cached_receipt.read_text())
    policy_sha = _sha(policy_receipt)
    if (package.get("arm_id") != arm_id or package.get("source_seed") != seed or
            package.get("policy_file_sha256") != _sha(policy_file) or
            not isinstance(package.get("source_binding_sha256"), str) or
            re.fullmatch(r"[0-9a-f]{64}", package["source_binding_sha256"]) is None or
            unit_receipt.get("unit_id") != fit_unit_id or
            unit_receipt.get("arm_id") != arm_id or
            unit_receipt.get("source_seed") != seed or
            not str(unit_receipt.get("status", "")).startswith("COMPLETE_") or
            fit.get("status") != unit_receipt.get("status") or
            fit.get("unit_id") != fit_unit_id or
            fit.get("source_binding_sha256") != package["source_binding_sha256"] or
            unit_receipt.get("fit_receipt_sha256") != _sha(fit_receipt) or
            package.get("fit_receipt_sha256") != _sha(fit_receipt) or
            unit_receipt.get("policy_receipt_sha256") != policy_sha or
            unit_receipt.get("source_binding_sha256") != package["source_binding_sha256"] or
            cached.get("arm_id") != arm_id or cached.get("source_seed") != seed or
            cached.get("role") != "OSF_DEFENSE_FIT" or cached.get("realization") != 0 or
            cached.get("policy_receipt_sha256") != policy_sha or
            cached.get("release_codes_sha256") != _sha(cached_release)):
        raise ValueError("selected R2 policy or cached FIT release differs")
    campaign_lock = pkg / "CAMPAIGN_LOCK.json"
    lock = json.loads(campaign_lock.read_text())
    if lock.get("selected_bank") != "W_TSTAR":
        raise ValueError("selected bank lock differs")
    pins = {
        "source_config": source_config, "source_receipt": source_receipt,
        "raw_sample": raw_sample, "schema_file": witness_dir / "schema.json",
        "b0_r0": witness_dir / "recipient0.npz",
        "b0_r1": witness_dir / "recipient1.npz",
        "public_table": table, "policy_receipt": policy_receipt,
        "policy_npz": policy_file, "campaign_lock": campaign_lock,
        "role_features": Path(source["pins"]["role_features"]),
        "cached_fit_release": cached_release,
        "cached_fit_receipt": cached_receipt,
    }
    if arm_id not in ("B0", "D"):
        stream_roots = pkg / "r2/.private/stream_roots_v1.json"
        if _sha(stream_roots) != lock["sampling_root_hashes"]["private_roots_source_sha256"]:
            raise ValueError("selected cached draw root differs from lock")
        pins["stream_roots"] = stream_roots
    config = {
        "schema": "pcrl-bank-r1-deployment-replay-config-v1",
        "unit_id": unit, "seed": seed, "arm_id": arm_id,
        "source_binding_sha256": package["source_binding_sha256"],
        "policy_receipt_sha256": policy_sha,
        "python": python_entrypoint, "cwd": str(root),
        "pins": {name: str(path.resolve(strict=True)) for name, path in pins.items()},
        "private_output_dir": str(private / unit),
        "public_receipt": str(r1 / f"{unit}_RECEIPT.json"),
    }
    config_path = private / f"{unit}_CONFIG.json"
    _new_json(config_path, config, private=True)
    code = [
        root / f"bank_anchored/{name}.py" for name in
        ("r1_deploy", "r1_runtime", "r1_release", "r1_deployment_replay")
    ] + [pkg / f"bank_anchored/{name}.py" for name in
         ("r2_laws", "r2_package")] + [
        pkg / "r0/compute_queue.py", root / "code_witness/bank.py",
        root / "code_witness/bank_deploy.py", root / "dpc/deploy.py",
        root / "dpc/partition.py",
        root / "results/pcrl_latent_and_sampled_methods_v1/r01/frozen_u.py",
        root / "results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/impl/release.py",
    ]
    source_paths = {str(path.resolve(strict=True)): path.resolve(strict=True)
                    for path in [config_path, *pins.values(), *map(Path, source["pins"].values()),
                                 fit_unit_receipt, fit_receipt, *code]}
    sources = [{"path": path, "sha256": _sha(value)}
               for path, value in sorted(source_paths.items())]
    spec = {"unit_id": unit, "stage": "verification",
            "argv": [config["python"], "-m", "bank_anchored.r1_deployment_replay",
                     "--config", str(config_path)],
            "cwd": str(root), "max_wall_s": 120, "cpu_reserve_s": 120,
            "max_rss_gib": 2.0, "projected_output_bytes": 2097152,
            "deps": [fit_unit_id], "outputs": [config["public_receipt"]],
            "env": {}, "sources": sources}
    spec_path = private / f"{unit}_QUEUE_SPEC.json"
    _new_json(spec_path, spec, private=True)
    index = {"schema": "pcrl-bank-r1-deployment-replay-spec-index-v1",
             "unit_id": unit, "arm_id": arm_id, "seed": seed,
             "fit_unit_id": fit_unit_id,
             "source_lock_sha256": _sha(campaign_lock),
             "policy_receipt_sha256": policy_sha,
             "private_config_sha256": _sha(config_path),
             "private_queue_spec_sha256": _sha(spec_path),
             "source_count": len(sources),
             "source_sha256_by_role": {name: _sha(path) for name, path in pins.items()},
             "public_output": f"r1/{unit}_RECEIPT.json"}
    _new_json(r1 / f"{unit}_SPEC_INDEX.json", index, private=False)
    return spec_path, index


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm-id", choices=sorted(ARMS), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--fit-unit-id", required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    path, index = prepare(args.root, arm_id=args.arm_id, seed=args.seed,
                          fit_unit_id=args.fit_unit_id, python=args.python)
    print(json.dumps({"queue_spec": str(path), "index": index}, sort_keys=True))


if __name__ == "__main__":
    main()
