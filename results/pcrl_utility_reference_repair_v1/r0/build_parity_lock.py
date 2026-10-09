"""Create the first private unit registry and public parity-stage lock.

This only hashes declared files. It never deserializes a role NPZ, source
release NPZ or label. The resulting lock still needs Git commit and an R3
review before the local seal is effective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROLES = ("OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT", "INNER_SELECTION")
SEEDS = (0, 1)
SOURCE_MODULES = (
    "results/pcrl_utility_reference_repair_v1/r0/admission.py",
    "results/pcrl_utility_reference_repair_v1/r0/queue_adapter.py",
    "results/pcrl_utility_reference_repair_v1/r1/reference_parity.py",
    "results/pcrl_utility_reference_repair_v1/r3/reference_replay_worker.py",
    "results/pcrl_utility_reference_repair_v1/r3/source_reference.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m0/queue.py",
    "dpc/deploy.py", "hcal/calib.py", "jcv/train.py",
)
PROTOCOL_FILES = (
    "EXECUTION_PLAN.md", "RESOURCE_PLAN.json", "AGENT_ROSTER.json",
    "m1/DATA_ACCESS_MANIFEST.json", "r3/ROLE_OVERLAP.json",
    "r3/PREFIT_SOURCE_AND_EXPOSURE_AUDIT.md",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build(repo: Path, hcal_private: Path, python: Path, slot: Path) -> dict:
    study = repo / "results" / "pcrl_utility_reference_repair_v1"
    private = study / ".private"
    manifest_path = study / "m1" / "DATA_ACCESS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    export = private / "role_export_v1"
    queue = private / "queue"
    lock = study / "CAMPAIGN_LOCK.json"
    if not python.is_file() or not slot.is_file():
        raise FileNotFoundError("Python runtime or global semaphore missing")
    if manifest.get("scientific_fit_permission") != "CONDITIONAL_ON_LOCAL_GIT_SEAL_AND_QUEUE_ADMISSION":
        raise PermissionError("role admission must remain conditional")
    hcal = hcal_private.resolve(strict=True)
    specs = {}
    for owner, script_rel in (
            ("R1", "r1/reference_parity.py"),
            ("R3", "r3/reference_replay_worker.py")):
        script = study / script_rel
        for role in ROLES:
            for seed in SEEDS:
                uid = f"{owner}_PARITY_{role}_S{seed}"
                role_keys = (f"{role}:features", f"{role}:references_s{seed}")
                role_inputs = {}
                inputs = {str(manifest_path): sha(manifest_path)}
                for key in role_keys:
                    role_file = (study / manifest["role_file_relative_path"][key]).resolve(strict=True)
                    expected = manifest["role_file_sha256"][key]
                    if sha(role_file) != expected:
                        raise PermissionError("admitted role file hash changed: " + key)
                    role_inputs[key] = {"path": str(role_file), "sha256": expected}
                    inputs[str(role_file)] = expected
                source = hcal / "admitted" / f"rel__s{seed}__U"
                cal = hcal / "run" / "units" / f"calU__s{seed}"
                for path in (
                    source / "COMPLETE.json", source / "model.pt",
                    source / "head_0.joblib", source / "head_1.joblib",
                    source / "record.json", cal / "COMPLETE.json", cal / "record.json",
                ):
                    inputs[str(path)] = sha(path)
                output = private / "reference_parity" / owner.lower() / f"{role}_s{seed}.json"
                argv = [str(python), str(script), "--unit-id", uid,
                        "--queue-root", str(queue), "--lock", str(lock),
                        "--manifest", str(manifest_path), "--export-root", str(export),
                        "--hcal-private", str(hcal), "--role", role,
                        "--seed", str(seed), "--output", str(output)]
                specs[uid] = {
                    "unit_id": uid, "mode": "real", "stage": "reference_parity",
                    "argv": argv, "script": str(script), "script_sha256": sha(script),
                    "cwd": str(repo), "cpu_reserve_s": 300,
                    "wall_limit_s": 300, "max_rss_gib": 3,
                    "inputs": inputs, "outputs": [str(output)], "deps": [],
                    "source_map_sha256": sha(manifest_path),
                    "role_access": [role], "role_inputs": role_inputs,
                    "allow_child_processes": False, "data_access": "ROLE_ONLY_FEATURES_AND_REFERENCES",
                    "seal_sha256": "$LOCAL_SEAL_SHA256",
                }
    registry_path = private / "UNIT_SPECS.json"
    write_once(registry_path, {"schema": "pcrl-utility-unit-registry-v1",
                               "stage": "reference_parity", "unit_specs": specs})
    lock_value = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "reference_parity", "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "development_status": "exploratory_reused_Adult_people",
        "paired_seeds": list(SEEDS), "roles": list(ROLES),
        "allowed_roles": list(ROLES),
        "reference_probability_max_abs_tolerance": 2e-7,
        "reference_parity_units": sorted(specs),
        "private_unit_registry_sha256": sha(registry_path),
        "source_map_sha256": sha(manifest_path),
        "prior_package_relative_to_worktree": "results/pcrl_utility_reference_repair_v1",
        "governed_source_sha256_by_path": {
            name: sha(repo / name) for name in SOURCE_MODULES},
        "governed_protocol_sha256_by_relative_path": {
            name: sha(study / name) for name in PROTOCOL_FILES},
        "shared_slot_sha256": sha(slot),
        "excluded_inputs": ["old assessment person arrays", "HEAD_VALIDATION labels",
                            "hcal release.npz"],
        "next_stage": "FIT-only A/B screen under new stage lock after parity",
    }
    write_once(lock, lock_value)
    return {"registry_sha256": sha(registry_path), "lock_sha256": sha(lock),
            "unit_count": len(specs), "unit_ids": sorted(specs)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--hcal-private", type=Path, required=True)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--slot", type=Path, required=True)
    args = p.parse_args()
    # Keep the venv launcher path. Resolving its symlink points at the base
    # framework Python, which has no project-installed numerical packages.
    python = args.python.absolute()
    if not python.is_file():
        raise FileNotFoundError(python)
    print(json.dumps(build(args.repo.resolve(strict=True), args.hcal_private,
                           python, args.slot.resolve(strict=True)),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
