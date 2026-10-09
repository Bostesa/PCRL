"""Seal two FIT-only independent no-go replays after the scientific screen.

This builder reads role-export hashes and archived receipt metadata, never NPZ
members. It replaces the effective queue lock with a verification-only one;
the original scientific lock, seal, queue and registry remain archived.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

from .queue import sha256


ROLE = "OSF_DEFENSE_FIT"
ARCHIVE_QUEUE = ".private/queue_science_no_go_20261009T0025Z/QUEUE.json"
SCIENCE_LOCK = "amendments/CAMPAIGN_LOCK_SCIENCE_NOGO.json"
SCIENCE_SEAL = "amendments/LOCAL_SEAL_RECEIPT_SCIENCE_NOGO.json"


def _write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        json.dump(payload, out, sort_keys=True, separators=(",", ":"))
        out.write("\n")


def build(study: Path, python: Path) -> dict:
    study = study.resolve(strict=True)
    repo = study.parents[1]
    if study.name != "pcrl_capacity_matched_feature_selection_v1":
        raise ValueError("unexpected study root")
    if not python.is_absolute() or not python.is_file():
        raise ValueError("concrete absolute Python launcher required")
    subprocess.run([str(python), "-c", "import psutil,numpy,sklearn,torch"],
                   check=True, timeout=30)
    lock_path = study / "CAMPAIGN_LOCK.json"
    registry_path = study / ".private/UNIT_SPECS.json"
    if lock_path.exists() or registry_path.exists():
        raise FileExistsError("effective queue lock or registry still exists")
    science_lock_path = study / SCIENCE_LOCK
    science_seal_path = study / SCIENCE_SEAL
    archived_queue_path = study / ARCHIVE_QUEUE
    if (sha256(science_lock_path) !=
            "93ed6271f30da39e6010784895d105f77a96a17fe573dcaf5209fb8bc932a6cb" or
            sha256(science_seal_path) !=
            "3db69ec7815f0ab6a4dcf3be0df752783bbb415a652ee65c2fe523d2cae90ef8" or
            sha256(archived_queue_path) !=
            "0a73102b04b06cee065f8a90ef83e9b5559244e9d3c7b206280476c145ddf402"):
        raise ValueError("scientific screen archive identity differs")
    original = json.loads(science_lock_path.read_text())
    old_seal = json.loads(science_seal_path.read_text())
    old_queue = json.loads(archived_queue_path.read_text())
    old_review = study / "amendments/PREFIT_REVIEW_SCIENCE_NOGO.json"
    old_commit_blob = subprocess.check_output([
        "git", "-C", str(repo), "show",
        f"{old_seal['lock_commit_sha']}:{(study / 'CAMPAIGN_LOCK.json').relative_to(repo)}",
    ], timeout=10)
    if (original["inference"]["bootstrap_seed_hex"] !=
            "a93797a57ccb7d091e63b4a2e8609113fe716fca195303dcbeeca730b6620c88" or
            old_seal["lock_sha256"] != sha256(science_lock_path) or
            old_seal["m5_review_sha256"] != sha256(old_review) or
            hashlib.sha256(old_commit_blob).hexdigest() != sha256(science_lock_path) or
            old_queue["epoch_start_utc"] != original["budget"]["epoch_start_utc"] or
            abs(old_queue["used_cpu_s"] - 602.7879501669668) > 1e-6 or
            set(old_queue["units"]) != {"M2-FIT-S0", "M2-FIT-S1"} or
            any(row["status"] != "DONE" for row in old_queue["units"].values())):
        raise ValueError("scientific no-go queue or seed identity differs")
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    prior = (repo / original["prior_package_relative_to_worktree"]).resolve(strict=True)
    if (sha256(manifest_path) != original["source_map_sha256"] or
            sha256(prior / "DATA_ACCESS_MANIFEST.json") !=
            original["prior_data_manifest_sha256"]):
        raise ValueError("admitted role manifest changed")
    script = (study / "m5/replay_task_no_go.py").resolve(strict=True)
    queue_root = study / ".private/queue"
    private = study / ".private"
    units = {}
    by_seed = {}
    for seed in (0, 1):
        fit_id = f"M2-FIT-S{seed}"
        old_row = old_queue["units"][fit_id]
        old_spec = Path(old_row["spec_path"])
        archived_spec = archived_queue_path.parent / "specs" / old_spec.name
        if sha256(archived_spec) != old_row["spec_sha256"]:
            raise ValueError("archived FIT unit spec differs")
        fit_receipt = private / "m2/fit" / f"seed_{seed}" / "FIT_RECEIPT.json"
        fit_sha = old_row["output_sha256"].get(str(fit_receipt))
        if fit_sha is None or sha256(fit_receipt) != fit_sha:
            raise ValueError("completed FIT receipt differs")
        fit = json.loads(fit_receipt.read_text())
        if fit["status"] != "NO_FEASIBLE_J" or fit["seed"] != seed:
            raise ValueError("archived FIT did not reach the frozen no-go")
        by_seed[str(seed)] = {"spec_sha256": old_row["spec_sha256"],
                              "fit_receipt_sha256": fit_sha}
        keys = (f"{ROLE}:features", f"{ROLE}:income", f"{ROLE}:occupation",
                f"{ROLE}:references_s{seed}")
        declared = {}
        inputs = {}
        for key in keys:
            path = (prior / manifest["role_file_relative_path"][key]).resolve(strict=True)
            expected = manifest["role_file_sha256"][key]
            if sha256(path) != expected:
                raise ValueError("admitted FIT role export differs")
            declared[key] = {"path": str(path), "sha256": expected}
            inputs[str(path)] = expected
        uid = f"M5-TASK-NOGO-REPLAY-S{seed}"
        output = private / "m5/task_no_go" / f"seed_{seed}" / "REPORT.json"
        if output.exists():
            raise FileExistsError("replay output already exists")
        units[uid] = {
            "unit_id": uid, "mode": "real", "stage": "independent_task_no_go",
            "data_access": "INDEPENDENT_TASK_NO_GO_ADMITTED_ROLES_ONLY",
            "argv": [str(python), str(script), "--unit-id", uid,
                     "--seed", str(seed), "--queue-root", str(queue_root),
                     "--output-report", str(output)],
            "cwd": str(repo), "script": str(script), "script_sha256": sha256(script),
            "inputs": inputs, "role_inputs": declared, "role_access": [ROLE],
            "outputs": [str(output)], "deps": [],
            "source_map_sha256": sha256(manifest_path),
            "seal_sha256": "$LOCAL_SEAL_SHA256",
            "allow_child_processes": False, "cpu_reserve_s": 120,
            "wall_limit_s": 120, "max_rss_gib": 2,
        }
    governed = {}
    for root in (study, repo / "results/pcrl_method_portfolio_v1/shared",
                 repo / "results/pcrl_method_portfolio_v1/teams/c"):
        for path in root.rglob("*.py"):
            if ".private" in path.parts or "__pycache__" in path.parts:
                continue
            governed[path.relative_to(repo).as_posix()] = sha256(path)
    protocol = dict(original["governed_protocol_sha256_by_relative_path"])
    for relative, expected in list(protocol.items()):
        if sha256(study / relative) != expected:
            raise ValueError("original governed protocol changed: " + relative)
    protocol["amendments/POSTFIT_VERIFICATION.md"] = sha256(
        study / "amendments/POSTFIT_VERIFICATION.md")
    historical_fit = {
        "lock_sha256": sha256(science_lock_path),
        "lock_commit_sha": old_seal["lock_commit_sha"],
        "archived_lock_relative_path": SCIENCE_LOCK,
        "local_seal_sha256": sha256(science_seal_path),
        "archived_seal_relative_path": SCIENCE_SEAL,
        "archived_queue_relative_path": ARCHIVE_QUEUE,
        "archived_queue_sha256": sha256(archived_queue_path),
        "by_seed": by_seed,
    }
    _write_new(registry_path, {"schema": "pcrl-focused-postfit-verification-units-v1",
                               "unit_count": 2, "unit_specs": units})
    os.chmod(registry_path, 0o600)
    source_tip = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    lock = dict(original)
    lock.update(
        schema="pcrl-focused-postfit-verification-lock-v1",
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_tip_before_lock=source_tip,
        adult_fit_permission=False, verification_only=True,
        verification_stage="independent_task_no_go",
        allowed_roles=[ROLE],
        private_unit_registry_sha256=sha256(registry_path),
        potential_unit_count=2,
        governed_source_sha256_by_path=governed,
        governed_protocol_sha256_by_relative_path=protocol,
        historical_fit=historical_fit,
    )
    _write_new(lock_path, lock)
    return {"lock_sha256": sha256(lock_path),
            "private_unit_registry_sha256": sha256(registry_path),
            "source_count": len(governed), "unit_count": 2,
            "carried_cpu_s": old_queue["used_cpu_s"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    print(json.dumps(build(Path(args.study), Path(args.python)), sort_keys=True))


if __name__ == "__main__":
    main()
