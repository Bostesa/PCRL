"""Build the one effective local prefit lock from frozen sources and role hashes.

Run exactly once after implementer commits and before any Adult scientific unit.
The private potential-unit registry contains absolute admitted paths; the public
lock contains only its SHA. This builder never deserializes any NPZ member.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

from .queue import sha256
from .unit_plan import (build_core_unit_specs, build_endpoint_inference_specs,
                        build_replay_specs)


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        json.dump(value, out, sort_keys=True, separators=(",", ":"))
        out.write("\n")


def build(study: Path, python: Path) -> dict:
    study = study.resolve(strict=True)
    repo = study.parents[1]
    if study.name != "pcrl_capacity_matched_feature_selection_v1":
        raise ValueError("unexpected focused study root")
    lock_path = study / "CAMPAIGN_LOCK.json"
    registry_path = study / ".private/UNIT_SPECS.json"
    if lock_path.exists() or registry_path.exists():
        raise FileExistsError("effective lock or potential registry already exists")
    if not python.is_absolute() or not python.is_file():
        raise ValueError("a concrete absolute Python launcher is required")
    subprocess.run([str(python), "-c", "import psutil,numpy,sklearn,torch"],
                   check=True, timeout=30)
    prior_relative = Path("../pcrl-method-portfolio-v1/results/pcrl_method_portfolio_v1")
    prior = (repo / prior_relative).resolve(strict=True)
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    prior_manifest = prior / "DATA_ACCESS_MANIFEST.json"
    if (len(manifest["role_file_sha256"]) != 25 or
            sha256(prior_manifest) != manifest["portfolio_manifest_sha256"]):
        raise ValueError("admitted role manifest or prior identity differs")
    geometry_path = study / "m1/CAPACITY_SPEC.json"
    geometry = json.loads(geometry_path.read_text())
    resource_path = study / "RESOURCE_PLAN.json"
    resource = json.loads(resource_path.read_text())
    caps = resource["unit_caps"]
    args = dict(worktree=repo, study=study, prior=prior, manifest=manifest,
                source_map_sha=sha256(manifest_path), caps=caps, python=python)
    registry = {}
    for part in (
        build_core_unit_specs(**args, geometry=geometry),
        build_endpoint_inference_specs(**args),
        build_replay_specs(**args, geometry=geometry),
    ):
        if set(registry) & set(part):
            raise ValueError("potential unit IDs collide")
        registry.update(part)
    if len(registry) != 628:
        raise ValueError("frozen potential unit roster is incomplete")
    if any(spec["argv"][0] != str(python) for spec in registry.values()):
        raise ValueError("a unit escaped the admitted Python launcher")
    archive = study / ".private/queue_pre_import_failure_20261009T0012Z/QUEUE.json"
    old = json.loads(archive.read_text())
    original_lock_path = study / "amendments/CAMPAIGN_LOCK_PRE_IMPORT_FAILURE.json"
    original_lock = json.loads(original_lock_path.read_text())
    if (sha256(archive) != "4300bde5ce246d570f2ff6c5050f476d89c4dbb51d8a55d04809c0addc0d14a5" or
            old["units"]["M2-FIT-S0"]["status"] != "FAILED" or
            abs(old["used_cpu_s"] - 589.4801633341704) > 1e-6 or
            sha256(original_lock_path) !=
            "7349da11b2dc228663e7bb8f43996bd34e1e029b47f9feece2586f8a8319cb34"):
        raise ValueError("failed-attempt archive identity differs")
    governed = {}
    source_roots = (study, repo / "results/pcrl_method_portfolio_v1/shared",
                    repo / "results/pcrl_method_portfolio_v1/teams/c")
    for root in source_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"governed source root absent: {root}")
        for path in root.rglob("*.py"):
            if ".private" in path.parts or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(repo).as_posix()
            governed[relative] = sha256(path)
    if len(governed) < 40:
        raise ValueError("governed source roster unexpectedly short")
    file_hashes = {}
    for relative in (
        "COMMON_PROTOCOL.md", "GUARD_AND_CONTROL_REFERENCES.json",
        "RESOURCE_PLAN.json", "AGENT_ROSTER.json", "MASK_BANK.json",
        "m1/CAPACITY_SPEC.json", "m1/DATA_ACCESS_MANIFEST.json",
        "m2/M2_METHOD_SPEC.md", "m3/M3_CONTROL_SPEC.md",
        "m3/M3_INFERENCE_SPEC.md", "EXECUTION_PLAN.md",
        "amendments/AMENDMENT_20261009_INTERPRETER.md",
    ):
        file_hashes[relative] = sha256(study / relative)
    # Finish source/protocol preflight before either one-shot seal artifact is
    # created. A failed check leaves no partial potential registry behind.
    _write_new(registry_path, {
        "schema": "pcrl-focused-presealed-potential-units-v1",
        "unit_count": len(registry), "unit_specs": registry,
    })
    source_tip = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    lock = {
        "schema": "pcrl-focused-capacity-matched-campaign-lock-v1",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE", "adult_fit_permission": True,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_branch": "research/pcrl-capacity-matched-feature-selection-v1",
        "source_tip_before_lock": source_tip,
        "amendment": {
            "id": "2026-10-09-interpreter-launcher",
            "previous_lock_sha256": sha256(study / "amendments/CAMPAIGN_LOCK_PRE_IMPORT_FAILURE.json"),
            "previous_seal_sha256": sha256(study / "amendments/LOCAL_SEAL_RECEIPT_PRE_IMPORT_FAILURE.json"),
            "failed_queue_sha256": sha256(archive),
            "carried_cpu_s": old["used_cpu_s"],
            "intermediate_unreviewed_lock_sha256": sha256(
                study / "amendments/CAMPAIGN_LOCK_PRE_SEED_REPAIR.json"),
            "retry_identity_rule": "lock_sha256 + seal_sha256 + queue_generation + logical_unit_id",
        },
        "prior_portfolio_tip": "38d641189d3e253ddff0ca372d295a034752017f",
        "prior_package_relative_to_worktree": prior_relative.as_posix(),
        "prior_data_manifest_sha256": sha256(prior_manifest),
        "source_map_sha256": sha256(manifest_path),
        "allowed_roles": ["OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT",
                          "ATTACK_FIT_NEW", "INNER_SELECTION"],
        "forbidden_roles": ["HEAD_VALIDATION", "OSF_DEVELOPMENT_ASSESSMENT"],
        "private_unit_registry_sha256": sha256(registry_path),
        "potential_unit_count": len(registry),
        "governed_source_sha256_by_path": governed,
        "governed_protocol_sha256_by_relative_path": file_hashes,
        "screen_seeds": [0, 1], "conditional_finalist_seed": 2,
        "inference": {"bootstrap_seed_hex": original_lock["inference"]["bootstrap_seed_hex"],
                      "replicates": 2048,
                      "family": "fixed-10-rule-207-clause-paired-max-t"},
        "budget": {"epoch_start_utc": resource["epoch_start_utc"],
                   "wall_ceiling_s": 36000, "cpu_ceiling_s": 72000,
                   "host_heavy_limit": 2, "focused_slot_count": 1,
                   "cloud_spend": 0},
        "development_exposure": (
            "Repeatedly used Adult development roles; no new confirmation population."
        ),
    }
    _write_new(lock_path, lock)
    return {"lock_sha256": sha256(lock_path),
            "private_unit_registry_sha256": sha256(registry_path),
            "potential_unit_count": len(registry),
            "governed_source_count": len(governed)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    print(json.dumps(build(Path(args.study), Path(args.python)), sort_keys=True))


if __name__ == "__main__":
    main()
