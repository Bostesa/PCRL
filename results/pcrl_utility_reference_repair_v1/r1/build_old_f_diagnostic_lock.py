"""Prepare optional, observational old validation-F same-INNER diagnostic.

R0 chooses the next queue only after privacy-stage closeout and executes this
builder. It never reads the predecessor's mixed-role release archive.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

from results.pcrl_utility_reference_repair_v1.r1.build_utility_interval_lock import (
    STUDY_RELATIVE, _add, _relative_file, _write_once, launcher_path,
)


OLD_TIP = "9e6bfea62e82297d024fd9588a5f4bfb30e0a9de"
OLD_MAP_SHA256 = "01f339d42e220733cca5f5f6eaa4f6ff5d3c04f53961bd54f5bfe5edc4d75096"
TASKS = ("income", "occupation_group")
TERMINAL_STATUSES = {
    "DONE", "FAILED", "FAILED_START", "FAILED_PROBE",
    "FAILED_MEMORY_LIMIT", "FAILED_ORPHAN_CHILD",
}
SOURCE_MODULES = (
    "results/pcrl_utility_reference_repair_v1/r0/admission.py",
    "results/pcrl_utility_reference_repair_v1/r0/queue_adapter.py",
    "results/pcrl_utility_reference_repair_v1/r1/old_f_inner_diagnostic.py",
    "results/pcrl_utility_reference_repair_v1/r1/fit_worker.py",
    "results/pcrl_utility_reference_repair_v1/r1/fit_splits.py",
    "results/pcrl_utility_reference_repair_v1/r1/role_store.py",
    "results/pcrl_utility_reference_repair_v1/r1/source_task.py",
    "results/pcrl_method_portfolio_v1/shared/common_eval.py",
    "results/pcrl_method_portfolio_v1/teams/c/prototype.py",
    "results/pcrl_method_portfolio_v1/teams/c/release_package.py",
    "results/pcrl_method_portfolio_v1/teams/c/training.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m0/queue.py",
)
PROTOCOL_FILES = (
    "EXECUTION_PLAN.md", "RESOURCE_PLAN.json", "AGENT_ROSTER.json",
    "m1/DATA_ACCESS_MANIFEST.json", "FULL_INPUT_GATE.json",
    "F_ARTIFACT_LOCK.json", "r1/OLD_F_DIAGNOSTIC_INPUTS.json",
    "r3/FULL_INPUT_TASK_NUMERICAL_VERIFICATION.json",
)


def sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_prior_queue(state: dict, state_sha: str,
                         generation: int) -> dict:
    """Carry every charged unit from the prior terminal queue, including failures."""
    units = state.get("units", {})
    previous = state.get("carry_forward", {})
    total = state.get("used_cpu_s")
    charges = [row.get("charged_cpu_s") for row in units.values()]
    if (generation < 16 or not units
            or any(row.get("status") not in TERMINAL_STATUSES
                   for row in units.values())
            or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                   or value < 0 for value in charges)
            or not isinstance(total, (int, float)) or not math.isfinite(total)
            or not isinstance(previous.get("charged_cpu_s"), (int, float))
            or not math.isfinite(previous["charged_cpu_s"])
            or abs(total - previous["charged_cpu_s"] - sum(charges)) > 1e-5):
        raise PermissionError("prior queue must be terminal with reconciled CPU ledger")
    return {"archived_queue_relative": f"queue_g{generation}",
            "archived_queue_sha256": state_sha,
            "charged_cpu_s": float(total), "unit_count": len(units)}


def _old_file(old: Path, relative: str, basename: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise PermissionError("old F component path escapes custody")
    path = (old / relative).resolve(strict=True)
    if not path.is_relative_to(old) or path.name != basename or basename == "release.npz":
        raise PermissionError("old F component differs from admitted file")
    return path


def collect_old_components(old: Path, catalog: dict, seed: int,
                           inputs: dict[str, str]) -> None:
    """Bind FIT receipt plus four files per purpose; never scan old directories."""
    old = old.resolve(strict=True)
    record = catalog["seed"][str(seed)]
    if set(record["tasks"]) != set(TASKS):
        raise PermissionError("both old purpose packages required")
    fit_path = _old_file(old, record["fit_receipt_relative_path"], "FIT_RECEIPT.json")
    fit_sha = sha(fit_path)
    if fit_sha != record["fit_receipt_sha256"]:
        raise PermissionError("old FIT receipt hash differs")
    inputs[str(fit_path)] = fit_sha
    fit = json.loads(fit_path.read_text())
    if fit.get("seed") != seed or fit.get("status") != "NO_FEASIBLE_J":
        raise PermissionError("old FIT receipt is not the frozen validation stage")
    components = (
        ("relative_package_path", "package_sha256", "release_package.json"),
        ("relative_head_pool_receipt_path", "head_pool_receipt_sha256", "HEAD_POOL_RECEIPT.json"),
        ("relative_head_path", "head_sha256", "head.npz"),
        ("relative_calibrator_path", "calibrator_sha256", "calibrator.json"),
    )
    for task in TASKS:
        entry = record["tasks"][task]
        inherited = fit["validation_head_packages"][task]["F_FULL83"]
        if (entry.get("stage") != "validation"
                or entry.get("mask_id") != "F_FULL83"
                or inherited.get("stage") != "validation"
                or inherited.get("mask_id") != "F_FULL83"
                or inherited.get("relative_package_path") != entry["relative_package_path"]
                or inherited.get("package_sha256") != entry["package_sha256"]
                or inherited.get("head_pool_receipt_relative_path") !=
                entry["relative_head_pool_receipt_path"]
                or inherited.get("head_pool_receipt_sha256") != entry["head_pool_receipt_sha256"]
                or inherited.get("head_sha256") != entry["head_sha256"]
                or inherited.get("calibrator_sha256") != entry["calibrator_sha256"]):
            raise PermissionError("old F package differs from FIT receipt")
        for path_key, sha_key, basename in components:
            path = _old_file(old, entry[path_key], basename)
            actual = sha(path)
            if actual != entry[sha_key]:
                raise PermissionError("old F component hash differs")
            inputs[str(path)] = actual


def bind_inner_role(study: Path, manifest: dict, field: str,
                    inputs: dict[str, str]) -> dict[str, str]:
    """Copy role SHA metadata without opening its archive before the seal."""
    if field not in ("features", "income", "occupation", "references_s0", "references_s1"):
        raise PermissionError("old F diagnostic role field not admitted")
    key = f"INNER_SELECTION:{field}"
    relative = f".private/role_export_v1/INNER_SELECTION/{field}.npz"
    if manifest["role_file_relative_path"][key] != relative:
        raise PermissionError("old F role path differs")
    path = _relative_file(study, relative)
    if path != (study / relative).absolute():
        raise PermissionError("old F role path symlink differs")
    digest = manifest["role_file_sha256"][key]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise PermissionError("old F role hash malformed")
    inputs[str(path)] = digest
    return {"path": str(path), "sha256": digest}


def build(repo: Path, python: Path, slot: Path, old_private: Path,
          prior_generation: int) -> dict:
    study = repo / STUDY_RELATIVE
    private = study / ".private"
    lock_path = study / "CAMPAIGN_LOCK.json"
    registry_path = private / "UNIT_SPECS.json"
    queue = private / f"queue_g{prior_generation + 1}"
    if lock_path.exists() or registry_path.exists() or queue.exists():
        raise FileExistsError("active seal, registry or next queue still exists")
    python = launcher_path(python)
    if not slot.is_file():
        raise FileNotFoundError("shared slot absent")
    expected_old = (repo.parent / "pcrl-capacity-matched-feature-selection-v1" /
                    "results/pcrl_capacity_matched_feature_selection_v1/.private").resolve(strict=True)
    if old_private.resolve(strict=True) != expected_old:
        raise PermissionError("old F private custody root differs")
    old_repo = expected_old.parents[2]
    tip = subprocess.run(["git", "-C", str(old_repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    if tip != OLD_TIP:
        raise PermissionError("old focused study tip differs")
    prior_path = private / f"queue_g{prior_generation}/QUEUE.json"
    carry = validate_prior_queue(json.loads(prior_path.read_text()), sha(prior_path),
                                 prior_generation)
    gate_path = study / "FULL_INPUT_GATE.json"
    gate = json.loads(gate_path.read_text())
    if (gate.get("schema") != "pcrl-utility-full-input-gate-v1"
            or gate.get("status") not in ("PASS_FULL_INPUT_POINT_GATE",
                                          "FAIL_FULL_INPUT_POINT_GATE")
            or gate.get("common_role") != "INNER_SELECTION"
            or gate.get("r3_numerical_verification_sha256") !=
            sha(study / "r3/FULL_INPUT_TASK_NUMERICAL_VERIFICATION.json")):
        raise PermissionError("completed independently replayed F gate required")
    map_path = study / "r1/OLD_F_DIAGNOSTIC_INPUTS.json"
    if sha(map_path) != OLD_MAP_SHA256:
        raise PermissionError("old F source map differs")
    catalog = json.loads(map_path.read_text())
    if (catalog.get("schema") != "pcrl-utility-old-f-diagnostic-inputs-v1"
            or catalog.get("prior_study_tip") != OLD_TIP
            or set(catalog.get("seed", {})) != {"0", "1"}):
        raise PermissionError("old F catalog identity differs")
    for name, digest in catalog["source_sha256_by_repo_path"].items():
        path = (repo / name).resolve(strict=True)
        if not path.is_relative_to(repo) or sha(path) != digest:
            raise PermissionError("old F deployment source differs")
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "pcrl-utility-repair-role-admission-v1":
        raise PermissionError("role admission manifest differs")
    shared: dict[str, str] = {}
    for path in (gate_path, map_path, manifest_path, prior_path,
                 study / "r3/FULL_INPUT_TASK_NUMERICAL_VERIFICATION.json",
                 study / "F_ARTIFACT_LOCK.json"):
        _add(shared, path)
    uid_by_seed = {}
    specs = {}
    script = study / "r1/old_f_inner_diagnostic.py"
    for seed in (0, 1):
        uid = f"R1_OLD_F_DIAGNOSTIC_S{seed}"
        uid_by_seed[str(seed)] = uid
        inputs = dict(shared)
        collect_old_components(expected_old, catalog, seed, inputs)
        role_inputs = {}
        for field in ("features", "income", "occupation", f"references_s{seed}"):
            key = f"INNER_SELECTION:{field}"
            role_inputs[key] = bind_inner_role(study, manifest, field, inputs)
        folder = private / "r1/old_f_inner" / f"seed_{seed}"
        receipt = folder / "DIAGNOSTIC_RECEIPT.json"
        prediction = folder / "OLD_F_INNER_PREDICTIONS.npz"
        if receipt.exists() or prediction.exists():
            raise FileExistsError("old F diagnostic seed output already exists")
        specs[uid] = {
            "unit_id": uid, "mode": "real", "stage": "r1_old_f_diagnostic",
            "argv": [str(python), str(script), "--unit-id", uid,
                     "--queue-root", str(queue), "--lock", str(lock_path),
                     "--manifest", str(manifest_path),
                     "--export-root", str(private / "role_export_v1"),
                     "--private-root", str(private),
                     "--old-private-root", str(expected_old),
                     "--old-inputs", str(map_path),
                     "--full-input-gate", str(gate_path),
                     "--seed", str(seed), "--output", str(receipt)],
            "script": str(script), "script_sha256": sha(script),
            "cwd": str(repo), "cpu_reserve_s": 300,
            "wall_limit_s": 600, "max_rss_gib": 2,
            "inputs": inputs, "outputs": [str(receipt), str(prediction)],
            "deps": [], "source_map_sha256": sha(manifest_path),
            "role_access": ["INNER_SELECTION"], "role_inputs": role_inputs,
            "allow_child_processes": False,
            "data_access": "INNER_TASK_FEATURE_REFERENCE_OLD_VALIDATION_F_ONLY_NO_SEX",
            "seal_sha256": "$LOCAL_SEAL_SHA256",
        }
    registry = {"schema": "pcrl-utility-unit-registry-v1",
                "stage": "optional_old_validation_f_same_inner_diagnostic",
                "unit_specs": specs}
    lock = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "optional_old_validation_f_same_inner_diagnostic",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "development_status": "observational_reused_Adult_INNER_people",
        "comparison_type": "old_validation_F_vs_new_all_FIT_F_same_INNER",
        "causal_attribution": "NOT_SUPPORTED_UNEQUAL_FIT_EXPOSURE",
        "prior_focused_tip": OLD_TIP,
        "paired_seeds": [0, 1], "allowed_roles": ["INNER_SELECTION"],
        "role_scope": "INNER features/task labels/seed references only; no SEX",
        "full_input_gate_sha256": sha(gate_path),
        "old_f_diagnostic_inputs_sha256": sha(map_path),
        "source_role_row_unit_pair_sha256": gate["common_row_unit_pair_sha256"],
        "prior_queue_generation": prior_generation,
        "prior_queue_state_sha256": sha(prior_path),
        "required_queue_carry_forward": carry,
        "unit_id_by_seed": uid_by_seed,
        "private_unit_registry_sha256": "__AFTER_REGISTRY_WRITE__",
        "source_map_sha256": sha(manifest_path),
        "prior_package_relative_to_worktree": str(STUDY_RELATIVE),
        "governed_source_sha256_by_path": {
            name: sha(repo / name) for name in SOURCE_MODULES},
        "governed_protocol_sha256_by_relative_path": {
            name: sha(study / name) for name in PROTOCOL_FILES},
        "shared_slot_sha256": sha(slot),
        "excluded_inputs": ["SEX labels", "old assessment person arrays",
                            "old mixed-role release.npz", "HEAD_VALIDATION labels"],
        "next_stage": "separate independent R3 same-INNER task replay only if both seeds DONE",
    }
    _write_once(registry_path, registry)
    lock["private_unit_registry_sha256"] = sha(registry_path)
    _write_once(lock_path, lock)
    return {"lock_sha256": sha(lock_path),
            "registry_sha256": sha(registry_path),
            "unit_id_by_seed": uid_by_seed,
            "input_count_by_seed": {uid: len(spec["inputs"]) for uid, spec in specs.items()},
            "required_queue_carry_forward": carry,
            "resource_reserve": {"cpu_s_total": 600, "wall_s_per_unit": 600,
                                 "max_rss_gib_per_unit": 2, "numerical_threads": 1}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--slot", type=Path, required=True)
    parser.add_argument("--old-private-root", type=Path, required=True)
    parser.add_argument("--prior-queue-generation", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo.resolve(strict=True), launcher_path(args.python),
                           args.slot.resolve(strict=True),
                           args.old_private_root.resolve(strict=True),
                           args.prior_queue_generation), sort_keys=True))
