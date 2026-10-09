"""Bind both full-FIT F packages before the common-role utility gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_MODULES = (
    "results/pcrl_utility_reference_repair_v1/r0/admission.py",
    "results/pcrl_utility_reference_repair_v1/r0/queue_adapter.py",
    "results/pcrl_utility_reference_repair_v1/r1/fit_worker.py",
    "results/pcrl_utility_reference_repair_v1/r1/fit_splits.py",
    "results/pcrl_utility_reference_repair_v1/r1/role_store.py",
    "results/pcrl_utility_reference_repair_v1/r1/source_task.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m0/queue.py",
    "results/pcrl_method_portfolio_v1/teams/c/prototype.py",
    "results/pcrl_method_portfolio_v1/teams/c/training.py",
    "results/pcrl_method_portfolio_v1/shared/common_eval.py",
    "jcv/train.py", "smf/train.py",
)
PROTOCOL_FILES = (
    "EXECUTION_PLAN.md", "RESOURCE_PLAN.json", "AGENT_ROSTER.json",
    "m1/DATA_ACCESS_MANIFEST.json", "REFERENCE_PARITY.json",
    "CONDITIONAL_UNIT_PLAN.json", "RECIPE_SELECTION.json",
    "F_ARTIFACT_LOCK.json", "r1/REFERENCE_PROVENANCE.md",
    "r3/ROLE_OVERLAP.json", "r3/RECIPE_SELECTION_AUDIT.json",
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
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _private_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise PermissionError("F artifact path escapes private root")
    return path


def build(repo: Path, python: Path, slot: Path) -> dict:
    study = repo / "results/pcrl_utility_reference_repair_v1"
    private = study / ".private"
    prior = private / "queue_g3/QUEUE.json"
    state = json.loads(prior.read_text())
    if (set(state["units"]) != {"R1_REFIT_S0", "R1_REFIT_S1"}
            or any(u["status"] != "DONE" for u in state["units"].values())):
        raise PermissionError("selected F refit units incomplete")
    selection_path = study / "RECIPE_SELECTION.json"
    f_lock_path = study / "F_ARTIFACT_LOCK.json"
    selection = json.loads(selection_path.read_text())
    frozen = json.loads(f_lock_path.read_text())
    if (selection.get("status") != "FROZEN"
            or frozen.get("schema") != "pcrl-utility-f-artifact-lock-v1"
            or frozen.get("status") != "FROZEN"
            or frozen.get("recipe_selection_sha256") != sha(selection_path)
            or set(frozen.get("full_refit_output_sha256_by_seed", {})) != {"0", "1"}
            or set(frozen.get("f_package_sha256_by_seed_task", {})) != {"0", "1"}):
        raise PermissionError("two-seed F package lock incomplete")
    if (python.name != "python" or python.parent.name != "bin"
            or python.parent.parent.name != ".venv" or not python.is_file()
            or not slot.is_file()):
        raise FileNotFoundError("project venv or shared slot missing")
    shared_inputs = {str(selection_path): sha(selection_path),
                     str(f_lock_path): sha(f_lock_path)}
    for seed in (0, 1):
        receipt_path = private / "r1/full" / f"seed_{seed}" / "F_PACKAGE_RECEIPT.json"
        expected = frozen["full_refit_output_sha256_by_seed"][str(seed)]
        if sha(receipt_path) != expected:
            raise PermissionError("F receipt hash differs from artifact lock")
        shared_inputs[str(receipt_path)] = expected
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("selected_recipe") != selection["selected_recipe"]
                or receipt.get("selection_lock_sha256") != sha(selection_path)):
            raise PermissionError("F receipt selected recipe differs")
        for task in ("income", "occupation_group"):
            entry = receipt["task_packages"][task]
            path = _private_path(private, entry["relative_path"])
            if (entry["sha256"] != frozen["f_package_sha256_by_seed_task"][str(seed)][task]
                    or sha(path) != entry["sha256"]):
                raise PermissionError("F task package differs from artifact lock")
            shared_inputs[str(path)] = entry["sha256"]
        for entry in receipt["packages"].values():
            if "relative_path" in entry:
                path = _private_path(private, entry["relative_path"])
                if sha(path) != entry["sha256"]:
                    raise PermissionError("F model or calibrator differs")
                shared_inputs[str(path)] = entry["sha256"]
            else:
                directory = _private_path(private, entry["relative_directory"])
                for name, expected_sha in entry["sha256"].items():
                    path = directory / name
                    if sha(path) != expected_sha:
                        raise PermissionError("F source bundle differs")
                    shared_inputs[str(path)] = expected_sha
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    source_map_sha = sha(manifest_path)
    script = study / "r1/fit_worker.py"
    queue = private / "queue_g4"
    lock = study / "CAMPAIGN_LOCK.json"
    specs = {}
    for seed in (0, 1):
        uid = f"R1_GATE_S{seed}"
        inputs = {**shared_inputs, str(manifest_path): source_map_sha}
        role_inputs = {}
        for field in ("features", "income", "occupation", f"references_s{seed}"):
            key = f"INNER_SELECTION:{field}"
            path = (study / manifest["role_file_relative_path"][key]).resolve(strict=True)
            expected = manifest["role_file_sha256"][key]
            if sha(path) != expected:
                raise PermissionError("INNER role input hash differs")
            inputs[str(path)] = expected
            role_inputs[key] = {"path": str(path), "sha256": expected}
        out_dir = private / "r1/gate" / f"seed_{seed}"
        receipt = out_dir / "GATE_RECEIPT.json"
        outputs = [out_dir / "GATE_PREDICTIONS.npz", receipt]
        argv = [str(python), str(script), "--stage", "gate", "--seed", str(seed),
                "--unit-id", uid, "--queue-root", str(queue), "--lock", str(lock),
                "--manifest", str(manifest_path), "--export-root",
                str(private / "role_export_v1"), "--private-root", str(private),
                "--output", str(receipt), "--recipe-selection", str(selection_path),
                "--f-artifact-lock", str(f_lock_path)]
        specs[uid] = {
            "unit_id": uid, "mode": "real", "stage": "r1_gate",
            "argv": argv, "script": str(script), "script_sha256": sha(script),
            "cwd": str(repo), "cpu_reserve_s": 300,
            "wall_limit_s": 300, "max_rss_gib": 3,
            "inputs": inputs, "outputs": [str(x) for x in outputs],
            "deps": [], "source_map_sha256": source_map_sha,
            "role_access": ["INNER_SELECTION"], "role_inputs": role_inputs,
            "allow_child_processes": False,
            "data_access": "INNER_TASK_LABELS_AND_FROZEN_REFERENCES_ONLY",
            "seal_sha256": "$LOCAL_SEAL_SHA256",
        }
    registry = private / "UNIT_SPECS.json"
    write_once(registry, {"schema": "pcrl-utility-unit-registry-v1",
                          "stage": "full_input_common_role_gate", "unit_specs": specs})
    lock_value = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "full_input_common_role_gate",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE", "adult_fit_permission": True,
        "verification_only": False,
        "development_status": "exploratory_reused_Adult_people",
        "paired_seeds": [0, 1], "allowed_roles": ["INNER_SELECTION"],
        "selected_recipe": selection["selected_recipe"],
        "common_role": "INNER_SELECTION",
        "direct_fit_and_CAL_role_overlap_with_common": 0,
        "Ucal_family_selection_exposure": "historical_INNER_selection",
        "utility_caps_inclusive": {
            "accuracy_loss": 0.010, "natural_nll_excess": 0.010,
            "summed_multiclass_brier_excess": 0.005},
        "compare_each_seed_task_against": ["U0", "Ucal"],
        "gate_units": sorted(specs),
        "previous_refit_queue_state_sha256": sha(prior),
        "recipe_selection_sha256": sha(selection_path),
        "f_artifact_lock_sha256": sha(f_lock_path),
        "private_unit_registry_sha256": sha(registry),
        "source_map_sha256": source_map_sha,
        "prior_package_relative_to_worktree": "results/pcrl_utility_reference_repair_v1",
        "governed_source_sha256_by_path": {
            name: sha(repo / name) for name in SOURCE_MODULES},
        "governed_protocol_sha256_by_relative_path": {
            name: sha(study / name) for name in PROTOCOL_FILES},
        "shared_slot_sha256": sha(slot),
        "excluded_inputs": ["SEX labels", "old assessment person arrays",
                            "HEAD_VALIDATION labels"],
        "next_stage": "independent full-input numerical replay before conditional admission",
    }
    write_once(lock, lock_value)
    return {"lock_sha256": sha(lock), "registry_sha256": sha(registry),
            "unit_ids": sorted(specs)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--slot", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(build(args.repo.resolve(strict=True), args.python.absolute(),
                           args.slot.resolve(strict=True)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
