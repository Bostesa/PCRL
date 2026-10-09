"""Create selected-recipe full-FIT refit and CAL unit registry/lock.

Requires the two sealed FIT-only screen outputs and an R0 recipe selection
that an independent reviewer has checked before this stage's new seal.
"""

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
    "r1/REFERENCE_PROVENANCE.md", "r3/ROLE_OVERLAP.json",
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
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def build(repo: Path, python: Path, slot: Path) -> dict:
    study = repo / "results/pcrl_utility_reference_repair_v1"
    private = study / ".private"
    prior = private / "queue_g2/QUEUE.json"
    state = json.loads(prior.read_text())
    if (set(state["units"]) != {"R1_SCREEN_S0", "R1_SCREEN_S1"}
            or any(u["status"] != "DONE" for u in state["units"].values())):
        raise PermissionError("FIT-only recipe screen units are incomplete")
    selection_path = study / "RECIPE_SELECTION.json"
    selection = json.loads(selection_path.read_text())
    if (selection.get("schema") != "pcrl-utility-recipe-selection-v1"
            or selection.get("status") != "FROZEN"
            or selection.get("rule") != "lowest_sum_outer_fit_validation_nll_A_tie"
            or selection.get("selected_recipe") not in ("A", "B")
            or set(selection.get("screen_output_sha256_by_seed", {})) != {"0", "1"}):
        raise PermissionError("recipe selection is not frozen")
    screen_receipts = {}
    for seed in (0, 1):
        p = private / "r1/screen" / f"seed_{seed}" / "SCREEN_RECEIPT.json"
        if sha(p) != selection["screen_output_sha256_by_seed"][str(seed)]:
            raise PermissionError("recipe selection differs from screen output")
        screen_receipts[seed] = p
    if selection["selected_recipe"] == "B":
        if set(selection.get("selected_C_by_seed_task", {})) != {"0", "1"}:
            raise PermissionError("selected source C values missing")
        for seed in (0, 1):
            receipt = json.loads(screen_receipts[seed].read_text())
            expected = {task: receipt["packages"]["B"]["C_by_task"][task]["selected_C"]
                        for task in ("income", "occupation_group")}
            if selection["selected_C_by_seed_task"][str(seed)] != expected:
                raise PermissionError("frozen source C differs from FIT screen")
    if (python.name != "python" or python.parent.name != "bin"
            or python.parent.parent.name != ".venv" or not python.is_file()
            or not slot.is_file()):
        raise FileNotFoundError("project venv or shared slot missing")
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    source_map_sha = sha(manifest_path)
    script = study / "r1/fit_worker.py"
    queue = private / "queue_g3"
    lock = study / "CAMPAIGN_LOCK.json"
    specs = {}
    for seed in (0, 1):
        uid = f"R1_REFIT_S{seed}"
        inputs = {str(manifest_path): source_map_sha,
                  str(selection_path): sha(selection_path),
                  **{str(p): sha(p) for p in screen_receipts.values()}}
        role_inputs = {}
        for role in ("OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT"):
            for field in ("features", "income", "occupation"):
                key = f"{role}:{field}"
                p = (study / manifest["role_file_relative_path"][key]).resolve(strict=True)
                expected = manifest["role_file_sha256"][key]
                if sha(p) != expected:
                    raise PermissionError("admitted role file hash differs: " + key)
                inputs[str(p)] = expected
                role_inputs[key] = {"path": str(p), "sha256": expected}
        out_dir = private / "r1/full" / f"seed_{seed}"
        model_names = (
            ("A_income_head.npz", "A_occupation_group_head.npz")
            if selection["selected_recipe"] == "A" else
            ("B_source_bundle/encoder.pt", "B_source_bundle/head_0.joblib",
             "B_source_bundle/head_1.joblib")
        )
        names = (*model_names, "income_calibrator.json",
                 "occupation_group_calibrator.json", "F_income_PACKAGE.json",
                 "F_occupation_group_PACKAGE.json", "F_PACKAGE_RECEIPT.json")
        outputs = [out_dir / name for name in names]
        receipt = out_dir / "F_PACKAGE_RECEIPT.json"
        argv = [str(python), str(script), "--stage", "refit", "--seed", str(seed),
                "--unit-id", uid, "--queue-root", str(queue), "--lock", str(lock),
                "--manifest", str(manifest_path), "--export-root",
                str(private / "role_export_v1"), "--private-root", str(private),
                "--output", str(receipt), "--recipe-selection", str(selection_path)]
        specs[uid] = {
            "unit_id": uid, "mode": "real", "stage": "r1_refit",
            "argv": argv, "script": str(script), "script_sha256": sha(script),
            "cwd": str(repo), "cpu_reserve_s": 3600,
            "wall_limit_s": 3600, "max_rss_gib": 3,
            "inputs": inputs, "outputs": [str(x) for x in outputs],
            "deps": [], "source_map_sha256": source_map_sha,
            "role_access": ["OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT"],
            "role_inputs": role_inputs, "allow_child_processes": False,
            "data_access": "FIT_TASK_LABELS_AND_CAL_TASK_LABELS_ONLY",
            "seal_sha256": "$LOCAL_SEAL_SHA256",
        }
    registry = private / "UNIT_SPECS.json"
    write_once(registry, {"schema": "pcrl-utility-unit-registry-v1",
                          "stage": "selected_full_FIT_refit", "unit_specs": specs})
    lock_value = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "selected_full_FIT_refit", "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "development_status": "exploratory_reused_Adult_people",
        "paired_seeds": [0, 1],
        "allowed_roles": ["OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT"],
        "selected_recipe": selection["selected_recipe"],
        "selection_rule": selection["rule"],
        "selected_C_by_seed_task": selection.get("selected_C_by_seed_task", {}),
        "full_fit_rows_expected": 15434, "calibration_rows_expected": 2000,
        "calibration_family": "single_H_GLOBAL_TEMP_per_task_seed",
        "no_common_role_access_before_F_artifact_freeze": True,
        "refit_units": sorted(specs),
        "previous_screen_queue_state_sha256": sha(prior),
        "recipe_selection_sha256": sha(selection_path),
        "private_unit_registry_sha256": sha(registry),
        "source_map_sha256": source_map_sha,
        "prior_package_relative_to_worktree": "results/pcrl_utility_reference_repair_v1",
        "governed_source_sha256_by_path": {
            name: sha(repo / name) for name in SOURCE_MODULES},
        "governed_protocol_sha256_by_relative_path": {
            name: sha(study / name) for name in PROTOCOL_FILES},
        "shared_slot_sha256": sha(slot),
        "excluded_inputs": ["INNER_SELECTION arrays", "SEX labels",
                            "old assessment person arrays", "HEAD_VALIDATION labels"],
        "next_stage": "bind both final F task packages before common-role gate",
    }
    write_once(lock, lock_value)
    return {"lock_sha256": sha(lock), "registry_sha256": sha(registry),
            "selected_recipe": selection["selected_recipe"], "unit_ids": sorted(specs)}


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
