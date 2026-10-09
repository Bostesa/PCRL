"""Create the FIT-only A/B screen's private registry and public stage lock."""

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
    "CONDITIONAL_UNIT_PLAN.json", "r1/REFERENCE_PROVENANCE.md",
    "r3/ROLE_OVERLAP.json", "r3/PREFIT_SOURCE_AND_EXPOSURE_AUDIT.md",
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


def build(repo: Path, python: Path, slot: Path) -> dict:
    study = repo / "results/pcrl_utility_reference_repair_v1"
    private = study / ".private"
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    previous = private / "queue/QUEUE.json"
    old = json.loads(previous.read_text())
    if (len(old["units"]) != 12 or
            any(u["status"] != "DONE" for u in old["units"].values())):
        raise PermissionError("reference parity queue is not complete")
    parity = json.loads((study / "REFERENCE_PARITY.json").read_text())
    if (parity.get("status") != "ENGINEERING_PARITY"
            or parity.get("bitwise_equal_cells") != 48
            or parity.get("queue_state_sha256") != sha(previous)):
        raise PermissionError("frozen U0/Ucal parity prerequisite differs")
    if (python.name != "python" or python.parent.name != "bin"
            or python.parent.parent.name != ".venv" or not python.is_file()):
        raise FileNotFoundError("project venv launcher required")
    if not slot.is_file():
        raise FileNotFoundError("global numerical slot missing")
    script = study / "r1/fit_worker.py"
    queue = private / "queue_g2"
    lock = study / "CAMPAIGN_LOCK.json"
    source_map_sha = sha(manifest_path)
    specs = {}
    for seed in (0, 1):
        uid = f"R1_SCREEN_S{seed}"
        keys = tuple(f"OSF_DEFENSE_FIT:{field}" for field in
                     ("features", "income", "occupation"))
        role_inputs = {}
        inputs = {str(manifest_path): source_map_sha}
        for key in keys:
            role_file = (study / manifest["role_file_relative_path"][key]).resolve(strict=True)
            expected = manifest["role_file_sha256"][key]
            if sha(role_file) != expected:
                raise PermissionError("admitted FIT role file differs: " + key)
            role_inputs[key] = {"path": str(role_file), "sha256": expected}
            inputs[str(role_file)] = expected
        output_dir = private / "r1/screen" / f"seed_{seed}"
        outputs = [output_dir / name for name in (
            "A_income_head.npz", "A_occupation_group_head.npz",
            "B_source_bundle/encoder.pt", "B_source_bundle/head_0.joblib",
            "B_source_bundle/head_1.joblib", "SCREEN_PREDICTIONS.npz",
            "SCREEN_RECEIPT.json",
        )]
        receipt = output_dir / "SCREEN_RECEIPT.json"
        argv = [str(python), str(script), "--stage", "screen", "--seed", str(seed),
                "--unit-id", uid, "--queue-root", str(queue), "--lock", str(lock),
                "--manifest", str(manifest_path), "--export-root",
                str(private / "role_export_v1"), "--private-root", str(private),
                "--output", str(receipt)]
        specs[uid] = {
            "unit_id": uid, "mode": "real", "stage": "r1_screen",
            "argv": argv, "script": str(script), "script_sha256": sha(script),
            "cwd": str(repo), "cpu_reserve_s": 3600,
            "wall_limit_s": 3600, "max_rss_gib": 3,
            "inputs": inputs, "outputs": [str(x) for x in outputs],
            "deps": [], "source_map_sha256": source_map_sha,
            "role_access": ["OSF_DEFENSE_FIT"], "role_inputs": role_inputs,
            "allow_child_processes": False, "data_access": "FIT_TASK_LABELS_ONLY",
            "seal_sha256": "$LOCAL_SEAL_SHA256",
        }
    registry = private / "UNIT_SPECS.json"
    write_once(registry, {"schema": "pcrl-utility-unit-registry-v1",
                          "stage": "fit_only_recipe_screen", "unit_specs": specs})
    lock_value = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "fit_only_recipe_screen", "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "development_status": "exploratory_reused_Adult_people",
        "paired_seeds": [0, 1], "allowed_roles": ["OSF_DEFENSE_FIT"],
        "selection_rule": "lowest_sum_outer_fit_validation_nll_A_tie",
        "screen_probability_convention": "uncalibrated_raw_task_head_probabilities",
        "screen_uses_calibration_labels": False,
        "screen_uses_sex": False,
        "source_B_numeric_C_rule": "nested_outer_FIT_per_seed_task_then_fixed_for_full_refit",
        "source_A_epoch_rule": "fixed_FIT_only_internal_early_stop_algorithm_per_fit",
        "usefulness_accuracy_gain_over_fit_constant_min": 0.010,
        "screen_units": sorted(specs),
        "previous_parity_queue_state_sha256": sha(previous),
        "previous_parity_summary_sha256": sha(study / "REFERENCE_PARITY.json"),
        "private_unit_registry_sha256": sha(registry),
        "source_map_sha256": source_map_sha,
        "prior_package_relative_to_worktree": "results/pcrl_utility_reference_repair_v1",
        "governed_source_sha256_by_path": {
            name: sha(repo / name) for name in SOURCE_MODULES},
        "governed_protocol_sha256_by_relative_path": {
            name: sha(study / name) for name in PROTOCOL_FILES},
        "shared_slot_sha256": sha(slot),
        "excluded_inputs": ["CALIBRATION_HELDOUT labels", "INNER_SELECTION arrays",
                            "SEX labels", "old assessment person arrays",
                            "HEAD_VALIDATION labels"],
        "next_stage": "R0 frozen recipe selection then selected-recipe full-FIT refit",
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
