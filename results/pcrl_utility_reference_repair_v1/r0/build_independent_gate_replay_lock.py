"""Seal independent, role-bound numerical replay of the full-input task gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_MODULES = (
    "results/pcrl_utility_reference_repair_v1/r0/admission.py",
    "results/pcrl_utility_reference_repair_v1/r0/queue_adapter.py",
    "results/pcrl_utility_reference_repair_v1/r3/task_gate_replay_worker.py",
    "results/pcrl_utility_reference_repair_v1/r3/task_replay.py",
    "results/pcrl_utility_reference_repair_v1/r3/independent_task.py",
    "results/pcrl_utility_reference_repair_v1/r3/source_reference.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m0/queue.py",
)
PROTOCOL_FILES = (
    "EXECUTION_PLAN.md", "RESOURCE_PLAN.json", "AGENT_ROSTER.json",
    "m1/DATA_ACCESS_MANIFEST.json", "REFERENCE_PARITY.json",
    "CONDITIONAL_UNIT_PLAN.json", "RECIPE_SELECTION.json",
    "F_ARTIFACT_LOCK.json", "r3/RECIPE_SELECTION_AUDIT.json",
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
    prior = private / "queue_g4/QUEUE.json"
    state = json.loads(prior.read_text())
    if (set(state["units"]) != {"R1_GATE_S0", "R1_GATE_S1"}
            or any(u["status"] != "DONE" for u in state["units"].values())):
        raise PermissionError("both common-role F gate units required")
    if (python.name != "python" or python.parent.name != "bin"
            or python.parent.parent.name != ".venv" or not python.is_file()
            or not slot.is_file()):
        raise FileNotFoundError("project venv or shared slot missing")
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    source_map_sha = sha(manifest_path)
    selection_path = study / "RECIPE_SELECTION.json"
    f_lock_path = study / "F_ARTIFACT_LOCK.json"
    selection = json.loads(selection_path.read_text())
    frozen = json.loads(f_lock_path.read_text())
    if (selection.get("selected_recipe") != frozen.get("selected_recipe")
            or frozen.get("recipe_selection_sha256") != sha(selection_path)):
        raise PermissionError("F artifact/recipe identity differs")
    uid = "R3_TASK_GATE_REPLAY"
    role_inputs = {}
    inputs = {str(manifest_path): source_map_sha,
              str(selection_path): sha(selection_path),
              str(f_lock_path): sha(f_lock_path)}
    for role, fields in (
        ("OSF_DEFENSE_FIT", ("features", "income", "occupation")),
        ("CALIBRATION_HELDOUT", ("features", "income", "occupation")),
        ("INNER_SELECTION", ("features", "income", "occupation",
                             "references_s0", "references_s1")),
    ):
        for field in fields:
            key = f"{role}:{field}"
            path = (study / manifest["role_file_relative_path"][key]).resolve(strict=True)
            expected = manifest["role_file_sha256"][key]
            if sha(path) != expected:
                raise PermissionError("admitted role file differs: " + key)
            inputs[str(path)] = expected
            role_inputs[key] = {"path": str(path), "sha256": expected}
    for seed in (0, 1):
        screen = private / "r1/screen" / f"seed_{seed}"
        full = private / "r1/full" / f"seed_{seed}"
        gate = private / "r1/gate" / f"seed_{seed}"
        screen_receipt_path = screen / "SCREEN_RECEIPT.json"
        screen_receipt = json.loads(screen_receipt_path.read_text())
        if sha(screen_receipt_path) != selection["screen_output_sha256_by_seed"][str(seed)]:
            raise PermissionError("screen receipt differs from recipe selection")
        screen_files = [screen_receipt_path, screen / "SCREEN_PREDICTIONS.npz",
                        screen / "A_income_head.npz",
                        screen / "A_occupation_group_head.npz",
                        screen / "B_source_bundle/encoder.pt",
                        screen / "B_source_bundle/head_0.joblib",
                        screen / "B_source_bundle/head_1.joblib"]
        for path in screen_files:
            inputs[str(path)] = sha(path)
        full_receipt_path = full / "F_PACKAGE_RECEIPT.json"
        full_receipt = json.loads(full_receipt_path.read_text())
        if sha(full_receipt_path) != frozen["full_refit_output_sha256_by_seed"][str(seed)]:
            raise PermissionError("full F receipt differs from artifact lock")
        inputs[str(full_receipt_path)] = sha(full_receipt_path)
        for task in ("income", "occupation_group"):
            entry = full_receipt["task_packages"][task]
            path = (private / entry["relative_path"]).resolve(strict=True)
            if not path.is_relative_to(private) or sha(path) != entry["sha256"]:
                raise PermissionError("F task package differs")
            inputs[str(path)] = entry["sha256"]
        for entry in full_receipt["packages"].values():
            if "relative_path" in entry:
                path = (private / entry["relative_path"]).resolve(strict=True)
                if not path.is_relative_to(private) or sha(path) != entry["sha256"]:
                    raise PermissionError("F component differs")
                inputs[str(path)] = entry["sha256"]
            else:
                directory = (private / entry["relative_directory"]).resolve(strict=True)
                if not directory.is_relative_to(private):
                    raise PermissionError("F bundle escapes private root")
                for name, expected in entry["sha256"].items():
                    path = directory / name
                    if sha(path) != expected:
                        raise PermissionError("F source component differs")
                    inputs[str(path)] = expected
        for name in ("GATE_RECEIPT.json", "GATE_PREDICTIONS.npz"):
            path = gate / name
            inputs[str(path)] = sha(path)
    script = study / "r3/task_gate_replay_worker.py"
    queue = private / "queue_g5"
    lock = study / "CAMPAIGN_LOCK.json"
    output = private / "r3/TASK_GATE_REPLAY.json"
    argv = [str(python), str(script), "--unit-id", uid,
            "--queue-root", str(queue), "--lock", str(lock),
            "--manifest", str(manifest_path),
            "--export-root", str(private / "role_export_v1"),
            "--private-root", str(private),
            "--recipe-selection", str(selection_path),
            "--f-artifact-lock", str(f_lock_path), "--output", str(output)]
    spec = {
        "unit_id": uid, "mode": "real", "stage": "r3_task_gate_replay",
        "argv": argv, "script": str(script), "script_sha256": sha(script),
        "cwd": str(repo), "cpu_reserve_s": 900,
        "wall_limit_s": 900, "max_rss_gib": 3,
        "inputs": inputs, "outputs": [str(output)], "deps": [],
        "source_map_sha256": source_map_sha,
        "role_access": ["OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT",
                        "INNER_SELECTION"], "role_inputs": role_inputs,
        "allow_child_processes": False,
        "data_access": "INDEPENDENT_FIT_CAL_INNER_TASK_REPLAY_NO_SEX",
        "seal_sha256": "$LOCAL_SEAL_SHA256",
    }
    registry = private / "UNIT_SPECS.json"
    write_once(registry, {"schema": "pcrl-utility-unit-registry-v1",
                          "stage": "independent_full_input_task_replay",
                          "unit_specs": {uid: spec}})
    lock_value = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "independent_full_input_task_replay",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "development_status": "exploratory_reused_Adult_people",
        "allowed_roles": ["OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT",
                          "INNER_SELECTION"],
        "independent_replay_unit": uid,
        "runner_code_imported": False,
        "replay_checks": ["FIT-only recipe selection", "constant-task gains",
                          "screen weight restore", "full-FIT F weight restore",
                          "CAL temperature replay", "U0/Ucal/F task metrics",
                          "exact inclusive point gate"],
        "prior_gate_queue_state_sha256": sha(prior),
        "prior_gate_science_lock_sha256": sha(private / "stage_seals/gate/CAMPAIGN_LOCK.json"),
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
        "next_stage": "R0 publishes full-input gate only after R3 numerical PASS",
    }
    write_once(lock, lock_value)
    return {"lock_sha256": sha(lock), "registry_sha256": sha(registry),
            "unit_id": uid, "input_count": len(inputs)}


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
