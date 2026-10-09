"""Seal independent numerical replay of the remaining fixed-bank task gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MASKS = ("M_DROP_EDUCATION_FACT", "M_DROP_RELATIONSHIP",
         "M_DROP_WORKCLASS")
SOURCE_MODULES = (
    "results/pcrl_utility_reference_repair_v1/r0/admission.py",
    "results/pcrl_utility_reference_repair_v1/r0/queue_adapter.py",
    "results/pcrl_utility_reference_repair_v1/r3/mask_task_replay_worker.py",
    "results/pcrl_utility_reference_repair_v1/r3/independent_task.py",
    "results/pcrl_utility_reference_repair_v1/r3/source_reference.py",
    "results/pcrl_utility_reference_repair_v1/r3/task_gate_replay_worker.py",
    "results/pcrl_utility_reference_repair_v1/r3/task_replay.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m0/queue.py",
)
PROTOCOL_FILES = (
    "EXECUTION_PLAN.md", "RESOURCE_PLAN.json", "AGENT_ROSTER.json",
    "m1/DATA_ACCESS_MANIFEST.json", "REFERENCE_PARITY.json",
    "CONDITIONAL_UNIT_PLAN.json", "RECIPE_SELECTION.json",
    "F_ARTIFACT_LOCK.json", "FULL_INPUT_GATE.json",
    "LEAD_BRANCH_RECEIPT.json",
    "r2/CONDITIONAL_METHOD_PLAN.md", "r2/CONDITIONAL_UNIT_PROPOSAL.json",
    "r3/FULL_INPUT_TASK_NUMERICAL_VERIFICATION.json",
    "r3/LEAD_MASK_TASK_NUMERICAL_VERIFICATION.json",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def add_private(inputs: dict, private: Path, record: dict) -> None:
    relative = Path(record["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("private receipt path escapes custody")
    path = (private / relative).resolve(strict=True)
    if not path.is_relative_to(private) or sha(path) != record["sha256"]:
        raise PermissionError("private receipt bytes differ")
    inputs[str(path)] = record["sha256"]


def build(repo: Path, python: Path, slot: Path) -> dict:
    study = repo / "results/pcrl_utility_reference_repair_v1"
    private = study / ".private"
    prior = private / "queue_g10/QUEUE.json"
    state = json.loads(prior.read_text())
    expected_units = {f"R2_BANK_GATE_{mask.removeprefix('M_DROP_')}_S{seed}"
                      for mask in MASKS for seed in (0, 1)}
    if (set(state["units"]) != expected_units
            or any(unit["status"] != "DONE" for unit in state["units"].values())):
        raise PermissionError("remaining-bank common task gates incomplete")
    if (python.name != "python" or python.parent.name != "bin"
            or python.parent.parent.name != ".venv" or not python.is_file()
            or not slot.is_file()):
        raise FileNotFoundError("project venv or shared slot missing")
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    gate_path = study / "FULL_INPUT_GATE.json"
    gate = json.loads(gate_path.read_text())
    bank_path = repo / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
    if gate.get("status") != "PASS_FULL_INPUT_POINT_GATE":
        raise PermissionError("full-input F gate ineligible")
    if sha(bank_path) != "a52750191a1e16051367cade59537bc65ed797d69efcb375e0b1cc2e9364722c":
        raise PermissionError("frozen bank differs")
    source_map_sha = sha(manifest_path)
    bank_gate_lock_path = private / "stage_seals/bank_gate/CAMPAIGN_LOCK.json"
    bank_fit_lock_path = private / "stage_seals/bank_fit/CAMPAIGN_LOCK.json"
    bank_gate_lock_sha = sha(bank_gate_lock_path)
    bank_fit_lock_sha = sha(bank_fit_lock_path)
    role_inputs = {}
    inputs = {str(manifest_path): source_map_sha,
              str(gate_path): sha(gate_path), str(bank_path): sha(bank_path),
              str(bank_gate_lock_path): bank_gate_lock_sha,
              str(bank_fit_lock_path): bank_fit_lock_sha}
    for field in ("features", "income", "occupation", "references_s0", "references_s1"):
        key = f"INNER_SELECTION:{field}"
        path = (study / manifest["role_file_relative_path"][key]).resolve(strict=True)
        expected = manifest["role_file_sha256"][key]
        if sha(path) != expected:
            raise PermissionError("INNER role bytes differ: " + key)
        role_inputs[key] = {"path": str(path), "sha256": expected}
        inputs[str(path)] = expected
    for seed in (0, 1):
        f_dir = private / "r1/gate" / f"seed_{seed}"
        for name, field in (("GATE_RECEIPT.json", "r1_gate_receipt_sha256_by_seed"),
                            ("GATE_PREDICTIONS.npz", "r1_gate_prediction_sha256_by_seed")):
            path = f_dir / name
            expected = gate[field][str(seed)]
            if sha(path) != expected:
                raise PermissionError("R1 F gate component differs")
            inputs[str(path)] = expected
        for mask in MASKS:
            fit_dir = private / "masks" / mask / f"seed{seed}"
            gate_dir = private / "mask_gate" / mask / f"seed{seed}"
            fit_path = fit_dir / "MASK_FIT_RECEIPT.json"
            fit = json.loads(fit_path.read_text())
            mask_gate_path = gate_dir / "MASK_GATE_RECEIPT.json"
            mask_gate = json.loads(mask_gate_path.read_text())
            if (fit.get("mask_id") != mask or fit.get("seed") != seed
                    or mask_gate.get("mask_id") != mask or mask_gate.get("seed") != seed
                    or fit.get("conditional_lock_sha256") != bank_fit_lock_sha
                    or mask_gate.get("mask_fit_receipt_sha256") != sha(fit_path)
                    or mask_gate.get("conditional_lock_sha256") != bank_gate_lock_sha
                    or mask_gate.get("mask_fit_source_lock_sha256") != bank_fit_lock_sha
                    or mask_gate.get("full_input_gate_sha256") != sha(gate_path)
                    or mask_gate.get("R1_gate_receipt_sha256") !=
                    gate["r1_gate_receipt_sha256_by_seed"][str(seed)]):
                raise PermissionError("mask FIT/gate lineage differs")
            inputs[str(fit_path)] = sha(fit_path)
            inputs[str(mask_gate_path)] = sha(mask_gate_path)
            pred_path = gate_dir / "MASK_GATE_PREDICTIONS.npz"
            if sha(pred_path) != mask_gate["prediction_sha256"]:
                raise PermissionError("mask gate vectors differ")
            inputs[str(pred_path)] = sha(pred_path)
            for task in ("income", "occupation_group"):
                for record in (fit["task_packages"][task],
                               fit["model_stages"]["final"][task],
                               fit["calibrators"][task]):
                    add_private(inputs, private, record)
            add_private(inputs, private, fit["fills"]["final"])
    script = study / "r3/mask_task_replay_worker.py"
    uid = "R3_BANK_MASK_TASK_REPLAY"
    queue = private / "queue_g11"
    lock = study / "CAMPAIGN_LOCK.json"
    output = private / "r3/BANK_MASK_TASK_REPLAY.json"
    spec = {
        "unit_id": uid, "mode": "real", "stage": "r3_mask_task_replay",
        "argv": [str(python), str(script), "--unit-id", uid,
                 "--queue-root", str(queue), "--lock", str(lock),
                 "--manifest", str(manifest_path),
                 "--export-root", str(private / "role_export_v1"),
                 "--private-root", str(private),
                 "--full-input-gate", str(gate_path),
                 "--mask-bank", str(bank_path), "--output", str(output)],
        "script": str(script), "script_sha256": sha(script),
        "cwd": str(repo), "cpu_reserve_s": 900,
        "wall_limit_s": 900, "max_rss_gib": 3,
        "inputs": inputs, "outputs": [str(output)], "deps": [],
        "source_map_sha256": source_map_sha,
        "role_access": ["INNER_SELECTION"], "role_inputs": role_inputs,
        "allow_child_processes": False,
        "data_access": "INDEPENDENT_INNER_MASK_TASK_REPLAY_NO_SEX",
        "seal_sha256": "$LOCAL_SEAL_SHA256",
    }
    registry = private / "UNIT_SPECS.json"
    if registry.exists():
        raise FileExistsError(registry)
    registry.write_text(json.dumps({"schema": "pcrl-utility-unit-registry-v1",
                                    "stage": "independent_bank_mask_task_replay",
                                    "unit_specs": {uid: spec}}, sort_keys=True, indent=2) + "\n")
    value = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "independent_bank_mask_task_replay",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "development_status": "exploratory_reused_Adult_people",
        "selected_recipe": gate["selected_recipe"],
        "admitted_mask_ids": list(MASKS), "paired_seeds": [0, 1],
        "allowed_roles": ["INNER_SELECTION"],
        "full_input_gate_sha256": sha(gate_path),
        "mask_bank_sha256": sha(bank_path),
        "common_row_unit_pair_sha256": gate["common_row_unit_pair_sha256"],
        "independent_replay_unit": uid,
        "runner_scoring_or_selection_imported": False,
        "prior_mask_gate_queue_state_sha256": sha(prior),
        "prior_mask_gate_lock_sha256": bank_gate_lock_sha,
        "prior_mask_fit_lock_sha256": bank_fit_lock_sha,
        "private_unit_registry_sha256": sha(registry),
        "source_map_sha256": source_map_sha,
        "prior_package_relative_to_worktree": "results/pcrl_utility_reference_repair_v1",
        "governed_source_sha256_by_path": {name: sha(repo / name) for name in SOURCE_MODULES},
        "governed_protocol_sha256_by_relative_path": {name: sha(study / name) for name in PROTOCOL_FILES},
        "shared_slot_sha256": sha(slot),
        "excluded_inputs": ["SEX labels", "old assessment person arrays", "HEAD_VALIDATION labels"],
        "next_stage": "frozen five-mask task-only table before FIT SEX pair selection",
    }
    if lock.exists():
        raise FileExistsError(lock)
    lock.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return {"lock_sha256": sha(lock), "registry_sha256": sha(registry),
            "input_count": len(inputs), "unit_id": uid}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--slot", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo.resolve(strict=True), args.python, args.slot), sort_keys=True))
