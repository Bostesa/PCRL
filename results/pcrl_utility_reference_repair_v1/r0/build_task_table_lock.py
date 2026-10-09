"""Seal five-mask task-only table/T choice before any protected-label read."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LEADS = ("M_DROP_NATIVE_COUNTRY", "M_DROP_MARITAL_STATUS")
REMAINING = ("M_DROP_EDUCATION_FACT", "M_DROP_RELATIONSHIP", "M_DROP_WORKCLASS")
MASKS = LEADS + REMAINING
SOURCE_MODULES = (
    "results/pcrl_utility_reference_repair_v1/r0/admission.py",
    "results/pcrl_utility_reference_repair_v1/r0/queue_adapter.py",
    "results/pcrl_utility_reference_repair_v1/r1/role_store.py",
    "results/pcrl_utility_reference_repair_v1/r2/admission.py",
    "results/pcrl_utility_reference_repair_v1/r2/frozen_bank.py",
    "results/pcrl_utility_reference_repair_v1/r2/lead_branch.py",
    "results/pcrl_utility_reference_repair_v1/r2/task_table.py",
    "results/pcrl_utility_reference_repair_v1/r2/task_table_worker.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m0/queue.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m2/bank.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m3/task_only.py",
)
PROTOCOL_FILES = (
    "EXECUTION_PLAN.md", "RESOURCE_PLAN.json", "AGENT_ROSTER.json",
    "m1/DATA_ACCESS_MANIFEST.json", "CONDITIONAL_UNIT_PLAN.json",
    "FULL_INPUT_GATE.json", "F_ARTIFACT_LOCK.json",
    "LEAD_BRANCH_RECEIPT.json",
    "r2/CONDITIONAL_METHOD_PLAN.md", "r2/CONDITIONAL_UNIT_PROPOSAL.json",
    "r3/FULL_INPUT_TASK_NUMERICAL_VERIFICATION.json",
    "r3/LEAD_MASK_TASK_NUMERICAL_VERIFICATION.json",
    "r3/BANK_MASK_TASK_NUMERICAL_VERIFICATION.json",
    "LOCK_AMENDMENTS.md",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def add_json(inputs: dict, path: Path) -> str:
    path = path.resolve(strict=True)
    if path.suffix != ".json":
        raise PermissionError("task table inputs must be JSON only")
    digest = sha(path)
    inputs[str(path)] = digest
    return digest


def add_private_json(inputs: dict, private: Path, record: dict) -> Path:
    relative = Path(record["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("task package path escapes private custody")
    path = (private / relative).resolve(strict=True)
    if (not path.is_relative_to(private) or path.suffix != ".json"
            or sha(path) != record["sha256"]):
        raise PermissionError("task package bytes or path differ")
    inputs[str(path)] = record["sha256"]
    return path


def build(repo: Path, python: Path, slot: Path) -> dict:
    study = repo / "results/pcrl_utility_reference_repair_v1"
    private = study / ".private"
    prior = private / "queue_g11/QUEUE.json"
    state = json.loads(prior.read_text())
    if (set(state["units"]) != {"R3_BANK_MASK_TASK_REPLAY"}
            or state["units"]["R3_BANK_MASK_TASK_REPLAY"]["status"] != "DONE"):
        raise PermissionError("independent bank replay incomplete")
    if (python.name != "python" or python.parent.name != "bin"
            or python.parent.parent.name != ".venv" or not python.is_file()
            or not slot.is_file()):
        raise FileNotFoundError("project venv or shared slot missing")
    gate_path = study / "FULL_INPUT_GATE.json"
    gate = json.loads(gate_path.read_text())
    branch_path = study / "LEAD_BRANCH_RECEIPT.json"
    branch = json.loads(branch_path.read_text())
    bank_path = repo / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    if (gate.get("status") != "PASS_FULL_INPUT_POINT_GATE"
            or gate.get("selected_recipe") != "A"
            or branch.get("status") != "LEAD_QUALIFIED"
            or branch.get("full_input_gate_sha256") != sha(gate_path)
            or branch.get("mask_bank_sha256") != sha(bank_path)
            or not set(branch.get("qualified_leads", [])).intersection(LEADS)):
        raise PermissionError("fixed bank task table prerequisite differs")
    independent_path = study / "r3/BANK_MASK_TASK_NUMERICAL_VERIFICATION.json"
    independent = json.loads(independent_path.read_text())
    if (independent.get("schema") !=
            "pcrl-utility-r3-bank-mask-task-numerical-verification-v1"
            or independent.get("status") != "PASS_NUMERICAL_REPLAY"
            or independent.get("private_replay_output_sha256") !=
            sha(private / "r3/BANK_MASK_TASK_REPLAY.json")):
        raise PermissionError("bank task numerical replay not verified")
    bank = json.loads(bank_path.read_text())
    if (set(item["mask_id"] for item in bank["masks"]) != set(MASKS)
            or sha(bank_path) !=
            "a52750191a1e16051367cade59537bc65ed797d69efcb375e0b1cc2e9364722c"):
        raise PermissionError("five-mask frozen bank differs")
    lead_gate_lock_path = private / "stage_seals/lead_gate/CAMPAIGN_LOCK.json"
    bank_gate_lock_path = private / "stage_seals/bank_gate/CAMPAIGN_LOCK.json"
    lead_fit_lock_path = private / "stage_seals/lead_fit/CAMPAIGN_LOCK.json"
    bank_fit_lock_path = private / "stage_seals/bank_fit/CAMPAIGN_LOCK.json"
    mask_gate_locks = {mask: sha(lead_gate_lock_path if mask in LEADS else bank_gate_lock_path)
                       for mask in MASKS}
    mask_fit_locks = {mask: sha(lead_fit_lock_path if mask in LEADS else bank_fit_lock_path)
                      for mask in MASKS}
    f_artifact_lock_path = study / "F_ARTIFACT_LOCK.json"
    r3_full_verification_path = study / "r3/FULL_INPUT_TASK_NUMERICAL_VERIFICATION.json"
    r3_full_replay_path = private / "r3/TASK_GATE_REPLAY.json"
    if (gate.get("F_artifact_lock_sha256") != sha(f_artifact_lock_path)
            or gate.get("r3_numerical_verification_sha256") != sha(r3_full_verification_path)
            or gate.get("r3_numerical_replay_sha256") != sha(r3_full_replay_path)):
        raise PermissionError("F aggregate package/replay lineage differs")
    shared = {}
    for path in (gate_path, f_artifact_lock_path, r3_full_verification_path,
                 r3_full_replay_path, branch_path, bank_path, manifest_path,
                 independent_path,
                 study / "r3/LEAD_MASK_TASK_NUMERICAL_VERIFICATION.json",
                 private / "r3/LEAD_MASK_TASK_REPLAY.json",
                 private / "r3/BANK_MASK_TASK_REPLAY.json",
                 lead_gate_lock_path, bank_gate_lock_path,
                 lead_fit_lock_path, bank_fit_lock_path):
        add_json(shared, path)
    source_lock_path = private / "stage_seals/gate/CAMPAIGN_LOCK.json"
    add_json(shared, source_lock_path)
    source_map_sha = sha(manifest_path)
    script = study / "r2/task_table_worker.py"
    queue = private / "queue_g12"
    lock = study / "CAMPAIGN_LOCK.json"
    specs = {}
    for seed in (0, 1):
        inputs = dict(shared)
        r1_gate = private / "r1/gate" / f"seed_{seed}" / "GATE_RECEIPT.json"
        if sha(r1_gate) != gate["r1_gate_receipt_sha256_by_seed"][str(seed)]:
            raise PermissionError("R1 F gate receipt differs")
        add_json(inputs, r1_gate)
        for mask in MASKS:
            fit_path = private / "masks" / mask / f"seed{seed}" / "MASK_FIT_RECEIPT.json"
            gate_receipt_path = private / "mask_gate" / mask / f"seed{seed}" / "MASK_GATE_RECEIPT.json"
            fit = json.loads(fit_path.read_text())
            task_gate = json.loads(gate_receipt_path.read_text())
            if (fit.get("seed") != seed or fit.get("mask_id") != mask
                    or fit.get("conditional_lock_sha256") != mask_fit_locks[mask]
                    or task_gate.get("seed") != seed or task_gate.get("mask_id") != mask
                    or task_gate.get("conditional_lock_sha256") != mask_gate_locks[mask]
                    or task_gate.get("mask_fit_source_lock_sha256") != mask_fit_locks[mask]
                    or task_gate.get("mask_fit_receipt_sha256") != sha(fit_path)
                    or task_gate.get("full_input_gate_sha256") != sha(gate_path)):
                raise PermissionError("five-mask fit/gate lineage differs")
            add_json(inputs, fit_path)
            add_json(inputs, gate_receipt_path)
            for task in ("income", "occupation_group"):
                add_private_json(inputs, private, fit["task_packages"][task])
        uid = f"R2_TASK_TABLE_S{seed}"
        out_dir = private / "task_table" / f"seed{seed}"
        receipt = out_dir / "TASK_TABLE_RECEIPT.json"
        outputs = [receipt, out_dir / "TASK_ONLY_TABLE.json",
                   out_dir / "T_SELECTION.json"]
        spec = {
            "unit_id": uid, "mode": "real", "stage": "r2_task_table",
            "argv": [str(python), str(script), "--seed", str(seed),
                     "--unit-id", uid, "--queue-root", str(queue),
                     "--lock", str(lock), "--gate", str(gate_path),
                     "--private-root", str(private), "--output", str(receipt)],
            "script": str(script), "script_sha256": sha(script),
            "cwd": str(repo), "cpu_reserve_s": 300,
            "wall_limit_s": 300, "max_rss_gib": 1,
            "inputs": inputs, "outputs": [str(path) for path in outputs],
            "deps": [], "source_map_sha256": source_map_sha,
            "role_access": [], "role_inputs": {},
            "allow_child_processes": False,
            "data_access": "PRIVATE_DERIVED_TASK_AGGREGATES_ONLY",
            "seal_sha256": "$LOCAL_SEAL_SHA256",
        }
        specs[uid] = spec
    registry = private / "UNIT_SPECS.json"
    if registry.exists():
        raise FileExistsError(registry)
    registry.write_text(json.dumps({"schema": "pcrl-utility-unit-registry-v1",
                                    "stage": "pre_sensitive_five_mask_task_table",
                                    "unit_specs": specs}, sort_keys=True, indent=2) + "\n")
    derived_inputs = {}
    for spec in specs.values():
        derived_inputs.update(spec["inputs"])
    value = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "pre_sensitive_five_mask_task_table",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "development_status": "exploratory_reused_Adult_people",
        "selected_recipe": "A", "admitted_mask_ids": list(MASKS),
        "paired_seeds": [0, 1], "allowed_roles": [],
        "role_scope": "PRIVATE_DERIVED_TASK_AGGREGATES_ONLY_NO_ROLE_ARRAYS_NO_SEX",
        "full_input_gate_sha256": sha(gate_path),
        "source_science_lock_sha256": sha(source_lock_path),
        "f_artifact_lock_sha256": sha(f_artifact_lock_path),
        "r3_full_input_numerical_verification_sha256": sha(r3_full_verification_path),
        "r3_full_input_replay_sha256": sha(r3_full_replay_path),
        "mask_bank_sha256": sha(bank_path),
        "common_row_unit_pair_sha256": gate["common_row_unit_pair_sha256"],
        "mask_lock_sha256_by_id": mask_gate_locks,
        "mask_fit_lock_sha256_by_id": mask_fit_locks,
        "lead_branch_receipt_relative_path": "LEAD_BRANCH_RECEIPT.json",
        "lead_branch_receipt_sha256": sha(branch_path),
        "bank_numerical_verification_sha256": sha(independent_path),
        "task_comparators": ["U0", "Ucal", "F"],
        "utility_caps_inclusive": {"accuracy_loss": 0.010,
                                   "natural_nll_excess": 0.010,
                                   "summed_multiclass_brier_excess": 0.005},
        "previous_bank_replay_queue_state_sha256": sha(prior),
        "derived_input_sha256_by_path": derived_inputs,
        "private_unit_registry_sha256": sha(registry),
        "source_map_sha256": source_map_sha,
        "prior_package_relative_to_worktree": "results/pcrl_utility_reference_repair_v1",
        "governed_source_sha256_by_path": {name: sha(repo / name) for name in SOURCE_MODULES},
        "governed_protocol_sha256_by_relative_path": {name: sha(study / name) for name in PROTOCOL_FILES},
        "shared_slot_sha256": sha(slot),
        "excluded_inputs": ["all role person arrays", "SEX labels", "old assessment arrays",
                            "HEAD_VALIDATION labels"],
        "next_stage": "only frozen T and table may admit separate FIT SEX pair search",
    }
    if lock.exists():
        raise FileExistsError(lock)
    lock.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return {"lock_sha256": sha(lock), "registry_sha256": sha(registry),
            "units": sorted(specs), "derived_json_input_count": len(derived_inputs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--slot", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo.resolve(strict=True), args.python, args.slot), sort_keys=True))
