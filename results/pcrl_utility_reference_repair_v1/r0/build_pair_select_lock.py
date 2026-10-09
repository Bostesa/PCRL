"""Seal the fixed FIT SEX proxy search after independent task-only T audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MASKS = ("M_DROP_NATIVE_COUNTRY", "M_DROP_MARITAL_STATUS",
         "M_DROP_EDUCATION_FACT", "M_DROP_RELATIONSHIP", "M_DROP_WORKCLASS")
SOURCE_MODULES = (
    "results/pcrl_utility_reference_repair_v1/r0/admission.py",
    "results/pcrl_utility_reference_repair_v1/r0/queue_adapter.py",
    "results/pcrl_utility_reference_repair_v1/r1/fit_splits.py",
    "results/pcrl_utility_reference_repair_v1/r1/role_store.py",
    "results/pcrl_utility_reference_repair_v1/r2/admission.py",
    "results/pcrl_utility_reference_repair_v1/r2/frozen_bank.py",
    "results/pcrl_utility_reference_repair_v1/r2/privacy_store.py",
    "results/pcrl_utility_reference_repair_v1/r2/raw_search.py",
    "results/pcrl_utility_reference_repair_v1/r2/pair_selection.py",
    "results/pcrl_utility_reference_repair_v1/r2/pair_select_worker.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m0/queue.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m2/bank.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m2/table.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m2/joint.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m3/task_only.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m3/control_rules.py",
)
PROTOCOL_FILES = (
    "EXECUTION_PLAN.md", "RESOURCE_PLAN.json", "AGENT_ROSTER.json",
    "m1/DATA_ACCESS_MANIFEST.json", "CONDITIONAL_UNIT_PLAN.json",
    "FULL_INPUT_GATE.json", "LEAD_BRANCH_RECEIPT.json",
    "r2/CONDITIONAL_METHOD_PLAN.md", "r2/CONDITIONAL_UNIT_PROPOSAL.json",
    "r2/R2_RESOURCE_FORECAST.json", "r2/RAW_UNION_ACCOUNTING_AMENDMENT.md",
    "r3/FULL_INPUT_TASK_NUMERICAL_VERIFICATION.json",
    "r3/LEAD_MASK_TASK_NUMERICAL_VERIFICATION.json",
    "r3/BANK_MASK_TASK_NUMERICAL_VERIFICATION.json",
    "r3/TASK_TABLE_SELECTION_AUDIT.json",
    "r3/TASK_TABLE_AUDIT_CORRECTION.md",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def add(inputs: dict, path: Path, expected: str | None = None) -> None:
    path = path.resolve(strict=True)
    actual = sha(path)
    if expected is not None and actual != expected:
        raise PermissionError("sealed predecessor bytes differ: " + str(path))
    inputs[str(path)] = actual


def add_private(inputs: dict, private: Path, record: dict) -> None:
    relative = Path(record["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("private package path escapes custody")
    path = (private / relative).resolve(strict=True)
    if not path.is_relative_to(private):
        raise PermissionError("private package escapes custody")
    add(inputs, path, record["sha256"])


def build(repo: Path, python: Path, slot: Path) -> dict:
    study = repo / "results/pcrl_utility_reference_repair_v1"
    private = study / ".private"
    prior = private / "queue_g12/QUEUE.json"
    state = json.loads(prior.read_text())
    if (set(state["units"]) != {"R2_TASK_TABLE_S0", "R2_TASK_TABLE_S1"}
            or any(unit["status"] != "DONE" for unit in state["units"].values())):
        raise PermissionError("both task-only T units incomplete")
    if (python.name != "python" or python.parent.name != "bin"
            or python.parent.parent.name != ".venv" or not python.is_file()
            or not slot.is_file()):
        raise FileNotFoundError("project venv or shared slot missing")
    gate_path = study / "FULL_INPUT_GATE.json"
    gate = json.loads(gate_path.read_text())
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    bank_path = repo / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
    branch_path = study / "LEAD_BRANCH_RECEIPT.json"
    audit_path = study / "r3/TASK_TABLE_SELECTION_AUDIT.json"
    audit = json.loads(audit_path.read_text())
    forecast_path = study / "r2/R2_RESOURCE_FORECAST.json"
    forecast = json.loads(forecast_path.read_text())
    if (gate.get("status") != "PASS_FULL_INPUT_POINT_GATE"
            or gate.get("selected_recipe") != "A"
            or json.loads(branch_path.read_text()).get("status") != "LEAD_QUALIFIED"
            or audit.get("schema") != "pcrl-utility-r3-task-table-selection-audit-v1"
            or audit.get("status") != "PASS_AGGREGATE_SELECTION_REPLAY"
            or audit.get("source_queue_state_sha256") != sha(prior)
            or audit.get("full_input_gate_sha256") != sha(gate_path)
            or audit.get("mask_bank_sha256") != sha(bank_path)
            or forecast.get("status") != "PROSPECTIVE_BEFORE_FIT_SEX_PAIR_SEARCH"
            or forecast.get("forecast_cpu_seconds", {}).get("twofold_stress_total", 10**9)
            > forecast.get("remaining_cpu_seconds_asof", 0)
            or forecast.get("forecast_cpu_seconds", {}).get("twofold_stress_total", 10**9)
            > forecast.get("remaining_wall_seconds_asof", 0)):
        raise PermissionError("task-only T audit or downstream resource reserve differs")
    source_lock_path = private / "stage_seals/gate/CAMPAIGN_LOCK.json"
    table_lock_path = private / "stage_seals/task_table/CAMPAIGN_LOCK.json"
    if (sha(source_lock_path) != gate["source_science_lock_sha256"]
            or sha(table_lock_path) != audit["source_lock_sha256"]):
        raise PermissionError("F/task table source lock differs")
    common_inputs = {}
    for path in (gate_path, manifest_path, bank_path, branch_path,
                 audit_path, forecast_path, prior, source_lock_path,
                 table_lock_path, study / "F_ARTIFACT_LOCK.json",
                 study / "r3/FULL_INPUT_TASK_NUMERICAL_VERIFICATION.json",
                 study / "r3/LEAD_MASK_TASK_NUMERICAL_VERIFICATION.json",
                 study / "r3/BANK_MASK_TASK_NUMERICAL_VERIFICATION.json"):
        add(common_inputs, path)
    # Deep F admission by the pair worker opens every frozen F package and
    # both R1 common-role prediction archives before FIT SEX is considered.
    for seed in (0, 1):
        key = str(seed)
        r1_dir = private / "r1/gate" / f"seed_{seed}"
        add(common_inputs, r1_dir / "GATE_RECEIPT.json",
            gate["r1_gate_receipt_sha256_by_seed"][key])
        add(common_inputs, r1_dir / "GATE_PREDICTIONS.npz",
            gate["r1_gate_prediction_sha256_by_seed"][key])
        add_private(common_inputs, private, gate["f_receipts"][key])
        fit_receipt = json.loads(
            (private / gate["f_receipts"][key]["relative_path"]).read_text())
        for task in ("income", "occupation_group"):
            add_private(common_inputs, private, gate["f_packages"][key][task])
        for record in fit_receipt["packages"].values():
            if "relative_path" in record:
                add_private(common_inputs, private, record)
            else:
                directory = (private / record["relative_directory"]).resolve(strict=True)
                if not directory.is_relative_to(private):
                    raise PermissionError("F source bundle escapes custody")
                for filename, digest in record["sha256"].items():
                    add(common_inputs, directory / filename, digest)
    script = study / "r2/pair_select_worker.py"
    queue = private / "queue_g13"
    lock = study / "CAMPAIGN_LOCK.json"
    specs = {}
    for seed in (0, 1):
        key = str(seed)
        task_dir = private / "task_table" / f"seed{seed}"
        task_receipt_path = task_dir / "TASK_TABLE_RECEIPT.json"
        receipt = json.loads(task_receipt_path.read_text())
        if (receipt.get("status") != "TASK_ONLY_T_FROZEN"
                or receipt.get("conditional_lock_sha256") != sha(table_lock_path)
                or receipt.get("T_pair_id") != audit["by_seed"][key]["T_pair_id"]
                or sha(task_receipt_path) != audit["by_seed"][key]["receipt_sha256"]):
            raise PermissionError("pre-sensitive T identity differs")
        inputs = dict(common_inputs)
        add(inputs, task_receipt_path, audit["by_seed"][key]["receipt_sha256"])
        add(inputs, task_dir / "TASK_ONLY_TABLE.json",
            audit["by_seed"][key]["table_sha256"])
        add(inputs, task_dir / "T_SELECTION.json",
            audit["by_seed"][key]["T_selection_sha256"])
        for mask in MASKS:
            add(inputs, private / "mask_gate" / mask / f"seed{seed}" /
                "MASK_GATE_RECEIPT.json")
        role_inputs = {}
        for field in ("features", "income", "occupation", "sex"):
            role_key = f"OSF_DEFENSE_FIT:{field}"
            path = (study / manifest["role_file_relative_path"][role_key]).resolve(strict=True)
            digest = manifest["role_file_sha256"][role_key]
            add(inputs, path, digest)
            role_inputs[role_key] = {"path": str(path), "sha256": digest}
        uid = f"R2_PAIR_SELECT_S{seed}"
        out_dir = private / "pair_select" / f"seed{seed}"
        receipt_path = out_dir / "PAIR_SELECT_RECEIPT.json"
        outputs = [receipt_path] + [out_dir / name for name in
                    ("PAIR_TABLE.json", "JOINT_SELECTION.json",
                     "CONTROL_PLAN.json", "RAW_SEARCH_RECEIPT.json")]
        specs[uid] = {
            "unit_id": uid, "mode": "real", "stage": "r2_pair_select",
            "argv": [str(python), str(script), "--seed", str(seed),
                     "--unit-id", uid, "--queue-root", str(queue),
                     "--lock", str(lock), "--gate", str(gate_path),
                     "--manifest", str(manifest_path),
                     "--export-root", str(private / "role_export_v1"),
                     "--private-root", str(private), "--output", str(receipt_path)],
            "script": str(script), "script_sha256": sha(script),
            "cwd": str(repo), "cpu_reserve_s": 5400,
            "wall_limit_s": 5400, "max_rss_gib": 4,
            "inputs": inputs, "outputs": [str(path) for path in outputs],
            "deps": [], "source_map_sha256": sha(manifest_path),
            "role_access": ["OSF_DEFENSE_FIT"], "role_inputs": role_inputs,
            "allow_child_processes": False,
            "data_access": "FIT_TASK_AND_SEX_FOR_FIXED_PAIR_SEARCH_ONLY",
            "seal_sha256": "$LOCAL_SEAL_SHA256",
        }
    registry = private / "UNIT_SPECS.json"
    if registry.exists():
        raise FileExistsError(registry)
    registry.write_text(json.dumps({"schema": "pcrl-utility-unit-registry-v1",
                                    "stage": "fixed_five_mask_pair_search",
                                    "unit_specs": specs}, sort_keys=True, indent=2) + "\n")
    value = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "fixed_five_mask_pair_search",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "development_status": "exploratory_reused_Adult_people",
        "selected_recipe": "A", "admitted_mask_ids": list(MASKS),
        "paired_seeds": [0, 1], "allowed_roles": ["OSF_DEFENSE_FIT"],
        "role_scope": "FIT features/income/occupation/SEX only; no ATT or INNER SEX",
        "full_input_gate_sha256": sha(gate_path),
        "source_science_lock_sha256": sha(source_lock_path),
        "common_row_unit_pair_sha256": gate["common_row_unit_pair_sha256"],
        "mask_bank_sha256": sha(bank_path),
        "task_table_source_lock_sha256": sha(table_lock_path),
        "task_table_selection_audit_sha256": sha(audit_path),
        "lead_branch_receipt_sha256": sha(branch_path),
        "resource_forecast_sha256": sha(forecast_path),
        "raw_search_reader_ids": ["LR_C1", "HGB_15x64"],
        "raw_search_fits_per_seed": 22,
        "selection_rule_ids": ["J", "T", "I", "S12", "S21", "SHARED",
                               "T_U", "I_U", "S12_U", "S21_U"],
        "utility_caps_inclusive": {"accuracy_loss": 0.010,
                                   "natural_nll_excess": 0.010,
                                   "summed_multiclass_brier_excess": 0.005},
        "previous_task_table_queue_state_sha256": sha(prior),
        "private_unit_registry_sha256": sha(registry),
        "source_map_sha256": sha(manifest_path),
        "prior_package_relative_to_worktree": "results/pcrl_utility_reference_repair_v1",
        "governed_source_sha256_by_path": {name: sha(repo / name) for name in SOURCE_MODULES},
        "governed_protocol_sha256_by_relative_path": {name: sha(study / name) for name in PROTOCOL_FILES},
        "shared_slot_sha256": sha(slot),
        "excluded_inputs": ["ATTACK_FIT_NEW labels", "INNER_SELECTION SEX",
                            "old assessment person arrays", "HEAD_VALIDATION labels"],
        "next_stage": "freeze selected pairs and full matched release/reader unit inventory",
    }
    if lock.exists():
        raise FileExistsError(lock)
    lock.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return {"lock_sha256": sha(lock), "registry_sha256": sha(registry),
            "units": sorted(specs), "input_count_by_unit":
            {uid: len(spec["inputs"]) for uid, spec in specs.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--slot", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo.resolve(strict=True), args.python, args.slot), sort_keys=True))
