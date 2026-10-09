"""Seal only the two historical task-only deletion leads after F gate PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LEADS = ("M_DROP_NATIVE_COUNTRY", "M_DROP_MARITAL_STATUS")
SEEDS = (0, 1)
SOURCE_MODULES = (
    "results/pcrl_utility_reference_repair_v1/r0/admission.py",
    "results/pcrl_utility_reference_repair_v1/r0/queue_adapter.py",
    "results/pcrl_utility_reference_repair_v1/r1/fit_splits.py",
    "results/pcrl_utility_reference_repair_v1/r1/role_store.py",
    "results/pcrl_utility_reference_repair_v1/r1/source_task.py",
    "results/pcrl_utility_reference_repair_v1/r2/admission.py",
    "results/pcrl_utility_reference_repair_v1/r2/frozen_bank.py",
    "results/pcrl_utility_reference_repair_v1/r2/mask_adapter.py",
    "results/pcrl_utility_reference_repair_v1/r2/mask_inputs.py",
    "results/pcrl_utility_reference_repair_v1/r2/mask_train.py",
    "results/pcrl_utility_reference_repair_v1/r2/mask_fit_worker.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m0/queue.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m2/bank.py",
    "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json",
    "results/pcrl_method_portfolio_v1/teams/c/prototype.py",
    "results/pcrl_method_portfolio_v1/teams/c/training.py",
    "results/pcrl_method_portfolio_v1/shared/common_eval.py",
    "jcv/train.py", "smf/train.py",
)
PROTOCOL_FILES = (
    "EXECUTION_PLAN.md", "RESOURCE_PLAN.json", "AGENT_ROSTER.json",
    "m1/DATA_ACCESS_MANIFEST.json", "REFERENCE_PARITY.json",
    "CONDITIONAL_UNIT_PLAN.json", "RECIPE_SELECTION.json",
    "F_ARTIFACT_LOCK.json", "FULL_INPUT_GATE.json",
    "r2/CONDITIONAL_METHOD_PLAN.md", "r2/CONDITIONAL_UNIT_PROPOSAL.json",
    "r3/FULL_INPUT_TASK_NUMERICAL_VERIFICATION.json",
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


def _private(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise PermissionError("F custody path escapes private root")
    return path


def build(repo: Path, python: Path, slot: Path) -> dict:
    study = repo / "results/pcrl_utility_reference_repair_v1"
    private = study / ".private"
    prior = private / "queue_g5/QUEUE.json"
    state = json.loads(prior.read_text())
    if (set(state["units"]) != {"R3_TASK_GATE_REPLAY"}
            or state["units"]["R3_TASK_GATE_REPLAY"]["status"] != "DONE"):
        raise PermissionError("independent F gate replay incomplete")
    gate_path = study / "FULL_INPUT_GATE.json"
    gate = json.loads(gate_path.read_text())
    if (gate.get("schema") != "pcrl-utility-full-input-gate-v1"
            or gate.get("status") != "PASS_FULL_INPUT_POINT_GATE"
            or gate.get("selected_recipe") not in ("A", "B")
            or set(gate.get("per_seed_task", {})) != {"0", "1"}):
        raise PermissionError("full-input point gate is not eligible")
    for tasks in gate["per_seed_task"].values():
        if set(tasks) != {"income", "occupation_group"} or any(
                set(refs) != {"U0", "Ucal"} or not all(refs.values())
                for refs in tasks.values()):
            raise PermissionError("full-input utility clause missing")
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if gate["common_row_unit_pair_sha256"] != manifest["roles"]["INNER_SELECTION"]["row_unit_pair_sha256"]:
        raise PermissionError("common role pairing differs")
    source_lock_path = private / "stage_seals/gate/CAMPAIGN_LOCK.json"
    if sha(source_lock_path) != gate["source_science_lock_sha256"]:
        raise PermissionError("F source science lock differs")
    bank_path = repo / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
    bank_sha = sha(bank_path)
    if bank_sha != "a52750191a1e16051367cade59537bc65ed797d69efcb375e0b1cc2e9364722c":
        raise PermissionError("frozen five-mask bank differs")
    if (python.name != "python" or python.parent.name != "bin"
            or python.parent.parent.name != ".venv" or not python.is_file()
            or not slot.is_file()):
        raise FileNotFoundError("project venv or shared slot missing")
    source_map_sha = sha(manifest_path)
    common_inputs = {str(gate_path): sha(gate_path),
                     str(manifest_path): source_map_sha,
                     str(source_lock_path): sha(source_lock_path),
                     str(bank_path): bank_sha}
    for seed in ("0", "1"):
        receipt_record = gate["f_receipts"][seed]
        receipt_path = _private(private, receipt_record["relative_path"])
        if sha(receipt_path) != receipt_record["sha256"]:
            raise PermissionError("F receipt differs from full gate")
        common_inputs[str(receipt_path)] = receipt_record["sha256"]
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("selected_recipe") != gate["selected_recipe"]:
            raise PermissionError("F receipt recipe differs")
        for task in ("income", "occupation_group"):
            package_entry = gate["f_packages"][seed][task]
            package_path = _private(private, package_entry["relative_path"])
            if (sha(package_path) != package_entry["sha256"]
                    or receipt["task_packages"][task] != package_entry):
                raise PermissionError("F task package differs")
            common_inputs[str(package_path)] = package_entry["sha256"]
        for entry in receipt["packages"].values():
            if "relative_path" in entry:
                path = _private(private, entry["relative_path"])
                if sha(path) != entry["sha256"]:
                    raise PermissionError("F model/calibrator differs")
                common_inputs[str(path)] = entry["sha256"]
            else:
                directory = _private(private, entry["relative_directory"])
                for name, expected in entry["sha256"].items():
                    path = directory / name
                    if sha(path) != expected:
                        raise PermissionError("F source component differs")
                    common_inputs[str(path)] = expected
        r1_gate = private / "r1/gate" / f"seed_{seed}" / "GATE_RECEIPT.json"
        if sha(r1_gate) != gate["r1_gate_receipt_sha256_by_seed"][seed]:
            raise PermissionError("R1 gate receipt differs")
        common_inputs[str(r1_gate)] = sha(r1_gate)
        r1_predictions = r1_gate.with_name("GATE_PREDICTIONS.npz")
        if sha(r1_predictions) != gate["r1_gate_prediction_sha256_by_seed"][seed]:
            raise PermissionError("R1 gate prediction archive differs")
        common_inputs[str(r1_predictions)] = sha(r1_predictions)
    script = study / "r2/mask_fit_worker.py"
    queue = private / "queue_g6"
    lock = study / "CAMPAIGN_LOCK.json"
    specs = {}
    for mask_id in LEADS:
        for seed in SEEDS:
            uid = f"R2_LEAD_FIT_{mask_id.removeprefix('M_DROP_')}_S{seed}"
            role_inputs = {}
            inputs = dict(common_inputs)
            for role in ("OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT"):
                for field in ("features", "income", "occupation"):
                    key = f"{role}:{field}"
                    path = (study / manifest["role_file_relative_path"][key]).resolve(strict=True)
                    expected = manifest["role_file_sha256"][key]
                    if sha(path) != expected:
                        raise PermissionError("admitted task role differs: " + key)
                    role_inputs[key] = {"path": str(path), "sha256": expected}
                    inputs[str(path)] = expected
            out_dir = private / "masks" / mask_id / f"seed{seed}"
            names = (["fill_inner.npy", "fill_outer.npy", "fill_final.npy"]
                     + ([f"{stage}_{task}_head.npz"
                         for stage in ("outer", "final")
                         for task in ("income", "occupation_group")]
                        if gate["selected_recipe"] == "A" else
                        ["inner_encoder.pt"] +
                        [f"{stage}_source_bundle/{name}"
                         for stage in ("outer", "final")
                         for name in ("encoder.pt", "head_0.joblib", "head_1.joblib")])
                     + [f"{task}_calibrator.json" for task in ("income", "occupation_group")]
                     + [f"FINAL_{task}_PACKAGE.json" for task in ("income", "occupation_group")]
                     + ["OUTER_TASK_METRICS.json", "MASK_FIT_RECEIPT.json"])
            outputs = [out_dir / name for name in names]
            receipt = out_dir / "MASK_FIT_RECEIPT.json"
            argv = [str(python), str(script), "--mask-id", mask_id,
                    "--seed", str(seed), "--unit-id", uid,
                    "--queue-root", str(queue), "--lock", str(lock),
                    "--gate", str(gate_path), "--manifest", str(manifest_path),
                    "--export-root", str(private / "role_export_v1"),
                    "--private-root", str(private), "--output", str(receipt)]
            specs[uid] = {
                "unit_id": uid, "mode": "real", "stage": "r2_mask_fit",
                "argv": argv, "script": str(script), "script_sha256": sha(script),
                "cwd": str(repo), "cpu_reserve_s": 1800,
                "wall_limit_s": 1800, "max_rss_gib": 3,
                "inputs": inputs, "outputs": [str(x) for x in outputs],
                "deps": [], "source_map_sha256": source_map_sha,
                "role_access": ["OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT"],
                "role_inputs": role_inputs, "allow_child_processes": False,
                "data_access": "FIT_AND_CAL_TASK_LABELS_ONLY_NO_SEX",
                "seal_sha256": "$LOCAL_SEAL_SHA256",
            }
    registry = private / "UNIT_SPECS.json"
    write_once(registry, {"schema": "pcrl-utility-unit-registry-v1",
                          "stage": "conditional_two_lead_task_fit",
                          "unit_specs": specs})
    lock_value = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "conditional_two_lead_task_fit",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "development_status": "exploratory_reused_Adult_people",
        "selected_recipe": gate["selected_recipe"],
        "admitted_mask_ids": list(LEADS),
        "paired_seeds": list(SEEDS),
        "allowed_roles": ["OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT"],
        "full_input_gate_sha256": sha(gate_path),
        "source_science_lock_sha256": sha(source_lock_path),
        "common_row_unit_pair_sha256": gate["common_row_unit_pair_sha256"],
        "mask_bank_sha256": bank_sha,
        "task_comparators": ["U0", "Ucal", "F"],
        "utility_caps_inclusive": {"accuracy_loss": 0.010,
                                   "natural_nll_excess": 0.010,
                                   "summed_multiclass_brier_excess": 0.005},
        "role_scope": "FIT/CAL task labels only; no SEX",
        "lead_units": sorted(specs),
        "previous_replay_queue_state_sha256": sha(prior),
        "private_unit_registry_sha256": sha(registry),
        "source_map_sha256": source_map_sha,
        "prior_package_relative_to_worktree": "results/pcrl_utility_reference_repair_v1",
        "governed_source_sha256_by_path": {
            name: sha(repo / name) for name in SOURCE_MODULES},
        "governed_protocol_sha256_by_relative_path": {
            name: sha(study / name) for name in PROTOCOL_FILES},
        "shared_slot_sha256": sha(slot),
        "excluded_inputs": ["SEX labels", "INNER_SELECTION person arrays",
                            "old assessment person arrays", "HEAD_VALIDATION labels"],
        "next_stage": "separate common-role task gate for two historical masks",
    }
    write_once(lock, lock_value)
    return {"lock_sha256": sha(lock), "registry_sha256": sha(registry),
            "unit_count": len(specs), "unit_ids": sorted(specs)}


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
