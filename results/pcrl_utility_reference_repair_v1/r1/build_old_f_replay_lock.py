"""Prepare the optional independent R3 replay of old-F same-INNER results.

R0 executes only after both sealed old-F diagnostic seed units finish. This
builder reads derived receipts, not role archives or old assessment arrays.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from results.pcrl_utility_reference_repair_v1.r1.build_utility_interval_lock import (
    STUDY_RELATIVE, _add, _write_once, launcher_path,
)
from results.pcrl_utility_reference_repair_v1.r1.build_old_f_diagnostic_lock import (
    OLD_MAP_SHA256, bind_inner_role, sha,
)


SOURCE_MODULES = (
    "results/pcrl_utility_reference_repair_v1/r0/admission.py",
    "results/pcrl_utility_reference_repair_v1/r0/queue_adapter.py",
    "results/pcrl_utility_reference_repair_v1/r3/independent_task.py",
    "results/pcrl_utility_reference_repair_v1/r3/source_reference.py",
    "results/pcrl_utility_reference_repair_v1/r3/task_replay.py",
    "results/pcrl_utility_reference_repair_v1/r3/old_f_replay_worker.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m0/queue.py",
)
PROTOCOL_FILES = (
    "EXECUTION_PLAN.md", "RESOURCE_PLAN.json", "AGENT_ROSTER.json",
    "m1/DATA_ACCESS_MANIFEST.json", "FULL_INPUT_GATE.json",
    "F_ARTIFACT_LOCK.json", "r1/OLD_F_DIAGNOSTIC_INPUTS.json",
    "r3/FULL_INPUT_TASK_NUMERICAL_VERIFICATION.json",
)


def validate_old_diagnostic_predecessor(
        state: dict, lock: dict, generation: int, state_sha: str,
        output_sha_by_seed: dict[int, dict[str, str]], gate_sha: str,
        old_map_sha: str) -> dict:
    """Require both qN old-F units DONE with exact outputs and CPU carry."""
    expected_ids = {f"R1_OLD_F_DIAGNOSTIC_S{seed}" for seed in (0, 1)}
    units = state.get("units", {})
    previous = state.get("carry_forward", {})
    total = state.get("used_cpu_s")
    costs = [units.get(uid, {}).get("charged_cpu_s") for uid in expected_ids]
    if (generation < 17 or set(units) != expected_ids
            or any(units[f"R1_OLD_F_DIAGNOSTIC_S{seed}"].get("status") != "DONE"
                   or units[f"R1_OLD_F_DIAGNOSTIC_S{seed}"].get("output_sha256")
                   != output_sha_by_seed[seed] for seed in (0, 1))
            or any(not isinstance(cost, (int, float)) or not math.isfinite(cost)
                   or cost < 0 for cost in costs)
            or not isinstance(total, (int, float)) or not math.isfinite(total)
            or not isinstance(previous.get("charged_cpu_s"), (int, float))
            or not math.isfinite(previous["charged_cpu_s"])
            or abs(total - previous["charged_cpu_s"] - sum(costs)) > 1e-5
            or lock.get("schema") != "pcrl-utility-campaign-lock-v1"
            or lock.get("stage") != "optional_old_validation_f_same_inner_diagnostic"
            or lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE"
            or lock.get("required_queue_carry_forward") != previous
            or lock.get("full_input_gate_sha256") != gate_sha
            or lock.get("old_f_diagnostic_inputs_sha256") != old_map_sha):
        raise PermissionError("both sealed old-F diagnostic units required")
    return {"archived_queue_relative": f"queue_g{generation}",
            "archived_queue_sha256": state_sha,
            "charged_cpu_s": float(total), "unit_count": 2}


def build(repo: Path, python: Path, slot: Path,
          prior_generation: int) -> dict:
    study = repo / STUDY_RELATIVE
    private = study / ".private"
    lock_path = study / "CAMPAIGN_LOCK.json"
    registry_path = private / "UNIT_SPECS.json"
    queue = private / f"queue_g{prior_generation + 1}"
    output = private / "r3/OLD_F_INNER_REPLAY.json"
    if (lock_path.exists() or registry_path.exists() or queue.exists()
            or output.exists()):
        raise FileExistsError("active seal, registry, queue or replay output exists")
    python = launcher_path(python)
    if not slot.is_file():
        raise FileNotFoundError("shared slot absent")
    prior_path = private / f"queue_g{prior_generation}/QUEUE.json"
    archive = private / "stage_seals/old_f_diagnostic"
    old_lock_path = archive / "CAMPAIGN_LOCK.json"
    old_registry_path = archive / "UNIT_SPECS.json"
    old_review_path = archive / "PREFIT_REVIEW.json"
    old_seal_path = archive / "LOCAL_SEAL_RECEIPT.json"
    gate_path = study / "FULL_INPUT_GATE.json"
    gate = json.loads(gate_path.read_text())
    old_map_path = study / "r1/OLD_F_DIAGNOSTIC_INPUTS.json"
    if (gate.get("schema") != "pcrl-utility-full-input-gate-v1"
            or gate.get("status") not in ("PASS_FULL_INPUT_POINT_GATE",
                                          "FAIL_FULL_INPUT_POINT_GATE")
            or gate.get("common_role") != "INNER_SELECTION"
            or sha(old_map_path) != OLD_MAP_SHA256):
        raise PermissionError("completed full-input gate or old F map differs")
    gate_sha = sha(gate_path)
    inputs: dict[str, str] = {}
    _add(inputs, gate_path)
    _add(inputs, old_map_path)
    _add(inputs, prior_path)
    for path in (old_lock_path, old_registry_path, old_review_path,
                 old_seal_path, study / "r3/FULL_INPUT_TASK_NUMERICAL_VERIFICATION.json"):
        _add(inputs, path)
    old_lock = json.loads(old_lock_path.read_text())
    review = json.loads(old_review_path.read_text())
    seal = json.loads(old_seal_path.read_text())
    prior = json.loads(prior_path.read_text())
    if (old_lock.get("private_unit_registry_sha256") != sha(old_registry_path)
            or review.get("status") != "PASS"
            or review.get("lock_sha256") != sha(old_lock_path)
            or seal.get("status") != "PASS"
            or seal.get("lock_sha256") != sha(old_lock_path)
            or seal.get("r3_review_sha256") != sha(old_review_path)
            or prior.get("gate", {}).get("seal_sha256") != sha(old_seal_path)):
        raise PermissionError("archived old-F science seal/review differs")
    output_sha_by_seed = {}
    for seed in (0, 1):
        folder = private / "r1/old_f_inner" / f"seed_{seed}"
        receipt_path = folder / "DIAGNOSTIC_RECEIPT.json"
        prediction_path = folder / "OLD_F_INNER_PREDICTIONS.npz"
        receipt_sha = _add(inputs, receipt_path)
        prediction_sha = _add(inputs, prediction_path)
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("schema") != "pcrl-utility-old-validation-f-inner-diagnostic-v1"
                or receipt.get("seed") != seed
                or receipt.get("causal_attribution") != "NOT_SUPPORTED"
                or receipt.get("primary_F_gate_sha256") != gate_sha
                or receipt.get("inner_row_unit_sha256") != gate["common_row_unit_pair_sha256"]
                or receipt.get("predictions_relative_path") !=
                str(prediction_path.relative_to(private))
                or receipt.get("predictions_sha256") != prediction_sha
                or receipt.get("old_fit_rows") != {
                    "income": 11574 if seed == 0 else 11575,
                    "occupation_group": 11574 if seed == 0 else 11575}):
            raise PermissionError("old F seed receipt identity differs")
        output_sha_by_seed[seed] = {str(receipt_path): receipt_sha,
                                    str(prediction_path): prediction_sha}
        new_dir = private / "r1/gate" / f"seed_{seed}"
        _add(inputs, new_dir / "GATE_RECEIPT.json",
             gate["r1_gate_receipt_sha256_by_seed"][str(seed)])
        _add(inputs, new_dir / "GATE_PREDICTIONS.npz",
             gate["r1_gate_prediction_sha256_by_seed"][str(seed)])
    carry = validate_old_diagnostic_predecessor(
        prior, old_lock, prior_generation, sha(prior_path),
        output_sha_by_seed, gate_sha, OLD_MAP_SHA256)
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "pcrl-utility-repair-role-admission-v1":
        raise PermissionError("role admission manifest differs")
    _add(inputs, manifest_path)
    role_inputs = {}
    for field in ("income", "occupation"):
        key = f"INNER_SELECTION:{field}"
        role_inputs[key] = bind_inner_role(study, manifest, field, inputs)
    script = study / "r3/old_f_replay_worker.py"
    uid = "R3_OLD_F_INNER_REPLAY"
    spec = {
        "unit_id": uid, "mode": "real", "stage": "r3_old_f_replay",
        "argv": [str(python), str(script), "--unit-id", uid,
                 "--queue-root", str(queue), "--lock", str(lock_path),
                 "--manifest", str(manifest_path),
                 "--export-root", str(private / "role_export_v1"),
                 "--private-root", str(private),
                 "--full-input-gate", str(gate_path), "--output", str(output)],
        "script": str(script), "script_sha256": sha(script),
        "cwd": str(repo), "cpu_reserve_s": 300,
        "wall_limit_s": 600, "max_rss_gib": 2,
        "inputs": inputs, "outputs": [str(output)], "deps": [],
        "source_map_sha256": sha(manifest_path),
        "role_access": ["INNER_SELECTION"], "role_inputs": role_inputs,
        "allow_child_processes": False,
        "data_access": "INDEPENDENT_OLD_F_SAME_INNER_TASK_REPLAY_NO_SEX",
        "seal_sha256": "$LOCAL_SEAL_SHA256",
    }
    registry = {"schema": "pcrl-utility-unit-registry-v1",
                "stage": "optional_independent_old_f_same_inner_replay",
                "unit_specs": {uid: spec}}
    lock = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "optional_independent_old_f_same_inner_replay",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "development_status": "observational_reused_Adult_INNER_people",
        "comparison_type": "old_validation_F_vs_new_all_FIT_F_same_INNER",
        "causal_attribution": "NOT_SUPPORTED_UNEQUAL_FIT_EXPOSURE",
        "allowed_roles": ["INNER_SELECTION"],
        "role_scope": "INNER task labels and frozen old/new F predictions only; no SEX",
        "full_input_gate_sha256": gate_sha,
        "old_f_diagnostic_inputs_sha256": OLD_MAP_SHA256,
        "prior_old_f_diagnostic_queue_state_sha256": sha(prior_path),
        "prior_old_f_diagnostic_lock_sha256": sha(old_lock_path),
        "required_queue_carry_forward": carry,
        "independent_replay_unit": uid,
        "runner_scoring_or_selection_imported": False,
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
        "next_stage": "publish only observational old/new task contrast after R3 replay",
    }
    _write_once(registry_path, registry)
    lock["private_unit_registry_sha256"] = sha(registry_path)
    _write_once(lock_path, lock)
    return {"lock_sha256": sha(lock_path),
            "registry_sha256": sha(registry_path),
            "unit_id": uid, "input_count": len(inputs),
            "required_queue_carry_forward": carry,
            "resource_reserve": {"cpu_s": 300, "wall_s": 600,
                                 "max_rss_gib": 2, "numerical_threads": 1}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--slot", type=Path, required=True)
    parser.add_argument("--prior-queue-generation", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo.resolve(strict=True), launcher_path(args.python),
                           args.slot.resolve(strict=True), args.prior_queue_generation),
                     sort_keys=True))
