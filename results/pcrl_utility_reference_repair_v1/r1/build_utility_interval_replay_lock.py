"""Prepare R3's independently sealed interval replay after q14 finishes.

R0 executes this builder; R3 reviews the resulting lock and runs the worker.
No role archive bytes are read while this preseal registry is constructed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from results.pcrl_utility_reference_repair_v1.r1.build_utility_interval_lock import (
    PROPOSAL_SHA256, STUDY_RELATIVE, _add, _relative_file, _write_once,
    bind_inner_task_role, launcher_path, sha,
)


SOURCE_MODULES = (
    "results/pcrl_utility_reference_repair_v1/r0/admission.py",
    "results/pcrl_utility_reference_repair_v1/r0/queue_adapter.py",
    "results/pcrl_utility_reference_repair_v1/r3/independent_task.py",
    "results/pcrl_utility_reference_repair_v1/r3/source_reference.py",
    "results/pcrl_utility_reference_repair_v1/r3/task_gate_replay_worker.py",
    "results/pcrl_utility_reference_repair_v1/r3/task_replay.py",
    "results/pcrl_utility_reference_repair_v1/r3/utility_interval_replay_worker.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m0/queue.py",
)
PROTOCOL_FILES = (
    "EXECUTION_PLAN.md", "RESOURCE_PLAN.json", "AGENT_ROSTER.json",
    "m1/DATA_ACCESS_MANIFEST.json", "FULL_INPUT_GATE.json",
    "F_ARTIFACT_LOCK.json", "r1/UTILITY_INTERVAL_PROTOCOL.md",
    "r1/INTERVAL_UNIT_PROPOSAL.json", "r1/INTERVAL_ACCESS_NOTE_20261009.md",
    "r3/FULL_INPUT_TASK_NUMERICAL_VERIFICATION.json",
)


def validate_interval_predecessor(state: dict, lock: dict, state_sha: str,
                                  output_path: str,
                                  output_sha: str, gate_sha: str,
                                  proposal_sha: str, manifest_sha: str) -> dict:
    """Require q14 DONE, its single frozen output, and exact CPU carry."""
    units = state.get("units", {})
    record = units.get("R1_UTILITY_INTERVALS", {})
    previous = state.get("carry_forward", {})
    charged = record.get("charged_cpu_s")
    total = state.get("used_cpu_s")
    if (set(units) != {"R1_UTILITY_INTERVALS"}
            or record.get("status") != "DONE"
            or record.get("output_sha256") != {output_path: output_sha}
            or not isinstance(charged, (int, float)) or not math.isfinite(charged)
            or charged < 0
            or not isinstance(total, (int, float)) or not math.isfinite(total)
            or not isinstance(previous.get("charged_cpu_s"), (int, float))
            or not math.isfinite(previous["charged_cpu_s"])
            or previous.get("archived_queue_relative") != "queue_g13"
            or previous.get("unit_count") != 2
            or abs(total - previous["charged_cpu_s"] - charged) > 1e-5
            or lock.get("schema") != "pcrl-utility-campaign-lock-v1"
            or lock.get("stage") != "r1_full_input_utility_intervals"
            or lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE"
            or lock.get("required_queue_carry_forward") != previous
            or lock.get("full_input_gate_sha256") != gate_sha
            or lock.get("interval_unit_proposal_sha256") != proposal_sha
            or lock.get("source_map_sha256") != manifest_sha):
        raise PermissionError("completed q14 interval or carry-forward differs")
    return {"archived_queue_relative": "queue_g14",
            "archived_queue_sha256": state_sha,
            "charged_cpu_s": float(total), "unit_count": 1}


def build(repo: Path, python: Path, slot: Path) -> dict:
    study = repo / STUDY_RELATIVE
    private = study / ".private"
    active_lock_path = study / "CAMPAIGN_LOCK.json"
    registry_path = private / "UNIT_SPECS.json"
    if active_lock_path.exists() or registry_path.exists():
        raise FileExistsError("q14 active lock/registry must be archived by R0")
    python = launcher_path(python)
    if not slot.is_file():
        raise FileNotFoundError("shared slot missing")
    proposal_path = study / "r1/INTERVAL_UNIT_PROPOSAL.json"
    if sha(proposal_path) != PROPOSAL_SHA256:
        raise PermissionError("reviewed interval proposal differs")
    proposal = json.loads(proposal_path.read_text())
    unit = proposal["r3_replay_unit"]
    if (proposal.get("status") != "PROPOSED_UNRUN"
            or unit.get("unit_id") != "R3_UTILITY_INTERVAL_REPLAY"
            or unit.get("stage") != "r3_utility_interval_replay"
            or unit.get("additional_input") !=
            ".private/r1/intervals/SIMULTANEOUS_UTILITY_INTERVALS.json"
            or unit.get("output") != ".private/r3/UTILITY_INTERVAL_REPLAY.json"
            or unit.get("role_access") != ["INNER_SELECTION:income", "INNER_SELECTION:occupation"]
            or unit.get("cpu_reserve_s") != 300
            or unit.get("wall_limit_s") != 600
            or unit.get("max_rss_gib") != 1
            or unit.get("numerical_threads") != 1
            or unit.get("allow_child_processes") is not False
            or unit.get("requires_post_r1_output_sha256_lock") is not True):
        raise PermissionError("R3 replay differs from reviewed unit proposal")
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "pcrl-utility-repair-role-admission-v1":
        raise PermissionError("role admission manifest differs")
    gate_path = study / "FULL_INPUT_GATE.json"
    gate = json.loads(gate_path.read_text())
    if (gate.get("status") != "PASS_FULL_INPUT_POINT_GATE"
            or gate.get("selected_recipe") != "A"
            or gate.get("common_role") != "INNER_SELECTION"):
        raise PermissionError("selected F point gate differs")
    old_queue = private / "queue_g14/QUEUE.json"
    archive = private / "stage_seals/utility_intervals"
    archived_lock_path = archive / "CAMPAIGN_LOCK.json"
    archived_registry_path = archive / "UNIT_SPECS.json"
    archived_seal_path = archive / "LOCAL_SEAL_RECEIPT.json"
    archived_review_path = archive / "PREFIT_REVIEW.json"
    original_path = study / unit["additional_input"]
    original_sha = sha(original_path)
    prior = json.loads(old_queue.read_text())
    old_lock = json.loads(archived_lock_path.read_text())
    carry = validate_interval_predecessor(
        prior, old_lock, sha(old_queue),
        str(original_path), original_sha, sha(gate_path),
        sha(proposal_path), sha(manifest_path))
    archived_seal = json.loads(archived_seal_path.read_text())
    archived_review = json.loads(archived_review_path.read_text())
    if (old_lock.get("private_unit_registry_sha256") != sha(archived_registry_path)
            or archived_seal.get("status") != "PASS"
            or archived_seal.get("lock_sha256") != sha(archived_lock_path)
            or archived_seal.get("r3_review_sha256") != sha(archived_review_path)
            or prior.get("gate", {}).get("seal_sha256") != sha(archived_seal_path)
            or archived_review.get("status") != "PASS"
            or archived_review.get("lock_sha256") != sha(archived_lock_path)):
        raise PermissionError("q14 archived science seal or independent review differs")
    original = json.loads(original_path.read_text())
    if (original.get("schema") != "pcrl-utility-simultaneous-intervals-v1"
            or original.get("primary_F_gate_sha256") != sha(gate_path)
            or original.get("family_size") != 24
            or original.get("replicates") != 2048
            or original.get("rng_seed") != 20261009
            or original.get("groups") != 2234
            or original.get("rows") != 2235):
        raise PermissionError("q14 output interval family differs")
    inputs: dict[str, str] = {}
    deferred_role_paths = {
        ".private/role_export_v1/INNER_SELECTION/income.npz",
        ".private/role_export_v1/INNER_SELECTION/occupation.npz",
    }
    for name, expected in proposal["shared_inputs_sha256_by_study_relative_path"].items():
        if name in deferred_role_paths:
            continue  # actual role bytes verified only after the q15 seal
        _add(inputs, _relative_file(study, name), expected)
    for name, expected in proposal["source_sha256_by_study_relative_path"].items():
        _add(inputs, _relative_file(study, name), expected)
    for path in (proposal_path, old_queue, archived_lock_path,
                 archived_registry_path, archived_seal_path,
                 archived_review_path, original_path,
                 study / "r1/UTILITY_INTERVAL_PROTOCOL.md"):
        _add(inputs, path)
    role_inputs = {}
    for field in ("income", "occupation"):
        key = f"INNER_SELECTION:{field}"
        role_inputs[key] = bind_inner_task_role(study, manifest, proposal,
                                                field, inputs)
    if (sha(gate_path) !=
            proposal["shared_inputs_sha256_by_study_relative_path"]["FULL_INPUT_GATE.json"]
            or sha(manifest_path) !=
            proposal["shared_inputs_sha256_by_study_relative_path"]["m1/DATA_ACCESS_MANIFEST.json"]):
        raise PermissionError("R3 gate/role manifest differs from reviewed proposal")
    uid = unit["unit_id"]
    script = study / unit["script"]
    queue = private / "queue_g15"
    output = study / unit["output"]
    if queue.exists() or output.exists():
        raise FileExistsError("q15 or independent interval output already exists")
    argv = [str(python), str(script), "--unit-id", uid,
            "--queue-root", str(queue), "--lock", str(active_lock_path)]
    path_args = {"m1/DATA_ACCESS_MANIFEST.json", ".private/role_export_v1",
                 ".private", "FULL_INPUT_GATE.json",
                 ".private/r1/intervals/SIMULTANEOUS_UTILITY_INTERVALS.json",
                 ".private/r3/UTILITY_INTERVAL_REPLAY.json"}
    argv.extend(str(study / word) if word in path_args else word
                for word in unit["argv_suffix"])
    spec = {
        "unit_id": uid, "mode": "real", "stage": unit["stage"],
        "argv": argv, "script": str(script), "script_sha256": sha(script),
        "cwd": str(repo), "cpu_reserve_s": unit["cpu_reserve_s"],
        "wall_limit_s": unit["wall_limit_s"], "max_rss_gib": unit["max_rss_gib"],
        "inputs": inputs, "outputs": [str(output)], "deps": [],
        "source_map_sha256": sha(manifest_path),
        "role_access": ["INNER_SELECTION"], "role_inputs": role_inputs,
        "allow_child_processes": False,
        "data_access": "INDEPENDENT_INNER_TASK_INTERVAL_REPLAY_NO_SEX",
        "seal_sha256": "$LOCAL_SEAL_SHA256",
    }
    registry = {"schema": "pcrl-utility-unit-registry-v1",
                "stage": "r3_independent_full_input_utility_interval_replay",
                "unit_specs": {uid: spec}}
    lock = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "r3_independent_full_input_utility_interval_replay",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "development_status": "conditional_post_selection_reused_Adult_people",
        "selected_recipe": "A", "paired_seeds": [0, 1],
        "allowed_roles": ["INNER_SELECTION"],
        "role_scope": "independent INNER task-label and frozen F/U prediction replay; no SEX",
        "full_input_gate_sha256": sha(gate_path),
        "common_row_unit_pair_sha256": gate["common_row_unit_pair_sha256"],
        "r1_interval_output_sha256": original_sha,
        "r1_interval_protocol_sha256": sha(study / "r1/UTILITY_INTERVAL_PROTOCOL.md"),
        "interval_unit_proposal_sha256": sha(proposal_path),
        "prior_interval_queue_state_sha256": sha(old_queue),
        "prior_interval_science_lock_sha256": sha(archived_lock_path),
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
                            "HEAD_VALIDATION labels", "ATTACK_FIT_NEW labels"],
        "next_stage": "publish interval results only after independent replay PASS",
    }
    _write_once(registry_path, registry)
    lock["private_unit_registry_sha256"] = sha(registry_path)
    _write_once(active_lock_path, lock)
    return {"lock_sha256": sha(active_lock_path),
            "registry_sha256": sha(registry_path),
            "unit_id": uid, "input_count": len(inputs),
            "r1_interval_output_sha256": original_sha,
            "required_queue_carry_forward": carry,
            "resource_reserve": {"cpu_s": 300, "wall_s": 600,
                                 "max_rss_gib": 1, "numerical_threads": 1}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--slot", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo.resolve(strict=True), launcher_path(args.python),
                           args.slot.resolve(strict=True)), sort_keys=True))
