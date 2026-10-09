"""Prepare the R1 interval science lock; R0 alone executes this builder.

This stage reads only frozen gate predictions and admitted INNER task labels.
The separate R3 replay needs a later lock bound to this stage's output hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


STUDY_RELATIVE = Path("results/pcrl_utility_reference_repair_v1")
PROPOSAL_SHA256 = "59685813ff8f139dc028c378282d354def6adcaccb03adea2f1566a72f22926d"
SOURCE_MODULES = (
    "results/pcrl_utility_reference_repair_v1/r0/admission.py",
    "results/pcrl_utility_reference_repair_v1/r0/queue_adapter.py",
    "results/pcrl_utility_reference_repair_v1/r1/fit_splits.py",
    "results/pcrl_utility_reference_repair_v1/r1/role_store.py",
    "results/pcrl_utility_reference_repair_v1/r1/utility_intervals.py",
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
    "r1/INTERVAL_UNIT_PROPOSAL.json",
    "r3/FULL_INPUT_TASK_NUMERICAL_VERIFICATION.json",
    "r3/PAIR_SELECTION_AUDIT.json",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_predecessor(state: dict, audit: dict, state_sha: str,
                         pair_lock_sha: str) -> dict:
    """Require completed q13 and the separately checked pair audit."""
    units = state.get("units", {})
    previous = state.get("carry_forward", {})
    total = state.get("used_cpu_s")
    charged = [unit.get("charged_cpu_s") for unit in units.values()]
    if (set(units) != {"R2_PAIR_SELECT_S0", "R2_PAIR_SELECT_S1"}
            or any(unit.get("status") != "DONE" for unit in units.values())
            or previous.get("archived_queue_relative") != "queue_g12"
            or previous.get("unit_count") != 2
            or not isinstance(total, (int, float)) or not math.isfinite(total)
            or total < 0
            or not isinstance(previous.get("charged_cpu_s"), (int, float))
            or not math.isfinite(previous["charged_cpu_s"])
            or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                   or value < 0 for value in charged)
            or abs(total - previous["charged_cpu_s"]
                   - sum(charged)) > 1e-5
            or audit.get("schema") != "pcrl-utility-r3-pair-selection-audit-v1"
            or audit.get("status") != "PASS_AGGREGATE_SELECTION_REPLAY"
            or audit.get("pair_search_queue_state_sha256") != state_sha
            or audit.get("pair_search_lock_sha256") != pair_lock_sha):
        raise PermissionError("completed q13 carry-forward or R3 pair audit differs")
    return {"archived_queue_relative": "queue_g13",
            "archived_queue_sha256": state_sha,
            "charged_cpu_s": float(total), "unit_count": len(units)}


def _relative_file(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("interval path escapes study")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise PermissionError("interval path escapes study")
    return path


def _add(inputs: dict[str, str], path: Path, expected: str | None = None) -> str:
    resolved = path.resolve(strict=True)
    actual = sha(resolved)
    if expected is not None and actual != expected:
        raise PermissionError("interval predecessor bytes differ: " + str(path))
    inputs[str(resolved)] = actual
    return actual


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + "\n")


def build(repo: Path, python: Path, slot: Path) -> dict:
    study = repo / STUDY_RELATIVE
    private = study / ".private"
    lock_path = study / "CAMPAIGN_LOCK.json"
    registry_path = private / "UNIT_SPECS.json"
    if lock_path.exists() or registry_path.exists():
        raise FileExistsError("active science lock or registry must be archived by R0")
    if (python.name != "python" or python.parent.name != "bin"
            or python.parent.parent.name != ".venv" or not python.is_file()
            or not slot.is_file()):
        raise FileNotFoundError("project venv or shared slot missing")
    proposal_path = study / "r1/INTERVAL_UNIT_PROPOSAL.json"
    if sha(proposal_path) != PROPOSAL_SHA256:
        raise PermissionError("reviewed R1 interval proposal differs")
    proposal = json.loads(proposal_path.read_text())
    unit = proposal["r1_unit"]
    if (proposal.get("schema") != "pcrl-utility-interval-unit-proposal-v1"
            or proposal.get("status") != "PROPOSED_UNRUN"
            or unit.get("unit_id") != "R1_UTILITY_INTERVALS"
            or unit.get("stage") != "r1_utility_intervals"
            or unit.get("role_access") != ["INNER_SELECTION:income", "INNER_SELECTION:occupation"]
            or unit.get("output") != ".private/r1/intervals/SIMULTANEOUS_UTILITY_INTERVALS.json"
            or unit.get("cpu_reserve_s") != 300
            or unit.get("wall_limit_s") != 600
            or unit.get("max_rss_gib") != 1
            or unit.get("numerical_threads") != 1
            or unit.get("allow_child_processes") is not False):
        raise PermissionError("R1 unit differs from reviewed resource/interface proposal")
    prior_path = private / "queue_g13/QUEUE.json"
    pair_lock_path = private / "stage_seals/pair_search/CAMPAIGN_LOCK.json"
    audit_path = study / "r3/PAIR_SELECTION_AUDIT.json"
    audit = json.loads(audit_path.read_text())
    carry = validate_predecessor(json.loads(prior_path.read_text()), audit,
                                 sha(prior_path), sha(pair_lock_path))
    gate_path = study / "FULL_INPUT_GATE.json"
    gate = json.loads(gate_path.read_text())
    if (gate.get("schema") != "pcrl-utility-full-input-gate-v1"
            or gate.get("status") != "PASS_FULL_INPUT_POINT_GATE"
            or gate.get("selected_recipe") != "A"
            or gate.get("common_role") != "INNER_SELECTION"
            or gate.get("r3_numerical_verification_sha256") !=
            sha(study / "r3/FULL_INPUT_TASK_NUMERICAL_VERIFICATION.json")):
        raise PermissionError("completed independently replayed F gate differs")
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "pcrl-utility-repair-role-admission-v1":
        raise PermissionError("role-only admission manifest differs")
    inputs: dict[str, str] = {}
    for name, expected in proposal["shared_inputs_sha256_by_study_relative_path"].items():
        _add(inputs, _relative_file(study, name), expected)
    for name, expected in proposal["source_sha256_by_study_relative_path"].items():
        _add(inputs, _relative_file(study, name), expected)
    for path in (proposal_path, prior_path, pair_lock_path, audit_path,
                 study / "F_ARTIFACT_LOCK.json",
                 study / "r3/FULL_INPUT_TASK_NUMERICAL_VERIFICATION.json",
                 private / "r3/TASK_GATE_REPLAY.json"):
        _add(inputs, path)
    role_inputs = {}
    for field in ("income", "occupation"):
        key = f"INNER_SELECTION:{field}"
        path = _relative_file(study, manifest["role_file_relative_path"][key])
        digest = manifest["role_file_sha256"][key]
        _add(inputs, path, digest)
        role_inputs[key] = {"path": str(path), "sha256": digest}
    if (sha(gate_path) != proposal["shared_inputs_sha256_by_study_relative_path"]["FULL_INPUT_GATE.json"]
            or sha(manifest_path) != proposal["shared_inputs_sha256_by_study_relative_path"]["m1/DATA_ACCESS_MANIFEST.json"]
            or gate["r3_numerical_replay_sha256"] != sha(private / "r3/TASK_GATE_REPLAY.json")):
        raise PermissionError("F gate source and role declarations differ")
    script = study / unit["script"]
    queue = private / "queue_g14"
    output = study / unit["output"]
    if output.exists() or queue.exists():
        raise FileExistsError("interval output or q14 already exists")
    uid = unit["unit_id"]
    argv = ([str(python), str(script), "--unit-id", uid,
             "--queue-root", str(queue), "--lock", str(lock_path)]
            + [str(study / word) if word in {
                "m1/DATA_ACCESS_MANIFEST.json", ".private/role_export_v1",
                ".private", "FULL_INPUT_GATE.json",
                ".private/r1/intervals/SIMULTANEOUS_UTILITY_INTERVALS.json"}
               else word for word in unit["argv_suffix"]])
    spec = {
        "unit_id": uid, "mode": "real", "stage": unit["stage"],
        "argv": argv, "script": str(script), "script_sha256": sha(script),
        "cwd": str(repo), "cpu_reserve_s": unit["cpu_reserve_s"],
        "wall_limit_s": unit["wall_limit_s"], "max_rss_gib": unit["max_rss_gib"],
        "inputs": inputs, "outputs": [str(output)], "deps": [],
        "source_map_sha256": sha(manifest_path),
        "role_access": ["INNER_SELECTION"], "role_inputs": role_inputs,
        "allow_child_processes": False,
        "data_access": "INNER_TASK_LABELS_AND_FROZEN_GATE_PREDICTIONS_NO_SEX",
        "seal_sha256": "$LOCAL_SEAL_SHA256",
    }
    registry = {"schema": "pcrl-utility-unit-registry-v1",
                "stage": "r1_full_input_utility_intervals",
                "unit_specs": {uid: spec}}
    lock = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "r1_full_input_utility_intervals",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "development_status": "conditional_post_selection_reused_Adult_people",
        "selected_recipe": "A", "paired_seeds": [0, 1],
        "allowed_roles": ["INNER_SELECTION"],
        "role_scope": "INNER income/occupation task labels and frozen F/U predictions only; no SEX",
        "full_input_gate_sha256": sha(gate_path),
        "common_row_unit_pair_sha256": gate["common_row_unit_pair_sha256"],
        "interval_protocol_sha256": sha(study / "r1/UTILITY_INTERVAL_PROTOCOL.md"),
        "interval_unit_proposal_sha256": sha(proposal_path),
        "prior_pair_search_queue_state_sha256": sha(prior_path),
        "prior_pair_search_lock_sha256": sha(pair_lock_path),
        "r3_pair_selection_audit_sha256": sha(audit_path),
        "required_queue_carry_forward": carry,
        "interval_family_size": 24, "interval_group_draws": 2048,
        "interval_seed": 20261009,
        "utility_caps_inclusive": {"accuracy_loss": 0.010,
                                   "natural_nll_excess": 0.010,
                                   "summed_multiclass_brier_excess": 0.005},
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
        "next_stage": "independent R3 interval replay from separately sealed R1 output hash",
    }
    _write_once(registry_path, registry)
    lock["private_unit_registry_sha256"] = sha(registry_path)
    _write_once(lock_path, lock)
    return {"lock_sha256": sha(lock_path),
            "registry_sha256": sha(registry_path),
            "unit_id": uid, "input_count": len(inputs),
            "required_queue_carry_forward": carry,
            "resource_reserve": {"cpu_s": 300, "wall_s": 600,
                                 "max_rss_gib": 1, "numerical_threads": 1}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--slot", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo.resolve(strict=True), args.python.resolve(strict=True),
                           args.slot.resolve(strict=True)), sort_keys=True))
