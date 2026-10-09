"""Prepare a new q22 seal after the preserved q21 verification failure.

R0 alone runs this write-once builder. It carries the failed unit's CPU charge,
rehashes every frozen nonrole input, and never opens a SEX archive preseal.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from results.pcrl_utility_reference_repair_v1.r3.build_privacy_replay_lock import (
    PROTOCOL_FILES as PRIOR_PROTOCOL_FILES, ROLE_KEYS,
    SOURCE_MODULES as PRIOR_SOURCE_MODULES, STUDY_RELATIVE,
    UNIT_ID as PRIOR_UNIT_ID, _add, _full_sha, _write_once, sha,
)


UNIT_ID = "R3_PRIVACY_REPLAY_RETRY"
SOURCE_MODULES = PRIOR_SOURCE_MODULES + (
    "results/pcrl_utility_reference_repair_v1/r3/build_privacy_replay_retry_lock.py",
)
PROTOCOL_FILES = PRIOR_PROTOCOL_FILES + ("LOCK_AMENDMENTS.md",)
FAILED_CPU_S = 1.0158803339581937


def failed_replay_carry(state: dict, old_spec: dict, state_sha: str,
                        old_seal_sha: str, prior_q20_sha: str) -> dict:
    """Require exactly one failed, output-free q21 unit and charge its cost."""
    units = state.get("units", {})
    row = units.get(PRIOR_UNIT_ID, {})
    previous = state.get("carry_forward", {})
    charge = row.get("charged_cpu_s")
    total = state.get("used_cpu_s")
    if (set(units) != {PRIOR_UNIT_ID}
            or row.get("status") != "FAILED" or row.get("exit_code") != 1
            or row.get("output_sha256") not in ({}, None)
            or old_spec.get("unit_id") != PRIOR_UNIT_ID
            or old_spec.get("stage") != "r3_privacy_replay"
            or len(old_spec.get("outputs", [])) != 1
            or any(Path(path).exists() for path in old_spec["outputs"])
            or state.get("gate", {}).get("seal_sha256") != old_seal_sha
            or previous.get("archived_queue_relative") != "queue_g20"
            or previous.get("archived_queue_sha256") != prior_q20_sha
            or previous.get("unit_count") != 2
            or not isinstance(previous.get("charged_cpu_s"), (int, float))
            or not math.isfinite(previous["charged_cpu_s"])
            or not isinstance(charge, (int, float)) or not math.isfinite(charge)
            or abs(charge - FAILED_CPU_S) > 1e-8
            or not isinstance(total, (int, float)) or not math.isfinite(total)
            or abs(total - previous["charged_cpu_s"] - charge) > 1e-5):
        raise PermissionError("q22 requires preserved failed q21 replay and CPU")
    return {"archived_queue_relative": "queue_g21",
            "archived_queue_sha256": state_sha,
            "charged_cpu_s": float(total), "unit_count": 1}


def inherit_inputs(old_inputs: dict[str, str], role_inputs: dict,
                   worker: Path) -> dict[str, str]:
    """Rehash old inputs, permitting only the reviewed replay-worker repair."""
    role_paths = {record["path"] for record in role_inputs.values()}
    if (set(role_inputs) != set(ROLE_KEYS)
            or not all(_full_sha(record.get("sha256"))
                       for record in role_inputs.values())
            or set(role_paths) - set(old_inputs)):
        raise PermissionError("q22 inherited role declaration differs")
    expected_worker = str(worker.resolve(strict=True))
    if expected_worker not in old_inputs:
        raise PermissionError("q22 repaired worker was not previously bound")
    inputs = {}
    for name, digest in old_inputs.items():
        if name in role_paths:
            record = next(value for value in role_inputs.values()
                          if value["path"] == name)
            if digest != record["sha256"]:
                raise PermissionError("q22 inherited role SHA differs")
            inputs[name] = digest
        elif name == expected_worker:
            current = _add(inputs, worker)
            if current == digest:
                raise PermissionError("q22 worker source was not repaired")
        else:
            _add(inputs, Path(name), digest)
    return inputs


def build(repo: Path, python: Path, slot: Path) -> dict:
    """R0-only retry builder; new unit/output and preserved failed predecessor."""
    repo = repo.resolve(strict=True)
    study = repo / STUDY_RELATIVE
    private = study / ".private"
    active_lock = study / "CAMPAIGN_LOCK.json"
    registry_path = private / "UNIT_SPECS.json"
    queue_dir = private / "queue_g22"
    output = private / "r3/PRIVACY_REPLAY_RETRY.json"
    if any(path.exists() for path in (active_lock, registry_path, queue_dir, output)):
        raise FileExistsError("q22 active lock, registry, queue or output exists")
    python = Path(python).absolute()
    if (str(python) != "/Users/nathansamson/PCRL/.venv/bin/python"
            or not python.is_file() or not slot.is_file()):
        raise FileNotFoundError("q22 project venv launcher or shared slot differs")
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    gate_path = study / "FULL_INPUT_GATE.json"
    bank_path = repo / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
    audit_path = study / "r3/PAIR_SELECTION_AUDIT.json"
    release_verification_path = study / "r3/RELEASE_NUMERICAL_VERIFICATION.json"
    manifest = json.loads(manifest_path.read_text())
    failed_root = private / "stage_seals/privacy_replay_failed"
    old_lock_path = failed_root / "CAMPAIGN_LOCK.json"
    old_registry_path = failed_root / "UNIT_SPECS.json"
    old_review_path = failed_root / "PREFIT_REVIEW.json"
    old_seal_path = failed_root / "LOCAL_SEAL_RECEIPT.json"
    failed_path = private / "queue_g21/QUEUE.json"
    q20_path = private / "queue_g20/QUEUE.json"
    old_lock = json.loads(old_lock_path.read_text())
    old_registry = json.loads(old_registry_path.read_text())
    old_review = json.loads(old_review_path.read_text())
    old_seal = json.loads(old_seal_path.read_text())
    failed = json.loads(failed_path.read_text())
    old_lock_sha = sha(old_lock_path)
    if (old_lock.get("schema") != "pcrl-utility-campaign-lock-v1"
            or old_lock.get("stage") != "independent_frozen_privacy_score_replay"
            or old_lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE"
            or old_lock.get("private_unit_registry_sha256") !=
            sha(old_registry_path)
            or old_review.get("status") != "PASS"
            or old_review.get("lock_sha256") != old_lock_sha
            or old_seal.get("status") != "PASS"
            or old_seal.get("lock_sha256") != old_lock_sha
            or old_seal.get("r3_review_sha256") != sha(old_review_path)
            or set(old_registry.get("unit_specs", {})) != {PRIOR_UNIT_ID}
            or old_lock.get("source_map_sha256") != sha(manifest_path)
            or old_lock.get("full_input_gate_sha256") != sha(gate_path)
            or old_lock.get("mask_bank_sha256") != sha(bank_path)
            or old_lock.get("pair_selection_audit_sha256") != sha(audit_path)
            or old_lock.get("independent_release_verification_sha256") !=
            sha(release_verification_path)):
        raise PermissionError("q22 archived failed replay science identity differs")
    old_spec = old_registry["unit_specs"][PRIOR_UNIT_ID]
    carry = failed_replay_carry(failed, old_spec, sha(failed_path),
                                sha(old_seal_path), sha(q20_path))
    role_inputs = old_spec["role_inputs"]
    if set(role_inputs) != set(ROLE_KEYS):
        raise PermissionError("q22 inherited sex role inventory differs")
    for key, record in role_inputs.items():
        relative = Path(manifest["role_file_relative_path"][key])
        if (relative.is_absolute() or ".." in relative.parts
                or record.get("path") != str(study / relative)
                or record.get("sha256") != manifest["role_file_sha256"][key]):
            raise PermissionError("q22 sex role differs from sealed manifest")
    script = study / "r3/privacy_replay_worker.py"
    inputs = inherit_inputs(old_spec["inputs"], role_inputs, script)
    for path in (failed_path, old_lock_path, old_registry_path,
                 old_review_path, old_seal_path, q20_path,
                 study / "LOCK_AMENDMENTS.md"):
        _add(inputs, path)
    for name in SOURCE_MODULES:
        _add(inputs, repo / name)
    for name in PROTOCOL_FILES:
        _add(inputs, study / name)
    argv = [str(python), str(script), "--unit-id", UNIT_ID,
            "--queue-root", str(queue_dir), "--lock", str(active_lock),
            "--manifest", str(manifest_path),
            "--export-root", str(private / "role_export_v1"),
            "--private-root", str(private),
            "--full-input-gate", str(gate_path),
            "--mask-bank", str(bank_path),
            "--pair-selection-audit", str(audit_path),
            "--release-verification", str(release_verification_path),
            "--output", str(output)]
    unit = {
        "unit_id": UNIT_ID, "mode": "real", "stage": "r3_privacy_replay",
        "argv": argv, "script": str(script), "script_sha256": sha(script),
        "cwd": str(repo), "cpu_reserve_s": 600, "wall_limit_s": 1200,
        "max_rss_gib": 3, "inputs": dict(sorted(inputs.items())),
        "outputs": [str(output)], "deps": [],
        "source_map_sha256": sha(manifest_path),
        "role_access": ["ATTACK_FIT_NEW", "INNER_SELECTION"],
        "role_inputs": role_inputs, "allow_child_processes": False,
        "data_access": "INDEPENDENT_FROZEN_ATT_AND_INNER_SEX_REPLAY_RETRY",
        "seal_sha256": "$LOCAL_SEAL_SHA256",
    }
    registry = {"schema": "pcrl-utility-unit-registry-v1",
                "stage": "independent_frozen_privacy_score_replay_retry",
                "unit_specs": {UNIT_ID: unit}}
    lock = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "independent_frozen_privacy_score_replay_retry",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "development_status": "exploratory_reused_Adult_people",
        "selected_recipe": "A", "paired_seeds": [0, 1],
        "selected_pairs_by_seed": old_lock["selected_pairs_by_seed"],
        "allowed_roles": ["ATTACK_FIT_NEW", "INNER_SELECTION"],
        "role_scope": "repair semantic bank check; replay frozen ATT/INNER SEX without fitting",
        "full_input_gate_sha256": old_lock["full_input_gate_sha256"],
        "source_science_lock_sha256": old_lock["source_science_lock_sha256"],
        "common_row_unit_pair_sha256": old_lock["common_row_unit_pair_sha256"],
        "mask_bank_sha256": old_lock["mask_bank_sha256"],
        "pair_selection_audit_sha256": old_lock["pair_selection_audit_sha256"],
        "independent_release_verification_sha256":
            old_lock["independent_release_verification_sha256"],
        "release_source_lock_sha256": old_lock["release_source_lock_sha256"],
        "local_attack_source_lock_sha256":
            old_lock["local_attack_source_lock_sha256"],
        "coalition_attack_source_lock_sha256":
            old_lock["coalition_attack_source_lock_sha256"],
        "privacy_score_source_lock_sha256":
            old_lock["privacy_score_source_lock_sha256"],
        "failed_replay_source_lock_sha256": old_lock_sha,
        "failed_replay_queue_state_sha256": sha(failed_path),
        "failed_replay_cpu_s": FAILED_CPU_S,
        "prior_failed_attempt_debit_cpu_s":
            old_lock["prior_failed_attempt_debit_cpu_s"],
        "effective_campaign_used_cpu_s_at_q22_start":
            carry["charged_cpu_s"] + old_lock["prior_failed_attempt_debit_cpu_s"],
        "required_queue_carry_forward": carry,
        "independent_replay_unit": UNIT_ID,
        "runner_scoring_or_selection_imported": False,
        "private_unit_registry_sha256": "__AFTER_REGISTRY_WRITE__",
        "source_map_sha256": sha(manifest_path),
        "prior_package_relative_to_worktree": str(STUDY_RELATIVE),
        "governed_source_sha256_by_path": {
            name: sha(repo / name) for name in SOURCE_MODULES},
        "governed_protocol_sha256_by_relative_path": {
            name: sha(study / name) for name in PROTOCOL_FILES},
        "shared_slot_sha256": sha(slot),
        "excluded_inputs": ["old assessment person arrays",
                            "HEAD_VALIDATION labels", "INNER task labels"],
        "next_stage": "publish only independently replayed privacy endpoints",
    }
    _write_once(registry_path, registry)
    lock["private_unit_registry_sha256"] = sha(registry_path)
    _write_once(active_lock, lock)
    return {"lock_sha256": sha(active_lock),
            "registry_sha256": sha(registry_path),
            "unit_id": UNIT_ID, "input_count": len(inputs),
            "failed_replay_cpu_s": FAILED_CPU_S,
            "required_queue_carry_forward": carry}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--slot", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.python, args.slot), sort_keys=True))
