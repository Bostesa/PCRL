"""Prepare q19 coalition ATT readers with literal q18 local-reader hashes.

R0 alone executes this write-once builder after eight q18 local units finish.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from results.pcrl_utility_reference_repair_v1.r2.build_local_att_lock import (
    PROTOCOL_FILES as BASE_PROTOCOL_FILES, ROLE_KEYS,
    SOURCE_MODULES as BASE_SOURCE_MODULES, resolve_att_inputs,
)
from results.pcrl_utility_reference_repair_v1.r2.build_release_raw_lock import (
    PROPOSAL_SHA256, STUDY_RELATIVE, _add, _write_once, sha,
)


SOURCE_MODULES = BASE_SOURCE_MODULES + (
    "results/pcrl_utility_reference_repair_v1/r2/build_coalition_att_lock.py",
)
PROTOCOL_FILES = BASE_PROTOCOL_FILES


def validate_local_predecessor(q18: dict, expected_ids: set[str],
                               q18_sha: str) -> dict:
    """Require all eight recipient readers with their exact CPU charge."""
    units = q18.get("units", {})
    previous = q18.get("carry_forward", {}).get("charged_cpu_s")
    total = q18.get("used_cpu_s")
    charges = [record.get("charged_cpu_s") for record in units.values()]
    if (len(expected_ids) != 8 or set(units) != expected_ids
            or any(record.get("status") != "DONE" for record in units.values())
            or not isinstance(previous, (int, float)) or not math.isfinite(previous)
            or not isinstance(total, (int, float)) or not math.isfinite(total)
            or any(not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0
                   for value in charges)
            or abs(total - previous - sum(charges)) > 1e-5):
        raise PermissionError("q18 local ATT predecessor incomplete")
    return {"archived_queue_relative": "queue_g18",
            "archived_queue_sha256": q18_sha,
            "charged_cpu_s": float(total), "unit_count": 8}


def build(repo: Path, python: Path, slot: Path) -> dict:
    """R0-only q19 coalition stage after exact local ATT custody is frozen."""
    repo = repo.resolve(strict=True)
    study = repo / STUDY_RELATIVE
    private = study / ".private"
    lock_path = study / "CAMPAIGN_LOCK.json"
    registry_path = private / "UNIT_SPECS.json"
    queue_path = private / "queue_g19"
    if lock_path.exists() or registry_path.exists() or queue_path.exists():
        raise FileExistsError("q19 shared lock, registry or queue exists")
    if (python.name != "python" or python.parent.name != "bin"
            or python.parent.parent.name != ".venv" or not python.is_file()
            or not slot.is_file()):
        raise FileNotFoundError("project venv or shared slot missing")
    proposal_path = study / "r2/DOWNSTREAM_UNIT_PROPOSAL.json"
    if sha(proposal_path) != PROPOSAL_SHA256:
        raise PermissionError("q19 selected proposal differs")
    proposal = json.loads(proposal_path.read_text())
    if (proposal.get("schema") != "pcrl-utility-r2-downstream-unit-proposal-v1"
            or proposal.get("unit_count_by_stage", {}).get("r2_attack_fit") != 12
            or proposal.get("new_att_fit_count") != 86):
        raise PermissionError("q19 coalition envelope differs")
    gate_path = study / "FULL_INPUT_GATE.json"
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    bank_path = repo / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
    audit_path = study / "r3/PAIR_SELECTION_AUDIT.json"
    verification_path = study / "r3/RELEASE_NUMERICAL_VERIFICATION.json"
    gate = json.loads(gate_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (gate.get("status") != "PASS_FULL_INPUT_POINT_GATE"
            or gate.get("selected_recipe") != "A"
            or sha(gate_path) != proposal["full_input_gate_sha256"]
            or sha(manifest_path) != proposal["source_manifest_sha256"]
            or sha(bank_path) != proposal["mask_bank_sha256"]
            or json.loads(audit_path.read_text()).get("status") !=
            "PASS_AGGREGATE_SELECTION_REPLAY"
            or json.loads(verification_path.read_text()).get("status") !=
            "PASS_INDEPENDENT_COMPLETE_VIEW_REPLAY"):
        raise PermissionError("q19 F/bank/role/selection/release source differs")
    q13_lock_path = private / "stage_seals/pair_search/CAMPAIGN_LOCK.json"
    q16_lock_path = private / "stage_seals/release_raw/CAMPAIGN_LOCK.json"
    q18_root = private / "stage_seals/local_att"
    q18_lock_path = q18_root / "CAMPAIGN_LOCK.json"
    q18_registry_path = q18_root / "UNIT_SPECS.json"
    q18_review_path = q18_root / "PREFIT_REVIEW.json"
    q18_seal_path = q18_root / "LOCAL_SEAL_RECEIPT.json"
    q16_path = private / "queue_g16/QUEUE.json"
    q18_path = private / "queue_g18/QUEUE.json"
    q16, q18 = json.loads(q16_path.read_text()), json.loads(q18_path.read_text())
    q18_lock = json.loads(q18_lock_path.read_text())
    q18_review = json.loads(q18_review_path.read_text())
    q18_seal = json.loads(q18_seal_path.read_text())
    local_ids = {uid for uid, record in proposal["unit_map"].items()
                 if record.get("stage") == "r2_attack_fit"
                 and record.get("view") in ("income", "occupation_group")}
    carry = validate_local_predecessor(q18, local_ids, sha(q18_path))
    local_lock_sha = sha(q18_lock_path)
    release_lock_sha = sha(q16_lock_path)
    if (q18_lock.get("schema") != "pcrl-utility-campaign-lock-v1"
            or q18_lock.get("stage") != "fresh_local_complete_view_readers"
            or q18_lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE"
            or q18_lock.get("private_unit_registry_sha256") !=
            sha(q18_registry_path)
            or q18_lock.get("downstream_unit_proposal_sha256") !=
            sha(proposal_path)
            or q18_lock.get("release_source_lock_sha256") != release_lock_sha
            or q18_lock.get("raw_union_source_lock_sha256") != release_lock_sha
            or q18_lock.get("pair_select_source_lock_sha256") !=
            sha(q13_lock_path)
            or q18_lock.get("full_input_gate_sha256") != sha(gate_path)
            or q18_lock.get("mask_bank_sha256") != sha(bank_path)
            or q18_review.get("status") != "PASS"
            or q18_review.get("lock_sha256") != local_lock_sha
            or q18_seal.get("status") != "PASS"
            or q18_seal.get("lock_sha256") != local_lock_sha
            or q18.get("gate", {}).get("seal_sha256") != sha(q18_seal_path)):
        raise PermissionError("q18 local ATT science seal differs")
    failed_debit = q18_lock.get("prior_failed_attempt_debit_cpu_s")
    if (not isinstance(failed_debit, (int, float))
            or abs(failed_debit - 0.2250022920779884) > 1e-8):
        raise PermissionError("failed q16 attempt debit was lost")
    merged_producers = {"units": {**q16["units"], **q18["units"]}}
    shared = {}
    for path in (proposal_path, gate_path, manifest_path, bank_path,
                 audit_path, verification_path, q13_lock_path,
                 q16_path, q16_lock_path, q18_path, q18_lock_path,
                 q18_registry_path, q18_review_path, q18_seal_path,
                 study / "r2/R2_RESOURCE_FORECAST.json",
                 study / "r2/ATT_STAGE_SPLIT_AMENDMENT.md"):
        _add(shared, path)
    from results.pcrl_utility_reference_repair_v1.r2.attack_fit import (
        expected_outputs,
    )
    units = {}
    external_deps = {}
    for uid, entry in proposal["unit_map"].items():
        if (entry.get("stage") != "r2_attack_fit"
                or entry.get("view") != "coalition"):
            continue
        deps = set(entry.get("deps", []))
        if (entry.get("unit_id") != uid or not uid.endswith("_COALITION")
                or entry.get("seed") not in (0, 1)
                or len(deps) != 4
                or sum(dep in q18["units"] for dep in deps) != 2
                or sum(dep in q16["units"] for dep in deps) != 2
                or set(entry.get("role_inputs", [])) != set(ROLE_KEYS)):
            raise PermissionError("q19 coalition unit/dependency inventory differs")
        inputs, role_inputs = resolve_att_inputs(
            entry["inputs"], repo, study, manifest, ROLE_KEYS,
            merged_producers, deps)
        inputs.update(shared)
        outputs = []
        for name in entry["outputs"]:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise PermissionError("q19 output path escapes worktree")
            output = (repo / relative).resolve(strict=False)
            if not output.is_relative_to(private) or output.exists():
                raise PermissionError("q19 output exists or escapes custody")
            outputs.append(str(output))
        primary = [name for name in outputs if name.endswith("/ATTACK_FIT_INDEX.json")]
        if (len(primary) != 1 or len(outputs) != 16
                or set(outputs) != expected_outputs(Path(primary[0]))):
            raise PermissionError("q19 complete-view output inventory differs")
        script = study / "r2/attack_fit_worker.py"
        argv = [str(python), str(script), "--seed", str(entry["seed"]),
                "--pair-id", entry["pair_id"], "--view", "coalition",
                "--unit-id", uid, "--queue-root", str(queue_path),
                "--lock", str(lock_path), "--gate", str(gate_path),
                "--manifest", str(manifest_path),
                "--export-root", str(private / "role_export_v1"),
                "--private-root", str(private), "--output", primary[0]]
        units[uid] = {
            "unit_id": uid, "mode": "real", "stage": "r2_attack_fit",
            "argv": argv, "script": str(script), "script_sha256": sha(script),
            "cwd": str(repo), "cpu_reserve_s": 900, "wall_limit_s": 1800,
            "max_rss_gib": 3, "inputs": dict(sorted(inputs.items())),
            "outputs": sorted(outputs), "deps": [],
            "source_map_sha256": sha(manifest_path),
            "role_access": ["ATTACK_FIT_NEW", "INNER_SELECTION"],
            "role_inputs": role_inputs, "allow_child_processes": False,
            "data_access": "ATT_FEATURES_AND_SEX_INNER_FEATURES_NO_INNER_SEX",
            "seal_sha256": "$LOCAL_SEAL_SHA256",
        }
        external_deps[uid] = sorted(deps)
    if len(units) != 4 or len({name for record in units.values()
                               for name in record["outputs"]}) != 64:
        raise PermissionError("q19 four coalition readers or outputs differ")
    registry = {"schema": "pcrl-utility-unit-registry-v1",
                "stage": "fresh_coalition_complete_view_readers",
                "unit_specs": dict(sorted(units.items()))}
    lock = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "fresh_coalition_complete_view_readers",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "development_status": "exploratory_reused_Adult_people",
        "selected_recipe": "A", "paired_seeds": [0, 1],
        "admitted_mask_ids": [row["mask_id"] for row in
                              json.loads(bank_path.read_text())["masks"]],
        "allowed_roles": ["ATTACK_FIT_NEW", "INNER_SELECTION"],
        "role_scope": "ATT features+SEX, INNER features; never INNER SEX",
        "full_input_gate_sha256": sha(gate_path),
        "source_science_lock_sha256": gate["source_science_lock_sha256"],
        "common_row_unit_pair_sha256": gate["common_row_unit_pair_sha256"],
        "mask_bank_sha256": sha(bank_path),
        "pair_select_source_lock_sha256": sha(q13_lock_path),
        "release_source_lock_sha256": release_lock_sha,
        "raw_union_source_lock_sha256": release_lock_sha,
        "local_attack_source_lock_sha256": local_lock_sha,
        "pair_selection_audit_sha256": sha(audit_path),
        "downstream_unit_proposal_sha256": sha(proposal_path),
        "prior_local_att_queue_state_sha256": sha(q18_path),
        "prior_local_att_science_lock_sha256": local_lock_sha,
        "prior_failed_attempt_debit_cpu_s": failed_debit,
        "effective_campaign_used_cpu_s_at_q19_start":
        carry["charged_cpu_s"] + failed_debit,
        "required_queue_carry_forward": carry,
        "coalition_att_units": 4, "coalition_att_core_fits": 16,
        "coalition_att_engineering_probe_fits": 8,
        "external_q16_q18_dependencies_by_unit": dict(sorted(external_deps.items())),
        "private_unit_registry_sha256": "__AFTER_REGISTRY_WRITE__",
        "source_map_sha256": sha(manifest_path),
        "prior_package_relative_to_worktree": str(STUDY_RELATIVE),
        "governed_source_sha256_by_path": {
            name: sha(repo / name) for name in SOURCE_MODULES},
        "governed_protocol_sha256_by_relative_path": {
            name: sha(study / name) for name in PROTOCOL_FILES},
        "shared_slot_sha256": sha(slot),
        "excluded_inputs": ["INNER_SELECTION SEX", "old assessment person arrays",
                            "HEAD_VALIDATION labels"],
        "next_stage": "seal two seed-level INNER SEX scores after all ATT fits",
    }
    _write_once(registry_path, registry)
    lock["private_unit_registry_sha256"] = sha(registry_path)
    _write_once(lock_path, lock)
    return {"lock_sha256": sha(lock_path),
            "registry_sha256": sha(registry_path),
            "unit_count": 4, "required_queue_carry_forward": carry,
            "input_count_by_unit": {uid: len(spec["inputs"])
                                    for uid, spec in units.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--slot", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.python, args.slot), sort_keys=True))
