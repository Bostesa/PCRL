"""Prepare two INNER SEX score units only after all fresh readers are frozen.

R0 alone executes this builder. Role bytes are not opened before the q20 seal.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from results.pcrl_utility_reference_repair_v1.r2.build_coalition_att_lock import (
    SOURCE_MODULES as BASE_SOURCE_MODULES,
    PROTOCOL_FILES as BASE_PROTOCOL_FILES,
)
from results.pcrl_utility_reference_repair_v1.r2.build_local_att_lock import (
    resolve_att_inputs,
)
from results.pcrl_utility_reference_repair_v1.r2.build_release_raw_lock import (
    PROPOSAL_SHA256, STUDY_RELATIVE, _full_sha, _write_once, sha,
)


SOURCE_MODULES = BASE_SOURCE_MODULES + (
    "results/pcrl_utility_reference_repair_v1/r2/privacy_score.py",
    "results/pcrl_utility_reference_repair_v1/r2/privacy_score_worker.py",
    "results/pcrl_utility_reference_repair_v1/r2/build_privacy_score_lock.py",
    "results/pcrl_method_portfolio_v1/shared/attack_controls_v2.py",
)
PROTOCOL_FILES = BASE_PROTOCOL_FILES + (
    "r2/PRIVACY_SCORE_ACCESS_AMENDMENT.md",
    "r2/PRIVACY_SCORE_STAGE_LINEAGE_20261009.md",
)


def validate_coalition_predecessor(q19: dict, expected_ids: set[str],
                                   q19_sha: str) -> dict:
    """Require four coalition ATT units and their exact CPU carry."""
    units = q19.get("units", {})
    previous = q19.get("carry_forward", {}).get("charged_cpu_s")
    total = q19.get("used_cpu_s")
    charges = [record.get("charged_cpu_s") for record in units.values()]
    if (len(expected_ids) != 4 or set(units) != expected_ids
            or any(record.get("status") != "DONE" for record in units.values())
            or not isinstance(previous, (int, float)) or not math.isfinite(previous)
            or not isinstance(total, (int, float)) or not math.isfinite(total)
            or any(not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0
                   for value in charges)
            or abs(total - previous - sum(charges)) > 1e-5):
        raise PermissionError("q19 coalition ATT predecessor incomplete")
    return {"archived_queue_relative": "queue_g19",
            "archived_queue_sha256": q19_sha,
            "charged_cpu_s": float(total), "unit_count": 4}


def build(repo: Path, python: Path, slot: Path) -> dict:
    """R0-only q20 score lock builder with no preseal INNER SEX read."""
    repo = repo.resolve(strict=True)
    study = repo / STUDY_RELATIVE
    private = study / ".private"
    lock_path = study / "CAMPAIGN_LOCK.json"
    registry_path = private / "UNIT_SPECS.json"
    queue_path = private / "queue_g20"
    if lock_path.exists() or registry_path.exists() or queue_path.exists():
        raise FileExistsError("q20 shared lock, registry or queue exists")
    if (python.name != "python" or python.parent.name != "bin"
            or python.parent.parent.name != ".venv" or not python.is_file()
            or not slot.is_file()):
        raise FileNotFoundError("project venv or shared slot missing")
    proposal_path = study / "r2/DOWNSTREAM_UNIT_PROPOSAL.json"
    if sha(proposal_path) != PROPOSAL_SHA256:
        raise PermissionError("q20 selected downstream proposal differs")
    proposal = json.loads(proposal_path.read_text())
    if (proposal.get("schema") != "pcrl-utility-r2-downstream-unit-proposal-v1"
            or proposal.get("unit_count_by_stage") != {
                "r2_release": 4, "r2_raw_union": 7,
                "r2_attack_fit": 12, "r2_privacy_score": 2}
            or proposal.get("new_att_fit_count") != 86):
        raise PermissionError("q20 frozen score/attack inventory differs")
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
        raise PermissionError("q20 gate/bank/role/audit/release source differs")
    q13_lock_path = private / "stage_seals/pair_search/CAMPAIGN_LOCK.json"
    q16_lock_path = private / "stage_seals/release_raw/CAMPAIGN_LOCK.json"
    q18_lock_path = private / "stage_seals/local_att/CAMPAIGN_LOCK.json"
    q19_root = private / "stage_seals/coalition_att"
    q19_lock_path = q19_root / "CAMPAIGN_LOCK.json"
    q19_registry_path = q19_root / "UNIT_SPECS.json"
    q19_review_path = q19_root / "PREFIT_REVIEW.json"
    q19_seal_path = q19_root / "LOCAL_SEAL_RECEIPT.json"
    q16_path = private / "queue_g16/QUEUE.json"
    q18_path = private / "queue_g18/QUEUE.json"
    q19_path = private / "queue_g19/QUEUE.json"
    q16 = json.loads(q16_path.read_text())
    q18 = json.loads(q18_path.read_text())
    q19 = json.loads(q19_path.read_text())
    q19_lock = json.loads(q19_lock_path.read_text())
    q19_review = json.loads(q19_review_path.read_text())
    q19_seal = json.loads(q19_seal_path.read_text())
    coalition_ids = {uid for uid, record in proposal["unit_map"].items()
                     if record.get("stage") == "r2_attack_fit"
                     and record.get("view") == "coalition"}
    carry = validate_coalition_predecessor(q19, coalition_ids, sha(q19_path))
    release_lock_sha, local_lock_sha = sha(q16_lock_path), sha(q18_lock_path)
    coalition_lock_sha = sha(q19_lock_path)
    if (q19_lock.get("schema") != "pcrl-utility-campaign-lock-v1"
            or q19_lock.get("stage") != "fresh_coalition_complete_view_readers"
            or q19_lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE"
            or q19_lock.get("private_unit_registry_sha256") !=
            sha(q19_registry_path)
            or q19_lock.get("downstream_unit_proposal_sha256") !=
            sha(proposal_path)
            or q19_lock.get("release_source_lock_sha256") != release_lock_sha
            or q19_lock.get("raw_union_source_lock_sha256") != release_lock_sha
            or q19_lock.get("local_attack_source_lock_sha256") !=
            local_lock_sha
            or q19_lock.get("pair_select_source_lock_sha256") !=
            sha(q13_lock_path)
            or q19_lock.get("full_input_gate_sha256") != sha(gate_path)
            or q19_lock.get("mask_bank_sha256") != sha(bank_path)
            or q19_review.get("status") != "PASS"
            or q19_review.get("lock_sha256") != coalition_lock_sha
            or q19_seal.get("status") != "PASS"
            or q19_seal.get("lock_sha256") != coalition_lock_sha
            or q19.get("gate", {}).get("seal_sha256") != sha(q19_seal_path)):
        raise PermissionError("q19 coalition ATT science seal differs")
    failed_debit = q19_lock.get("prior_failed_attempt_debit_cpu_s")
    if (not isinstance(failed_debit, (int, float))
            or abs(failed_debit - 0.2250022920779884) > 1e-8):
        raise PermissionError("failed q16 attempt debit was lost")
    producers = {"units": {**q16["units"], **q18["units"], **q19["units"]}}
    units = {}
    for uid, entry in proposal["unit_map"].items():
        if entry.get("stage") != "r2_privacy_score":
            continue
        deps = set(entry.get("deps", []))
        if (entry.get("unit_id") != uid
                or uid not in ("R2_PRIVACY_SCORE_S0", "R2_PRIVACY_SCORE_S1")
                or entry.get("seed") not in (0, 1)
                or entry.get("distinct_pairs") != 2
                or entry.get("exact_input_count") != 126
                or len(entry.get("inputs", {})) != 126
                or len(deps) != 8
                or sum(dep in q16["units"] for dep in deps) != 2
                or sum(dep in q18["units"] for dep in deps) != 4
                or sum(dep in q19["units"] for dep in deps) != 2
                or entry.get("role_inputs") != ["INNER_SELECTION:sex"]):
            raise PermissionError("q20 seed score unit/dependency inventory differs")
        inputs, role_inputs = resolve_att_inputs(
            entry["inputs"], repo, study, manifest,
            ("INNER_SELECTION:sex",), producers, deps)
        if len(inputs) != 126:
            raise PermissionError("q20 score exact input count differs")
        outputs = []
        for name in entry["outputs"]:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise PermissionError("q20 output path escapes worktree")
            output = (repo / relative).resolve(strict=False)
            if not output.is_relative_to(private) or output.exists():
                raise PermissionError("q20 output exists or escapes custody")
            outputs.append(str(output))
        expected_dir = private / "privacy_score" / f"seed{entry['seed']}"
        if set(outputs) != {str(expected_dir / "PRIVACY_SCORES.json"),
                            str(expected_dir / "PRIVACY_SCORE_RECEIPT.json")}:
            raise PermissionError("q20 seed score outputs differ")
        script = study / "r2/privacy_score_worker.py"
        argv = [str(python), str(script), "--seed", str(entry["seed"]),
                "--unit-id", uid, "--queue-root", str(queue_path),
                "--lock", str(lock_path), "--gate", str(gate_path),
                "--manifest", str(manifest_path),
                "--export-root", str(private / "role_export_v1"),
                "--private-root", str(private),
                "--output", str(expected_dir / "PRIVACY_SCORE_RECEIPT.json")]
        units[uid] = {
            "unit_id": uid, "mode": "real", "stage": "r2_privacy_score",
            "argv": argv, "script": str(script), "script_sha256": sha(script),
            "cwd": str(repo), "cpu_reserve_s": 900, "wall_limit_s": 1800,
            "max_rss_gib": 3, "inputs": dict(sorted(inputs.items())),
            "outputs": sorted(outputs), "deps": [],
            "source_map_sha256": sha(manifest_path),
            "role_access": ["INNER_SELECTION"],
            "role_inputs": role_inputs, "allow_child_processes": False,
            "data_access": "INNER_SEX_FROZEN_SCORE_ONLY",
            "seal_sha256": "$LOCAL_SEAL_SHA256",
        }
    if set(units) != {"R2_PRIVACY_SCORE_S0", "R2_PRIVACY_SCORE_S1"}:
        raise PermissionError("q20 two seed score units differ")
    registry = {"schema": "pcrl-utility-unit-registry-v1",
                "stage": "frozen_reader_inner_privacy_scoring",
                "unit_specs": dict(sorted(units.items()))}
    lock = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "frozen_reader_inner_privacy_scoring",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "development_status": "exploratory_reused_Adult_people",
        "selected_recipe": "A", "paired_seeds": [0, 1],
        "selected_pairs_by_seed": proposal["selected_pairs_by_seed"],
        "admitted_mask_ids": [row["mask_id"] for row in
                              json.loads(bank_path.read_text())["masks"]],
        "allowed_roles": ["INNER_SELECTION"],
        "role_scope": "INNER_SELECTION SEX only after frozen ATT readers; no fit",
        "full_input_gate_sha256": sha(gate_path),
        "source_science_lock_sha256": gate["source_science_lock_sha256"],
        "common_row_unit_pair_sha256": gate["common_row_unit_pair_sha256"],
        "mask_bank_sha256": sha(bank_path),
        "pair_select_source_lock_sha256": sha(q13_lock_path),
        "release_source_lock_sha256": release_lock_sha,
        "local_attack_source_lock_sha256": local_lock_sha,
        "coalition_attack_source_lock_sha256": coalition_lock_sha,
        "pair_selection_audit_sha256": sha(audit_path),
        "independent_release_verification_sha256": sha(verification_path),
        "downstream_unit_proposal_sha256": sha(proposal_path),
        "prior_coalition_att_queue_state_sha256": sha(q19_path),
        "prior_failed_attempt_debit_cpu_s": failed_debit,
        "effective_campaign_used_cpu_s_at_q20_start":
        carry["charged_cpu_s"] + failed_debit,
        "required_queue_carry_forward": carry,
        "privacy_score_units": 2,
        "private_unit_registry_sha256": "__AFTER_REGISTRY_WRITE__",
        "source_map_sha256": sha(manifest_path),
        "prior_package_relative_to_worktree": str(STUDY_RELATIVE),
        "governed_source_sha256_by_path": {
            name: sha(repo / name) for name in SOURCE_MODULES},
        "governed_protocol_sha256_by_relative_path": {
            name: sha(study / name) for name in PROTOCOL_FILES},
        "shared_slot_sha256": sha(slot),
        "excluded_inputs": ["old assessment person arrays",
                            "HEAD_VALIDATION labels", "ATTACK_FIT_NEW SEX",
                            "INNER task labels"],
        "next_stage": "R3 independent numerical replay of all saved privacy scores",
    }
    _write_once(registry_path, registry)
    lock["private_unit_registry_sha256"] = sha(registry_path)
    _write_once(lock_path, lock)
    return {"lock_sha256": sha(lock_path),
            "registry_sha256": sha(registry_path),
            "unit_count": 2,
            "input_count_by_unit": {uid: len(spec["inputs"])
                                    for uid, spec in units.items()},
            "required_queue_carry_forward": carry}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--slot", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.python, args.slot), sort_keys=True))
