"""Prepare q18 local ATT readers after independent release replay.

R0 alone executes this builder and writes the shared lock. Coalition ATT
readers are sealed separately after these local outputs have literal hashes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from results.pcrl_utility_reference_repair_v1.r2.build_release_raw_lock import (
    PROPOSAL_SHA256, STUDY_RELATIVE, _add, _full_sha, _write_once,
    materialize_unit_inputs, sha,
)


SOURCE_MODULES = (
    "results/pcrl_utility_reference_repair_v1/r0/admission.py",
    "results/pcrl_utility_reference_repair_v1/r0/queue_adapter.py",
    "results/pcrl_utility_reference_repair_v1/r1/role_store.py",
    "results/pcrl_utility_reference_repair_v1/r2/admission.py",
    "results/pcrl_utility_reference_repair_v1/r2/frozen_bank.py",
    "results/pcrl_utility_reference_repair_v1/r2/privacy_store.py",
    "results/pcrl_utility_reference_repair_v1/r2/mask_adapter.py",
    "results/pcrl_utility_reference_repair_v1/r2/release_loader.py",
    "results/pcrl_utility_reference_repair_v1/r2/complete_views.py",
    "results/pcrl_utility_reference_repair_v1/r2/release_custody.py",
    "results/pcrl_utility_reference_repair_v1/r2/release_worker.py",
    "results/pcrl_utility_reference_repair_v1/r2/raw_union_worker.py",
    "results/pcrl_utility_reference_repair_v1/r2/raw_search.py",
    "results/pcrl_utility_reference_repair_v1/r2/attack_fit.py",
    "results/pcrl_utility_reference_repair_v1/r2/control_fit.py",
    "results/pcrl_utility_reference_repair_v1/r2/attack_fit_worker.py",
    "results/pcrl_utility_reference_repair_v1/r2/build_release_raw_lock.py",
    "results/pcrl_utility_reference_repair_v1/r2/build_local_att_lock.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m0/queue.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m4/raw_union.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m4/attack_runner.py",
    "results/pcrl_method_portfolio_v1/shared/attack_eval.py",
)
PROTOCOL_FILES = (
    "EXECUTION_PLAN.md", "RESOURCE_PLAN.json", "AGENT_ROSTER.json",
    "m1/DATA_ACCESS_MANIFEST.json", "FULL_INPUT_GATE.json",
    "r2/DOWNSTREAM_UNIT_PROPOSAL.json", "r2/R2_RESOURCE_FORECAST.json",
    "r2/RAW_UNION_ACCOUNTING_AMENDMENT.md",
    "r2/ATT_STAGE_SPLIT_AMENDMENT.md",
    "r2/R2_ACCESS_DEVIATION_20261009.md",
    "r3/PAIR_SELECTION_AUDIT.json",
    "r3/RELEASE_NUMERICAL_VERIFICATION.json",
)
ROLE_KEYS = ("ATTACK_FIT_NEW:features", "ATTACK_FIT_NEW:sex",
             "INNER_SELECTION:features")


def resolve_att_inputs(
    proposal_inputs: dict, repo: Path, study: Path, manifest: dict,
    role_keys: tuple[str, ...], q16: dict, required_producers: set[str],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Resolve only completed q16 outputs; preserve manifest-only role access."""
    root = repo.resolve(strict=True)
    amended = {}
    seen_producers = set()
    for name, record in proposal_inputs.items():
        if not isinstance(record, dict):
            raise PermissionError("q18 proposal input entry is incomplete")
        if record.get("sha256") != "FROM_PRODUCER_UNIT":
            amended[name] = record
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise PermissionError("q18 future input path escapes worktree")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise PermissionError("q18 future input escapes worktree")
        producer = record.get("producer_unit")
        if producer not in required_producers:
            raise PermissionError("q18 producer is not a required q16 unit")
        old = q16.get("units", {}).get(producer, {})
        expected = old.get("output_sha256", {}).get(str(path))
        if (old.get("status") != "DONE" or not _full_sha(expected)
                or sha(path) != expected):
            raise PermissionError("q18 producer output is incomplete or changed")
        amended[name] = {"source": "FROZEN_EXISTING", "sha256": expected}
        seen_producers.add(producer)
    if seen_producers != required_producers:
        raise PermissionError("q18 missing q16 producer dependencies")
    return materialize_unit_inputs(amended, repo, study, manifest, role_keys)


def validate_prior_stage(
    q16: dict, q17: dict, q16_sha: str, q17_sha: str, replay: dict,
    replay_sha: str, replay_path: Path, release_lock_sha: str,
    selected_pairs: dict,
) -> dict:
    """Require all release/raw units and feature-only replay to be DONE."""
    first = q16.get("units", {})
    second = q17.get("units", {})
    if (len(first) != 11
            or sum(uid.startswith("R2_RELEASE_") for uid in first) != 4
            or sum(uid.startswith("R2_RAW_UNION_") for uid in first) != 7
            or any(row.get("status") != "DONE" for row in first.values())
            or set(second) != {"R3_RELEASE_REPLAY"}
            or second["R3_RELEASE_REPLAY"].get("status") != "DONE"
            or q17.get("carry_forward", {}).get("archived_queue_relative") !=
            "queue_g16"
            or q17["carry_forward"].get("archived_queue_sha256") != q16_sha
            or q17["carry_forward"].get("unit_count") != 11
            or second["R3_RELEASE_REPLAY"].get("output_sha256", {}).get(
                str(replay_path)) != replay_sha
            or replay.get("schema") !=
            "pcrl-utility-r3-conditional-release-replay-v1"
            or replay.get("status") != "PASS_RELEASE_REPLAY"
            or replay.get("source_release_lock_sha256") != release_lock_sha
            or replay.get("selected_pairs_by_seed") != selected_pairs
            or replay.get("role_access") !=
            ["ATTACK_FIT_NEW:features", "INNER_SELECTION:features"]):
        raise PermissionError("q16/q17 release custody or replay incomplete")
    for name in ("max_abs_probability_difference",
                 "max_abs_complete_view_difference"):
        value = replay.get(name)
        if (not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < 0 or value > 2e-7):
            raise PermissionError("q17 independent release difference exceeds cap")
    for state in (q16, q17):
        previous = state.get("carry_forward", {}).get("charged_cpu_s")
        total = state.get("used_cpu_s")
        charges = [row.get("charged_cpu_s") for row in state["units"].values()]
        if (not isinstance(previous, (int, float)) or not math.isfinite(previous)
                or not isinstance(total, (int, float)) or not math.isfinite(total)
                or any(not isinstance(x, (int, float)) or not math.isfinite(x)
                       or x < 0 for x in charges)
                or abs(total - previous - sum(charges)) > 1e-5):
            raise PermissionError("q16/q17 compute carry-forward differs")
    return {"archived_queue_relative": "queue_g17",
            "archived_queue_sha256": q17_sha,
            "charged_cpu_s": float(q17["used_cpu_s"]), "unit_count": 1}


def build(repo: Path, python: Path, slot: Path) -> dict:
    """R0-only write-once q18 builder; q17 must be independently complete."""
    repo = repo.resolve(strict=True)
    study = repo / STUDY_RELATIVE
    private = study / ".private"
    lock_path = study / "CAMPAIGN_LOCK.json"
    registry_path = private / "UNIT_SPECS.json"
    queue_path = private / "queue_g18"
    if lock_path.exists() or registry_path.exists() or queue_path.exists():
        raise FileExistsError("q18 shared lock, registry or queue exists")
    if (python.name != "python" or python.parent.name != "bin"
            or python.parent.parent.name != ".venv" or not python.is_file()
            or not slot.is_file()):
        raise FileNotFoundError("project venv or shared slot missing")
    proposal_path = study / "r2/DOWNSTREAM_UNIT_PROPOSAL.json"
    if sha(proposal_path) != PROPOSAL_SHA256:
        raise PermissionError("reviewed selected ATT proposal differs")
    proposal = json.loads(proposal_path.read_text())
    if (proposal.get("schema") != "pcrl-utility-r2-downstream-unit-proposal-v1"
            or proposal.get("unit_count_by_stage", {}).get("r2_attack_fit") != 12
            or proposal.get("new_att_fit_count") != 86):
        raise PermissionError("q18 fixed ATT reader envelope differs")
    gate_path = study / "FULL_INPUT_GATE.json"
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    bank_path = repo / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
    audit_path = study / "r3/PAIR_SELECTION_AUDIT.json"
    gate = json.loads(gate_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (gate.get("status") != "PASS_FULL_INPUT_POINT_GATE"
            or gate.get("selected_recipe") != "A"
            or sha(gate_path) != proposal["full_input_gate_sha256"]
            or sha(manifest_path) != proposal["source_manifest_sha256"]
            or sha(bank_path) != proposal["mask_bank_sha256"]
            or json.loads(audit_path.read_text()).get("status") !=
            "PASS_AGGREGATE_SELECTION_REPLAY"):
        raise PermissionError("selected F/bank/manifest/pair audit differs")
    q16_path = private / "queue_g16/QUEUE.json"
    q17_path = private / "queue_g17/QUEUE.json"
    q13_lock_path = private / "stage_seals/pair_search/CAMPAIGN_LOCK.json"
    q16_lock_path = private / "stage_seals/release_raw/CAMPAIGN_LOCK.json"
    q17_lock_path = private / "stage_seals/release_replay/CAMPAIGN_LOCK.json"
    replay_path = private / "r3/RELEASE_REPLAY.json"
    verification_path = study / "r3/RELEASE_NUMERICAL_VERIFICATION.json"
    q16 = json.loads(q16_path.read_text())
    q17 = json.loads(q17_path.read_text())
    replay = json.loads(replay_path.read_text())
    q16_sha, q17_sha = sha(q16_path), sha(q17_path)
    release_lock_sha = sha(q16_lock_path)
    q16_lock = json.loads(q16_lock_path.read_text())
    q17_lock = json.loads(q17_lock_path.read_text())
    verification = json.loads(verification_path.read_text())
    carry = validate_prior_stage(
        q16, q17, q16_sha, q17_sha, replay, sha(replay_path),
        replay_path, release_lock_sha, proposal["selected_pairs_by_seed"])
    if (q17_lock.get("stage") != "independent_selected_release_replay"
            or q17_lock.get("prior_release_lock_sha256") != release_lock_sha
            or q17_lock.get("full_input_gate_sha256") != sha(gate_path)
            or q17_lock.get("mask_bank_sha256") != sha(bank_path)
            or q17_lock.get("pair_selection_audit_sha256") != sha(audit_path)
            or replay.get("full_input_gate_sha256") != sha(gate_path)
            or replay.get("pair_selection_audit_sha256") != sha(audit_path)):
        raise PermissionError("q17 science lock or source replay differs")
    if (verification.get("schema") !=
            "pcrl-utility-r3-release-numerical-verification-v1"
            or verification.get("status") !=
            "PASS_INDEPENDENT_COMPLETE_VIEW_REPLAY"
            or verification.get("q17_queue_sha256") != q17_sha
            or verification.get("q17_science_lock_sha256") !=
            sha(q17_lock_path)
            or verification.get("independent_replay_output_sha256") !=
            sha(replay_path)
            or verification.get("source_release_lock_sha256") !=
            release_lock_sha
            or verification.get("selected_pairs_by_seed") !=
            proposal["selected_pairs_by_seed"]):
        raise PermissionError("public R3 q17 verification differs")
    failed_debit = q16_lock.get("failed_attempt_debit_cpu_s")
    if (q16_lock.get("stage") != "selected_release_and_raw_reader_custody"
            or q16_lock.get("downstream_unit_proposal_sha256") !=
            sha(proposal_path)
            or q16_lock.get("full_input_gate_sha256") != sha(gate_path)
            or q16_lock.get("mask_bank_sha256") != sha(bank_path)
            or not isinstance(failed_debit, (int, float))
            or not math.isfinite(failed_debit)
            or abs(failed_debit - 0.2250022920779884) > 1e-8):
        raise PermissionError("q16 release science or failed-attempt debit differs")
    shared = {}
    for path in (proposal_path, gate_path, manifest_path, bank_path,
                 audit_path, q13_lock_path, q16_path, q16_lock_path,
                 q17_path, q17_lock_path, replay_path, verification_path,
                 study / "r2/R2_RESOURCE_FORECAST.json",
                 study / "r2/ATT_STAGE_SPLIT_AMENDMENT.md"):
        _add(shared, path)
    from results.pcrl_utility_reference_repair_v1.r2.attack_fit import (
        expected_outputs,
    )
    units = {}
    external_deps = {}
    for uid, entry in proposal["unit_map"].items():
        if entry.get("stage") != "r2_attack_fit" or entry.get("view") == "coalition":
            continue
        if (entry.get("unit_id") != uid or entry.get("seed") not in (0, 1)
                or entry.get("view") not in ("income", "occupation_group")
                or len(entry.get("deps", [])) != 2
                or sum(dep.startswith("R2_RELEASE_") for dep in entry["deps"]) != 1
                or sum(dep.startswith("R2_RAW_UNION_") for dep in entry["deps"]) != 1
                or set(entry.get("role_inputs", [])) != set(ROLE_KEYS)):
            raise PermissionError("q18 local ATT unit or external deps differ")
        inputs, role_inputs = resolve_att_inputs(
            entry["inputs"], repo, study, manifest, ROLE_KEYS, q16,
            set(entry["deps"]))
        inputs.update(shared)
        outputs = []
        for name in entry["outputs"]:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise PermissionError("q18 output path escapes worktree")
            output = (repo / relative).resolve(strict=False)
            if not output.is_relative_to(private) or output.exists():
                raise PermissionError("q18 ATT output exists or escapes custody")
            outputs.append(str(output))
        primary = [name for name in outputs if name.endswith("/ATTACK_FIT_INDEX.json")]
        if (len(primary) != 1 or len(outputs) != 16
                or set(outputs) != expected_outputs(Path(primary[0]))):
            raise PermissionError("q18 local ATT file inventory differs")
        script = study / "r2/attack_fit_worker.py"
        argv = [str(python), str(script), "--seed", str(entry["seed"]),
                "--pair-id", entry["pair_id"], "--view", entry["view"],
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
        external_deps[uid] = sorted(entry["deps"])
    if (len(units) != 8
            or sum(uid.endswith("_INCOME") for uid in units) != 4
            or sum(uid.endswith("_OCCUPATION_GROUP") for uid in units) != 4):
        raise PermissionError("q18 eight-local-view inventory differs")
    outputs = [name for spec in units.values() for name in spec["outputs"]]
    if len(outputs) != len(set(outputs)):
        raise PermissionError("q18 ATT output ownership collides")
    registry = {"schema": "pcrl-utility-unit-registry-v1",
                "stage": "fresh_local_complete_view_readers",
                "unit_specs": dict(sorted(units.items()))}
    lock = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "fresh_local_complete_view_readers",
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
        "pair_selection_audit_sha256": sha(audit_path),
        "downstream_unit_proposal_sha256": sha(proposal_path),
        "prior_release_queue_state_sha256": q16_sha,
        "prior_release_replay_queue_state_sha256": q17_sha,
        "prior_release_replay_sha256": sha(replay_path),
        "independent_release_verification_sha256": sha(verification_path),
        "prior_failed_attempt_debit_cpu_s": failed_debit,
        "effective_campaign_used_cpu_s_at_q18_start":
        carry["charged_cpu_s"] + failed_debit,
        "required_queue_carry_forward": carry,
        "local_att_units": 8, "local_att_core_fits": 32,
        "local_att_engineering_probe_fits": 16,
        "external_q16_dependencies_by_unit": dict(sorted(external_deps.items())),
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
        "next_stage": "seal four coalition ATT units with exact local output hashes",
    }
    _write_once(registry_path, registry)
    lock["private_unit_registry_sha256"] = sha(registry_path)
    _write_once(lock_path, lock)
    return {"lock_sha256": sha(lock_path),
            "registry_sha256": sha(registry_path),
            "unit_count": 8, "required_queue_carry_forward": carry,
            "input_count_by_unit": {uid: len(spec["inputs"])
                                    for uid, spec in units.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--slot", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.python, args.slot), sort_keys=True))
