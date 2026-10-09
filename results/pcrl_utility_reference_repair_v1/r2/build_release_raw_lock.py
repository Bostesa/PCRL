"""R2 proposal to seal selected complete releases and canonical raw readers.

R0 is the only party that executes this builder or writes the shared lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


STUDY_RELATIVE = Path("results/pcrl_utility_reference_repair_v1")
PROPOSAL_SHA256 = "79bfaae53e3cdf2415e1147740f52f3e0a8ae42ada3a0cef041009c7f175933a"
QUEUE_UID = re.compile(r"[A-Z0-9][A-Z0-9_-]{2,119}\Z")
STAGES = {"r2_release": "r2/release_worker.py",
          "r2_raw_union": "r2/raw_union_worker.py"}
SOURCE_MODULES = (
    "results/pcrl_utility_reference_repair_v1/r0/admission.py",
    "results/pcrl_utility_reference_repair_v1/r0/queue_adapter.py",
    "results/pcrl_utility_reference_repair_v1/r1/role_store.py",
    "results/pcrl_utility_reference_repair_v1/r1/source_task.py",
    "results/pcrl_utility_reference_repair_v1/r2/admission.py",
    "results/pcrl_utility_reference_repair_v1/r2/frozen_bank.py",
    "results/pcrl_utility_reference_repair_v1/r2/privacy_store.py",
    "results/pcrl_utility_reference_repair_v1/r2/mask_adapter.py",
    "results/pcrl_utility_reference_repair_v1/r2/release_loader.py",
    "results/pcrl_utility_reference_repair_v1/r2/complete_views.py",
    "results/pcrl_utility_reference_repair_v1/r2/release_custody.py",
    "results/pcrl_utility_reference_repair_v1/r2/release_worker.py",
    "results/pcrl_utility_reference_repair_v1/r2/raw_union_worker.py",
    "results/pcrl_utility_reference_repair_v1/r2/build_release_raw_lock.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m0/queue.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m4/raw_union.py",
    "results/pcrl_method_portfolio_v1/teams/c/training.py",
    "results/pcrl_method_portfolio_v1/teams/c/prototype.py",
    "results/pcrl_method_portfolio_v1/shared/common_eval.py",
)
PROTOCOL_FILES = (
    "EXECUTION_PLAN.md", "RESOURCE_PLAN.json", "AGENT_ROSTER.json",
    "m1/DATA_ACCESS_MANIFEST.json", "FULL_INPUT_GATE.json",
    "r2/DOWNSTREAM_UNIT_PROPOSAL.json", "r2/R2_RESOURCE_FORECAST.json",
    "r2/RAW_UNION_ACCOUNTING_AMENDMENT.md",
    "r2/PRIVACY_SCORE_ACCESS_AMENDMENT.md",
    "r2/R2_ACCESS_DEVIATION_20261009.md",
    "r2/ATT_FEATURE_LOADER_AMENDMENT_20261009.md",
    "LOCK_AMENDMENTS.md",
    "r3/PAIR_SELECTION_AUDIT.json", "r1/UTILITY_INTERVAL_PROTOCOL.md",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _full_sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def materialize_unit_inputs(
    proposal_inputs: dict, repo: Path, study: Path, manifest: dict,
    role_keys: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Map exact proposal paths to queue inputs without opening role files."""
    root = repo.resolve(strict=True)
    study_root = study.resolve(strict=True)
    role_root = study_root / ".private/role_export_v1"
    expected_roles = {}
    for key in role_keys:
        relative_value = Path(manifest["role_file_relative_path"][key])
        digest = manifest["role_file_sha256"][key]
        if (relative_value.is_absolute() or ".." in relative_value.parts
                or relative_value.parts[:2] != (".private", "role_export_v1")
                or not _full_sha(digest)):
            raise PermissionError("role manifest identity is incomplete")
        role_path = (study_root / relative_value).resolve(strict=False)
        if not role_path.is_relative_to(role_root):
            raise PermissionError("role manifest path escapes custody")
        expected_roles[str(role_path)] = (key, digest)
    inputs = {}
    declared = {}
    seen_roles = set()
    for name, record in proposal_inputs.items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts
                or not isinstance(record, dict)
                or not _full_sha(record.get("sha256"))):
            raise PermissionError("proposal input identity is incomplete")
        path = (root / relative).resolve(strict=False)
        if not path.is_relative_to(root):
            raise PermissionError("proposal input escapes worktree")
        source = record.get("source")
        if source == "FROZEN_MANIFEST_ROLE_METADATA":
            if (str(path) not in expected_roles
                    or expected_roles[str(path)][1] != record["sha256"]):
                raise PermissionError("proposal role differs from sealed manifest")
            key = expected_roles[str(path)][0]
            declared[key] = {"path": str(path), "sha256": record["sha256"]}
            seen_roles.add(key)
        elif source == "FROZEN_EXISTING":
            if (path.is_relative_to(role_root) or not path.is_file()
                    or sha(path) != record["sha256"]):
                raise PermissionError("frozen non-role proposal bytes differ")
        else:
            raise PermissionError("q16 cannot consume a future unit input")
        inputs[str(path)] = record["sha256"]
    if seen_roles != set(role_keys):
        raise PermissionError("proposal role inventory differs from manifest")
    return dict(sorted(inputs.items())), dict(sorted(declared.items()))


def _add(inputs: dict[str, str], path: Path, expected: str | None = None) -> str:
    resolved = path.resolve(strict=True)
    actual = sha(resolved)
    if expected is not None and actual != expected:
        raise PermissionError("q16 predecessor bytes differ")
    inputs[str(resolved)] = actual
    return actual


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2,
                                allow_nan=False) + "\n")


def validate_interval_predecessors(
    q14: dict, q15: dict, q14_sha: str, q15_sha: str,
    replay: dict, replay_sha: str, replay_path: Path,
) -> dict:
    """Require q14/q15 to be complete, hash-bound, and budget carried."""
    interval_units = q14.get("units", {})
    verifier_units = q15.get("units", {})
    if (set(interval_units) != {"R1_UTILITY_INTERVALS"}
            or interval_units["R1_UTILITY_INTERVALS"].get("status") != "DONE"
            or set(verifier_units) != {"R3_UTILITY_INTERVAL_REPLAY"}
            or verifier_units["R3_UTILITY_INTERVAL_REPLAY"].get("status") != "DONE"
            or q15.get("carry_forward", {}).get("archived_queue_relative") !=
            "queue_g14"
            or q15["carry_forward"].get("archived_queue_sha256") != q14_sha
            or q15["carry_forward"].get("unit_count") != 1
            or replay.get("schema") !=
            "pcrl-utility-r3-independent-interval-replay-v1"
            or replay.get("status") != "REPLAYED"
            or replay.get("family_size") != 24
            or replay.get("replicates") != 2048
            or replay.get("all_clause_statuses_match") is not True
            or verifier_units["R3_UTILITY_INTERVAL_REPLAY"].get(
                "output_sha256", {}).get(str(replay_path)) != replay_sha):
        raise PermissionError("q14/q15 interval or independent replay incomplete")
    for state in (q14, q15):
        total = state.get("used_cpu_s")
        carried = state.get("carry_forward", {}).get("charged_cpu_s")
        charges = [record.get("charged_cpu_s") for record in state["units"].values()]
        if (not isinstance(total, (int, float)) or not math.isfinite(total)
                or not isinstance(carried, (int, float)) or not math.isfinite(carried)
                or any(not isinstance(x, (int, float)) or not math.isfinite(x)
                       or x < 0 for x in charges)
                or abs(total - carried - sum(charges)) > 1e-5):
            raise PermissionError("q14/q15 compute carry-forward differs")
    return {"archived_queue_relative": "queue_g15",
            "archived_queue_sha256": q15_sha,
            "charged_cpu_s": float(q15["used_cpu_s"]), "unit_count": 1}


def failed_loader_debit(failed: dict, carry: dict,
                        expected_uids: set[str]) -> float:
    """Count the unrun archived q16 failure outside the fresh queue carry."""
    units = failed.get("units", {})
    failures = [(uid, row) for uid, row in units.items()
                if row.get("status") == "FAILED"]
    if (set(units) != expected_uids or failed.get("carry_forward") != carry
            or len(failures) != 1
            or not failures[0][0].startswith("R2_RAW_UNION_")
            or any(row.get("status") not in ("FAILED", "QUEUED")
                   for row in units.values())
            or any(row.get("output_sha256") not in (None, {})
                   for row in units.values())):
        raise PermissionError("archived failed loader attempt differs")
    charge = failures[0][1].get("charged_cpu_s")
    total = failed.get("used_cpu_s")
    if (not isinstance(charge, (int, float)) or not math.isfinite(charge)
            or charge <= 0
            or not isinstance(total, (int, float)) or not math.isfinite(total)
            or abs(total - carry["charged_cpu_s"] - charge) > 1e-5):
        raise PermissionError("archived failed loader CPU debit differs")
    return float(charge)


def build(repo: Path, python: Path, slot: Path, *, generation: str = "q16") -> dict:
    """R0-only write-once q16 builder; never run while q15 is incomplete."""
    if generation != "q16":
        raise ValueError("release/raw stage generation must be q16")
    repo = repo.resolve(strict=True)
    study = repo / STUDY_RELATIVE
    private = study / ".private"
    lock_path = study / "CAMPAIGN_LOCK.json"
    registry_path = private / "UNIT_SPECS.json"
    queue_path = private / "queue_g16"
    if lock_path.exists() or registry_path.exists() or queue_path.exists():
        raise FileExistsError("q16 shared lock, registry or queue already exists")
    if (python.name != "python" or python.parent.name != "bin"
            or python.parent.parent.name != ".venv" or not python.is_file()
            or not slot.is_file()):
        raise FileNotFoundError("project venv or shared slot missing")
    proposal_path = study / "r2/DOWNSTREAM_UNIT_PROPOSAL.json"
    if sha(proposal_path) != PROPOSAL_SHA256:
        raise PermissionError("independently reviewed q16 proposal differs")
    proposal = json.loads(proposal_path.read_text())
    expected_counts = {"r2_release": 4, "r2_raw_union": 7,
                       "r2_attack_fit": 12, "r2_privacy_score": 2}
    if (proposal.get("schema") != "pcrl-utility-r2-downstream-unit-proposal-v1"
            or proposal.get("status") !=
            "PROSPECTIVE_AFTER_FIXED_PAIR_SELECTION_BEFORE_ATT_OR_INNER_SEX"
            or proposal.get("unit_count_by_stage") != expected_counts
            or proposal.get("new_att_fit_count") != 86
            or not proposal.get("unit_id_convention", "").startswith(
                "Uppercase ASCII queue IDs")
            or any(not QUEUE_UID.fullmatch(uid) or record.get("unit_id") != uid
                   for uid, record in proposal.get("unit_map", {}).items())
            or any(dep not in proposal["unit_map"] or not QUEUE_UID.fullmatch(dep)
                   for record in proposal["unit_map"].values()
                   for dep in record.get("deps", []))):
        raise PermissionError("selected downstream design or counted units differ")
    gate_path = study / "FULL_INPUT_GATE.json"
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    bank_path = repo / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
    gate = json.loads(gate_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (gate.get("schema") != "pcrl-utility-full-input-gate-v1"
            or gate.get("status") != "PASS_FULL_INPUT_POINT_GATE"
            or gate.get("selected_recipe") != "A"
            or sha(gate_path) != proposal["full_input_gate_sha256"]
            or sha(manifest_path) != proposal["source_manifest_sha256"]
            or sha(bank_path) != proposal["mask_bank_sha256"]):
        raise PermissionError("F, role manifest or fixed bank differs")
    q13_path = private / "queue_g13/QUEUE.json"
    pair_lock_path = private / "stage_seals/pair_search/CAMPAIGN_LOCK.json"
    pair_audit_path = study / "r3/PAIR_SELECTION_AUDIT.json"
    pair_audit = json.loads(pair_audit_path.read_text())
    if (pair_audit.get("schema") != "pcrl-utility-r3-pair-selection-audit-v1"
            or pair_audit.get("status") != "PASS_AGGREGATE_SELECTION_REPLAY"
            or pair_audit.get("pair_search_lock_sha256") != sha(pair_lock_path)
            or pair_audit.get("pair_search_queue_state_sha256") != sha(q13_path)
            or pair_audit.get("mask_bank_sha256") != sha(bank_path)):
        raise PermissionError("q13 selected pairs lack independent audit")
    for seed in ("0", "1"):
        source = proposal["selected_source_by_seed"][seed]
        audited = pair_audit["by_seed"][seed]
        if (source["pair_receipt_sha256"] != audited["pair_receipt_sha256"]
                or source["control_plan_sha256"] != audited["control_plan_sha256"]
                or source["distinct_pair_count"] != audited["distinct_selected_pairs"]):
            raise PermissionError("q16 selected pair receipts differ from R3 audit")
    q14_path = private / "queue_g14/QUEUE.json"
    q15_path = private / "queue_g15/QUEUE.json"
    q14_lock_path = private / "stage_seals/utility_intervals/CAMPAIGN_LOCK.json"
    q15_lock_path = private / "stage_seals/utility_interval_replay/CAMPAIGN_LOCK.json"
    replay_path = private / "r3/UTILITY_INTERVAL_REPLAY.json"
    q14, q15 = json.loads(q14_path.read_text()), json.loads(q15_path.read_text())
    replay = json.loads(replay_path.read_text())
    carry = validate_interval_predecessors(
        q14, q15, sha(q14_path), sha(q15_path), replay, sha(replay_path),
        replay_path)
    if (replay.get("full_input_gate_sha256") != sha(gate_path)
            or replay.get("r1_interval_output_sha256") !=
            sha(private / "r1/intervals/SIMULTANEOUS_UTILITY_INTERVALS.json")
            or json.loads(q14_lock_path.read_text()).get("full_input_gate_sha256") !=
            sha(gate_path)
            or json.loads(q15_lock_path.read_text()).get("full_input_gate_sha256") !=
            sha(gate_path)):
        raise PermissionError("utility intervals changed reference gate")
    failed_root = private / "lock_attempts/q16_att_feature_loader"
    failed_queue_path = failed_root / "queue_g16/QUEUE.json"
    failed_lock_path = failed_root / "CAMPAIGN_LOCK.json"
    failed_registry_path = failed_root / "UNIT_SPECS.json"
    failed_review_path = failed_root / "PREFIT_REVIEW.json"
    failed_seal_path = failed_root / "LOCAL_SEAL_RECEIPT.json"
    failed_queue = json.loads(failed_queue_path.read_text())
    failed_lock = json.loads(failed_lock_path.read_text())
    failed_review = json.loads(failed_review_path.read_text())
    failed_seal = json.loads(failed_seal_path.read_text())
    expected_failed_uids = {uid for uid, entry in proposal["unit_map"].items()
                            if entry["stage"] in STAGES}
    failed_debit = failed_loader_debit(
        failed_queue, carry, expected_failed_uids)
    if (abs(failed_debit - 0.2250022920779884) > 1e-8
            or failed_lock.get("stage") !=
            "selected_release_and_raw_reader_custody"
            or failed_lock.get("downstream_unit_proposal_sha256") !=
            sha(proposal_path)
            or failed_lock.get("full_input_gate_sha256") != sha(gate_path)
            or failed_lock.get("required_queue_carry_forward") != carry
            or failed_lock.get("private_unit_registry_sha256") !=
            sha(failed_registry_path)
            or failed_review.get("status") != "PASS"
            or failed_review.get("lock_sha256") != sha(failed_lock_path)
            or failed_seal.get("status") != "PASS"
            or failed_seal.get("lock_sha256") != sha(failed_lock_path)
            or failed_queue.get("gate", {}).get("seal_sha256") !=
            sha(failed_seal_path)):
        raise PermissionError("archived q16 failed attempt custody differs")
    # Existing admitted model/task bytes may be rehashed here. Role-only
    # exports are declared solely from the sealed manifest until q16 admission.
    shared = {}
    for path in (proposal_path, study / "r2/R2_ACCESS_DEVIATION_20261009.md",
                 study / "r2/R2_RESOURCE_FORECAST.json",
                 study / "r2/RAW_UNION_ACCOUNTING_AMENDMENT.md",
                 pair_audit_path, q13_path, pair_lock_path,
                 q14_path, q14_lock_path, q15_path, q15_lock_path,
                 replay_path,
                 private / "r1/intervals/SIMULTANEOUS_UTILITY_INTERVALS.json",
                 failed_queue_path, failed_lock_path, failed_registry_path,
                 failed_review_path, failed_seal_path,
                 study / "r2/ATT_FEATURE_LOADER_AMENDMENT_20261009.md"):
        _add(shared, path)
    units = {}
    for uid, entry in proposal["unit_map"].items():
        stage = entry.get("stage")
        if stage not in STAGES:
            continue
        if (entry.get("unit_id") != uid or entry.get("deps") != []
                or entry.get("seed") not in (0, 1)):
            raise PermissionError("q16 unit identity or stage dependency differs")
        role_keys = tuple(entry["role_inputs"])
        expected_roles = ({"ATTACK_FIT_NEW:features", "INNER_SELECTION:features"}
                          if stage == "r2_release" else
                          {"ATTACK_FIT_NEW:features", "ATTACK_FIT_NEW:sex",
                           "INNER_SELECTION:features"})
        if set(role_keys) != expected_roles:
            raise PermissionError("q16 role scope differs from selected proposal")
        inputs, role_inputs = materialize_unit_inputs(
            entry["inputs"], repo, study, manifest, role_keys)
        inputs.update(shared)
        script = study / STAGES[stage]
        outputs = []
        for name in entry["outputs"]:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise PermissionError("q16 output path escapes worktree")
            output = (repo / relative).resolve(strict=False)
            if (not output.is_relative_to(private) or output.exists()):
                raise PermissionError("q16 output exists or escapes private custody")
            outputs.append(str(output))
        expected_count = 3 if stage == "r2_release" else 11
        target = "RELEASE_RECEIPT.json" if stage == "r2_release" else "RAW_UNION_INDEX.json"
        primary = [path for path in outputs if path.endswith("/" + target)]
        if len(outputs) != expected_count or len(primary) != 1:
            raise PermissionError("q16 stage output inventory differs")
        args = (["--pair-id", entry["pair_id"]] if stage == "r2_release"
                else ["--union-sha256", entry["union_sha256"]])
        argv = [str(python), str(script), "--seed", str(entry["seed"])] + args + [
            "--unit-id", uid, "--queue-root", str(queue_path),
            "--lock", str(lock_path), "--gate", str(gate_path),
            "--manifest", str(manifest_path),
            "--export-root", str(private / "role_export_v1"),
            "--private-root", str(private), "--output", primary[0]]
        units[uid] = {
            "unit_id": uid, "mode": "real", "stage": stage,
            "argv": argv, "script": str(script), "script_sha256": sha(script),
            "cwd": str(repo),
            "cpu_reserve_s": 300 if stage == "r2_release" else 900,
            "wall_limit_s": 900 if stage == "r2_release" else 1800,
            "max_rss_gib": 3, "inputs": dict(sorted(inputs.items())),
            "outputs": sorted(outputs), "deps": [],
            "source_map_sha256": sha(manifest_path),
            "role_access": ["ATTACK_FIT_NEW", "INNER_SELECTION"],
            "role_inputs": role_inputs, "allow_child_processes": False,
            "data_access": ("ATT_AND_INNER_FEATURES_ONLY_NO_SEX"
                            if stage == "r2_release" else
                            "ATT_FEATURES_AND_SEX_INNER_FEATURES_NO_INNER_SEX"),
            "seal_sha256": "$LOCAL_SEAL_SHA256",
        }
    if (len(units) != 11 or
            sum(value["stage"] == "r2_release" for value in units.values()) != 4
            or sum(value["stage"] == "r2_raw_union" for value in units.values()) != 7):
        raise PermissionError("q16 complete release/raw unit count differs")
    all_outputs = [name for unit_spec in units.values()
                   for name in unit_spec["outputs"]]
    if len(all_outputs) != len(set(all_outputs)):
        raise PermissionError("q16 output ownership collides")
    registry = {"schema": "pcrl-utility-unit-registry-v1",
                "stage": "selected_release_and_raw_reader_custody",
                "unit_specs": dict(sorted(units.items()))}
    lock = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "selected_release_and_raw_reader_custody",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "development_status": "exploratory_reused_Adult_people",
        "selected_recipe": "A", "paired_seeds": [0, 1],
        "admitted_mask_ids": [item["mask_id"] for item in
                              json.loads(bank_path.read_text())["masks"]],
        "allowed_roles": ["ATTACK_FIT_NEW", "INNER_SELECTION"],
        "role_scope": "release ATT/INNER features; raw ATT features+SEX and INNER features; never INNER SEX",
        "full_input_gate_sha256": sha(gate_path),
        "source_science_lock_sha256": gate["source_science_lock_sha256"],
        "common_row_unit_pair_sha256": gate["common_row_unit_pair_sha256"],
        "mask_bank_sha256": sha(bank_path),
        "pair_select_source_lock_sha256": sha(pair_lock_path),
        "pair_selection_audit_sha256": sha(pair_audit_path),
        "downstream_unit_proposal_sha256": sha(proposal_path),
        "access_deviation_note_sha256": sha(
            study / "r2/R2_ACCESS_DEVIATION_20261009.md"),
        "prior_interval_queue_state_sha256": sha(q14_path),
        "prior_interval_replay_queue_state_sha256": sha(q15_path),
        "prior_interval_replay_sha256": sha(replay_path),
        "failed_attempt_queue_state_sha256": sha(failed_queue_path),
        "failed_attempt_science_lock_sha256": sha(failed_lock_path),
        "failed_attempt_debit_cpu_s": failed_debit,
        "effective_campaign_used_cpu_s_at_q16_start":
        carry["charged_cpu_s"] + failed_debit,
        "att_feature_loader_amendment_sha256": sha(
            study / "r2/ATT_FEATURE_LOADER_AMENDMENT_20261009.md"),
        "interval_family_status": replay["family_status"],
        "required_queue_carry_forward": carry,
        "release_units": 4, "canonical_raw_union_units": 7,
        "raw_reader_fits": 14,
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
        "next_stage": "independent complete release replay before fresh ATT readers",
    }
    _write_once(registry_path, registry)
    lock["private_unit_registry_sha256"] = sha(registry_path)
    _write_once(lock_path, lock)
    return {"lock_sha256": sha(lock_path),
            "registry_sha256": sha(registry_path),
            "unit_count": len(units), "release_count": 4,
            "raw_union_count": 7,
            "input_count_by_unit": {uid: len(spec["inputs"])
                                    for uid, spec in units.items()},
            "required_queue_carry_forward": carry}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--slot", type=Path, required=True)
    parser.add_argument("--queue-generation", default="q16")
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.python, args.slot,
                           generation=args.queue_generation), sort_keys=True))
