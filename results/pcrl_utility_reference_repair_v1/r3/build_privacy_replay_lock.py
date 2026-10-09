"""Prepare R3's separate post-score privacy replay seal for R0 to execute.

All role-export SHAs come from the frozen manifest and prior registries. This
builder never opens protected-label archives before the q21 queue seal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


STUDY_RELATIVE = Path("results/pcrl_utility_reference_repair_v1")
UNIT_ID = "R3_PRIVACY_REPLAY"
STAGE = "r3_privacy_replay"
ROLE_KEYS = ("ATTACK_FIT_NEW:sex", "INNER_SELECTION:sex")
SOURCE_MODULES = (
    "results/pcrl_utility_reference_repair_v1/r0/admission.py",
    "results/pcrl_utility_reference_repair_v1/r0/queue_adapter.py",
    "results/pcrl_utility_reference_repair_v1/r3/build_privacy_replay_lock.py",
    "results/pcrl_utility_reference_repair_v1/r3/privacy_math.py",
    "results/pcrl_utility_reference_repair_v1/r3/privacy_replay.py",
    "results/pcrl_utility_reference_repair_v1/r3/privacy_replay_worker.py",
    "results/pcrl_utility_reference_repair_v1/r3/source_reference.py",
    "results/pcrl_utility_reference_repair_v1/r3/task_replay.py",
    "results/pcrl_utility_reference_repair_v1/r3/independent_task.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m0/queue.py",
)
PROTOCOL_FILES = (
    "EXECUTION_PLAN.md", "RESOURCE_PLAN.json", "AGENT_ROSTER.json",
    "m1/DATA_ACCESS_MANIFEST.json", "FULL_INPUT_GATE.json",
    "r3/PAIR_SELECTION_AUDIT.json",
    "r3/RELEASE_NUMERICAL_VERIFICATION.json",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _full_sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _add(inputs: dict[str, str], path: Path,
         expected: str | None = None) -> str:
    real = path.resolve(strict=True)
    digest = sha(real)
    if expected is not None and digest != expected:
        raise PermissionError("q21 frozen nonrole input bytes differ")
    old = inputs.get(str(real))
    if old is not None and old != digest:
        raise PermissionError("q21 duplicate input identity differs")
    inputs[str(real)] = digest
    return digest


def bind_sex_roles(study: Path, manifest: dict, prior_specs: dict) -> dict:
    """Bind q18 ATT SEX and q20 INNER SEX metadata without opening either NPZ."""
    root = study.resolve(strict=True)
    declarations = {}
    for key, prior in (("ATTACK_FIT_NEW:sex", prior_specs["q18"]),
                       ("INNER_SELECTION:sex", prior_specs["q20"])):
        relative = Path(manifest["role_file_relative_path"][key])
        digest = manifest["role_file_sha256"][key]
        if (relative.is_absolute() or ".." in relative.parts
                or relative.parts[:2] != (".private", "role_export_v1")
                or not _full_sha(digest)):
            raise PermissionError("q21 sex role manifest identity differs")
        path = root / relative
        for spec in prior.values():
            record = spec.get("role_inputs", {}).get(key)
            if (record is None or record.get("path") != str(path)
                    or record.get("sha256") != digest
                    or spec.get("inputs", {}).get(str(path)) != digest):
                raise PermissionError("q21 role was not sealed in predecessor")
        declarations[key] = {"path": str(path), "sha256": digest}
    return declarations


def completed_score_carry(state: dict, state_sha: str,
                          prior_sha: str) -> dict:
    expected = {"R2_PRIVACY_SCORE_S0", "R2_PRIVACY_SCORE_S1"}
    units = state.get("units", {})
    previous = state.get("carry_forward", {})
    total = state.get("used_cpu_s")
    charges = [row.get("charged_cpu_s") for row in units.values()]
    if (set(units) != expected
            or any(row.get("status") != "DONE" for row in units.values())
            or previous.get("archived_queue_relative") != "queue_g19"
            or previous.get("archived_queue_sha256") != prior_sha
            or previous.get("unit_count") != 4
            or not isinstance(previous.get("charged_cpu_s"), (int, float))
            or not math.isfinite(previous["charged_cpu_s"])
            or not isinstance(total, (int, float)) or not math.isfinite(total)
            or any(not isinstance(x, (int, float)) or not math.isfinite(x)
                   or x < 0 for x in charges)
            or abs(total - previous["charged_cpu_s"] - sum(charges)) > 1e-5):
        raise PermissionError("q21 requires two completed q20 score units")
    return {"archived_queue_relative": "queue_g20",
            "archived_queue_sha256": state_sha,
            "charged_cpu_s": float(total), "unit_count": 2}


def _selected(audit: dict) -> dict[str, list[str]]:
    chosen = {}
    for seed in (0, 1):
        rules = audit["by_seed"][str(seed)]["selected_rule_pair_ids"]
        pair_ids = sorted({value for value in rules.values() if value is not None})
        if len(pair_ids) != 2:
            raise PermissionError("q21 frozen selected pair set differs")
        chosen[str(seed)] = pair_ids
    return chosen


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2,
                                allow_nan=False) + "\n")


def build(repo: Path, python: Path, slot: Path) -> dict:
    """R0-only q21 lock builder after both INNER SEX score units are frozen."""
    repo = repo.resolve(strict=True)
    study = repo / STUDY_RELATIVE
    private = study / ".private"
    active_lock = study / "CAMPAIGN_LOCK.json"
    registry_path = private / "UNIT_SPECS.json"
    queue_dir = private / "queue_g21"
    output = private / "r3/PRIVACY_REPLAY.json"
    if any(path.exists() for path in (active_lock, registry_path, queue_dir, output)):
        raise FileExistsError("q21 active lock, registry, queue or output exists")
    # The inherited queue requires the visible .venv launcher path verbatim.
    python = Path(python).absolute()
    if (str(python) != "/Users/nathansamson/PCRL/.venv/bin/python"
            or not python.is_file() or not slot.is_file()):
        raise FileNotFoundError("q21 project venv launcher or shared slot differs")
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    gate_path = study / "FULL_INPUT_GATE.json"
    bank_path = repo / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
    audit_path = study / "r3/PAIR_SELECTION_AUDIT.json"
    release_verification_path = study / "r3/RELEASE_NUMERICAL_VERIFICATION.json"
    manifest = json.loads(manifest_path.read_text())
    gate = json.loads(gate_path.read_text())
    audit = json.loads(audit_path.read_text())
    verified = json.loads(release_verification_path.read_text())
    if (manifest.get("schema") != "pcrl-utility-repair-role-admission-v1"
            or gate.get("status") != "PASS_FULL_INPUT_POINT_GATE"
            or gate.get("selected_recipe") != "A"
            or audit.get("status") != "PASS_AGGREGATE_SELECTION_REPLAY"
            or verified.get("status") !=
            "PASS_INDEPENDENT_COMPLETE_VIEW_REPLAY"):
        raise PermissionError("q21 prerequisite identity differs")
    selected = _selected(audit)
    q19_path = private / "queue_g19/QUEUE.json"
    q20_path = private / "queue_g20/QUEUE.json"
    archive = private / "stage_seals/privacy_score"
    score_lock_path = archive / "CAMPAIGN_LOCK.json"
    score_registry_path = archive / "UNIT_SPECS.json"
    score_review_path = archive / "PREFIT_REVIEW.json"
    score_seal_path = archive / "LOCAL_SEAL_RECEIPT.json"
    q20 = json.loads(q20_path.read_text())
    score_lock = json.loads(score_lock_path.read_text())
    score_registry = json.loads(score_registry_path.read_text())
    review = json.loads(score_review_path.read_text())
    seal = json.loads(score_seal_path.read_text())
    score_lock_sha = sha(score_lock_path)
    carry = completed_score_carry(q20, sha(q20_path), sha(q19_path))
    q16_lock_path = private / "stage_seals/release_raw/CAMPAIGN_LOCK.json"
    q18_lock_path = private / "stage_seals/local_att/CAMPAIGN_LOCK.json"
    q18_registry_path = private / "stage_seals/local_att/UNIT_SPECS.json"
    q19_lock_path = private / "stage_seals/coalition_att/CAMPAIGN_LOCK.json"
    release_lock_sha = sha(q16_lock_path)
    local_lock_sha = sha(q18_lock_path)
    coalition_lock_sha = sha(q19_lock_path)
    if (score_lock.get("schema") != "pcrl-utility-campaign-lock-v1"
            or score_lock.get("stage") != "frozen_reader_inner_privacy_scoring"
            or score_lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE"
            or score_lock.get("private_unit_registry_sha256") !=
            sha(score_registry_path)
            or score_lock.get("selected_pairs_by_seed") != selected
            or score_lock.get("full_input_gate_sha256") != sha(gate_path)
            or score_lock.get("source_map_sha256") != sha(manifest_path)
            or score_lock.get("mask_bank_sha256") != sha(bank_path)
            or score_lock.get("pair_selection_audit_sha256") != sha(audit_path)
            or score_lock.get("independent_release_verification_sha256") !=
            sha(release_verification_path)
            or score_lock.get("release_source_lock_sha256") != release_lock_sha
            or score_lock.get("local_attack_source_lock_sha256") != local_lock_sha
            or score_lock.get("coalition_attack_source_lock_sha256") !=
            coalition_lock_sha
            or review.get("status") != "PASS"
            or review.get("lock_sha256") != score_lock_sha
            or seal.get("status") != "PASS"
            or seal.get("lock_sha256") != score_lock_sha
            or seal.get("r3_review_sha256") != sha(score_review_path)
            or q20.get("gate", {}).get("seal_sha256") != sha(score_seal_path)):
        raise PermissionError("q21 archived q20 score science seal differs")
    score_specs = score_registry.get("unit_specs", {})
    if (set(score_specs) != {"R2_PRIVACY_SCORE_S0", "R2_PRIVACY_SCORE_S1"}
            or score_registry.get("stage") !=
            "frozen_reader_inner_privacy_scoring"):
        raise PermissionError("q21 q20 score registry differs")
    local_specs = json.loads(q18_registry_path.read_text())["unit_specs"]
    roles = bind_sex_roles(study, manifest,
                           {"q18": local_specs, "q20": score_specs})
    role_paths = {record["path"] for record in roles.values()}
    inputs: dict[str, str] = {}
    for key, record in roles.items():
        inputs[record["path"]] = record["sha256"]
    expected_score_outputs = set()
    for seed in (0, 1):
        uid = f"R2_PRIVACY_SCORE_S{seed}"
        spec = score_specs[uid]
        queue_record = q20["units"][uid]
        output_dir = private / "privacy_score" / f"seed{seed}"
        expected = {str(output_dir / "PRIVACY_SCORE_RECEIPT.json"),
                    str(output_dir / "PRIVACY_SCORES.json")}
        if (spec.get("stage") != "r2_privacy_score"
                or set(spec.get("outputs", [])) != expected
                or queue_record.get("output_sha256", {}).keys() != expected
                or set(spec.get("role_inputs", {})) != {"INNER_SELECTION:sex"}):
            raise PermissionError("q21 q20 score unit/output inventory differs")
        for name, digest in spec["inputs"].items():
            if name in role_paths:
                if inputs[name] != digest:
                    raise PermissionError("q21 inherited role SHA differs")
            else:
                _add(inputs, Path(name), digest)
        for name in spec["outputs"]:
            _add(inputs, Path(name), queue_record["output_sha256"][name])
            expected_score_outputs.add(name)
        receipt = json.loads((output_dir / "PRIVACY_SCORE_RECEIPT.json").read_text())
        scores = json.loads((output_dir / "PRIVACY_SCORES.json").read_text())
        if (receipt.get("seed") != seed
                or receipt.get("conditional_lock_sha256") != score_lock_sha
                or receipt.get("score_file_sha256") !=
                sha(output_dir / "PRIVACY_SCORES.json")
                or receipt.get("status") not in ("SCORED", "CONTROL_FAILED")
                or scores.get("seed") != seed
                or sorted(scores.get("pairs", {})) != selected[str(seed)]):
            raise PermissionError("q21 frozen score receipt or pair set differs")
    if len(expected_score_outputs) != 4:
        raise PermissionError("q21 score output set differs")
    for path in (q19_path, q20_path, score_lock_path, score_registry_path,
                 score_review_path, score_seal_path, q16_lock_path,
                 q18_lock_path, q18_registry_path, q19_lock_path,
                 manifest_path, gate_path, bank_path, audit_path,
                 release_verification_path):
        _add(inputs, path)
    for name in SOURCE_MODULES:
        _add(inputs, repo / name)
    for name in PROTOCOL_FILES:
        _add(inputs, study / name)
    script = study / "r3/privacy_replay_worker.py"
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
        "unit_id": UNIT_ID, "mode": "real", "stage": STAGE,
        "argv": argv, "script": str(script), "script_sha256": sha(script),
        "cwd": str(repo), "cpu_reserve_s": 600, "wall_limit_s": 1200,
        "max_rss_gib": 3, "inputs": dict(sorted(inputs.items())),
        "outputs": [str(output)], "deps": [],
        "source_map_sha256": sha(manifest_path),
        "role_access": ["ATTACK_FIT_NEW", "INNER_SELECTION"],
        "role_inputs": roles, "allow_child_processes": False,
        "data_access": "INDEPENDENT_FROZEN_ATT_AND_INNER_SEX_REPLAY",
        "seal_sha256": "$LOCAL_SEAL_SHA256",
    }
    registry = {"schema": "pcrl-utility-unit-registry-v1",
                "stage": "independent_frozen_privacy_score_replay",
                "unit_specs": {UNIT_ID: unit}}
    failed_debit = score_lock.get("prior_failed_attempt_debit_cpu_s")
    if (not isinstance(failed_debit, (int, float))
            or abs(failed_debit - 0.2250022920779884) > 1e-8):
        raise PermissionError("q21 failed q16 debit was lost")
    lock = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "independent_frozen_privacy_score_replay",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "development_status": "exploratory_reused_Adult_people",
        "selected_recipe": "A", "paired_seeds": [0, 1],
        "selected_pairs_by_seed": selected,
        "allowed_roles": ["ATTACK_FIT_NEW", "INNER_SELECTION"],
        "role_scope": "read ATT and INNER SEX only after frozen score stage; no fitting",
        "full_input_gate_sha256": sha(gate_path),
        "source_science_lock_sha256": gate["source_science_lock_sha256"],
        "common_row_unit_pair_sha256": gate["common_row_unit_pair_sha256"],
        "mask_bank_sha256": sha(bank_path),
        "pair_selection_audit_sha256": sha(audit_path),
        "independent_release_verification_sha256":
            sha(release_verification_path),
        "release_source_lock_sha256": release_lock_sha,
        "local_attack_source_lock_sha256": local_lock_sha,
        "coalition_attack_source_lock_sha256": coalition_lock_sha,
        "privacy_score_source_lock_sha256": score_lock_sha,
        "prior_privacy_score_queue_state_sha256": sha(q20_path),
        "prior_failed_attempt_debit_cpu_s": failed_debit,
        "effective_campaign_used_cpu_s_at_q21_start":
            carry["charged_cpu_s"] + failed_debit,
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
            "required_queue_carry_forward": carry}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--slot", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.python, args.slot), sort_keys=True))
