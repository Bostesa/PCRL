"""R3's feature-only q17 replay lock; R0 is the sole shared-lock writer.

The builder reads completed q16 release files and source metadata. It records
ATT and INNER feature-role hashes from the sealed manifest without opening
either role export before q17 is sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


STUDY_RELATIVE = Path("results/pcrl_utility_reference_repair_v1")
UNIT_ID = "R3_RELEASE_REPLAY"
STAGE = "r3_release_replay"
FEATURE_KEYS = ("ATTACK_FIT_NEW:features", "INNER_SELECTION:features")
SOURCE_MODULES = (
    "results/pcrl_utility_reference_repair_v1/r0/admission.py",
    "results/pcrl_utility_reference_repair_v1/r0/queue_adapter.py",
    "results/pcrl_utility_reference_repair_v1/r1/role_store.py",
    "results/pcrl_utility_reference_repair_v1/r3/build_release_replay_lock.py",
    "results/pcrl_utility_reference_repair_v1/r3/mask_task_replay_worker.py",
    "results/pcrl_utility_reference_repair_v1/r3/release_replay_worker.py",
    "results/pcrl_utility_reference_repair_v1/r3/source_reference.py",
    "results/pcrl_utility_reference_repair_v1/r3/task_gate_replay_worker.py",
    "results/pcrl_utility_reference_repair_v1/r3/task_replay.py",
    "results/pcrl_capacity_matched_feature_selection_v1/m0/queue.py",
)
PROTOCOL_FILES = (
    "EXECUTION_PLAN.md", "RESOURCE_PLAN.json", "AGENT_ROSTER.json",
    "m1/DATA_ACCESS_MANIFEST.json", "FULL_INPUT_GATE.json",
    "r3/PAIR_SELECTION_AUDIT.json",
    "r3/UTILITY_INTERVAL_NUMERICAL_VERIFICATION.json",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _full_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value)


def _put(inputs: dict[str, str], path: Path, expected: str | None = None) -> str:
    resolved = path.resolve(strict=True)
    digest = sha(resolved)
    if expected is not None and digest != expected:
        raise PermissionError("q17 frozen nonrole input differs")
    old = inputs.get(str(resolved))
    if old is not None and old != digest:
        raise PermissionError("q17 duplicate input identity conflicts")
    inputs[str(resolved)] = digest
    return digest


def bind_feature_roles(study: Path, manifest: dict,
                       prior_inputs: dict[str, str],
                       keys: tuple[str, ...] = FEATURE_KEYS) -> dict:
    """Bind prior release role inputs from manifest metadata only."""
    study = study.resolve(strict=True)
    declared = {}
    for key in keys:
        relative = Path(manifest["role_file_relative_path"][key])
        digest = manifest["role_file_sha256"][key]
        if (relative.is_absolute() or ".." in relative.parts
                or relative.parts[:2] != (".private", "role_export_v1")
                or not _full_sha(digest)):
            raise PermissionError("q17 role metadata path or SHA differs")
        path = study / relative
        if prior_inputs.get(str(path)) != digest:
            raise PermissionError("q17 role metadata differs from q16 release")
        declared[key] = {"path": str(path), "sha256": digest}
    return declared


def bind_completed_release_outputs(record: dict,
                                   outputs: list[str]) -> dict[str, str]:
    """Hash the three exact released feature-only files after q16 DONE."""
    expected = record.get("output_sha256", {})
    if (record.get("status") != "DONE" or len(outputs) != 3
            or len(set(outputs)) != 3 or set(expected) != set(outputs)
            or not any(name.endswith("/RELEASE_RECEIPT.json") for name in outputs)
            or not any(name.endswith("/ATTACK_FIT_NEW.npz") for name in outputs)
            or not any(name.endswith("/INNER_SELECTION.npz") for name in outputs)):
        raise PermissionError("q16 release output inventory or status differs")
    for name in outputs:
        path = Path(name)
        if not path.is_file() or sha(path) != expected[name]:
            raise PermissionError("q16 release output bytes differ")
    return dict(expected)


def q16_carry(state: dict, state_sha: str, prior_q15_sha: str) -> dict:
    units = state.get("units", {})
    previous = state.get("carry_forward", {})
    total = state.get("used_cpu_s")
    charges = [value.get("charged_cpu_s") for value in units.values()]
    if (len(units) != 11 or any(value.get("status") != "DONE"
                                for value in units.values())
            or previous.get("archived_queue_relative") != "queue_g15"
            or previous.get("archived_queue_sha256") != prior_q15_sha
            or previous.get("unit_count") != 1
            or not isinstance(previous.get("charged_cpu_s"), (int, float))
            or not math.isfinite(previous["charged_cpu_s"])
            or not isinstance(total, (int, float)) or not math.isfinite(total)
            or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
                   for x in charges)
            or abs(total - previous["charged_cpu_s"] - sum(charges)) > 1e-5):
        raise PermissionError("q16 release/raw queue is incomplete or carry differs")
    return {"archived_queue_relative": "queue_g16",
            "archived_queue_sha256": state_sha,
            "charged_cpu_s": float(total), "unit_count": 11}


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2,
                                allow_nan=False) + "\n")


def _selected(audit: dict) -> dict[str, list[str]]:
    result = {}
    for seed in (0, 1):
        rules = audit["by_seed"][str(seed)]["selected_rule_pair_ids"]
        values = sorted({name for name in rules.values() if name is not None})
        if len(values) != 2 or any("__" not in name for name in values):
            raise PermissionError("q17 selected pair set differs")
        result[str(seed)] = values
    return result


def build(repo: Path, python: Path, slot: Path) -> dict:
    repo = repo.resolve(strict=True)
    study = repo / STUDY_RELATIVE
    private = study / ".private"
    active_lock = study / "CAMPAIGN_LOCK.json"
    registry_path = private / "UNIT_SPECS.json"
    queue_dir = private / "queue_g17"
    output = private / "r3/RELEASE_REPLAY.json"
    if any(path.exists() for path in (active_lock, registry_path, queue_dir, output)):
        raise FileExistsError("q17 lock, registry, queue or output already exists")
    # Keep the literal launcher; resolving the venv symlink changes its visible
    # path and defeats the inherited queue's exact argv identity.
    python = Path(python).absolute()
    if (str(python) != "/Users/nathansamson/PCRL/.venv/bin/python"
            or not python.is_file() or not slot.is_file()):
        raise FileNotFoundError("q17 project venv launcher or shared slot differs")
    manifest_path = study / "m1/DATA_ACCESS_MANIFEST.json"
    gate_path = study / "FULL_INPUT_GATE.json"
    bank_path = repo / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
    audit_path = study / "r3/PAIR_SELECTION_AUDIT.json"
    interval_verification = study / "r3/UTILITY_INTERVAL_NUMERICAL_VERIFICATION.json"
    manifest = json.loads(manifest_path.read_text())
    gate = json.loads(gate_path.read_text())
    audit = json.loads(audit_path.read_text())
    verified = json.loads(interval_verification.read_text())
    if (gate.get("status") != "PASS_FULL_INPUT_POINT_GATE"
            or gate.get("selected_recipe") != "A"
            or audit.get("status") != "PASS_AGGREGATE_SELECTION_REPLAY"
            or verified.get("status") != "PASS_INDEPENDENT_NUMERICAL_REPLAY"
            or verified.get("full_input_gate_sha256") != sha(gate_path)):
        raise PermissionError("q17 selected F/interval/pair identities differ")
    selected = _selected(audit)
    q16_path = private / "queue_g16/QUEUE.json"
    q15_path = private / "queue_g15/QUEUE.json"
    archive = private / "stage_seals/release_raw"
    prior_lock_path = archive / "CAMPAIGN_LOCK.json"
    prior_registry_path = archive / "UNIT_SPECS.json"
    prior_review_path = archive / "PREFIT_REVIEW.json"
    prior_seal_path = archive / "LOCAL_SEAL_RECEIPT.json"
    q16 = json.loads(q16_path.read_text())
    prior_lock = json.loads(prior_lock_path.read_text())
    prior_registry = json.loads(prior_registry_path.read_text())
    review = json.loads(prior_review_path.read_text())
    seal = json.loads(prior_seal_path.read_text())
    prior_sha = sha(prior_lock_path)
    carry = q16_carry(q16, sha(q16_path), sha(q15_path))
    if (prior_lock.get("schema") != "pcrl-utility-campaign-lock-v1"
            or prior_lock.get("stage") != "selected_release_and_raw_reader_custody"
            or prior_lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE"
            or prior_lock.get("required_queue_carry_forward") !=
            q16["carry_forward"]
            or prior_lock.get("source_map_sha256") != sha(manifest_path)
            or prior_lock.get("mask_bank_sha256") != sha(bank_path)
            or prior_lock.get("full_input_gate_sha256") != sha(gate_path)
            or prior_lock.get("pair_selection_audit_sha256") != sha(audit_path)
            or prior_lock.get("private_unit_registry_sha256") !=
            sha(prior_registry_path)
            or review.get("status") != "PASS"
            or review.get("lock_sha256") != prior_sha
            or seal.get("status") != "PASS"
            or seal.get("lock_sha256") != prior_sha
            or seal.get("r3_review_sha256") != sha(prior_review_path)
            or q16.get("gate", {}).get("seal_sha256") != sha(prior_seal_path)):
        raise PermissionError("q16 archived lock, review or seal differs")
    release_units = {}
    for uid, spec in prior_registry["unit_specs"].items():
        if spec["stage"] != "r2_release":
            continue
        argv = spec["argv"]
        if (argv.count("--seed") != 1 or argv.count("--pair-id") != 1
                or spec.get("unit_id") != uid
                or set(spec.get("role_inputs", {})) != set(FEATURE_KEYS)):
            raise PermissionError("q16 release unit identity differs")
        seed = argv[argv.index("--seed") + 1]
        pair_id = argv[argv.index("--pair-id") + 1]
        if seed not in selected or pair_id not in selected[seed]:
            raise PermissionError("q16 release pair is outside selected set")
        key = (seed, pair_id)
        if key in release_units:
            raise PermissionError("q16 selected release is duplicated")
        release_units[key] = (uid, spec)
    if set(release_units) != {(str(seed), pair) for seed in (0, 1)
                             for pair in selected[str(seed)]}:
        raise PermissionError("q16 release set differs from independent pair audit")
    inputs: dict[str, str] = {}
    role_prior: dict[str, str] = {}
    for key, (uid, spec) in release_units.items():
        record = q16["units"][uid]
        if (spec["stage"] != "r2_release" or len(spec["outputs"]) != 3
                or spec["script_sha256"] != sha(Path(spec["script"]))):
            raise PermissionError("q16 release worker source or inventory differs")
        bind_completed_release_outputs(record, spec["outputs"])
        for name, digest in spec["inputs"].items():
            path = Path(name)
            if "/role_export_v1/" in name:
                if name not in (value["path"] for value in spec["role_inputs"].values()):
                    raise PermissionError("q16 undisclosed role input differs")
                if name in role_prior and role_prior[name] != digest:
                    raise PermissionError("q16 feature role SHA changes across units")
                role_prior[name] = digest
                inputs[name] = digest
            else:
                _put(inputs, path, digest)
        for name in spec["outputs"]:
            _put(inputs, Path(name), record["output_sha256"][name])
    role_inputs = bind_feature_roles(study, manifest, role_prior)
    if set(role_prior) != {record["path"] for record in role_inputs.values()}:
        raise PermissionError("q17 release has extra role inputs")
    # A packages do not consult fill constants during deployment, but the
    # independent restore checks their sealed public bytes as part of the
    # complete package. They are therefore explicit q17 inputs.
    for name in tuple(inputs):
        if not name.endswith(("FINAL_income_PACKAGE.json",
                              "FINAL_occupation_group_PACKAGE.json")):
            continue
        package = json.loads(Path(name).read_text())
        fill = package["fill_constants"]
        relative = Path(fill["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or not _full_sha(fill["sha256"]):
            raise PermissionError("q17 package fill path or SHA differs")
        fill_path = (private / relative).resolve(strict=True)
        if not fill_path.is_relative_to(private):
            raise PermissionError("q17 package fill escapes private custody")
        _put(inputs, fill_path, fill["sha256"])
    for path in (q16_path, q15_path, prior_lock_path, prior_registry_path,
                 prior_review_path, prior_seal_path, interval_verification,
                 audit_path, gate_path, bank_path, manifest_path):
        _put(inputs, path)
    for name in SOURCE_MODULES:
        _put(inputs, repo / name)
    for name in PROTOCOL_FILES:
        _put(inputs, study / name)
    script = study / "r3/release_replay_worker.py"
    argv = [str(python), str(script), "--unit-id", UNIT_ID,
            "--queue-root", str(queue_dir), "--lock", str(active_lock),
            "--manifest", str(manifest_path),
            "--export-root", str(private / "role_export_v1"),
            "--private-root", str(private),
            "--full-input-gate", str(gate_path),
            "--mask-bank", str(bank_path),
            "--pair-selection-audit", str(audit_path),
            "--output", str(output)]
    spec = {
        "unit_id": UNIT_ID, "mode": "real", "stage": STAGE,
        "argv": argv, "script": str(script), "script_sha256": sha(script),
        "cwd": str(repo), "cpu_reserve_s": 600, "wall_limit_s": 1200,
        "max_rss_gib": 3, "inputs": dict(sorted(inputs.items())),
        "outputs": [str(output)], "deps": [],
        "source_map_sha256": sha(manifest_path),
        "role_access": ["ATTACK_FIT_NEW", "INNER_SELECTION"],
        "role_inputs": role_inputs, "allow_child_processes": False,
        "data_access": "INDEPENDENT_ATT_AND_INNER_FEATURE_RELEASE_REPLAY_NO_SEX",
        "seal_sha256": "$LOCAL_SEAL_SHA256",
    }
    registry = {"schema": "pcrl-utility-unit-registry-v1",
                "stage": "independent_selected_release_replay",
                "unit_specs": {UNIT_ID: spec}}
    lock = {
        "schema": "pcrl-utility-campaign-lock-v1",
        "stage": "independent_selected_release_replay",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "development_status": "exploratory_reused_Adult_people",
        "selected_recipe": "A", "paired_seeds": [0, 1],
        "allowed_roles": ["ATTACK_FIT_NEW", "INNER_SELECTION"],
        "role_scope": "independent complete release replay on ATT/INNER features; no SEX or task labels",
        "selected_pairs_by_seed": selected,
        "prior_release_lock_sha256": prior_sha,
        "prior_release_queue_state_sha256": sha(q16_path),
        "full_input_gate_sha256": sha(gate_path),
        "mask_bank_sha256": sha(bank_path),
        "pair_selection_audit_sha256": sha(audit_path),
        "interval_numerical_verification_sha256": sha(interval_verification),
        "required_queue_carry_forward": carry,
        "private_unit_registry_sha256": "__AFTER_REGISTRY_WRITE__",
        "source_map_sha256": sha(manifest_path),
        "prior_package_relative_to_worktree": str(STUDY_RELATIVE),
        "governed_source_sha256_by_path": {
            name: sha(repo / name) for name in SOURCE_MODULES},
        "governed_protocol_sha256_by_relative_path": {
            name: sha(study / name) for name in PROTOCOL_FILES},
        "shared_slot_sha256": sha(slot),
        "excluded_inputs": ["SEX labels", "task labels", "HEAD_VALIDATION labels",
                            "old assessment person arrays"],
        "next_stage": "fresh ATT attackers only after independent complete release replay PASS",
    }
    _write_once(registry_path, registry)
    lock["private_unit_registry_sha256"] = sha(registry_path)
    _write_once(active_lock, lock)
    return {"lock_sha256": sha(active_lock),
            "registry_sha256": sha(registry_path),
            "unit_id": UNIT_ID, "input_count": len(inputs),
            "release_count": len(release_units),
            "required_queue_carry_forward": carry}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--slot", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.python, args.slot), sort_keys=True))
