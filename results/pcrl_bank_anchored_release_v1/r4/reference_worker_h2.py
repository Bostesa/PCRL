"""H2 technical retry of the historical P* fresh complete-view attack.

P* has its own public fixed bank and release wire. This worker must not
substitute W_TSTAR's table or treat historical attacks as fresh. All Adult
label reads occur after the active campaign queue/stage/source checks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Mapping

from results.pcrl_bank_anchored_release_v1.r0.compute_queue import (
    assert_active_from_env,
)
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, assert_same_people, sha_file,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.protocol import (
    run_three_view_attack,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.reference_wire import (
    load_public_fixed_map, load_reference_role_package,
)
from .pstar_views_h2 import prepare_pstar_three_views_h2


SPEC_SCHEMA = "pcrl-bank-r4-pstar-unit-h2-v1"
LOCK_SCHEMA = "pcrl-bank-attack-stage-lock-v1"
SPEC_KEYS = frozenset({
    "schema", "unit_id", "arm_id", "source_seed", "realization",
    "attacker_seed", "campaign_seal_path", "campaign_seal_sha256",
    "role_manifest_path", "role_manifest_sha256",
    "fixed_map_receipt_path", "fixed_map_receipt_sha256",
    "fixed_map_archive_path", "fixed_map_archive_sha256",
    "att_package_dir", "att_receipt_sha256",
    "inner_package_dir", "inner_receipt_sha256",
    "eligibility_receipt_path", "eligibility_receipt_sha256",
    "source_pins", "private_output_path", "public_receipt_path",
})
REQUIRED_SOURCES = frozenset({
    "reference_worker_h2.py", "pstar_views_h2.py", "reference_wire.py", "role_store.py",
    "protocol.py", "complete_view.py", "attack_eval.py",
    "attack_controls_v2.py", "common_eval.py", "safe_data_guard.py",
})


def _hex(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _file(path: str, expected_sha256: str, *, private: bool = False) -> Path:
    supplied = Path(path)
    if supplied.is_symlink() or not _hex(expected_sha256):
        raise PermissionError("pinned file path or SHA-256 invalid")
    found = supplied.resolve(strict=True)
    if not found.is_file() or _sha(found) != expected_sha256:
        raise PermissionError("pinned file bytes differ")
    if private and (".private" not in found.parts or found.stat().st_mode & 0o077):
        raise PermissionError("private artifact has loose custody")
    return found


def validate_spec(spec: Mapping) -> None:
    """Pure structural validation, before any source file or label is opened."""
    if (not isinstance(spec, Mapping) or set(spec) != SPEC_KEYS or
            spec.get("schema") != SPEC_SCHEMA or spec.get("arm_id") != "P_STAR" or
            type(spec.get("source_seed")) is not int or spec["source_seed"] not in (0, 1) or
            type(spec.get("realization")) is not int or spec["realization"] != 0 or
            type(spec.get("attacker_seed")) is not int or spec["attacker_seed"] < 0 or
            type(spec.get("unit_id")) is not str or not spec["unit_id"]):
        raise ValueError("P* unit schema/identity differs")
    for key in ("campaign_seal_sha256", "role_manifest_sha256",
                "fixed_map_receipt_sha256", "fixed_map_archive_sha256",
                "att_receipt_sha256", "inner_receipt_sha256",
                "eligibility_receipt_sha256"):
        if not _hex(spec[key]):
            raise ValueError(f"P* unit {key} is not a SHA-256")
    pins = spec["source_pins"]
    if (type(pins) is not list or {Path(item.get("path", "")).name for item in pins
                                   if type(item) is dict} != REQUIRED_SOURCES or
            len(pins) != len(REQUIRED_SOURCES) or
            any(type(item) is not dict or set(item) != {"path", "sha256"} or
                type(item["path"]) is not str or not _hex(item["sha256"])
                for item in pins)):
        raise ValueError("P* unit transitive source inventory differs")
    for key in ("campaign_seal_path", "role_manifest_path", "fixed_map_receipt_path",
                "fixed_map_archive_path", "att_package_dir", "inner_package_dir",
                "eligibility_receipt_path", "private_output_path", "public_receipt_path"):
        if type(spec[key]) is not str or not spec[key]:
            raise ValueError(f"P* unit {key} path absent")


def validate_stage_lock(lock: Mapping, spec: Mapping, spec_sha256: str,
                        *, stage_lock_sha256: str) -> None:
    if (not isinstance(lock, Mapping) or lock.get("schema") != LOCK_SCHEMA or
            lock.get("status") != "LOCALLY_SEALED_ATTACKS" or
            lock.get("campaign_seal_sha256") != spec["campaign_seal_sha256"]):
        raise PermissionError("R0 attack-stage lock/status/source differs")
    unit = lock.get("units", {}).get(spec["unit_id"])
    if unit != {"spec_sha256": spec_sha256, "arm_id": "P_STAR",
                "source_seed": spec["source_seed"], "realization": 0,
                "point_eligible": True}:
        raise PermissionError("R0 attack-stage lock did not admit exact P* unit")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    if not any(item["sha256"] == stage_lock_sha256 and "attacks" in item["stages"]
               for item in state.get("amendments", [])):
        raise PermissionError("attack-stage lock was not registered in the queue")


def validate_live_sources(pins: list[dict]) -> None:
    """The hashed paths must be the modules this worker actually imported."""
    names = {
        "reference_worker_h2.py": __file__,
        "pstar_views_h2.py": importlib.import_module(
            "results.pcrl_bank_anchored_release_v1.r4.pstar_views_h2").__file__,
        "reference_wire.py": importlib.import_module(
            "results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.reference_wire").__file__,
        "role_store.py": importlib.import_module(
            "results.pcrl_latent_and_sampled_methods_v1.r01.role_store").__file__,
        "protocol.py": importlib.import_module(
            "results.pcrl_latent_and_sampled_methods_v1.r02.protocol").__file__,
        "complete_view.py": importlib.import_module(
            "results.pcrl_latent_and_sampled_methods_v1.r02.complete_view").__file__,
        "attack_eval.py": importlib.import_module("attack_eval").__file__,
        "attack_controls_v2.py": importlib.import_module("attack_controls_v2").__file__,
        "common_eval.py": importlib.import_module("common_eval").__file__,
        "safe_data_guard.py": importlib.import_module("safe_data_guard").__file__,
    }
    if set(names) != REQUIRED_SOURCES:
        raise AssertionError("transitive source inventory changed")
    supplied = {Path(item["path"]).name: item for item in pins}
    for name, actual in names.items():
        if (Path(supplied[name]["path"]).resolve(strict=True) !=
                Path(actual).resolve(strict=True) or
                _sha(Path(actual)) != supplied[name]["sha256"]):
            raise PermissionError("pinned source is not the live imported module")


def _private_new_directory(path: str) -> Path:
    supplied = Path(path)
    if supplied.is_symlink() or ".private" not in supplied.parts or supplied.exists():
        raise PermissionError("attack output needs a new owner-only private directory")
    parent = supplied.parent.resolve(strict=True)
    if not parent.is_dir() or parent.stat().st_mode & 0o077:
        raise PermissionError("attack output parent has loose custody")
    return supplied.resolve(strict=False)


def _write_public_once(path: str, payload: Mapping) -> str:
    target = Path(path).resolve(strict=False)
    if target.exists() or not target.parent.is_dir() or ".private" in target.parts:
        raise PermissionError("public attack receipt must be a new public file")
    data = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(target, flags, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return sha_file(target)


def run(spec_path: str, spec_sha256: str, lock_path: str,
        lock_sha256: str) -> dict:
    # Queue guard is deliberately first, before even a role-only release opens.
    active = assert_active_from_env(stage="attacks")
    source_spec = _file(spec_path, spec_sha256, private=True)
    spec = json.loads(source_spec.read_text())
    validate_spec(spec)
    if (active["unit_id"] != spec["unit_id"] or
            active["seal_sha256"] != spec["campaign_seal_sha256"]):
        raise PermissionError("active queue unit or seal differs from P* spec")
    seal = _file(spec["campaign_seal_path"], spec["campaign_seal_sha256"])
    lock = json.loads(_file(lock_path, lock_sha256).read_text())
    validate_stage_lock(lock, spec, spec_sha256,
                        stage_lock_sha256=lock_sha256)
    for item in spec["source_pins"]:
        _file(item["path"], item["sha256"])
    validate_live_sources(spec["source_pins"])
    _file(spec["role_manifest_path"], spec["role_manifest_sha256"])
    store = RoleOnlyStore(spec["role_manifest_path"])
    gate = json.loads(_file(spec["eligibility_receipt_path"],
                            spec["eligibility_receipt_sha256"]).read_text())
    if (gate.get("schema") != "pcrl-bank-r5-pstar-common-inner-replay-v1" or
            gate.get("status") != "PASS_UTILITY_POINT_ONLY" or
            gate.get("all_24_point_clauses_pass") is not True):
        raise PermissionError("P* common INNER point utility was not established")
    fixed, map_receipt = load_public_fixed_map(
        spec["fixed_map_receipt_path"], spec["fixed_map_archive_path"],
        expected_receipt_sha256=spec["fixed_map_receipt_sha256"],
        expected_archive_sha256=spec["fixed_map_archive_sha256"],
        arm_id="P_STAR", seed=spec["source_seed"])
    att = load_reference_role_package(
        spec["att_package_dir"], receipt_sha256=spec["att_receipt_sha256"],
        full_bank1=fixed["bank_r1"], full_bank2=fixed["bank_r2"])
    inner = load_reference_role_package(
        spec["inner_package_dir"], receipt_sha256=spec["inner_receipt_sha256"],
        full_bank1=fixed["bank_r1"], full_bank2=fixed["bank_r2"])
    if (att.receipt["arm_id"] != "P_STAR" or inner.receipt["arm_id"] != "P_STAR" or
            att.receipt["source_seed"] != spec["source_seed"] or
            inner.receipt["source_seed"] != spec["source_seed"] or
            att.receipt["source_binding_sha256"] != map_receipt["source_binding_sha256"] or
            inner.receipt["source_binding_sha256"] != map_receipt["source_binding_sha256"]):
        raise PermissionError("P* fixed map and role releases have different source")
    for role, package in (("ATTACK_FIT_NEW", att), ("INNER_SELECTION", inner)):
        meta = store.manifest["roles"][role]
        if (len(package.row_id) != meta["rows"] or
                len(set(package.unit.tolist())) != meta["groups"] or
                package.receipt["row_unit_pair_sha256"] !=
                meta["row_unit_pair_sha256"]):
            raise PermissionError("P* release differs from admitted ordered role")
    prepared = prepare_pstar_three_views_h2(att, inner)
    private_out = _private_new_directory(spec["private_output_path"])
    if Path(spec["public_receipt_path"]).exists():
        raise FileExistsError("public P* attack receipt already exists")

    def guard() -> None:
        current = assert_active_from_env(unit_id=spec["unit_id"], stage="attacks")
        if current["seal_sha256"] != spec["campaign_seal_sha256"]:
            raise PermissionError("campaign seal changed during P* attack")

    guard()
    sex_att = store.read("ATTACK_FIT_NEW", "sex", procedure="attack")
    assert_same_people(sex_att, {"row_id": att.row_id, "unit": att.unit})

    def load_inner_sex() -> dict:
        guard()
        if not (private_out / "FROZEN_COMPLETE.json").is_file():
            raise PermissionError("INNER SEX remains closed until every reader freezes")
        sex_inner = store.read("INNER_SELECTION", "sex", procedure="screen")
        assert_same_people(sex_inner, {"row_id": inner.row_id, "unit": inner.unit})
        return sex_inner

    attack = run_three_view_attack(
        prepared, sex_att, load_inner_sex=load_inner_sex,
        family="A", seed=spec["attacker_seed"], private_root=private_out,
        lock_path=seal, lock_sha256=spec["campaign_seal_sha256"],
        queue_guard=guard)
    public = {"schema": "pcrl-bank-r4-pstar-fresh-attack-h2-v1",
              "status": "COMPLETE_PRIVATE_DEVELOPMENT_ONLY",
              "unit_id": spec["unit_id"], "arm_id": "P_STAR",
              "source_seed": spec["source_seed"], "realization": 0,
              "attacker_seed": spec["attacker_seed"],
              "campaign_seal_sha256": spec["campaign_seal_sha256"],
              "stage_lock_sha256": lock_sha256,
              "unit_spec_sha256": spec_sha256,
              "fixed_map_receipt_sha256": spec["fixed_map_receipt_sha256"],
              "fixed_map_archive_sha256": spec["fixed_map_archive_sha256"],
              "release_receipt_sha256": {
                  "ATTACK_FIT_NEW": spec["att_receipt_sha256"],
                  "INNER_SELECTION": spec["inner_receipt_sha256"]},
              "attack": attack["public"]}
    receipt_sha256 = _write_public_once(spec["public_receipt_path"], public)
    return {"status": public["status"], "unit_id": spec["unit_id"],
            "receipt_sha256": receipt_sha256}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("unit-spec", "unit-spec-sha256", "stage-lock", "stage-lock-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.unit_spec, args.unit_spec_sha256,
                         args.stage_lock, args.stage_lock_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
