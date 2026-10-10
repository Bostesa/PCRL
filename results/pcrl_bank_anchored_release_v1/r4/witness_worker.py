"""Queue-only fresh attack for immutable B0 and compatible W_Q references.

Each reference restores its own exact ten-bit public bank table and role-only
releases. B0 is W_TSTAR identity; W_Q is a distinct inherited task-only bank.
No protected label opens until R0's point-eligibility and source seal pass.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Mapping
from zipfile import ZipFile

import numpy as np

from results.pcrl_bank_anchored_release_v1.r0.compute_queue import assert_active_from_env
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, assert_same_people, sha_array, sha_file,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.protocol import run_three_view_attack
from results.pcrl_code_utility_witness_v1.r4.bank_service import make_bank_service

from .reference_worker import (_file, _hex, _private_new_directory,
                               _write_public_once)
from .views import prepare_complete_bank_views


SCHEMA = "pcrl-bank-r4-witness-unit-v1"
STAGE_SCHEMA = "pcrl-bank-attack-stage-lock-v1"
ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")
SPEC_KEYS = frozenset({
    "schema", "unit_id", "arm_id", "source_seed", "realization",
    "attacker_seed", "campaign_seal_path", "campaign_seal_sha256",
    "role_manifest_path", "role_manifest_sha256",
    "public_table_path", "public_table_sha256",
    "source_receipt_path", "source_receipt_sha256",
    "source_model_sha256", "feature_schema_sha256",
    "policy_sha256", "decoder_sha256",
    "release_path_by_role", "release_sha256_by_role",
    "eligibility_receipt_path", "eligibility_receipt_sha256",
    "source_pins", "private_output_path", "public_receipt_path",
})
SOURCES = frozenset({
    "witness_worker.py", "views.py", "reference_worker.py",
    "bank_service.py", "role_store.py",
    "protocol.py", "complete_view.py", "attack_eval.py",
    "attack_controls_v2.py", "common_eval.py", "safe_data_guard.py",
})


def validate_spec(spec: Mapping) -> None:
    if (not isinstance(spec, Mapping) or set(spec) != SPEC_KEYS or
            spec.get("schema") != SCHEMA or spec.get("arm_id") not in ("B0", "W_Q") or
            type(spec.get("source_seed")) is not int or spec["source_seed"] not in (0, 1) or
            type(spec.get("realization")) is not int or spec["realization"] != 0 or
            type(spec.get("attacker_seed")) is not int or spec["attacker_seed"] < 0 or
            type(spec.get("unit_id")) is not str or not spec["unit_id"]):
        raise ValueError("witness attack arm/source/realization differs")
    for key in ("campaign_seal_sha256", "role_manifest_sha256",
                "public_table_sha256", "source_receipt_sha256",
                "source_model_sha256", "feature_schema_sha256",
                "policy_sha256", "eligibility_receipt_sha256"):
        if not _hex(spec[key]):
            raise ValueError("witness attack SHA-256 absent")
    if spec["arm_id"] == "B0" and not _hex(spec["decoder_sha256"]):
        raise ValueError("W_TSTAR held-out decoder SHA missing")
    if spec["arm_id"] == "W_Q" and spec["decoder_sha256"] is not None:
        raise ValueError("W_Q does not have a held-out decoder")
    for key in ("release_path_by_role", "release_sha256_by_role"):
        if type(spec[key]) is not dict or set(spec[key]) != set(ROLES):
            raise ValueError("both witness role releases required")
    if (any(type(v) is not str or not v for v in spec["release_path_by_role"].values()) or
            any(not _hex(v) for v in spec["release_sha256_by_role"].values())):
        raise ValueError("witness role release path/hash invalid")
    pins = spec["source_pins"]
    if (type(pins) is not list or len(pins) != len(SOURCES) or
            {Path(item.get("path", "")).name for item in pins if type(item) is dict} != SOURCES or
            any(type(item) is not dict or set(item) != {"path", "sha256"} or
                type(item["path"]) is not str or not _hex(item["sha256"])
                for item in pins)):
        raise ValueError("witness transitive source inventory differs")
    for key in ("campaign_seal_path", "role_manifest_path", "public_table_path",
                "source_receipt_path", "eligibility_receipt_path",
                "private_output_path", "public_receipt_path"):
        if type(spec[key]) is not str or not spec[key]:
            raise ValueError("witness attack path absent")


def _live_sources(pins: list[dict]) -> None:
    modules = {
        "witness_worker.py": __file__,
        "views.py": importlib.import_module(__package__ + ".views").__file__,
        "reference_worker.py": importlib.import_module(__package__ + ".reference_worker").__file__,
        "bank_service.py": importlib.import_module(
            "results.pcrl_code_utility_witness_v1.r4.bank_service").__file__,
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
    supplied = {Path(item["path"]).name: item for item in pins}
    for name, live in modules.items():
        if (Path(supplied[name]["path"]).resolve(strict=True) !=
                Path(live).resolve(strict=True) or
                sha_file(Path(live)) != supplied[name]["sha256"]):
            raise PermissionError("witness source hash is not the imported module")


def _load_witness_release(path: Path, role_meta: Mapping,
                          service) -> dict[str, np.ndarray]:
    """Replay the old wire with the torch-free fixed-table service only."""
    members = frozenset({"row_id", "unit", "code0", "code1",
                         "prob0", "prob1", "decision0", "decision1"})
    with ZipFile(path) as archive:
        if (len(archive.infolist()) != len(members) or
                {entry.filename for entry in archive.infolist()} !=
                {name + ".npy" for name in members}):
            raise ValueError("witness release NPZ member inventory differs")
    with np.load(path, allow_pickle=False) as archive:
        release = {name: np.array(archive[name], copy=True) for name in members}
    row, unit = release["row_id"], release["unit"]
    n = role_meta["rows"]
    if (row.dtype != np.int64 or unit.dtype != np.int64 or
            row.shape != (n,) or unit.shape != (n,) or
            not np.all(np.diff(row) > 0) or
            len(np.unique(unit)) != role_meta["groups"] or
            sha_array(np.column_stack((row, unit))) !=
            role_meta["row_unit_pair_sha256"]):
        raise PermissionError("witness release differs from admitted ordered role")
    for purpose in (0, 1):
        code, prob, decision = (release[f"{field}{purpose}"]
                                for field in ("code", "prob", "decision"))
        if (code.dtype != np.uint8 or code.shape != (n, 10) or
                prob.dtype != np.float64 or prob.shape != (n, (2, 6)[purpose]) or
                decision.dtype != np.int64 or decision.shape != (n,) or
                not np.array_equal(prob,
                    service.probabilities_from_code(purpose, code)) or
                not np.array_equal(decision, prob.argmax(axis=1))):
            raise ValueError("witness emitted wire differs from its public table")
    return release


def run(spec_path: str, spec_sha256: str, lock_path: str,
        lock_sha256: str) -> dict:
    active = assert_active_from_env(stage="attacks")
    spec = json.loads(_file(spec_path, spec_sha256, private=True).read_text())
    validate_spec(spec)
    if (active["unit_id"] != spec["unit_id"] or
            active["seal_sha256"] != spec["campaign_seal_sha256"]):
        raise PermissionError("active queue does not match witness attack spec")
    seal = _file(spec["campaign_seal_path"], spec["campaign_seal_sha256"])
    lock = json.loads(_file(lock_path, lock_sha256).read_text())
    expected = {"spec_sha256": spec_sha256, "arm_id": spec["arm_id"],
                "source_seed": spec["source_seed"],
                "realization": 0, "point_eligible": True}
    if (lock.get("schema") != STAGE_SCHEMA or
            lock.get("status") != "LOCALLY_SEALED_ATTACKS" or
            lock.get("campaign_seal_sha256") != spec["campaign_seal_sha256"] or
            lock.get("units", {}).get(spec["unit_id"]) != expected):
        raise PermissionError("R0 did not admit exact point-eligible witness unit")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    if not any(item["sha256"] == lock_sha256 and "attacks" in item["stages"]
               for item in state.get("amendments", [])):
        raise PermissionError("witness attack lock was not registered")
    for item in spec["source_pins"]:
        _file(item["path"], item["sha256"])
    _live_sources(spec["source_pins"])
    _file(spec["role_manifest_path"], spec["role_manifest_sha256"])
    store = RoleOnlyStore(spec["role_manifest_path"])
    gate = json.loads(_file(spec["eligibility_receipt_path"],
                            spec["eligibility_receipt_sha256"]).read_text())
    if (gate.get("arm_id") != spec["arm_id"] or
            gate.get("all_two_seed_task_reference_point_caps_pass") is not True):
        raise PermissionError("reference did not pass original common point utility")
    table_doc = json.loads(_file(spec["public_table_path"],
                                 spec["public_table_sha256"]).read_text())
    inherited = "W_TSTAR" if spec["arm_id"] == "B0" else "W_Q"
    service = make_bank_service(
        table_doc, arm_id=inherited, seed=spec["source_seed"],
        source_model_sha256=spec["source_model_sha256"],
        feature_schema_sha256=spec["feature_schema_sha256"],
        public_table_sha256=spec["public_table_sha256"],
        policy_sha256=spec["policy_sha256"],
        decoder_sha256=spec["decoder_sha256"])
    source = json.loads(_file(spec["source_receipt_path"],
                              spec["source_receipt_sha256"]).read_text())
    if (source.get("schema") != "code-utility-witness-worker-v1" or
            source.get("stage") != "bank" or source.get("status") != "COMPLETE" or
            source.get("seed") != spec["source_seed"] or
            source.get("role_manifest_sha256") != spec["role_manifest_sha256"] or
            source.get("source_model_sha256") != spec["source_model_sha256"] or
            inherited not in source.get("result", {}).get("arm_ids", [])):
        raise PermissionError("witness source compilation receipt differs")
    roles = {}
    for role in ROLES:
        item = source["result"]["roles"][role]["arms"][inherited]
        if (item["private_release_sha256"] != spec["release_sha256_by_role"][role] or
                item["policy_sha256"] != spec["policy_sha256"] or
                item["decoder_sha256"] != spec["decoder_sha256"] or
                item["public_table_sha256"] != spec["public_table_sha256"]):
            raise PermissionError("witness release bytes/policy/table differ")
        path = _file(spec["release_path_by_role"][role],
                     spec["release_sha256_by_role"][role], private=True)
        if path.name != item["private_release_basename"]:
            raise PermissionError("witness release filename differs from source")
        roles[role] = _load_witness_release(
            path, store.manifest["roles"][role], service)
    tables = service.tables
    prepared = prepare_complete_bank_views(
        roles["ATTACK_FIT_NEW"], roles["INNER_SELECTION"], tables,
        metadata_att=({}, {}), metadata_inner=({}, {}))
    private_out = _private_new_directory(spec["private_output_path"])
    if Path(spec["public_receipt_path"]).exists():
        raise FileExistsError("public witness attack receipt already exists")

    def guard() -> None:
        current = assert_active_from_env(unit_id=spec["unit_id"], stage="attacks")
        if current["seal_sha256"] != spec["campaign_seal_sha256"]:
            raise PermissionError("campaign seal changed during witness attack")

    guard()
    sex_att = store.read("ATTACK_FIT_NEW", "sex", procedure="attack")
    assert_same_people(sex_att, roles["ATTACK_FIT_NEW"])

    def load_inner_sex() -> dict:
        guard()
        if not (private_out / "FROZEN_COMPLETE.json").is_file():
            raise PermissionError("INNER SEX remains closed until reader freeze")
        record = store.read("INNER_SELECTION", "sex", procedure="screen")
        assert_same_people(record, roles["INNER_SELECTION"])
        return record

    attack = run_three_view_attack(
        prepared, sex_att, load_inner_sex=load_inner_sex,
        family="A", seed=spec["attacker_seed"], private_root=private_out,
        lock_path=seal, lock_sha256=spec["campaign_seal_sha256"],
        queue_guard=guard)
    public = {"schema": "pcrl-bank-r4-witness-fresh-attack-v1",
              "status": "COMPLETE_PRIVATE_DEVELOPMENT_ONLY",
              "unit_id": spec["unit_id"], "arm_id": spec["arm_id"],
              "source_seed": spec["source_seed"], "realization": 0,
              "attacker_seed": spec["attacker_seed"],
              "campaign_seal_sha256": spec["campaign_seal_sha256"],
              "stage_lock_sha256": lock_sha256, "unit_spec_sha256": spec_sha256,
              "public_table_sha256": spec["public_table_sha256"],
              "source_receipt_sha256": spec["source_receipt_sha256"],
              "release_sha256_by_role": spec["release_sha256_by_role"],
              "attack": attack["public"]}
    receipt_sha256 = _write_public_once(spec["public_receipt_path"], public)
    return {"unit_id": spec["unit_id"], "status": public["status"],
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
