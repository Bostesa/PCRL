"""Queue-only complete-view attack for one eligible bank-channel release.

The selected policy and exact role-only cached release are restored from R2's
write-once private packages. The attack matrix is built solely from the
actual emitted ten-bit code, its fixed probability and decision. No source
teacher score, fine cell, input action position, SEX or private coin enters a
reader feature. ATT SEX opens only after all source and utility gates pass.
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
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_package import load_policy

from .reference_worker import (_file, _hex, _private_new_directory,
                               _write_public_once)
from .views import materialize_role_from_ids, prepare_complete_bank_views


SCHEMA = "pcrl-bank-r4-policy-unit-v1"
STAGE_SCHEMA = "pcrl-bank-attack-stage-lock-v1"
ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")
ARMS = ("T", "D", "P", "L", "J", "S12", "S21")
SPEC_KEYS = frozenset({
    "schema", "unit_id", "arm_id", "source_seed", "realization",
    "attacker_seed", "campaign_seal_path", "campaign_seal_sha256",
    "role_manifest_path", "role_manifest_sha256",
    "public_table_path", "public_table_sha256",
    "policy_directory", "policy_receipt_sha256",
    "release_directory_by_role", "release_receipt_sha256_by_role",
    "eligibility_receipt_path", "eligibility_receipt_sha256",
    "source_pins", "private_output_path", "public_receipt_path",
})
SOURCES = frozenset({
    "policy_worker.py", "views.py", "reference_worker.py", "r2_package.py",
    "r2_materialize.py", "r2_laws.py", "r1_release.py", "role_store.py",
    "protocol.py", "complete_view.py", "attack_eval.py",
    "attack_controls_v2.py", "common_eval.py", "safe_data_guard.py",
})
LAW_MEMBERS = frozenset({"row_id", "unit", "code_r1", "code_r2", "mass",
                         "valid", "b0_code_r1", "b0_code_r2"})
RELEASE_MEMBERS = frozenset({"row_id", "unit", "code_r1", "code_r2"})


def validate_spec(spec: Mapping) -> None:
    if (not isinstance(spec, Mapping) or set(spec) != SPEC_KEYS or
            spec.get("schema") != SCHEMA or spec.get("arm_id") not in ARMS or
            type(spec.get("source_seed")) is not int or spec["source_seed"] not in (0, 1) or
            type(spec.get("realization")) is not int or
            spec["realization"] not in ((0,) if spec["arm_id"] == "D" else (0, 1, 2)) or
            type(spec.get("attacker_seed")) is not int or spec["attacker_seed"] < 0 or
            type(spec.get("unit_id")) is not str or not spec["unit_id"]):
        raise ValueError("selected policy attack identity differs")
    for key in ("campaign_seal_sha256", "role_manifest_sha256",
                "public_table_sha256", "policy_receipt_sha256",
                "eligibility_receipt_sha256"):
        if not _hex(spec[key]):
            raise ValueError("selected policy attack SHA-256 missing")
    for key in ("release_directory_by_role", "release_receipt_sha256_by_role"):
        if type(spec[key]) is not dict or set(spec[key]) != set(ROLES):
            raise ValueError("both role-only release packages are required")
    if (any(type(v) is not str or not v for v in spec["release_directory_by_role"].values()) or
            any(not _hex(v) for v in spec["release_receipt_sha256_by_role"].values())):
        raise ValueError("role release path/receipt missing")
    pins = spec["source_pins"]
    if (type(pins) is not list or len(pins) != len(SOURCES) or
            {Path(item.get("path", "")).name for item in pins if type(item) is dict} != SOURCES or
            any(type(item) is not dict or set(item) != {"path", "sha256"} or
                type(item["path"]) is not str or not _hex(item["sha256"])
                for item in pins)):
        raise ValueError("selected policy transitive source inventory differs")
    for key in ("campaign_seal_path", "role_manifest_path", "public_table_path",
                "policy_directory", "eligibility_receipt_path",
                "private_output_path", "public_receipt_path"):
        if type(spec[key]) is not str or not spec[key]:
            raise ValueError("selected policy attack path missing")


def _live_sources(pins: list[dict]) -> None:
    modules = {
        "policy_worker.py": __file__,
        "views.py": importlib.import_module(__package__ + ".views").__file__,
        "reference_worker.py": importlib.import_module(__package__ + ".reference_worker").__file__,
        "r2_package.py": importlib.import_module(
            "results.pcrl_bank_anchored_release_v1.bank_anchored.r2_package").__file__,
        "r2_materialize.py": importlib.import_module(
            "results.pcrl_bank_anchored_release_v1.bank_anchored.r2_materialize").__file__,
        "r2_laws.py": importlib.import_module(
            "results.pcrl_bank_anchored_release_v1.bank_anchored.r2_laws").__file__,
        "r1_release.py": importlib.import_module("bank_anchored.r1_release").__file__,
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
            raise PermissionError("pinned policy attack source is not the imported module")


def _npz(path: Path, members: frozenset[str]) -> dict[str, np.ndarray]:
    with ZipFile(path) as archive:
        if (len(archive.infolist()) != len(members) or
                {entry.filename for entry in archive.infolist()} !=
                {name + ".npy" for name in members}):
            raise ValueError("private release NPZ member inventory differs")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in members}


def load_role_release(directory: str, receipt_sha256: str, *,
                      arm_id: str, seed: int, realization: int,
                      role: str, policy_receipt_sha256: str,
                      source_binding_sha256: str,
                      role_meta: Mapping,
                      tables: tuple[np.ndarray, np.ndarray]) -> tuple[dict, dict]:
    """Validate R2's law/cached public draw before a protected label read."""
    root = Path(directory)
    if (root.is_symlink() or ".private" not in root.parts or
            not root.is_dir() or root.stat().st_mode & 0o077):
        raise PermissionError("R2 release package lacks owner-only custody")
    receipt_path = _file(str(root / "RELEASE_RECEIPT.json"), receipt_sha256,
                         private=True)
    receipt = json.loads(receipt_path.read_text())
    if (set(receipt) != {"schema", "arm_id", "source_seed", "role",
                         "realization", "source_binding_sha256",
                         "policy_receipt_sha256", "rows", "law_sha256",
                         "release_codes_sha256"} or
            receipt["schema"] != "BAR1-role-release-v1" or
            receipt["arm_id"] != arm_id or receipt["source_seed"] != seed or
            receipt["realization"] != realization or receipt["role"] != role or
            receipt["policy_receipt_sha256"] != policy_receipt_sha256 or
            receipt["source_binding_sha256"] != source_binding_sha256 or
            receipt["rows"] != role_meta["rows"]):
        raise PermissionError("R2 release role/source/policy identity differs")
    law = _npz(_file(str(root / "law.npz"), receipt["law_sha256"], private=True),
               LAW_MEMBERS)
    release = _npz(_file(str(root / "release_codes.npz"),
                         receipt["release_codes_sha256"], private=True),
                   RELEASE_MEMBERS)
    n = role_meta["rows"]
    row = release["row_id"]
    unit = release["unit"]
    if (row.dtype != np.int64 or unit.dtype != np.int64 or
            row.shape != (n,) or unit.shape != (n,) or
            not np.all(np.diff(row) > 0) or
            len(np.unique(unit)) != role_meta["groups"] or
            sha_array(np.column_stack((row, unit))) !=
            role_meta["row_unit_pair_sha256"] or
            any(not np.array_equal(law[key], release[key]) for key in ("row_id", "unit"))):
        raise PermissionError("R2 release differs from admitted ordered role")
    ids = (release["code_r1"], release["code_r2"])
    if (any(code.dtype != np.int64 or code.shape != (n,) or
            np.any((code < 0) | (code >= len(tables[i])))
            for i, code in enumerate(ids))):
        raise ValueError("R2 cached release code is outside public bank")
    if (law["code_r1"].dtype != np.int64 or law["code_r2"].dtype != np.int64 or
            law["code_r1"].shape != (n, 16) or law["code_r2"].shape != (n, 16) or
            law["mass"].dtype != np.float64 or law["mass"].shape != (n, 16) or
            law["valid"].dtype != np.bool_ or law["valid"].shape != (n, 16) or
            any(law[key].dtype != np.int64 or law[key].shape != (n,)
                for key in ("b0_code_r1", "b0_code_r2")) or
            not np.isfinite(law["mass"]).all() or np.any(law["mass"] < 0) or
            not np.allclose(law["mass"].sum(axis=1), 1.0, rtol=0, atol=1e-12) or
            not np.all(law["valid"][:, 0]) or
            not np.all(law["code_r1"][:, 0] == law["b0_code_r1"]) or
            not np.all(law["code_r2"][:, 0] == law["b0_code_r2"]) or
            np.any(law["mass"][~law["valid"]] != 0)):
        raise ValueError("R2 cached release law shape, mass or B0 action differs")
    if (np.any((law["code_r1"] < 0) | (law["code_r1"] >= len(tables[0]))) or
            np.any((law["code_r2"] < 0) | (law["code_r2"] >= len(tables[1])))):
        raise ValueError("R2 law contains code outside public bank")
    positive = law["valid"] & (law["mass"] > 0)
    matching = ((law["code_r1"] == ids[0][:, None]) &
                (law["code_r2"] == ids[1][:, None]) & positive)
    if not np.all(matching.any(axis=1)):
        raise ValueError("cached output is absent from positive law support")
    classes = tuple(np.argmax(table, axis=1) for table in tables)
    for i in range(n):
        active = np.flatnonzero(law["valid"][i])
        pairs = [(int(law["code_r1"][i, j]), int(law["code_r2"][i, j]))
                 for j in active]
        if len(pairs) != len(set(pairs)):
            raise ValueError("law repeated a global public output pair")
        base_class = (classes[0][law["b0_code_r1"][i]],
                      classes[1][law["b0_code_r2"][i]])
        if any((classes[0][a], classes[1][b]) != base_class
               for a, b in pairs):
            raise ValueError("law changed the source bank decision class")
    role_view = materialize_role_from_ids(row, unit, ids[0], ids[1], tables)
    return role_view, receipt


def run(spec_path: str, spec_sha256: str, lock_path: str,
        lock_sha256: str) -> dict:
    active = assert_active_from_env(stage="attacks")
    spec = json.loads(_file(spec_path, spec_sha256, private=True).read_text())
    validate_spec(spec)
    if (active["unit_id"] != spec["unit_id"] or
            active["seal_sha256"] != spec["campaign_seal_sha256"]):
        raise PermissionError("queue unit or campaign seal differs")
    seal = _file(spec["campaign_seal_path"], spec["campaign_seal_sha256"])
    lock = json.loads(_file(lock_path, lock_sha256).read_text())
    expected = {"spec_sha256": spec_sha256, "arm_id": spec["arm_id"],
                "source_seed": spec["source_seed"],
                "realization": spec["realization"], "point_eligible": True}
    if (lock.get("schema") != STAGE_SCHEMA or
            lock.get("status") != "LOCALLY_SEALED_ATTACKS" or
            lock.get("campaign_seal_sha256") != spec["campaign_seal_sha256"] or
            lock.get("units", {}).get(spec["unit_id"]) != expected):
        raise PermissionError("R0 did not admit exact point-eligible policy release")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    if not any(item["sha256"] == lock_sha256 and "attacks" in item["stages"]
               for item in state.get("amendments", [])):
        raise PermissionError("policy attack lock was not registered")
    for item in spec["source_pins"]:
        _file(item["path"], item["sha256"])
    _live_sources(spec["source_pins"])
    _file(spec["role_manifest_path"], spec["role_manifest_sha256"])
    store = RoleOnlyStore(spec["role_manifest_path"])
    eligibility = json.loads(_file(spec["eligibility_receipt_path"],
                                   spec["eligibility_receipt_sha256"]).read_text())
    if (eligibility.get("arm_id") != spec["arm_id"] or
            eligibility.get("all_two_seed_task_reference_point_caps_pass") is not True):
        raise PermissionError("arm did not pass original common point utility gate")
    policy, policy_receipt = load_policy(
        spec["policy_directory"], receipt_sha256=spec["policy_receipt_sha256"])
    if (policy_receipt["arm_id"] != spec["arm_id"] or
            policy_receipt["source_seed"] != spec["source_seed"]):
        raise PermissionError("selected policy differs from attack arm/source seed")
    table_doc = json.loads(_file(spec["public_table_path"],
                                 spec["public_table_sha256"]).read_text())
    if (table_doc.get("schema") != "code-utility-public-bank-tables-v1" or
            table_doc.get("seed") != spec["source_seed"] or
            table_doc.get("bit_count") != 10):
        raise PermissionError("source public bank table identity differs")
    family = table_doc["families"]["W_TSTAR"]
    tables = tuple(np.asarray(family[f"table{i}"], dtype=np.float64)
                   for i in (0, 1))
    if (not np.array_equal(policy.bank1.probability, tables[0]) or
            not np.array_equal(policy.bank2.probability, tables[1])):
        raise PermissionError("selected policy bank differs from frozen W_TSTAR table")
    roles, receipts = {}, {}
    for role in ROLES:
        roles[role], receipts[role] = load_role_release(
            spec["release_directory_by_role"][role],
            spec["release_receipt_sha256_by_role"][role],
            arm_id=spec["arm_id"], seed=spec["source_seed"],
            realization=spec["realization"], role=role,
            policy_receipt_sha256=spec["policy_receipt_sha256"],
            source_binding_sha256=policy_receipt["source_binding_sha256"],
            role_meta=store.manifest["roles"][role], tables=tables)
    prepared = prepare_complete_bank_views(
        roles["ATTACK_FIT_NEW"], roles["INNER_SELECTION"], tables,
        metadata_att=({}, {}), metadata_inner=({}, {}))
    private_out = _private_new_directory(spec["private_output_path"])
    if Path(spec["public_receipt_path"]).exists():
        raise FileExistsError("public policy attack receipt already exists")

    def guard() -> None:
        current = assert_active_from_env(unit_id=spec["unit_id"], stage="attacks")
        if current["seal_sha256"] != spec["campaign_seal_sha256"]:
            raise PermissionError("campaign seal changed during attack")

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
    public = {"schema": "pcrl-bank-r4-policy-fresh-attack-v1",
              "status": "COMPLETE_PRIVATE_DEVELOPMENT_ONLY",
              "unit_id": spec["unit_id"], "arm_id": spec["arm_id"],
              "source_seed": spec["source_seed"],
              "realization": spec["realization"],
              "attacker_seed": spec["attacker_seed"],
              "campaign_seal_sha256": spec["campaign_seal_sha256"],
              "stage_lock_sha256": lock_sha256, "unit_spec_sha256": spec_sha256,
              "policy_receipt_sha256": spec["policy_receipt_sha256"],
              "public_table_sha256": spec["public_table_sha256"],
              "release_receipt_sha256_by_role":
                  spec["release_receipt_sha256_by_role"],
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
