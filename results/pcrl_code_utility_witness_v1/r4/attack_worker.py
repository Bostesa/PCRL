"""Queue-only, hash-pinned complete-view attack worker for hard public codes.

This worker opens only role-specific admitted releases and labels. It restores
the public code-only head before fitting readers. INNER SEX is loaded only after
the shared attack runner writes its frozen-reader custody receipt.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from code_witness.service import load_service
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, assert_same_people, sha_array, sha_file,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.protocol import run_three_view_attack
from results.pcrl_code_utility_witness_v1.r0.compute_queue import assert_active_from_env

from .code_views import prepare_complete_views


RELEASE_MEMBERS = frozenset({"row_id", "unit", "code0", "code1", "prob0", "prob1",
                             "decision0", "decision1"})
ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")
LOCK_SCHEMA = "pcrl-code-witness-attack-lock-v1"
READER_PLAN = {"core": ["LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
                        "CELL_A1", "CODE_BIT_LR_C1"],
               "probability": ["PROB_LR_C1", "PROB_HGB_15x200"],
               "controls": ["SHUFFLED_LR", "PLANTED_LR"],
               "views": ["r1", "r2", "pair"]}


def validate_attack_lock(lock: Mapping, *, campaign_seal_sha256: str,
                         role_manifest_sha256: str, worker_sha256: str,
                         unit_id: str, arm_id: str, seed: int, attacker_seed: int,
                         bits: int,
                         source_model_sha256: str, service_manifest_sha256: str,
                         release_sha256: Mapping[str, str]) -> None:
    """Bind every scientific input and the full mandatory reader slate."""
    if (not isinstance(lock, Mapping) or lock.get("schema") != LOCK_SCHEMA
            or lock.get("status") != "LOCALLY_SEALED_ATTACKS"
            or lock.get("campaign_seal_sha256") != campaign_seal_sha256
            or lock.get("role_manifest_sha256") != role_manifest_sha256
            or lock.get("worker_sha256") != worker_sha256
            or lock.get("reader_plan") != READER_PLAN):
        raise PermissionError("attack lock source, worker or slate differs")
    expected = {"arm_id": arm_id, "seed": seed,
                "attacker_seed": attacker_seed, "bits": bits,
                "source_model_sha256": source_model_sha256,
                "service_manifest_sha256": service_manifest_sha256,
                "release_sha256": dict(release_sha256)}
    if (bits not in (10, 16) or seed not in (0, 1)
            or set(release_sha256) != set(ROLES)
            or lock.get("units", {}).get(unit_id) != expected):
        raise PermissionError("attack lock unit/release registry differs")


def validate_service_manifest_binding(manifest: Mapping, *, arm_id: str,
                                      seed: int, source_model_sha256: str,
                                      role_manifest_sha256: str,
                                      source_schema_sha256: str,
                                      bits: int) -> None:
    if (manifest.get("arm_id") != arm_id
            or manifest.get("source_seed") != seed
            or manifest.get("role_manifest_sha256") != role_manifest_sha256
            or manifest.get("source_sha256") != source_model_sha256
            or manifest.get("source_schema_sha256") != source_schema_sha256
            or manifest.get("bit_count") != bits):
        raise PermissionError("service manifest arm/seed/source/schema/capacity differs")


def validate_archive_arrays(arrays: Mapping[str, np.ndarray], role_meta: Mapping,
                            service, *, bits: int) -> dict:
    """Replay saved role-only code-head wire without consulting any label."""
    if not isinstance(arrays, Mapping) or set(arrays) != RELEASE_MEMBERS:
        raise ValueError("release member inventory differs from hard public-code wire")
    rows, groups = np.asarray(arrays["row_id"]), np.asarray(arrays["unit"])
    n = role_meta["rows"]
    if (rows.dtype != np.int64 or groups.dtype != np.int64
            or rows.shape != (n,) or groups.shape != (n,)
            or len(np.unique(rows)) != n or not np.all(np.diff(rows) > 0)
            or len(np.unique(groups)) != role_meta["groups"]
            or sha_array(np.column_stack((rows, groups))) != role_meta["row_unit_pair_sha256"]):
        raise PermissionError("release row/group order differs from admitted role")
    output = {"row_id": rows, "unit": groups}
    for purpose, name in ((0, "first"), (1, "second")):
        code, q, decision = (np.asarray(arrays[f"{field}{purpose}"])
                             for field in ("code", "prob", "decision"))
        classes = (2, 6)[purpose]
        if (bits not in (10, 16) or code.dtype != np.uint8 or code.shape != (n, bits)
                or not np.isin(code, (0, 1)).all()):
            raise ValueError("release hard code dtype, width or values differ")
        if (q.dtype != np.float64 or q.shape != (n, classes)
                or decision.dtype != np.int64 or decision.shape != (n,)):
            raise ValueError("release probability or decision dtype/shape differs")
        expected = service.probabilities_from_code(purpose, code)
        if (not np.array_equal(q, expected)
                or not np.array_equal(decision, expected.argmax(axis=1))):
            raise ValueError("saved task output does not replay from code-only public head")
        fixed = service.recipient(
            purpose, np.empty((0, service.model.latent_dim), dtype=np.float32))
        if (set(fixed) != {"code", "probabilities", "decision", "metadata"}
                or fixed["metadata"].get("bits") != bits
                or fixed["metadata"].get("purpose") != purpose):
            raise ValueError("public service metadata or wire contract differs")
        output[name] = {"code": code, "probabilities": q,
                        "decision": decision, "metadata": fixed["metadata"]}
    return output


def _pinned(path: str, expected: str, *, private: bool = False) -> Path:
    item = Path(path).resolve(strict=True)
    if (Path(path).is_symlink() or not item.is_file() or sha_file(item) != expected):
        raise PermissionError("pinned source file missing or byte hash changed")
    if private and (".private" not in item.parts or item.stat().st_mode & 0o077):
        raise PermissionError("role release or model must stay under private custody")
    return item


def _private_output(path: str) -> Path:
    item = Path(path).resolve(strict=False)
    if (".private" not in item.parts or item.exists() or not item.parent.is_dir()
            or item.parent.stat().st_mode & 0o077):
        raise PermissionError("new attack custody requires owner-only private parent")
    return item


def _load_release(path: Path, expected: str, role_meta: Mapping,
                  service, *, bits: int) -> dict:
    _pinned(str(path), expected, private=True)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != RELEASE_MEMBERS:
            raise ValueError("release member inventory differs")
        arrays = {key: np.array(archive[key], copy=True) for key in RELEASE_MEMBERS}
    return validate_archive_arrays(arrays, role_meta, service, bits=bits)


def _write_once(path: Path, value: Mapping) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def run(args: argparse.Namespace) -> dict:
    def guard() -> None:
        active = assert_active_from_env(unit_id=args.unit_id, stage="attacks")
        if active["seal_sha256"] != args.seal_sha256:
            raise PermissionError("attack worker campaign seal differs from active queue")

    guard()
    seal = _pinned(args.seal, args.seal_sha256)
    attack_lock = _pinned(args.attack_lock, args.attack_lock_sha256)
    role_path = _pinned(args.role_manifest, args.role_manifest_sha256, private=True)
    service_path = _pinned(args.service_manifest, args.service_manifest_sha256, private=True)
    release = {role: _pinned(getattr(args, f"release_{role.lower()}"),
                             getattr(args, f"release_{role.lower()}_sha256"),
                             private=True) for role in ROLES}
    private_out = _private_output(args.private_output)
    public_out = Path(args.public_receipt).resolve(strict=False)
    if public_out.exists() or not public_out.parent.is_dir():
        raise FileExistsError("new public attack receipt path required")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    if not any(item["sha256"] == args.attack_lock_sha256 and "attacks" in item["stages"]
               for item in state.get("amendments", [])):
        raise PermissionError("R0 did not register the attack stage lock")
    lock = json.loads(attack_lock.read_text())
    worker_sha = sha_file(Path(__file__))
    release_hashes = {role: getattr(args, f"release_{role.lower()}_sha256") for role in ROLES}
    validate_attack_lock(
        lock, campaign_seal_sha256=args.seal_sha256,
        role_manifest_sha256=args.role_manifest_sha256, worker_sha256=worker_sha,
        unit_id=args.unit_id, arm_id=args.arm_id, seed=args.seed,
        attacker_seed=args.attacker_seed, bits=args.bits,
        source_model_sha256=args.source_model_sha256,
        service_manifest_sha256=args.service_manifest_sha256,
        release_sha256=release_hashes)
    store = RoleOnlyStore(role_path)
    manifest = json.loads(service_path.read_text())
    validate_service_manifest_binding(
        manifest, arm_id=args.arm_id, seed=args.seed,
        role_manifest_sha256=args.role_manifest_sha256,
        source_model_sha256=args.source_model_sha256,
        source_schema_sha256=store.manifest["source_schema_sha256"],
        bits=args.bits)
    service = load_service(service_path,
                           expected_manifest_sha256=args.service_manifest_sha256,
                           expected_source_sha256=args.source_model_sha256,
                           expected_bits=args.bits)
    att = _load_release(release["ATTACK_FIT_NEW"], release_hashes["ATTACK_FIT_NEW"],
                        store.manifest["roles"]["ATTACK_FIT_NEW"], service, bits=args.bits)
    inner = _load_release(release["INNER_SELECTION"], release_hashes["INNER_SELECTION"],
                          store.manifest["roles"]["INNER_SELECTION"], service, bits=args.bits)
    prepared = prepare_complete_views(att, inner, bits=args.bits)
    guard()
    sex_att = store.read("ATTACK_FIT_NEW", "sex", procedure="attack")
    assert_same_people(sex_att, att)

    def load_inner_sex() -> dict:
        guard()
        if not (private_out / "FROZEN_COMPLETE.json").is_file():
            raise PermissionError("INNER SEX cannot open before reader freeze receipt")
        record = store.read("INNER_SELECTION", "sex", procedure="screen")
        assert_same_people(record, inner)
        return record

    attack = run_three_view_attack(
        prepared, sex_att, load_inner_sex=load_inner_sex, family="D",
        seed=args.attacker_seed, private_root=private_out,
        lock_path=seal, lock_sha256=args.seal_sha256, queue_guard=guard)
    receipt = {"schema": "pcrl-code-witness-attack-receipt-v1",
               "status": "COMPLETE_PRIVATE_DEVELOPMENT_ONLY",
               "unit_id": args.unit_id, "arm_id": args.arm_id,
               "source_seed": args.seed, "attacker_seed": args.attacker_seed,
               "bits": args.bits, "campaign_seal_sha256": args.seal_sha256,
               "attack_lock_sha256": args.attack_lock_sha256,
               "role_manifest_sha256": args.role_manifest_sha256,
               "source_model_sha256": args.source_model_sha256,
               "service_manifest_sha256": args.service_manifest_sha256,
               "release_sha256": release_hashes,
               "attack": attack["public"]}
    _write_once(public_out, receipt)
    return {"status": receipt["status"], "unit_id": args.unit_id,
            "receipt_sha256": sha_file(public_out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("seal", "seal-sha256", "attack-lock", "attack-lock-sha256",
                 "unit-id", "role-manifest", "role-manifest-sha256",
                 "service-manifest", "service-manifest-sha256", "source-model-sha256",
                 "release_attack_fit_new", "release_attack_fit_new_sha256",
                 "release_inner_selection", "release_inner_selection_sha256",
                 "arm-id", "private-output", "public-receipt"):
        parser.add_argument("--" + flag.replace("_", "-"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--attacker-seed", type=int, required=True)
    parser.add_argument("--bits", type=int, required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
