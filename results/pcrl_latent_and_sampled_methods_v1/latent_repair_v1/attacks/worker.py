"""R03-queued first-seed D complete-view attacker worker.

Reads only R07's role-specific public release wires and R01's role-only SEX
archives. INNER SEX is requested lazily after R02 seals the frozen readers.
Direct execution without an inherited active R03 queue slot is refused.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ...r01.role_store import RoleOnlyStore, assert_same_people, sha_array, sha_file
from ...r02.protocol import run_three_view_attack
from ...r03.compute_queue import assert_active_from_env
from ..impl.service import load_service
from .adapter import prepared_d_views


RELEASE_MEMBERS = frozenset({"row_id", "unit", "code0", "code1", "prob0", "prob1",
                             "decision0", "decision1"})
ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")
ARMS = ("T", "D-K")


def _json_once(path: Path, data: dict) -> None:
    payload = json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _private(path: Path, *, existing: bool) -> Path:
    p = path.resolve(strict=existing)
    if ".private" not in p.parts:
        raise PermissionError("per-person or model artifacts require .private custody")
    if existing and (not p.is_dir() or (p.stat().st_mode & 0o077)):
        raise PermissionError("source private directory must be owner-only")
    if not existing and (p.exists() or not p.parent.is_dir() or (p.parent.stat().st_mode & 0o077)):
        raise PermissionError("attack private output must be new below owner-only custody")
    return p


def _source_receipt(args, store: RoleOnlyStore, source_root: Path) -> dict:
    path = Path(args.source_receipt).resolve(strict=True)
    if sha_file(path) != args.source_receipt_sha256:
        raise PermissionError("first-seed source receipt SHA differs")
    receipt = json.loads(path.read_text())
    if (receipt.get("schema") != "latent-repair-first-paired-seed-receipt-v1"
            or receipt.get("status") != "T_DK_RELEASES_READY"
            or receipt.get("seed") != args.seed
            or receipt.get("unit_id") != args.source_unit_id
            or receipt.get("seal_sha256") != args.seal_sha256
            or receipt.get("role_manifest_sha256") != args.role_manifest_sha256
            or set(receipt.get("packages", {})) != set(ARMS)
            or set(receipt.get("roles", {})) != set(ROLES)):
        raise PermissionError("first-seed release receipt is not the registered T/D-K pair")
    for role in ROLES:
        expected = set(ARMS) | ({"endpoint_labels_references_sha256"}
                                if role == "INNER_SELECTION" else set())
        if set(receipt["roles"][role]) != expected:
            raise PermissionError("first-seed role release/endpoint inventory is incomplete")
        if receipt["roles"][role][args.arm_id]["rows"] != store.manifest["roles"][role]["rows"]:
            raise PermissionError("source role row count differs from admitted role")
    endpoint_path = source_root / "endpoint__INNER_SELECTION.npz"
    endpoint_sha = receipt["roles"]["INNER_SELECTION"]["endpoint_labels_references_sha256"]
    if (endpoint_path.is_symlink() or not endpoint_path.is_file()
            or not isinstance(endpoint_sha, str) or len(endpoint_sha) != 64
            or sha_file(endpoint_path) != endpoint_sha):
        raise PermissionError("first-seed private endpoint sidecar hash differs")
    return receipt


def _load_wire(root: Path, role: str, arm: str, receipt: dict,
               store: RoleOnlyStore, service) -> dict:
    path = root / f"release__{arm}__{role}.npz"
    if path.is_symlink() or not path.is_file() or sha_file(path) != \
            receipt["roles"][role][arm]["private_release_sha256"]:
        raise PermissionError("role-specific D wire missing or SHA changed")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != RELEASE_MEMBERS:
            raise ValueError("D release member inventory includes missing/extra arrays")
        z = {name: np.array(archive[name], copy=True) for name in RELEASE_MEMBERS}
    rows, units = z["row_id"], z["unit"]
    meta = store.manifest["roles"][role]
    if (rows.dtype != np.int64 or units.dtype != np.int64
            or rows.shape != units.shape or rows.shape != (meta["rows"],)
            or len(np.unique(units)) != meta["groups"]
            or not np.all(np.diff(rows) > 0)
            or sha_array(np.column_stack((rows, units))) != meta["row_unit_pair_sha256"]):
        raise PermissionError("D release row/group order differs from admitted role")
    receivers = []
    for i, classes in ((0, 2), (1, 6)):
        bits, q, d = z[f"code{i}"], z[f"prob{i}"], z[f"decision{i}"]
        if (bits.dtype != np.uint8 or bits.shape != (len(rows), 10)
                or q.dtype != np.float64 or q.shape != (len(rows), classes)
                or d.dtype != np.int64 or d.shape != (len(rows),)):
            raise ValueError("D wire dtype or dimensional contract differs")
        expected = service.probabilities_from_code(i, bits)
        if not np.array_equal(q, expected) or not np.array_equal(d, expected.argmax(axis=1)):
            raise ValueError("D public probability/decision is not exactly code-only")
        # The metadata is globally fixed and saved in the hash-pinned public
        # service package, not as a redundant per-person NPZ member.
        public_metadata = service.recipient(i, np.empty((0, service.model.latent_dim),
                                                        dtype=np.float32))["metadata"]
        receivers.append({"code": bits, "probabilities": q, "decision": d,
                          "metadata": public_metadata})
    return {"row_id": rows, "unit": units, "first": receivers[0], "second": receivers[1]}


def run(args: argparse.Namespace) -> dict:
    if args.stage not in ("d_first", "d_followup"):
        raise PermissionError("D attack stage is not registered")
    if args.stage == "d_first" and (args.seed != 0 or args.arm_id not in ARMS):
        raise PermissionError("first D attack stage is seed-0 T or D-K only")
    active = assert_active_from_env(unit_id=args.unit_id, stage=args.stage)
    if active["seal_sha256"] != args.seal_sha256:
        raise PermissionError("worker and R03 effective seal differ")
    queue_state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    if queue_state.get("units", {}).get(args.source_unit_id, {}).get("status") != "done":
        raise PermissionError("source T/D-K release queue unit is not complete")
    seal = Path(args.seal).resolve(strict=True)
    if sha_file(seal) != args.seal_sha256:
        raise PermissionError("effective seal bytes changed")
    lock = json.loads(seal.read_text())
    if (lock.get("schema") != "pcrl-latent-and-sampled-campaign-lock-v1"
            or lock.get("status") != "LOCALLY_SEALED_D_STAGE"):
        raise PermissionError("R00 lock does not admit scientific fits")
    manifest = Path(args.role_manifest).resolve(strict=True)
    if sha_file(manifest) != args.role_manifest_sha256:
        raise PermissionError("admitted repair role manifest SHA changed")
    source_root = _private(Path(args.source_private), existing=True)
    private_out = _private(Path(args.private_output), existing=False)
    public_out = Path(args.public_receipt).resolve(strict=False)
    if public_out.exists() or not public_out.parent.is_dir():
        raise FileExistsError("attack public receipt must be a new file")
    store = RoleOnlyStore(manifest)
    receipt = _source_receipt(args, store, source_root)
    package = source_root / args.arm_id / "manifest.json"
    package_sha = receipt["packages"][args.arm_id]["manifest_sha256"]
    package_record = json.loads(package.read_text())
    if (package_record.get("arm_id") != args.arm_id
            or package_record.get("source_seed") != args.seed):
        raise ValueError("public service package arm/seed identity differs")
    service = load_service(package, expected_manifest_sha256=package_sha,
                           expected_source_sha256=receipt["source_sha256"])
    if service.calibrators is None:
        raise ValueError("first-seed scored release lacks held-out calibration")
    att = _load_wire(source_root, "ATTACK_FIT_NEW", args.arm_id, receipt, store, service)
    inner = _load_wire(source_root, "INNER_SELECTION", args.arm_id, receipt, store, service)
    prepared = prepared_d_views(att["row_id"], att["unit"], att["first"], att["second"],
                                inner["row_id"], inner["unit"], inner["first"], inner["second"])

    def guard() -> None:
        current = assert_active_from_env(unit_id=args.unit_id, stage=args.stage)
        if current["seal_sha256"] != args.seal_sha256:
            raise PermissionError("active slot seal changed during attack")

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
    public = {"schema": "latent-repair-first-seed-attack-receipt-v1",
              "status": "COMPLETE_PRIVATE_DEVELOPMENT_ONLY", "arm_id": args.arm_id,
              "source_seed": args.seed, "attacker_seed": args.attacker_seed,
              "unit_id": args.unit_id, "source_unit_id": args.source_unit_id,
              "source_receipt_sha256": args.source_receipt_sha256,
              "source_release_sha256": {role: receipt["roles"][role][args.arm_id]["private_release_sha256"]
                                        for role in ROLES},
              "service_manifest_sha256": package_sha,
              "role_manifest_sha256": args.role_manifest_sha256,
              "effective_lock_sha256": args.seal_sha256,
              "attack": attack["public"]}
    _json_once(public_out, public)
    return {"status": public["status"], "arm_id": args.arm_id,
            "unit_id": args.unit_id, "public_receipt_sha256": sha_file(public_out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seal", "seal-sha256", "unit-id", "stage", "role-manifest",
                 "role-manifest-sha256", "source-private", "source-receipt",
                 "source-receipt-sha256", "source-unit-id", "arm-id",
                 "private-output", "public-receipt"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attacker-seed", type=int, default=0)
    result = run(parser.parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
