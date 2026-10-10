"""R03-queued complete-view attacks for the five fixed D control arms.

This worker is separate from the original hash-sealed T/D-K attacker. It uses
the same R02 reader bank and lazy INNER label boundary. R00's dated A3
amendment must pin this module's exact byte hash before any ATT role read.
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
from ..controls.f_adapter import prepared_f_views
from ..controls.f_service import load_f_service
from ..impl.service import load_service
from .adapter import prepared_d_views
from .worker import _json_once, _load_wire, _private


ARMS = ("L", "S12", "S21", "C", "F")
ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")
F_MEMBERS = frozenset({"row_id", "unit", "features0", "features1", "prob0",
                       "prob1", "decision0", "decision1"})


def _source_arg(argv: list[str], name: str) -> str:
    flag = "--" + name
    if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
        raise PermissionError(f"completed parent argv lacks exactly one {flag}")
    return argv[argv.index(flag) + 1]


def _admit(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if (args.stage != "d_followup" or args.seed not in (0, 1)
            or args.arm_id not in ARMS or args.source_unit_id != f"D_FOLLOWUP_CONTROLS_S{args.seed}"):
        raise PermissionError("follow-up attack identity is outside the authorized slate")
    active = assert_active_from_env(unit_id=args.unit_id, stage=args.stage)
    if active["seal_sha256"] != args.seal_sha256:
        raise PermissionError("R03 active seal and worker argument differ")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    parent = state.get("units", {}).get(args.source_unit_id, {})
    if parent.get("status") != "done" or parent.get("stage") != "d_followup":
        raise PermissionError("five-arm source release queue unit is incomplete")
    parent_spec = parent.get("spec", {})
    parent_argv = parent_spec.get("argv", [])
    if (parent_spec.get("unit_id") != args.source_unit_id
            or parent_spec.get("stage") != "d_followup"
            or _source_arg(parent_argv, "unit-id") != args.source_unit_id
            or _source_arg(parent_argv, "seed") != str(args.seed)
            or _source_arg(parent_argv, "seal-sha256") != args.seal_sha256
            or _source_arg(parent_argv, "role-manifest-sha256") != args.role_manifest_sha256
            or _source_arg(parent_argv, "private-output") != args.source_private
            or _source_arg(parent_argv, "public-receipt") != args.source_receipt
            or parent.get("output_sha256", {}).get(args.source_receipt)
               != args.source_receipt_sha256):
        raise PermissionError("attack source path/hash differs from completed parent output")
    current = next((a["sha256"] for a in reversed(state.get("amendments", []))
                    if "d_followup" in a["stages"]), None)
    if current != args.attack_amendment_sha256:
        raise PermissionError("latest registered d_followup amendment differs")
    seal = Path(args.seal).resolve(strict=True)
    amendment = Path(args.attack_amendment).resolve(strict=True)
    manifest = Path(args.role_manifest).resolve(strict=True)
    if (sha_file(seal) != args.seal_sha256
            or sha_file(amendment) != args.attack_amendment_sha256
            or sha_file(manifest) != args.role_manifest_sha256):
        raise PermissionError("sealed lock, amendment or role manifest changed")
    lock = json.loads(seal.read_text())
    dated = json.loads(amendment.read_text())
    if (lock.get("schema") != "pcrl-latent-and-sampled-campaign-lock-v1"
            or lock.get("status") != "LOCALLY_SEALED_D_STAGE"
            or dated.get("schema") != "pcrl-d-followup-amendment-v1"
            or dated.get("status") != "LOCALLY_SEALED_D_FOLLOWUP"
            or dated.get("effective_seal_sha256") != args.seal_sha256
            or dated.get("control_attack_worker_sha256") != sha_file(Path(__file__))
            or dated.get("role_manifest_sha256") != args.role_manifest_sha256
            or args.unit_id not in dated.get("authorized_unit_ids", ())):
        raise PermissionError("A3 does not authorize this exact control attacker")
    return seal, amendment, manifest


def _source_receipt(args: argparse.Namespace, source_root: Path,
                    store: RoleOnlyStore, amendment: Path) -> dict:
    path = Path(args.source_receipt).resolve(strict=True)
    digest = sha_file(path)
    dated = json.loads(amendment.read_text())
    if (digest != args.source_receipt_sha256
            or dated.get("control_receipt_sha256_by_seed", {}).get(str(args.seed)) != digest):
        raise PermissionError("A3 control source receipt SHA differs")
    receipt = json.loads(path.read_text())
    if (receipt.get("schema") != "latent-repair-followup-controls-receipt-v1"
            or receipt.get("status") != "D_FOLLOWUP_RELEASES_READY"
            or receipt.get("seed") != args.seed
            or receipt.get("unit_id") != args.source_unit_id
            or receipt.get("seal_sha256") != args.seal_sha256
            or receipt.get("role_manifest_sha256") != args.role_manifest_sha256
            or set(receipt.get("packages", {})) != set(ARMS)
            or set(receipt.get("roles", {})) != set(ROLES)):
        raise PermissionError("five-arm source release receipt is incomplete or changed")
    for role in ROLES:
        if set(receipt["roles"][role]) != set(ARMS):
            raise PermissionError("follow-up role/arm inventory differs")
        if receipt["roles"][role][args.arm_id]["rows"] != store.manifest["roles"][role]["rows"]:
            raise PermissionError("source release row count differs from admitted role")
    package = source_root / args.arm_id / "manifest.json"
    expected = receipt["packages"][args.arm_id]
    if (package.is_symlink() or sha_file(package) != expected["manifest_sha256"]
            or sha_file(package.with_name("weights.npz")) != expected["weights_sha256"]):
        raise PermissionError("source service package SHA differs")
    return receipt


def _f_probabilities(service, purpose: int, features: np.ndarray) -> np.ndarray:
    probability = service.uncalibrated_probabilities_from_features(purpose, features)
    if service.calibrators is None:
        raise ValueError("F public wire lacks held-out calibration")
    alpha = float(service.calibrators[purpose]["alpha"])
    clipped = np.maximum(probability, 1e-12)
    clipped /= clipped.sum(axis=1, keepdims=True)
    values = clipped ** alpha
    return values / values.sum(axis=1, keepdims=True)


def _load_f_wire(root: Path, role: str, receipt: dict,
                 store: RoleOnlyStore, service) -> dict:
    path = root / f"release__F__{role}.npz"
    if (path.is_symlink() or not path.is_file()
            or sha_file(path) != receipt["roles"][role]["F"]["private_release_sha256"]):
        raise PermissionError("role-specific F release missing or SHA changed")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != F_MEMBERS:
            raise ValueError("F release member inventory differs")
        arrays = {name: np.array(archive[name], copy=True) for name in F_MEMBERS}
    rows, units = arrays["row_id"], arrays["unit"]
    meta = store.manifest["roles"][role]
    if (rows.dtype != np.int64 or units.dtype != np.int64
            or rows.shape != units.shape or rows.shape != (meta["rows"],)
            or len(np.unique(units)) != meta["groups"]
            or not np.all(np.diff(rows) > 0)
            or sha_array(np.column_stack((rows, units))) != meta["row_unit_pair_sha256"]):
        raise PermissionError("F row/group order differs from admitted role")
    receivers = []
    for purpose, classes in ((0, 2), (1, 6)):
        features = arrays[f"features{purpose}"]
        probabilities = arrays[f"prob{purpose}"]
        decision = arrays[f"decision{purpose}"]
        if (features.dtype != np.float32 or features.shape != (len(rows), 16)
                or probabilities.dtype != np.float64
                or probabilities.shape != (len(rows), classes)
                or decision.dtype != np.int64 or decision.shape != (len(rows),)):
            raise ValueError("F public wire dtype/shape differs")
        expected = _f_probabilities(service, purpose, features)
        if (not np.array_equal(probabilities, expected)
                or not np.array_equal(decision, expected.argmax(axis=1))):
            raise ValueError("F probabilities/decisions do not replay from transmitted features")
        metadata = service.recipient(
            purpose, np.empty((0, service.scaler.dim), dtype=np.float32))["metadata"]
        receivers.append({"features": features, "probabilities": probabilities,
                          "decision": decision, "metadata": metadata})
    return {"row_id": rows, "unit": units, "first": receivers[0], "second": receivers[1]}


def run(args: argparse.Namespace) -> dict:
    seal, amendment, manifest = _admit(args)
    source_root = _private(Path(args.source_private), existing=True)
    private_out = _private(Path(args.private_output), existing=False)
    public_out = Path(args.public_receipt).resolve(strict=False)
    if public_out.exists() or not public_out.parent.is_dir():
        raise FileExistsError("new aggregate public receipt path required")
    store = RoleOnlyStore(manifest)
    receipt = _source_receipt(args, source_root, store, amendment)
    package = source_root / args.arm_id / "manifest.json"
    package_sha = receipt["packages"][args.arm_id]["manifest_sha256"]
    package_record = json.loads(package.read_text())
    if (package_record.get("arm_id") != args.arm_id
            or package_record.get("source_seed") != args.seed):
        raise ValueError("control service package arm or seed differs")
    if args.arm_id == "F":
        service = load_f_service(package, expected_manifest_sha256=package_sha,
                                 expected_source_sha256=receipt["source_sha256"])
        load_wire = lambda role: _load_f_wire(source_root, role, receipt, store, service)
        prepare = prepared_f_views
    else:
        service = load_service(package, expected_manifest_sha256=package_sha,
                               expected_source_sha256=receipt["source_sha256"])
        if service.calibrators is None:
            raise ValueError("hard-code control lacks held-out calibration")
        load_wire = lambda role: _load_wire(source_root, role, args.arm_id,
                                            receipt, store, service)
        prepare = prepared_d_views
    att = load_wire("ATTACK_FIT_NEW")
    inner = load_wire("INNER_SELECTION")
    prepared = prepare(att["row_id"], att["unit"], att["first"], att["second"],
                       inner["row_id"], inner["unit"], inner["first"], inner["second"])

    def guard() -> None:
        active = assert_active_from_env(unit_id=args.unit_id, stage="d_followup")
        if active["seal_sha256"] != args.seal_sha256:
            raise PermissionError("R03 active seal changed during attack")

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
        seed=args.attacker_seed, private_root=private_out, lock_path=seal,
        lock_sha256=args.seal_sha256, queue_guard=guard)
    public = {"schema": "latent-repair-followup-attack-receipt-v1",
              "status": "COMPLETE_PRIVATE_DEVELOPMENT_ONLY", "arm_id": args.arm_id,
              "source_seed": args.seed, "attacker_seed": args.attacker_seed,
              "unit_id": args.unit_id, "source_unit_id": args.source_unit_id,
              "source_receipt_sha256": args.source_receipt_sha256,
              "source_release_sha256": {role: receipt["roles"][role][args.arm_id]["private_release_sha256"]
                                        for role in ROLES},
              "service_manifest_sha256": package_sha,
              "role_manifest_sha256": args.role_manifest_sha256,
              "effective_lock_sha256": args.seal_sha256,
              "attack_amendment_sha256": args.attack_amendment_sha256,
              "attack": attack["public"]}
    _json_once(public_out, public)
    return {"status": public["status"], "arm_id": args.arm_id,
            "unit_id": args.unit_id, "public_receipt_sha256": sha_file(public_out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seal", "seal-sha256", "attack-amendment", "attack-amendment-sha256",
                 "unit-id", "stage", "role-manifest", "role-manifest-sha256",
                 "source-private", "source-receipt", "source-receipt-sha256",
                 "source-unit-id", "arm-id", "private-output", "public-receipt"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--attacker-seed", type=int, required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
