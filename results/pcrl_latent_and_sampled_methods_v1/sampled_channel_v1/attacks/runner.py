"""Assemble one sampled A realization for R02's fixed three-view attack.

The caller supplies already admitted ATT and INNER release packages. This
module does not open role files or fit readers. Its exact materialization must
be independently replayed by R04 against saved R12 release/cache artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r02.complete_view import (
    PreparedViews, WireBatch, prepare_views,
)
from results.pcrl_latent_and_sampled_methods_v1.r03.compute_queue import (
    assert_active_from_env,
)
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, sha_array,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.protocol import (
    run_three_view_attack,
)
from .q_aware import (PrototypeFeatureTable, fit_q_posterior_reader,
                      table_content_sha256, with_q_features)
from .q_artifact import load_q_table_artifact
from .wire import (CacheIdentity, PublicBank, materialize_pair_batch,
                   materialize_recipient_batch)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls import control_package


@dataclass(frozen=True)
class SampledRoleRelease:
    role: str
    wires1: Sequence[Mapping[str, object]]
    wires2: Sequence[Mapping[str, object]]
    row_id: np.ndarray
    unit: np.ndarray
    left_cache: CacheIdentity
    right_cache: CacheIdentity
    release_receipt_sha256: str | None = None
    private_stream_root_sha256: str | None = None


@dataclass(frozen=True)
class PreparedSampledViews:
    prepared: dict[str, PreparedViews]
    att_batches: dict[str, WireBatch]
    inner_batches: dict[str, WireBatch]
    wire_content_sha256: dict[str, str]
    source_seed: int
    realization: int
    release_id: str
    q_table_sha256: str | None


RELEASE_RECEIPT_KEYS = frozenset({
    "schema", "status", "release_id", "source_seed", "role", "realization",
    "rows", "groups", "row_unit_pair_sha256", "source_binding_sha256",
    "source_identity", "canonical_bank_sha256_by_recipient",
    "mechanism_sha256", "private_stream_root_sha256",
    "cache_file_sha256", "codes_file_sha256",
    "wire_sha256_by_recipient",
})
SOURCE_IDENTITY_KEYS = frozenset({
    "tstar_bank_admission_receipt_sha256", "tstar_role_tokens_file_sha256",
    "reference_file_sha256",
})
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _sha_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _wire_sha256(wires: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for wire in wires:
        digest.update(json.dumps(wire, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8") + b"\n")
    return digest.hexdigest()


def load_role_release_package(
    directory: str | Path, bank1: PublicBank, bank2: PublicBank, *,
    expected_receipt_sha256: str,
) -> SampledRoleRelease:
    """Read R12's exact private one-draw code package, then rebuild sent wires.

    The queue must pin receipt, codes and cache byte hashes before this call.
    The helper never reads source scores, fine cells or any labels. R04 still
    independently replays that code IDs match the cached policy draw.
    """
    supplied_root = Path(directory)
    if supplied_root.is_symlink():
        raise PermissionError("saved release directory cannot be a symlink")
    root = supplied_root.resolve(strict=True)
    if (".private" not in root.parts or not root.is_dir() or
            stat.S_IMODE(root.stat().st_mode) & 0o077):
        raise PermissionError("saved release directory is not private owner-only custody")
    paths = {name: root / name for name in
             ("RELEASE_RECEIPT.json", "release_codes.npz", "cache.sqlite3")}
    for path in paths.values():
        if (path.is_symlink() or not path.is_file() or
                stat.S_IMODE(path.stat().st_mode) & 0o077):
            raise PermissionError("saved release artifact is not a private regular file")
    if (type(expected_receipt_sha256) is not str or
            HEX64.fullmatch(expected_receipt_sha256) is None or
            _sha_file(paths["RELEASE_RECEIPT.json"]) != expected_receipt_sha256):
        raise ValueError("saved release receipt SHA differs from frozen source")
    receipt = json.loads(paths["RELEASE_RECEIPT.json"].read_text())
    if (type(receipt) is not dict or receipt.keys() != RELEASE_RECEIPT_KEYS or
            receipt["schema"] != "SCV1-private-realization-v1" or
            receipt["status"] != "FROZEN_ONE_DRAW" or
            type(receipt["rows"]) is not int or receipt["rows"] < 1 or
            type(receipt["groups"]) is not int or receipt["groups"] < 1):
        raise ValueError("saved release receipt schema/status differs")
    for key in ("row_unit_pair_sha256", "source_binding_sha256",
                "mechanism_sha256", "private_stream_root_sha256",
                "cache_file_sha256", "codes_file_sha256"):
        if type(receipt[key]) is not str or HEX64.fullmatch(receipt[key]) is None:
            raise ValueError("saved release receipt has malformed SHA-256")
    identity = receipt["source_identity"]
    if (type(identity) is not dict or identity.keys() != SOURCE_IDENTITY_KEYS or
            any(type(value) is not str or HEX64.fullmatch(value) is None
                for value in identity.values())):
        raise ValueError("saved release source identity differs")
    if (_sha_file(paths["cache.sqlite3"]) != receipt["cache_file_sha256"] or
            _sha_file(paths["release_codes.npz"]) != receipt["codes_file_sha256"]):
        raise ValueError("saved cache or code artifact SHA differs")
    if (bank1.recipient != 1 or bank2.recipient != 2 or
            bank1.release_id != bank2.release_id or
            receipt["release_id"] != bank1.release_id or
            receipt["source_binding_sha256"] !=
            bank1.public_metadata["source_binding_sha256"] or
            receipt["source_binding_sha256"] !=
            bank2.public_metadata["source_binding_sha256"] or
            receipt["mechanism_sha256"] != bank1.public_metadata["mechanism_sha256"] or
            receipt["mechanism_sha256"] != bank2.public_metadata["mechanism_sha256"] or
            receipt["canonical_bank_sha256_by_recipient"] != {
                "r1": bank1.public_metadata["canonical_bank_sha256"],
                "r2": bank2.public_metadata["canonical_bank_sha256"]}):
        raise ValueError("saved release public bank/source/mechanism identity differs")
    with ZipFile(paths["release_codes.npz"]) as archive:
        if ({member.filename for member in archive.infolist()} != {
                "row_id.npy", "unit.npy", "code_r1.npy", "code_r2.npy"} or
                len(archive.infolist()) != 4):
            raise ValueError("saved release codes NPZ member set differs")
    with np.load(paths["release_codes.npz"], allow_pickle=False) as archive:
        row_id, unit, code1, code2 = (
            np.array(archive[name], copy=True)
            for name in ("row_id", "unit", "code_r1", "code_r2"))
    n = receipt["rows"]
    if (any(value.dtype != np.int64 or value.shape != (n,)
            for value in (row_id, unit, code1, code2)) or
            not np.all(np.diff(row_id) > 0) or
            len(np.unique(unit)) != receipt["groups"] or
            sha_array(np.column_stack((row_id, unit))) !=
            receipt["row_unit_pair_sha256"] or
            np.any((code1 < 0) | (code1 >= len(bank1.probability))) or
            np.any((code2 < 0) | (code2 >= len(bank2.probability)))):
        raise ValueError("saved release ordered rows or code IDs differ")
    cache = CacheIdentity(
        receipt["source_seed"], receipt["role"], receipt["realization"],
        receipt["release_id"], receipt["source_binding_sha256"],
        receipt["cache_file_sha256"])
    def reconstruct(codes: np.ndarray, bank: PublicBank) -> tuple[dict, ...]:
        return tuple({
            "schema": "SCV1-score-release-v1", "recipient": bank.recipient,
            "code_id": int(code), "probability": bank.probability[code].tolist(),
            "decision": int(np.argmax(bank.probability[code])),
            "release_id": bank.release_id,
            "public_metadata": dict(bank.public_metadata),
        } for code in codes)
    wires1, wires2 = reconstruct(code1, bank1), reconstruct(code2, bank2)
    if receipt["wire_sha256_by_recipient"] != {
            "r1": _wire_sha256(wires1), "r2": _wire_sha256(wires2)}:
        raise ValueError("saved release reconstructed wire SHA differs")
    return SampledRoleRelease(receipt["role"], wires1, wires2, row_id, unit,
                              cache, cache, expected_receipt_sha256,
                              receipt["private_stream_root_sha256"])


def validate_attack_identity(
    spec: dict, att: SampledRoleRelease, inner: SampledRoleRelease,
    bank1: PublicBank, bank2: PublicBank, *, policy_receipt_sha256: str,
    policy_arm_id: str,
) -> dict:
    """Bind both R12 role packages and public policy to one immutable A unit."""
    actual = spec.get("attack_identity")
    if type(actual) is not dict or actual.keys() != {
            "arm_id", "source_seed", "realization", "release_id",
            "source_binding_sha256", "mechanism_sha256",
            "canonical_bank_sha256_by_recipient",
            "release_receipt_sha256_by_role",
            "private_stream_root_sha256_by_role", "policy_receipt_sha256"}:
        raise ValueError("A immutable attack identity descriptor is missing")
    for name, package in (("ATTACK_FIT_NEW", att), ("INNER_SELECTION", inner)):
        if package.role != name:
            raise ValueError("A release role differs from immutable attack unit")
    expected = {
        "arm_id": policy_arm_id,
        "source_seed": att.left_cache.source_seed,
        "realization": att.left_cache.realization,
        "release_id": bank1.release_id,
        "source_binding_sha256": bank1.public_metadata["source_binding_sha256"],
        "mechanism_sha256": bank1.public_metadata["mechanism_sha256"],
        "canonical_bank_sha256_by_recipient": {
            "r1": bank1.public_metadata["canonical_bank_sha256"],
            "r2": bank2.public_metadata["canonical_bank_sha256"]},
        "release_receipt_sha256_by_role": {
            "ATTACK_FIT_NEW": att.release_receipt_sha256,
            "INNER_SELECTION": inner.release_receipt_sha256},
        "private_stream_root_sha256_by_role": {
            "ATTACK_FIT_NEW": att.private_stream_root_sha256,
            "INNER_SELECTION": inner.private_stream_root_sha256},
        "policy_receipt_sha256": policy_receipt_sha256,
    }
    if (actual != expected or
            inner.left_cache.source_seed != expected["source_seed"] or
            inner.left_cache.realization != expected["realization"] or
            any(type(value) is not str or HEX64.fullmatch(value) is None
                for value in (*expected["release_receipt_sha256_by_role"].values(),
                              *expected["private_stream_root_sha256_by_role"].values(),
                              policy_receipt_sha256))):
        raise ValueError("A saved release/policy differs from immutable attack identity")
    return expected


def load_fixed_saved_release_pair(
    policy_directory: str | Path, att_directory: str | Path,
    inner_directory: str | Path, *, attack_identity: dict,
) -> tuple[SampledRoleRelease, SampledRoleRelease, PublicBank, PublicBank, dict]:
    """Restore one selected stochastic or matched-control policy and role packages."""
    if type(attack_identity) is not dict:
        raise ValueError("immutable A attack identity is required before policy loading")
    policy, receipt = control_package.load_selected_policy_package(
        policy_directory,
        receipt_sha256=attack_identity["policy_receipt_sha256"])
    if (receipt["arm_id"] != attack_identity["arm_id"] or
            receipt["source_seed"] != attack_identity["source_seed"] or
            receipt["release_id"] != attack_identity["release_id"] or
            receipt["source_binding_sha256"] !=
            attack_identity["source_binding_sha256"] or
            receipt["mechanism_sha256"] != attack_identity["mechanism_sha256"] or
            receipt["bank_sha256_by_recipient"] !=
            attack_identity["canonical_bank_sha256_by_recipient"]):
        raise ValueError("selected policy differs from immutable A attack unit")
    banks = []
    for recipient, fixed in ((1, policy.bank1), (2, policy.bank2)):
        banks.append(PublicBank(
            recipient, fixed.probability, receipt["release_id"], {
                "model_id": receipt["release_id"],
                "source_binding_sha256": receipt["source_binding_sha256"],
                "canonical_bank_sha256": receipt["bank_sha256_by_recipient"][f"r{recipient}"],
                "mechanism_sha256": receipt["mechanism_sha256"],
                "code_length_bits": (len(fixed.probability) - 1).bit_length(),
                "fallback": False,
                "sampling_contract": "cached_one_joint_draw_v1",
            }))
    bank1, bank2 = banks
    att = load_role_release_package(
        att_directory, bank1, bank2,
        expected_receipt_sha256=attack_identity[
            "release_receipt_sha256_by_role"]["ATTACK_FIT_NEW"])
    inner = load_role_release_package(
        inner_directory, bank1, bank2,
        expected_receipt_sha256=attack_identity[
            "release_receipt_sha256_by_role"]["INNER_SELECTION"])
    validate_attack_identity({"attack_identity": attack_identity},
                             att, inner, bank1, bank2,
                             policy_receipt_sha256=attack_identity["policy_receipt_sha256"],
                             policy_arm_id=receipt["arm_id"])
    return att, inner, bank1, bank2, receipt


def assert_a_attack_admission(
    *, queue_unit_id: str, original_lock_path: str | Path,
    original_lock_sha256: str, a_amendment_path: str | Path,
    a_amendment_sha256: str, role_manifest_sha256: str,
    source_paths: Sequence[str | Path],
) -> tuple[dict, dict]:
    """Check original R00 seal, latest registered A amendment and source bytes.

    This is the first call of the protected worker, before release/role reads.
    R03's inherited slot FD and process-group state are checked independently
    by ``assert_active_from_env``; an ordinary local call fails closed.
    """
    active = assert_active_from_env(unit_id=queue_unit_id, stage="a")
    lock = Path(original_lock_path).resolve(strict=True)
    amendment_path = Path(a_amendment_path).resolve(strict=True)
    if (_sha_file(lock) != original_lock_sha256 or
            _sha_file(amendment_path) != a_amendment_sha256 or
            active["seal_sha256"] != original_lock_sha256):
        raise ValueError("A original seal or amendment file SHA differs")
    original = json.loads(lock.read_text())
    if (original.get("schema") != "pcrl-latent-and-sampled-campaign-lock-v1"
            or original.get("status") != "LOCALLY_SEALED_D_STAGE"):
        raise ValueError("A original campaign lock is not locally sealed")
    source_pins = original.get("source_files_sha256", {})
    source_audit = lock.parent / "r01" / "SOURCE_AUDIT.json"
    if (source_pins.get("source_role_manifest") != role_manifest_sha256 or
            source_pins.get("r01/SOURCE_AUDIT.json") != _sha_file(source_audit)):
        raise ValueError("A role manifest/source audit differs from original seal")
    amendment = json.loads(amendment_path.read_text())
    if (amendment.get("schema") != "pcrl-latent-sampled-A-stage-amendment-v1"
            or amendment.get("status") != "LOCALLY_SEALED_A_STAGE"
            or amendment.get("original_lock_sha256") != original_lock_sha256):
        raise ValueError("A stage amendment is not an effective matched seal")
    root = Path(os.environ["PCRL_QUEUE_ROOT"]).resolve(strict=True)
    state = json.loads((root / "state.json").read_text())
    unit = state.get("units", {}).get(queue_unit_id, {})
    spec = unit.get("spec", {})
    stage_amendments = [item for item in state.get("amendments", [])
                        if "a" in item.get("stages", [])]
    if (not stage_amendments or stage_amendments[-1].get("sha256") != a_amendment_sha256
            or Path(stage_amendments[-1].get("path", "")).resolve() != amendment_path
            or spec.get("amendment_sha256") != a_amendment_sha256):
        raise ValueError("A unit does not bind latest registered stage amendment")
    identity = spec.get("attack_identity")
    unit_plan = amendment.get("unit_plan")
    if type(unit_plan) is not dict:
        raise ValueError("A amendment has no sealed unit plan")
    planned = unit_plan.get(queue_unit_id)
    if type(identity) is not dict or type(planned) is not dict:
        raise ValueError("A attack lacks sealed unit plan or immutable identity")
    release_units = planned.get("release_unit_ids")
    if type(release_units) is not dict or set(release_units) != {"ATT", "INNER"}:
        raise ValueError("A attack release unit plan is incomplete")
    expected_deps = [planned.get("fit_unit_id"), release_units["ATT"],
                     release_units["INNER"], planned.get("q_unit_id")]
    q_planned = unit_plan.get(planned.get("q_unit_id"))
    if (planned.get("kind") != "attack" or
            planned.get("output_id") != queue_unit_id or
            planned.get("arm_id") != identity.get("arm_id") or
            planned.get("seed") != identity.get("source_seed") or
            planned.get("realization") != identity.get("realization") or
            planned.get("trained_reader_and_probe_fits") != 33 or
            any(type(value) is not str or not value for value in expected_deps) or
            planned.get("deps") != expected_deps or spec.get("deps") != expected_deps or
            type(q_planned) is not dict or
            q_planned.get("kind") != "q_preprocess" or
            q_planned.get("output_id") != planned.get("q_unit_id") or
            q_planned.get("arm_id") != planned.get("arm_id") or
            q_planned.get("seed") != planned.get("seed") or
            q_planned.get("fit_unit_id") != planned.get("fit_unit_id")):
        raise ValueError("A attack spec differs from sealed arm/seed/draw/Q unit plan")
    stream_path = (lock.parent / "sampled_channel_v1" / "design" /
                   "STREAM_ROOT_HASH_MANIFEST.json").resolve(strict=True)
    streams = json.loads(stream_path.read_text())
    realization = amendment.get("realizations", {})
    identity = spec.get("attack_identity", {})
    draw = identity.get("realization")
    roots = streams.get("release_realization_root_sha256", {})
    if (streams.get("schema") != "scv1-public-stream-root-hashes-v1"
            or set(roots) != {"0", "1", "2"}
            or type(draw) is not int or draw not in (0, 1, 2)
            or realization.get("ids") != [0, 1, 2]
            or realization.get("stream_root_hash_manifest_sha256") != _sha_file(stream_path)
            or realization.get("release_root_sha256_by_realization") != roots
            or identity.get("private_stream_root_sha256_by_role") != {
                "ATTACK_FIT_NEW": roots[str(draw)],
                "INNER_SELECTION": roots[str(draw)]}):
        raise ValueError("A attack stream roots differ from precommitted realization")
    declared = {Path(item["path"]).resolve(): item["sha256"]
                for item in spec.get("sources", [])}
    implementation = amendment.get("implementation_source_sha256")
    required_implementation = {
        "sampled_channel_v1/attacks/runner.py",
        "sampled_channel_v1/attacks/wire.py",
        "sampled_channel_v1/attacks/q_aware.py",
        "sampled_channel_v1/attacks/q_artifact.py",
        "sampled_channel_v1/controls/control_package.py",
        "r02/complete_view.py", "r02/protocol.py", "r03/compute_queue.py",
    }
    if (type(implementation) is not dict or
            not required_implementation <= set(implementation)):
        raise ValueError("A amendment lacks mandatory attack implementation pins")
    for relative, digest in implementation.items():
        part = Path(relative)
        if (type(relative) is not str or part.is_absolute() or
                ".." in part.parts or type(digest) is not str or
                HEX64.fullmatch(digest) is None):
            raise ValueError("A implementation source path or SHA is malformed")
        path = (lock.parent / part).resolve(strict=True)
        if (not path.is_relative_to(lock.parent) or
                declared.get(path) != digest or _sha_file(path) != digest):
            raise ValueError("A implementation source differs from amendment/queue")
    for path in (lock, amendment_path, stream_path,
                 *(Path(p).resolve(strict=True) for p in source_paths)):
        if declared.get(path) != _sha_file(path):
            raise ValueError("A unit source path/SHA differs from immutable queue spec")
    return amendment, spec


def _wire_content_sha256(package: SampledRoleRelease) -> str:
    digest = hashlib.sha256(b"SCV1-ordered-two-recipient-wire-v1\n")
    for value in (package.row_id, package.unit):
        array = np.asarray(value)
        if array.dtype != np.dtype("int64") or array.ndim != 1:
            raise ValueError("private ordered release IDs must be int64 vectors")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode() + b"\n")
        digest.update(np.ascontiguousarray(array, dtype="<i8").tobytes())
    for left, right in zip(package.wires1, package.wires2, strict=True):
        for wire in (left, right):
            digest.update(json.dumps(wire, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode() + b"\n")
    return digest.hexdigest()


def _assert_q_unit_custody(q_dir: Path, *, spec: dict, amendment: dict,
                           queue_unit_id: str, q_identity: dict) -> None:
    """Require the one public Q preprocessing unit named by the sealed plan."""
    q_id = amendment["unit_plan"][queue_unit_id]["q_unit_id"]
    root = Path(os.environ["PCRL_QUEUE_ROOT"]).resolve(strict=True)
    state = json.loads((root / "state.json").read_text())
    q_unit = state.get("units", {}).get(q_id, {})
    files = [str(q_dir / "Q_TABLE_RECEIPT.json"), str(q_dir / "q_table.npz")]
    if (q_id not in spec.get("deps", []) or q_unit.get("status") != "done" or
            q_unit.get("spec", {}).get("outputs") != files):
        raise ValueError("A public Q dependency/output differs from sealed unit")
    queue_receipt_path = root / "units" / q_id / "RECEIPT.json"
    queue_receipt = json.loads(queue_receipt_path.read_text())
    if (queue_receipt.get("status") != "done" or
            queue_receipt.get("returncode") != 0 or
            queue_receipt.get("spec_sha256") != q_unit.get("spec_sha256") or
            queue_receipt.get("output_sha256") !=
                {path: _sha_file(Path(path)) for path in files} or
            q_identity.get("receipt_sha256") != _sha_file(Path(files[0]))):
        raise ValueError("A public Q dependency receipt/content differs")


def prepare_sampled_three_views(
    att: SampledRoleRelease, inner: SampledRoleRelease,
    bank1: PublicBank, bank2: PublicBank,
    *, q_table: PrototypeFeatureTable | None = None,
) -> PreparedSampledViews:
    """Freeze one ATT/INNER realization; never stack repeated samples."""
    if not isinstance(att, SampledRoleRelease) or not isinstance(inner, SampledRoleRelease):
        raise ValueError("ATT and INNER sampled role packages are required")
    if (att.role != "ATTACK_FIT_NEW" or inner.role != "INNER_SELECTION"
            or att.left_cache.role != att.role or att.right_cache.role != att.role
            or inner.left_cache.role != inner.role or inner.right_cache.role != inner.role):
        raise ValueError("sampled attack requires ATTACK_FIT_NEW and INNER_SELECTION packages")
    if (att.left_cache.source_seed != inner.left_cache.source_seed
            or att.left_cache.realization != inner.left_cache.realization):
        raise ValueError("ATT and INNER releases need same source seed and realization")
    if (att.left_cache.release_id != inner.left_cache.release_id
            or att.left_cache.release_id != bank1.release_id
            or bank1.release_id != bank2.release_id):
        raise ValueError("ATT and INNER releases need one frozen paired arm")
    if (att.left_cache.source_binding_sha256 !=
            inner.left_cache.source_binding_sha256):
        raise ValueError("ATT and INNER releases need one frozen source binding")
    batches: dict[str, dict[str, WireBatch]] = {}
    hashes: dict[str, str] = {}
    for role_name, package in (("att", att), ("inner", inner)):
        left = materialize_recipient_batch(package.wires1, package.row_id,
                                           package.unit, bank1)
        right = materialize_recipient_batch(package.wires2, package.row_id,
                                            package.unit, bank2)
        pair = materialize_pair_batch(package.wires1, package.wires2,
                                      package.row_id, package.unit, bank1, bank2,
                                      left_cache=package.left_cache,
                                      right_cache=package.right_cache)
        view_batches = {"r1": left, "r2": right, "pair": pair}
        if q_table is not None:
            view_batches = {name: with_q_features(value, q_table, view=name)
                            for name, value in view_batches.items()}
        batches[role_name] = view_batches
        hashes[role_name] = _wire_content_sha256(package)
    prepared = {name: prepare_views(batches["att"][name],
                                    batches["inner"][name])
                for name in ("r1", "r2", "pair")}
    return PreparedSampledViews(prepared, batches["att"], batches["inner"],
                                hashes, att.left_cache.source_seed,
                                att.left_cache.realization, bank1.release_id,
                                None if q_table is None else q_table.table_sha256)


def run_sampled_attack(
    *, policy_package_dir: str | Path,
    att_release_dir: str | Path, inner_release_dir: str | Path,
    q_table_dir: str | Path | None,
    role_manifest_path: str | Path, role_manifest_sha256: str,
    original_lock_path: str | Path, original_lock_sha256: str,
    a_amendment_path: str | Path, a_amendment_sha256: str,
    queue_unit_id: str,
    private_root: str | Path,
) -> dict:
    """Queue-gated A attacker from exact R12 policy/code/cache and Q artifacts."""
    # The active inherited slot check occurs before any path or role value read.
    assert_active_from_env(unit_id=queue_unit_id, stage="a")
    from . import q_aware, q_artifact, wire
    from results.pcrl_latent_and_sampled_methods_v1.r01 import role_store
    from results.pcrl_latent_and_sampled_methods_v1.r02 import complete_view, protocol
    from results.pcrl_latent_and_sampled_methods_v1.r03 import compute_queue
    from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl import (
        bank, channel, fit, materialize, policy_package, release, source_bridge)
    from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls import (
        deterministic_policy, factor_controls)
    for supplied in (policy_package_dir, att_release_dir, inner_release_dir,
                     *((q_table_dir,) if q_table_dir is not None else ())):
        if Path(supplied).is_symlink():
            raise PermissionError("A source package directory cannot be a symlink")
    policy_dir = Path(policy_package_dir).resolve(strict=True)
    att_dir = Path(att_release_dir).resolve(strict=True)
    inner_dir = Path(inner_release_dir).resolve(strict=True)
    q_dir = None if q_table_dir is None else Path(q_table_dir).resolve(strict=True)
    manifest_path = Path(role_manifest_path).resolve(strict=True)
    source_audit_path = Path(original_lock_path).resolve(strict=True).parent / \
        "r01" / "SOURCE_AUDIT.json"
    stream_path = Path(original_lock_path).resolve(strict=True).parent / \
        "sampled_channel_v1" / "design" / "STREAM_ROOT_HASH_MANIFEST.json"
    inherited_shared = Path(protocol.__file__).resolve().parents[2] / \
        "pcrl_method_portfolio_v1" / "shared"
    mandatory = [
        policy_dir / "POLICY_RECEIPT.json", policy_dir / "policy.npz",
        *(directory / name for directory in (att_dir, inner_dir)
          for name in ("RELEASE_RECEIPT.json", "release_codes.npz", "cache.sqlite3")),
        manifest_path, source_audit_path, stream_path,
        *(Path(module.__file__).resolve(strict=True) for module in (
            control_package, deterministic_policy, factor_controls,
            policy_package, bank, channel, fit, materialize, release, source_bridge,
            q_aware, q_artifact, wire, role_store, complete_view, protocol,
            compute_queue)),
        *(inherited_shared / name for name in (
            "attack_eval.py", "attack_controls_v2.py", "common_eval.py",
            "safe_data_guard.py")),
        Path(__file__).resolve(strict=True),
    ]
    if q_dir is not None:
        mandatory.extend((q_dir / "Q_TABLE_RECEIPT.json", q_dir / "q_table.npz"))
    sources = tuple(Path(path).resolve(strict=True) for path in mandatory)
    amendment, spec = assert_a_attack_admission(
        queue_unit_id=queue_unit_id,
        original_lock_path=original_lock_path,
        original_lock_sha256=original_lock_sha256,
        a_amendment_path=a_amendment_path,
        a_amendment_sha256=a_amendment_sha256,
        role_manifest_sha256=role_manifest_sha256,
        source_paths=sources)
    if Path(private_root).is_symlink():
        raise PermissionError("A attack output root cannot be a symlink")
    root = Path(private_root).resolve()
    if (".private" not in root.parts or root.exists()
            or str(root / "ATTACK_ATTEMPT_INDEX.json") not in spec.get("outputs", [])):
        raise ValueError("A attack output differs from immutable private queue custody")
    if _sha_file(manifest_path) != role_manifest_sha256:
        raise ValueError("admitted role-only manifest SHA differs")
    identity = spec.get("attack_identity")
    if type(identity) is not dict:
        raise ValueError("immutable A attack identity descriptor is missing")
    attack_plan = amendment.get("attack_plan")
    declared_readers = (attack_plan.get("reader_ids")
                        if type(attack_plan) is dict else None)
    plan = protocol.reader_plan(family="A", code_bits=True)
    common_readers = (*plan.core, *plan.probability)
    if (type(declared_readers) is not list or
            declared_readers not in (list(common_readers),
                                     [*common_readers, "Q_POST_LR_C1"])):
        raise ValueError("A amendment attack_plan.reader_ids differs from fixed slate")
    q_required = "Q_POST_LR_C1" in declared_readers
    reason = attack_plan.get("q_inapplicable_reason")
    if ((q_required and reason is not None) or
            (not q_required and (type(reason) is not str or not reason.strip()))):
        raise ValueError("A Q reader applicability lacks its prelocked reason")
    q_identity = spec.get("q_table_identity")
    if (not isinstance(declared_readers, list) or
            q_required != (q_dir is not None) or
            (q_required and (type(q_identity) is not dict or
                             q_identity.keys() != {"receipt_sha256", "content_sha256"})) or
            (not q_required and q_identity is not None)):
        raise ValueError("A amendment Q reader/table applicability differs")
    att, inner, bank1, bank2, policy_receipt = load_fixed_saved_release_pair(
        policy_dir, att_dir, inner_dir, attack_identity=identity)
    if policy_receipt["fit_unit_id"] not in spec.get("deps", []):
        raise ValueError("A attack unit lacks its exact selected fit dependency")
    q_table = None
    expected_q_table_sha256 = None
    if q_required:
        _assert_q_unit_custody(q_dir, spec=spec, amendment=amendment,
                               queue_unit_id=queue_unit_id, q_identity=q_identity)
        expected_q_table_sha256 = q_identity["content_sha256"]
        q_table = load_q_table_artifact(
            q_dir, expected_receipt_sha256=q_identity["receipt_sha256"],
            expected_content_sha256=expected_q_table_sha256,
            expected_policy_receipt_sha256=identity["policy_receipt_sha256"])
    views = prepare_sampled_three_views(att, inner, bank1, bank2, q_table=q_table)
    store = RoleOnlyStore(manifest_path)
    for name, package in (("ATTACK_FIT_NEW", att), ("INNER_SELECTION", inner)):
        expected = store.manifest["roles"][name]["row_unit_pair_sha256"]
        actual = sha_array(np.column_stack((package.row_id, package.unit)))
        if actual != expected:
            raise ValueError("saved sampled release rows differ from admitted role")
    queue_guard = lambda: assert_active_from_env(unit_id=queue_unit_id, stage="a")
    queue_guard()
    sex_att = store.read("ATTACK_FIT_NEW", "sex", procedure="attack")
    root.mkdir(parents=True, mode=0o700)
    extras = None
    if q_table is not None:
        extras = {}
        for name in ("r1", "r2", "pair"):
            extras[name] = (fit_q_posterior_reader(
                views.att_batches[name], views.inner_batches[name],
                sex_att, q_table, view=name, attacker_seed=views.source_seed,
                private_dir=root / "q_readers" / name,
                lock_path=original_lock_path,
                lock_sha256=original_lock_sha256,
                expected_table_sha256=expected_q_table_sha256,
                queue_unit_id=queue_unit_id, queue_guard=queue_guard),)

    def load_inner_sex() -> dict:
        queue_guard()
        freeze = root / "common" / "FROZEN_COMPLETE.json"
        if not freeze.is_file() or json.loads(freeze.read_text()).get("status") != "FROZEN_COMPLETE":
            raise RuntimeError("INNER SEX is embargoed until all readers freeze")
        return store.read("INNER_SELECTION", "sex", procedure="screen")

    outcome = run_three_view_attack(
        views.prepared, sex_att, load_inner_sex=load_inner_sex,
        family="A", seed=views.source_seed, private_root=root / "common",
        lock_path=original_lock_path, lock_sha256=original_lock_sha256,
        queue_guard=queue_guard, specialists_by_view=extras)
    index = {"schema": "SCV1-sampled-attack-attempt-v1",
             "status": "COMPLETE" if all(
                 outcome["public"]["controls"][name]["scores"]["status"] == "PASS"
                 for name in ("r1", "r2", "pair")) else "CONTROL_FAIL",
             "queue_unit_id": queue_unit_id,
             "original_lock_sha256": original_lock_sha256,
             "a_amendment_sha256": a_amendment_sha256,
             "role_manifest_sha256": role_manifest_sha256,
             "saved_source_sha256": {str(p): _sha_file(p) for p in sources},
             "source_seed": views.source_seed,
             "realization": views.realization,
             "release_id": views.release_id,
             "arm_id": identity["arm_id"],
             "selected_fit_unit_id": policy_receipt["fit_unit_id"],
             "policy_receipt_sha256": identity["policy_receipt_sha256"],
             "cache_identity": {
                 "att": att.left_cache.__dict__,
                 "inner": inner.left_cache.__dict__,
             },
             "canonical_bank_sha256": {
                 "r1": bank1.public_metadata["canonical_bank_sha256"],
                 "r2": bank2.public_metadata["canonical_bank_sha256"],
             },
             "mechanism_sha256": bank1.public_metadata["mechanism_sha256"],
             "wire_content_sha256": views.wire_content_sha256,
             "q_table_sha256": views.q_table_sha256,
             "view_content_sha256": {name: {
                 "att": views.prepared[name].att.content_hashes(),
                 "inner": views.prepared[name].inner.content_hashes()}
                 for name in ("r1", "r2", "pair")},
             "reader_freeze_sha256": outcome["public"]["freeze_receipt_sha256"],
             "scores": outcome["public"]["scores"],
             "control_status": {name: outcome["public"]["controls"][name]["scores"]["status"]
                                for name in ("r1", "r2", "pair")}}
    index_path = root / "ATTACK_ATTEMPT_INDEX.json"
    fd = os.open(index_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(index, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"index_path": str(index_path), "index_sha256": _sha_file(index_path),
            "status": index["status"], "scores": index["scores"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Queue-gated sampled A attacker")
    parser.add_argument("--policy-package-dir", required=True)
    parser.add_argument("--att-release-dir", required=True)
    parser.add_argument("--inner-release-dir", required=True)
    parser.add_argument("--q-table-dir")
    parser.add_argument("--role-manifest-path", required=True)
    parser.add_argument("--role-manifest-sha256", required=True)
    parser.add_argument("--original-lock-path", required=True)
    parser.add_argument("--original-lock-sha256", required=True)
    parser.add_argument("--a-amendment-path", required=True)
    parser.add_argument("--a-amendment-sha256", required=True)
    parser.add_argument("--queue-unit-id", required=True)
    parser.add_argument("--private-root", required=True)
    args = parser.parse_args(argv)
    result = run_sampled_attack(**vars(args))
    print(json.dumps({"status": result["status"],
                      "index_sha256": result["index_sha256"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
