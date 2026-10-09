"""Write-once exact selected-input wires and complete attacker view custody.

Only admitted in-memory feature records enter this module. The ATT/INNER
release stage opens no SEX or task label. Restoring rebuilds each wire with
the public packages and compares it to every stored array and constant.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from results.pcrl_utility_reference_repair_v1.r1.role_store import sha_file
from results.pcrl_utility_reference_repair_v1.r2.complete_views import (
    build_conditional_attack_views, release_conditional_pair,
)


ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")
VIEWS = ("income", "occupation_group", "coalition")
VIEW_ARRAYS = ("row_id", "unit", "x", "x_defense_aware", "x_raw_only")
SCHEMA = "pcrl-utility-conditional-complete-release-v1"


def _sha_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    for block in (str(array.dtype).encode(), str(array.shape).encode(),
                  array.tobytes()):
        digest.update(block)
    return digest.hexdigest()


def _full_sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
    return sha_file(path)


def _atomic_json(path: Path, content: dict) -> str:
    payload = (json.dumps(content, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def _check_context(
    roles: Mapping[str, object], public: Mapping[str, object],
    pair: object, bank: object, root: Path, output: Path, seed: int,
    hashes: Mapping[str, str],
) -> None:
    if (root.name != ".private" or not output.is_relative_to(root)
            or set(roles) != set(ROLES) or set(public) != set(VIEWS[:2])
            or set(hashes) != set(ROLES) or not all(map(_full_sha, hashes.values()))
            or seed not in (0, 1, 2)):
        raise PermissionError("conditional release context is incomplete")
    if (pair.pair_id != f"{pair.income_mask_id}__{pair.occupation_mask_id}"
            or public["income"].mask_id != pair.income_mask_id
            or public["occupation_group"].mask_id != pair.occupation_mask_id):
        raise PermissionError("conditional release pair or public mask identity differs")
    for task, mask_id in (("income", pair.income_mask_id),
                          ("occupation_group", pair.occupation_mask_id)):
        mask = bank.mask(mask_id)
        release = public[task]
        if (release.mask_sha256 != mask.mask_sha256
                or tuple(release.selected_atoms) != tuple(mask.atoms)):
            raise PermissionError("conditional release mask differs from frozen bank")


def _materialize_role(source: Mapping[str, object], public: Mapping[str, object]):
    wire = release_conditional_pair(source, public)
    views, union = build_conditional_attack_views(source, wire, public)
    arrays: dict[str, np.ndarray] = {}
    constants = {}
    for name in VIEWS:
        constants[name] = {}
        for field, value in wire[name].items():
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    raise ValueError("object wire arrays are not admitted")
                arrays[f"{name}__wire__{field}"] = value.copy()
            else:
                constants[name][field] = value
        for field in VIEW_ARRAYS:
            value = getattr(views[name], field)
            if value is None:
                raise ValueError("complete attack view lacks required matrix")
            arrays[f"{name}__view__{field}"] = np.asarray(value).copy()
    return wire, views, union, arrays, constants


def write_complete_release(
    role_sources: Mapping[str, Mapping[str, object]],
    public_releases: Mapping[str, object], pair: object, bank: object,
    output_dir: str | Path, *, private_root: str | Path, seed: int,
    lock_sha256: str, gate_sha256: str, manifest_sha256: str,
    role_feature_sha256: Mapping[str, str],
) -> str:
    """Save both actual role releases and return the index SHA-256."""
    root = Path(private_root).resolve(strict=True)
    output = Path(output_dir).resolve()
    _check_context(role_sources, public_releases, pair, bank, root, output,
                   seed, role_feature_sha256)
    if (not all(_full_sha(value) for value in
                (lock_sha256, gate_sha256, manifest_sha256, bank.bank_sha256))
            or output.exists()):
        raise PermissionError("conditional release lock or output differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.",
                                       dir=output.parent))
    temporary.chmod(0o700)
    role_receipts = {}
    for role in ROLES:
        _, views, union, arrays, constants = _materialize_role(
            role_sources[role], public_releases)
        npz_path = temporary / f"{role}.npz"
        npz_sha = _atomic_npz(npz_path, arrays)
        role_receipts[role] = {
            "npz_sha256": npz_sha,
            "source_feature_sha256": role_feature_sha256[role],
            "array_sha256": {key: _sha_array(value)
                             for key, value in arrays.items()},
            "wire_constants": constants,
            "view_content_sha256": {name: views[name].content_hashes()
                                    for name in VIEWS},
            "union_content_sha256": union.content_sha256,
            "union_source_columns": list(union.source_columns),
            "ordered_row_unit_sha256": _sha_array(np.column_stack((
                union.row_id, union.unit))),
        }
    receipt = {
        "schema": SCHEMA, "status": "FROZEN", "seed": seed,
        "pair_id": pair.pair_id, "bank_sha256": bank.bank_sha256,
        "lock_sha256": lock_sha256, "gate_sha256": gate_sha256,
        "manifest_sha256": manifest_sha256,
        "task_package_sha256_by_purpose": {
            name: public_releases[name].task_package_sha256
            for name in VIEWS[:2]},
        "public_mask_sha256_by_purpose": {
            name: public_releases[name].mask_sha256
            for name in VIEWS[:2]},
        "roles": role_receipts,
        "builder_source_sha256": {
            name: sha_file(Path(__file__).with_name(name))
            for name in ("complete_views.py", "release_custody.py")},
    }
    digest = _atomic_json(temporary / "RELEASE_RECEIPT.json", receipt)
    if output.exists():
        raise PermissionError("conditional release output appeared during write")
    os.rename(temporary, output)
    fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return digest


def restore_complete_release(
    index_path: str | Path, *, expected_index_sha256: str,
    source_roles: Mapping[str, Mapping[str, object]],
    public_releases: Mapping[str, object], pair: object, bank: object,
    private_root: str | Path, seed: int, lock_sha256: str,
    gate_sha256: str, manifest_sha256: str,
    role_feature_sha256: Mapping[str, str],
) -> dict[str, dict]:
    """Rebuild from public packages and exact source values before ATT fitting."""
    root = Path(private_root).resolve(strict=True)
    path = Path(index_path).resolve(strict=True)
    _check_context(source_roles, public_releases, pair, bank, root,
                   path.parent, seed, role_feature_sha256)
    if (not path.is_relative_to(root) or path.name != "RELEASE_RECEIPT.json"
            or sha_file(path) != expected_index_sha256):
        raise PermissionError("conditional release index bytes or path differ")
    receipt = json.loads(path.read_text())
    if (receipt.get("schema") != SCHEMA or receipt.get("status") != "FROZEN"
            or receipt.get("seed") != seed or receipt.get("pair_id") != pair.pair_id
            or receipt.get("bank_sha256") != bank.bank_sha256
            or receipt.get("lock_sha256") != lock_sha256
            or receipt.get("gate_sha256") != gate_sha256
            or receipt.get("manifest_sha256") != manifest_sha256
            or receipt.get("builder_source_sha256") != {
                name: sha_file(Path(__file__).with_name(name))
                for name in ("complete_views.py", "release_custody.py")}
            or set(receipt.get("roles", {})) != set(ROLES)):
        raise PermissionError("conditional release context or builder differs")
    for task in VIEWS[:2]:
        if (receipt["task_package_sha256_by_purpose"].get(task) !=
                public_releases[task].task_package_sha256
                or receipt["public_mask_sha256_by_purpose"].get(task) !=
                public_releases[task].mask_sha256):
            raise PermissionError("conditional release public package differs")
    result = {}
    for role in ROLES:
        entry = receipt["roles"][role]
        npz = path.parent / f"{role}.npz"
        if (entry.get("source_feature_sha256") != role_feature_sha256[role]
                or sha_file(npz) != entry.get("npz_sha256")):
            raise PermissionError("conditional release source or NPZ bytes differ")
        with np.load(npz, allow_pickle=False) as archive:
            if set(archive.files) != set(entry.get("array_sha256", {})):
                raise PermissionError("conditional release array inventory differs")
            saved = {key: archive[key].copy() for key in archive.files}
        if any(_sha_array(value) != entry["array_sha256"][key]
               for key, value in saved.items()):
            raise PermissionError("conditional release stored array bytes differ")
        wire, views, union, fresh, constants = _materialize_role(
            source_roles[role], public_releases)
        if (set(saved) != set(fresh)
                or any(saved[key].dtype != value.dtype
                       or saved[key].shape != value.shape
                       or saved[key].tobytes() != value.tobytes()
                       for key, value in fresh.items())
                or json.dumps(constants, sort_keys=True) !=
                json.dumps(entry.get("wire_constants"), sort_keys=True)
                or any(views[name].content_hashes() !=
                       entry["view_content_sha256"][name] for name in VIEWS)
                or union.content_sha256 != entry["union_content_sha256"]
                or list(union.source_columns) != entry["union_source_columns"]
                or _sha_array(np.column_stack((union.row_id, union.unit))) !=
                entry["ordered_row_unit_sha256"]):
            raise PermissionError("conditional release source/view replay differs")
        result[role] = {"wire": wire, "views": views, "union": union}
    return result
