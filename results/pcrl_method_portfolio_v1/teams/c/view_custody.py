"""Private write-once custody for Team C's complete release and S2 views."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from s2_bridge import to_s2_attack_views


SCHEMA = "team-c-private-complete-view-v1"
VIEWS = ("income", "occupation_group", "coalition")
MATRICES = ("row_id", "unit", "x", "x_defense_aware", "x_raw_only")


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha_arr(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def _inside_private(path: Path, root: Path) -> None:
    if root.name != ".private" or not path.resolve().is_relative_to(root.resolve()):
        raise PermissionError("view custody path must remain under .private")


def _link_once(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except FileExistsError as exc:
        raise PermissionError("private view artifact is immutable") from exc


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_view_bundle(
    release: Mapping[str, Mapping[str, object]],
    views: Mapping[str, object],
    *, npz_path: str | Path, private_root: str | Path,
    role: str, arm_id: str, seed: int,
    campaign_lock_sha256: str, data_manifest_sha256: str,
    role_feature_sha256: str,
    package_sha256_by_purpose: Mapping[str, str],
) -> tuple[Path, str]:
    """Persist every wire field and reader matrix, with exact array hashes."""
    if role not in ("ATTACK_FIT_NEW", "INNER_SELECTION") or set(release) != set(VIEWS) or set(views) != set(VIEWS):
        raise ValueError("one complete registered role and all three views required")
    path = Path(npz_path)
    root = Path(private_root)
    _inside_private(path, root)
    if path.suffix != ".npz" or path.exists():
        raise PermissionError("view NPZ must be a fresh .npz path")
    if set(package_sha256_by_purpose) != {"income", "occupation_group"}:
        raise ValueError("both release package hashes required")
    arrays = {}
    wire_constants = {}
    view_schemas = {}
    for name in VIEWS:
        view = views[name].validate()
        if set(view.wire_components) != set(release[name]):
            raise ValueError("view does not bind every transmitted wire key")
        constants = {}
        for field, value in release[name].items():
            key = f"{name}__wire__{field}"
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    raise ValueError("object wire array cannot be stored privately")
                arrays[key] = np.asarray(value).copy()
            else:
                constants[field] = value
        wire_constants[name] = constants
        for field in MATRICES:
            value = getattr(view, field)
            if value is None:
                raise ValueError("required Team C view matrix is absent")
            arrays[f"{name}__view__{field}"] = np.asarray(value).copy()
        view_schemas[name] = {
            "components": view.components,
            "wire_components": view.wire_components,
            "component_columns": view.component_columns,
            "wire_constants": view.wire_constants,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".view-", suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        _link_once(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    path.chmod(0o600)
    receipt = {
        "schema": SCHEMA, "role": role, "arm_id": arm_id, "seed": seed,
        "campaign_lock_sha256": campaign_lock_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "role_feature_sha256": role_feature_sha256,
        "package_sha256_by_purpose": dict(package_sha256_by_purpose),
        "builder_source_sha256": {
            name: _sha_file(Path(__file__).with_name(name))
            for name in ("prototype.py", "evaluation.py", "s2_bridge.py", "view_custody.py")
        },
        "npz_sha256": _sha_file(path),
        "array_sha256": {key: _sha_arr(value) for key, value in arrays.items()},
        "wire_constants": wire_constants,
        "view_schemas": view_schemas,
    }
    receipt_path = path.with_suffix(".json")
    if receipt_path.exists():
        raise PermissionError("view receipt already exists")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".view-receipt-", delete=False) as stream:
        temporary_receipt = Path(stream.name)
        stream.write(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        stream.flush()
        os.fsync(stream.fileno())
    try:
        _link_once(temporary_receipt, receipt_path)
        _sync_directory(receipt_path.parent)
    finally:
        temporary_receipt.unlink(missing_ok=True)
    receipt_path.chmod(0o600)
    return receipt_path, _sha_file(receipt_path)


def restore_view_bundle(
    receipt_path: str | Path, *, expected_receipt_sha256: str,
    private_root: str | Path, expected_campaign_lock_sha256: str,
    expected_data_manifest_sha256: str,
    expected_role: str, expected_arm_id: str, expected_seed: int,
    expected_role_feature_sha256: str,
    expected_package_sha256_by_purpose: Mapping[str, str],
    public_releases: Mapping[str, object], attack_view_type: type,
    return_release: bool = False,
) -> dict[str, object] | tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Verify all bytes and replay C's bridge before S2 reads SEX labels."""
    receipt_path = Path(receipt_path)
    root = Path(private_root)
    _inside_private(receipt_path, root)
    if _sha_file(receipt_path) != expected_receipt_sha256:
        raise PermissionError("complete-view receipt SHA changed")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != SCHEMA
            or receipt.get("role") != expected_role
            or receipt.get("arm_id") != expected_arm_id
            or receipt.get("seed") != expected_seed
            or receipt.get("campaign_lock_sha256") != expected_campaign_lock_sha256
            or receipt.get("data_manifest_sha256") != expected_data_manifest_sha256
            or receipt.get("role_feature_sha256") != expected_role_feature_sha256
            or receipt.get("package_sha256_by_purpose") != dict(expected_package_sha256_by_purpose)):
        raise PermissionError("complete-view source or package identity changed")
    for name, checksum in receipt["builder_source_sha256"].items():
        if _sha_file(Path(__file__).with_name(name)) != checksum:
            raise PermissionError("complete-view builder source changed")
    npz_path = receipt_path.with_suffix(".npz")
    _inside_private(npz_path, root)
    if _sha_file(npz_path) != receipt["npz_sha256"]:
        raise PermissionError("complete-view NPZ changed")
    with np.load(npz_path, allow_pickle=False) as package:
        if set(package.files) != set(receipt["array_sha256"]):
            raise PermissionError("complete-view array set changed")
        arrays = {name: package[name].copy() for name in package.files}
    if any(_sha_arr(array) != receipt["array_sha256"][key]
           for key, array in arrays.items()):
        raise PermissionError("complete-view array content changed")
    release = {}
    alignment = {}
    for name in VIEWS:
        wire = dict(receipt["wire_constants"][name])
        for field in ("feature_names", "selected_groups", "income_groups",
                      "occupation_group_groups", "flags"):
            if field in wire:
                wire[field] = tuple(wire[field])
        prefix = f"{name}__wire__"
        wire.update({key[len(prefix):]: value for key, value in arrays.items()
                     if key.startswith(prefix)})
        release[name] = wire
        alignment[name] = {
            "row_id": arrays[f"{name}__view__row_id"],
            "unit": arrays[f"{name}__view__unit"],
        }
    rebuilt = to_s2_attack_views(release, alignment, public_releases, attack_view_type)
    for name in VIEWS:
        view = rebuilt[name]
        stored_schema = receipt["view_schemas"][name]
        if (list(view.components) != stored_schema["components"]
                or list(view.wire_components) != stored_schema["wire_components"]
                or [list(x) for x in view.component_columns] != stored_schema["component_columns"]
                or [list(x) for x in view.wire_constants] != stored_schema["wire_constants"]):
            raise PermissionError("restored complete-view schema differs from source")
        for field in MATRICES:
            if not np.array_equal(np.asarray(getattr(view, field)),
                                  arrays[f"{name}__view__{field}"]):
                raise PermissionError("restored complete-view matrix differs from source")
    return (rebuilt, release) if return_release else rebuilt
