"""Independent custody decoding for frozen private M4 release artifacts.

This module checks bytes, identities and transmitted raw values directly. It
does not import M4's release builder, view scorer, or selection implementation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


VIEWS = ("income", "occupation_group", "coalition")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _private(path: Path, root: Path) -> Path:
    resolved = Path(path).resolve(strict=True)
    if (root.name != ".private"
            or not resolved.is_relative_to(root.resolve(strict=True))):
        raise PermissionError("release artifact escaped admitted private root")
    return resolved


def _same_wire_bits(first: np.ndarray, second: np.ndarray) -> bool:
    a, b = np.asarray(first), np.asarray(second)
    return (a.dtype == b.dtype and a.shape == b.shape
            and np.ascontiguousarray(a).tobytes() ==
            np.ascontiguousarray(b).tobytes())


def read_frozen_release(*, index_path: Path, index_sha256: str,
                        private_root: Path, role: str, seed: int,
                        pair_id: str, lock_sha256: str,
                        manifest_sha256: str, role_feature_sha256: str,
                        source_x: np.ndarray, source_row_id: np.ndarray,
                        source_unit: np.ndarray,
                        selected_columns: dict[str, tuple[int, ...]]) -> dict:
    """Decode a saved complete release only after its S3 parent SHA is known."""
    if role not in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        raise PermissionError("unregistered release role")
    root = Path(private_root)
    index_path = _private(index_path, root)
    if file_sha256(index_path) != index_sha256:
        raise PermissionError("release index SHA differs from completed parent")
    index = json.loads(index_path.read_text())
    if (index.get("schema") != "pcrl-focused-complete-release-index-v1"
            or index.get("status") != "FROZEN"
            or index.get("seed") != seed or index.get("pair_id") != pair_id
            or index.get("lock_sha256") != lock_sha256
            or index.get("manifest_sha256") != manifest_sha256
            or role not in index.get("roles", {})):
        raise PermissionError("release index role/seed/pair/lock identity changed")
    entry = index["roles"][role]
    name = entry["view_receipt_file"]
    if not isinstance(name, str) or Path(name).name != name:
        raise PermissionError("view receipt name escapes release directory")
    receipt_path = _private(index_path.parent / name, root)
    if file_sha256(receipt_path) != entry["view_receipt_sha256"]:
        raise PermissionError("view receipt SHA changed")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "team-c-private-complete-view-v1"
            or receipt.get("role") != role
            or receipt.get("arm_id") != pair_id
            or receipt.get("seed") != seed
            or receipt.get("campaign_lock_sha256") != lock_sha256
            or receipt.get("data_manifest_sha256") != manifest_sha256
            or receipt.get("role_feature_sha256") != role_feature_sha256
            or receipt.get("package_sha256_by_purpose") !=
            index.get("package_sha256_by_purpose")):
        raise PermissionError("view receipt source/package identity changed")
    npz_path = _private(receipt_path.with_suffix(".npz"), root)
    if (file_sha256(npz_path) != receipt.get("npz_sha256")
            or file_sha256(npz_path) != entry["view_npz_sha256"]):
        raise PermissionError("release NPZ SHA changed")
    expected = receipt.get("array_sha256")
    if not isinstance(expected, dict) or not expected:
        raise PermissionError("release NPZ member registry absent")
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise PermissionError("release NPZ member set changed")
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    if any(array_sha256(array) != expected[key] for key, array in arrays.items()):
        raise PermissionError("release NPZ member SHA changed")
    x = np.asarray(source_x)
    row, unit = np.asarray(source_row_id), np.asarray(source_unit)
    if (x.dtype != np.dtype("float32") or x.ndim != 2
            or not np.isfinite(x).all() or row.dtype != np.dtype("int64")
            or unit.dtype != np.dtype("int64") or row.shape != (len(x),)
            or unit.shape != row.shape or len(np.unique(row)) != len(row)
            or set(selected_columns) != set(VIEWS)):
        raise PermissionError("admitted source predictor/identity schema changed")
    for view in VIEWS:
        columns = tuple(selected_columns[view])
        if (not columns or len(set(columns)) != len(columns)
                or any(type(column) is not int or not 0 <= column < x.shape[1]
                       for column in columns)):
            raise PermissionError("source selected columns invalid")
        selected = arrays[f"{view}__wire__selected_values"]
        if not _same_wire_bits(selected, x[:, columns]):
            raise PermissionError("transmitted selected values differ from source bits")
        if (not np.array_equal(arrays[f"{view}__view__row_id"], row)
                or not np.array_equal(arrays[f"{view}__view__unit"], unit)):
            raise PermissionError("complete view row/group identity differs from source")
    if list(selected_columns["coalition"]) != entry["raw_union_columns"]:
        raise PermissionError("coalition selected source columns differ from index")
    candidate = {}
    for purpose, width in (("income", 2), ("occupation_group", 6)):
        probability = arrays[f"{purpose}__wire__probabilities"]
        duplicate = arrays[f"coalition__wire__{purpose}_probabilities"]
        if (probability.dtype != np.dtype("float64")
                or probability.shape != (len(row), width)
                or not np.isfinite(probability).all()
                or np.any((probability < 0) | (probability > 1))
                or not np.allclose(probability.sum(axis=1), 1, rtol=0, atol=1e-8)
                or not _same_wire_bits(probability, duplicate)):
            raise PermissionError("coalition and local probability wire differs")
        candidate[purpose] = probability.copy()
    return {
        "candidate_probabilities": candidate,
        "source_row_unit_count": len(row),
        "release_index_sha256": index_sha256,
        "view_receipt_sha256": entry["view_receipt_sha256"],
        "view_npz_sha256": entry["view_npz_sha256"],
        "array_sha256": dict(expected),
    }
