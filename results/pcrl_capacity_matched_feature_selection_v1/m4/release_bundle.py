"""Write-once custody for selected focused-study complete releases.

The caller provides M1-admitted role records and M2-restored public packages.
This module saves the actual two local wires and aligned coalition wire for
ATTACK_FIT_NEW and INNER_SELECTION; it opens no role files or labels itself.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from results.pcrl_capacity_matched_feature_selection_v1.m1.capacity import (
    assert_role_alignment, reconstruct_union, validate_release_input,
)
from view_custody import restore_view_bundle, save_view_bundle
from views import build_complete_attack_views


ROLE_ORDER = ("ATTACK_FIT_NEW", "INNER_SELECTION")
VIEW_NAMES = ("income", "occupation_group", "coalition")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once_json(path: Path, value: dict) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha_file(path)


def _validate_role(
    source_record: Mapping[str, object], result: Mapping[str, object],
    pair: object, bank: object,
) -> None:
    wire = result["wire"]
    aligned = result["private_alignment"]
    first = bank.mask(pair.income_mask_id).public_record()
    second = bank.mask(pair.occupation_mask_id).public_record()
    x = np.asarray(source_record["X"])
    names = tuple(source_record["feature_names"])
    assert_role_alignment(source_record["row_id"], source_record["unit"],
                          aligned["income"]["row_id"], aligned["income"]["unit"])
    assert_role_alignment(source_record["row_id"], source_record["unit"],
                          aligned["occupation_group"]["row_id"],
                          aligned["occupation_group"]["unit"])
    assert_role_alignment(source_record["row_id"], source_record["unit"],
                          aligned["coalition"]["row_id"], aligned["coalition"]["unit"])
    columns_i = validate_release_input(x, names, first, wire["income"]["selected_values"])
    columns_o = validate_release_input(x, names, second,
                                       wire["occupation_group"]["selected_values"])
    columns_u, raw_u = reconstruct_union(
        first, wire["income"]["selected_values"],
        second, wire["occupation_group"]["selected_values"],
        expected_full_x=x,
    )
    actual_u = np.asarray(wire["coalition"]["selected_values"])
    if (columns_i != bank.mask(pair.income_mask_id).column_indices
            or columns_o != bank.mask(pair.occupation_mask_id).column_indices
            or columns_u != pair.union_columns
            or actual_u.dtype != raw_u.dtype or actual_u.shape != raw_u.shape
            or np.ascontiguousarray(actual_u).tobytes() !=
            np.ascontiguousarray(raw_u).tobytes()):
        raise ValueError("focused complete release differs from pinned source masks")
    if result["receipt"]["pair_id"] != pair.pair_id or \
            result["receipt"]["bank_sha256"] != bank.bank_sha256:
        raise ValueError("M2 deployment receipt differs from frozen bank")


def save_release_bundle(
    *, role_sources: Mapping[str, Mapping[str, object]],
    release_results: Mapping[str, Mapping[str, object]],
    public_releases: Mapping[str, object], pair: object, bank: object,
    schema: object, bridge: object, attack_view_type: type,
    private_root: str | Path, output_dir: str | Path, seed: int,
    lock_sha256: str, manifest_sha256: str,
    role_feature_sha256: Mapping[str, str],
    package_sha256_by_purpose: Mapping[str, str],
) -> dict:
    """Save both real release roles, every wire field and S2 reader matrix."""
    if set(role_sources) != set(ROLE_ORDER) or set(release_results) != set(ROLE_ORDER):
        raise ValueError("release needs exact ATT and INNER role records")
    if set(role_feature_sha256) != set(ROLE_ORDER):
        raise ValueError("release role feature hashes are incomplete")
    root = Path(private_root).resolve(strict=True)
    output = Path(output_dir).resolve()
    if root.name != ".private" or not output.is_relative_to(root) or output.exists():
        raise PermissionError("release output must be a fresh private directory")
    output.mkdir(parents=True, mode=0o700)
    roles = {}
    for role in ROLE_ORDER:
        source = role_sources[role]
        result = release_results[role]
        _validate_role(source, result, pair, bank)
        views, union = build_complete_attack_views(
            source, result["wire"], public_releases, schema,
            bridge=bridge, attack_view_type=attack_view_type,
        )
        receipt_path, receipt_sha = save_view_bundle(
            result["wire"], views, npz_path=output / f"{role}.npz",
            private_root=root, role=role, arm_id=pair.pair_id,
            seed=seed, campaign_lock_sha256=lock_sha256,
            data_manifest_sha256=manifest_sha256,
            role_feature_sha256=role_feature_sha256[role],
            package_sha256_by_purpose=package_sha256_by_purpose,
        )
        roles[role] = {
            "view_receipt_file": receipt_path.name,
            "view_receipt_sha256": receipt_sha,
            "view_npz_sha256": _sha_file(receipt_path.with_suffix(".npz")),
            "raw_union_content_sha256": union.content_sha256,
            "raw_union_columns": list(union.source_columns),
            "view_content_sha256": {
                name: views[name].content_hashes() for name in VIEW_NAMES
            },
            "m2_alignment_receipt": result["receipt"],
        }
    index = {
        "schema": "pcrl-focused-complete-release-index-v1",
        "status": "FROZEN", "seed": seed, "pair_id": pair.pair_id,
        "bank_sha256": bank.bank_sha256,
        "lock_sha256": lock_sha256, "manifest_sha256": manifest_sha256,
        "package_sha256_by_purpose": dict(package_sha256_by_purpose),
        "roles": roles,
    }
    _write_once_json(output / "RELEASE_RECEIPT.json", index)
    return index


def restore_release_role(
    *, index_path: str | Path, expected_index_sha256: str,
    role: str, source_record: Mapping[str, object],
    public_releases: Mapping[str, object], pair: object,
    bank: object, schema: object, bridge: object,
    attack_view_type: type, private_root: str | Path,
    lock_sha256: str, manifest_sha256: str,
    role_feature_sha256: str,
) -> tuple[dict[str, object], object]:
    """Restore, then independently recheck each emitted byte against source."""
    path = Path(index_path).resolve(strict=True)
    root = Path(private_root).resolve(strict=True)
    if (root.name != ".private" or not path.is_relative_to(root)
            or _sha_file(path) != expected_index_sha256 or role not in ROLE_ORDER):
        raise PermissionError("release index identity differs")
    index = json.loads(path.read_text())
    if (index.get("schema") != "pcrl-focused-complete-release-index-v1"
            or index.get("status") != "FROZEN"
            or index.get("pair_id") != pair.pair_id
            or index.get("bank_sha256") != bank.bank_sha256
            or index.get("lock_sha256") != lock_sha256
            or index.get("manifest_sha256") != manifest_sha256):
        raise PermissionError("release index differs from expected pair or lock")
    entry = index["roles"][role]
    restored, wire = restore_view_bundle(
        path.parent / entry["view_receipt_file"],
        expected_receipt_sha256=entry["view_receipt_sha256"],
        private_root=root, expected_campaign_lock_sha256=lock_sha256,
        expected_data_manifest_sha256=manifest_sha256,
        expected_role=role, expected_arm_id=pair.pair_id,
        expected_seed=index["seed"],
        expected_role_feature_sha256=role_feature_sha256,
        expected_package_sha256_by_purpose=index["package_sha256_by_purpose"],
        public_releases=public_releases, attack_view_type=attack_view_type,
        return_release=True,
    )
    views, union = build_complete_attack_views(
        source_record, wire, public_releases, schema, bridge=bridge,
        attack_view_type=attack_view_type,
    )
    if (union.content_sha256 != entry["raw_union_content_sha256"]
            or list(union.source_columns) != entry["raw_union_columns"]
            or any(views[name].content_hashes() != entry["view_content_sha256"][name]
                   for name in VIEW_NAMES)):
        raise PermissionError("restored release differs from pinned source and view")
    return views, union
