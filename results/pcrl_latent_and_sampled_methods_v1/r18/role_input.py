"""One-row, role-only deployment input reader for an R03-admitted child.

This module must not be invoked before the effective role and queue seal. It
opens only the previously admitted feature export, never a mixed-role source
or a label/SEX file. The returned row and IDs remain private to the child.
"""

from __future__ import annotations

import json
import hashlib
import zipfile
from pathlib import Path

import numpy as np

from .acceptance_contract import (
    FEATURE_NAMES_SHA256, check_raw_schema, sha256_file,
)


FEATURE_MEMBERS = frozenset({"X.npy", "feature_names.npy", "row_id.npy", "unit.npy"})
ADMITTED_ROLES = frozenset({"OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT",
                            "ATTACK_FIT_NEW", "INNER_SELECTION"})


def _sha_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode() + str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_one_admitted_feature_row(package_root: Path, manifest_path: Path,
                                  expected_manifest_sha256: str, role: str,
                                  row_index: int, pinned_schema_sha256: str =
                                  FEATURE_NAMES_SHA256) -> tuple[np.ndarray, tuple[str, ...], object, object]:
    """Hash-check a role export, validate its exact members, and select one row."""
    if role not in ADMITTED_ROLES:
        raise ValueError("deployment role is not admitted")
    root = Path(package_root).resolve(strict=True)
    manifest_file = Path(manifest_path).resolve(strict=True)
    if not manifest_file.is_relative_to(root):
        raise ValueError("manifest is outside the admitted package")
    if sha256_file(manifest_file) != expected_manifest_sha256:
        raise ValueError("role manifest hash differs")
    manifest = json.loads(manifest_file.read_text())
    if (manifest.get("schema") != "pcrl-utility-repair-role-admission-v1"
            or manifest.get("source_schema_sha256") != pinned_schema_sha256):
        raise ValueError("role manifest or predictor schema differs")
    key = f"{role}:features"
    relative = manifest["role_file_relative_path"][key]
    export = (root / relative).resolve(strict=True)
    if not export.is_relative_to(root) or export.name != "features.npz":
        raise ValueError("feature export path differs")
    if sha256_file(export) != manifest["role_file_sha256"][key]:
        raise ValueError("admitted feature export hash differs")
    with zipfile.ZipFile(export) as archive:
        if set(archive.namelist()) != FEATURE_MEMBERS:
            raise ValueError("role feature export has unexpected members")
    with np.load(export, allow_pickle=False) as contents:
        x = contents["X"]
        names = tuple(str(name) for name in contents["feature_names"].tolist())
        row_ids = contents["row_id"]
        units = contents["unit"]
        if (x.ndim != 2 or x.shape[1] != 83 or len(row_ids) != len(x)
                or len(units) != len(x) or not 0 <= row_index < len(x)):
            raise ValueError("role row identity or shape differs")
        identity = manifest["roles"][role]
        if (len(x) != identity["rows"]
                or len(np.unique(units)) != identity["groups"]
                or _sha_array(np.sort(row_ids)) != identity["row_id_sha256"]
                or _sha_array(np.unique(units)) != identity["group_id_sha256"]
                or _sha_array(np.column_stack((row_ids, units)).astype(np.int64))
                != identity["row_unit_pair_sha256"]):
            raise ValueError("admitted role row/group identity differs")
        row = np.ascontiguousarray(x[row_index:row_index + 1])
        check_raw_schema(row, names, names, pinned_schema_sha256)
        # IDs are returned for private cache binding only; callers must not
        # place them in a public release or public result manifest.
        return row, names, row_ids[row_index].item(), units[row_index].item()
