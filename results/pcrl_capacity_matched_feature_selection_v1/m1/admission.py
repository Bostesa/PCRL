"""Metadata and identity-only admission of the prior verified role exports.

The caller supplies the prior portfolio package path. This module never opens
the inherited mixed-role source archive, old assessment labels, or HEAD labels.
It reads only ZIP directories and admitted role-only row/unit/name members;
scientific workers still need their own effective seal and queue admission.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

try:
    from .capacity import SOURCE_SCHEMA_SHA256, feature_name_hash, source_groups
except ImportError:  # direct synthetic test invocation from this directory
    from capacity import SOURCE_SCHEMA_SHA256, feature_name_hash, source_groups


EXPECTED_MANIFEST_SHA256 = "c228dfdbef8a8e0a46281716e89c986e888fb4f64553fc35b3573020c0c83a11"
EXPECTED_S4_SHA256 = "5d455806634fbe3046d5d6487f9b9c411c600344f72e1374c1d66f3c19ebd9e9"
EXPECTED_S4_EXPORT_REPLAY_SHA256 = "314fd7dd86ee815d084abf95581f8c6d7711e58110c188e32e09408c6f5dd8e0"
EXPECTED_HCAL_ROLE_MANIFEST_SHA256 = "0994e84d426f2de87f7fe3ad2b70b49b19504bfd209419d308c17337f57f36f0"
EXPECTED_SOURCE_SHA256 = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
ROLE_ORDER = (
    "OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT", "ATTACK_FIT_NEW", "INNER_SELECTION"
)
EXPECTED_COUNTS = {
    "OSF_DEFENSE_FIT": (15434, 15428),
    "CALIBRATION_HELDOUT": (2000, 2000),
    "ATTACK_FIT_NEW": (4064, 4061),
    "INNER_SELECTION": (2235, 2234),
}
BANNED_BASENAMES = {"adult_jcv.npz", "teacher.npz", "release.npz", "u.npz"}


def sha_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_arr(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    h = sha256()
    h.update(str(arr.dtype).encode() + str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _bounded_role_file(package: Path, relative_path: str) -> Path:
    declared = Path(relative_path)
    if declared.is_absolute() or ".." in declared.parts:
        raise ValueError("role export path is not package relative")
    path = (package / declared).resolve()
    if not path.is_relative_to(package) or path.name in BANNED_BASENAMES:
        raise ValueError("role export path escapes the admitted package")
    if path.suffix != ".npz" or not path.is_file():
        raise ValueError("admitted role export is missing or not NPZ")
    return path


def _identity(path: Path, expected_members: set[str]) -> tuple[np.ndarray, np.ndarray,
                                                                 tuple[str, ...] | None]:
    with ZipFile(path) as archive:
        actual = {entry.filename for entry in archive.infolist()}
    if actual != {name + ".npy" for name in expected_members}:
        raise ValueError("role-only ZIP member schema changed")
    with np.load(path, allow_pickle=False) as archive:
        row_id = np.asarray(archive["row_id"])
        unit = np.asarray(archive["unit"])
        names = tuple(str(name) for name in archive["feature_names"]) if "feature_names" in expected_members else None
    if row_id.dtype != np.int64 or unit.dtype != np.int64 or row_id.ndim != 1 or unit.shape != row_id.shape:
        raise ValueError("role identity dtype or shape changed")
    if row_id.size > 1 and not np.all(np.diff(row_id) > 0):
        raise ValueError("role rows are not in ascending source order")
    return row_id, unit, names


def admit_existing_exports(package_path: str | Path) -> dict:
    """Return public hashes/counts only; never return IDs or private arrays."""
    package = Path(package_path).resolve()
    manifest_path = package / "DATA_ACCESS_MANIFEST.json"
    s4_path = package / "S4_ROLE_EXPORT_VERIFICATION.json"
    replay_path = package / "S4_EXPORT_REPLAY.json"
    hcal_role_path = package.parent / "pcrl_heldout_calibration_v1/ROLE_MANIFEST.json"
    if sha_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("portfolio role manifest bytes differ from admitted identity")
    if sha_file(s4_path) != EXPECTED_S4_SHA256:
        raise ValueError("S4 role audit bytes differ from admitted identity")
    if sha_file(replay_path) != EXPECTED_S4_EXPORT_REPLAY_SHA256:
        raise ValueError("S4 deterministic export replay bytes differ")
    if sha_file(hcal_role_path) != EXPECTED_HCAL_ROLE_MANIFEST_SHA256:
        raise ValueError("hcal role definition bytes differ")
    manifest = json.loads(manifest_path.read_text())
    s4 = json.loads(s4_path.read_text())
    replay = json.loads(replay_path.read_text())
    if manifest.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED" or s4.get("status") != "PASS":
        raise ValueError("role exports lack S4 admission")
    if manifest["source"]["source_npz_sha256"] != EXPECTED_SOURCE_SHA256:
        raise ValueError("source NPZ identity differs")
    if manifest["source"]["feature_names_sha256_newline_join"] != SOURCE_SCHEMA_SHA256:
        raise ValueError("83-column feature identity differs")
    if manifest["custodian_export"]["s4_report_sha256"] != EXPECTED_S4_SHA256:
        raise ValueError("custodian receipt does not bind S4 role audit")
    if manifest["custodian_export"]["role_manifest_sha256"] != EXPECTED_HCAL_ROLE_MANIFEST_SHA256:
        raise ValueError("custodian receipt does not bind hcal role definition")
    if s4["source_sha256"] != EXPECTED_SOURCE_SHA256 or s4["export_npz_count"] != 25:
        raise ValueError("S4 report source or file count differs")
    if s4["barred_id_overlap"] != 0 or s4["export_group_overlap"] != 0:
        raise ValueError("S4 found excluded-person or inter-role overlap")
    if replay.get("status") != "PASS" or replay.get("first_second_byte_identical") is not True:
        raise ValueError("independent deterministic export replay did not pass")
    if replay.get("role_npz_count") != 25 or replay.get("receipt_sha256") != s4["export_receipt_sha256"]:
        raise ValueError("export replay receipt differs")
    specs = manifest["real_file_specs"]
    if len(specs) != 25:
        raise ValueError("admitted role-only file count differs")
    role_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    file_hashes: dict[str, str] = {}
    relative_paths: dict[str, str] = {}
    checked_names: tuple[str, ...] | None = None
    for key, spec in sorted(specs.items()):
        role, field = key.split(":", 1)
        if role not in ROLE_ORDER or spec["role"] != role or spec["field"] != field:
            raise ValueError("role export key/role/field mismatch")
        path = _bounded_role_file(package, spec["path"])
        actual_sha = sha_file(path)
        if actual_sha != spec["sha256"]:
            raise ValueError("role-only file byte hash mismatch")
        members = set(spec["members"])
        ids, units, names = _identity(path, members)
        expected = manifest["roles"][role]
        if (ids.size, np.unique(units).size) != EXPECTED_COUNTS[role]:
            raise ValueError("role row/group count mismatch")
        if ids.size != expected["rows"] or np.unique(units).size != expected["groups"]:
            raise ValueError("role count differs from admitted manifest")
        if sha_arr(np.sort(ids)) != expected["row_id_sha256"]:
            raise ValueError("role row-ID hash differs")
        if sha_arr(np.unique(units)) != expected["group_id_sha256"]:
            raise ValueError("role group-ID hash differs")
        pair_sha = sha_arr(np.column_stack((ids, units)).astype(np.int64))
        if pair_sha != spec["row_unit_pair_sha256"] or pair_sha != s4["roles"][role]["row_unit_pair_sha256"]:
            raise ValueError("role row-to-group assignment differs")
        if role in role_data:
            prior_ids, prior_units = role_data[role]
            if not np.array_equal(ids, prior_ids) or not np.array_equal(units, prior_units):
                raise ValueError("role files have inconsistent row/group order")
        else:
            role_data[role] = ids, units
        if field == "features":
            if members != {"row_id", "unit", "X", "feature_names"} or names is None:
                raise ValueError("feature export member schema differs")
            source_groups(names)
            if checked_names is not None and names != checked_names:
                raise ValueError("role feature-name order differs")
            checked_names = names
        file_hashes[key] = actual_sha
        relative_paths[key] = spec["path"]
    overlaps = {}
    for i, left in enumerate(ROLE_ORDER):
        for right in ROLE_ORDER[i + 1:]:
            l_ids, l_units = role_data[left]
            r_ids, r_units = role_data[right]
            overlaps[left + "__" + right] = {
                "row_overlap": int(np.intersect1d(l_ids, r_ids).size),
                "group_overlap": int(np.intersect1d(l_units, r_units).size),
            }
    if any(v["row_overlap"] or v["group_overlap"] for v in overlaps.values()):
        raise ValueError("admitted development roles overlap by row or group")
    return {
        "schema": "pcrl-focused-role-admission-v1",
        "status": "METADATA_AND_ROLE_IDENTITY_PASS",
        "source_npz_sha256": EXPECTED_SOURCE_SHA256,
        "portfolio_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "s4_role_audit_sha256": EXPECTED_S4_SHA256,
        "s4_export_replay_sha256": sha_file(replay_path),
        "role_manifest_sha256": manifest["custodian_export"]["role_manifest_sha256"],
        "source_schema_sha256": SOURCE_SCHEMA_SHA256,
        "role_file_count": len(file_hashes),
        "role_file_sha256": file_hashes,
        "role_file_relative_path": relative_paths,
        "roles": {role: {
            "rows": EXPECTED_COUNTS[role][0], "groups": EXPECTED_COUNTS[role][1],
            "row_id_sha256": manifest["roles"][role]["row_id_sha256"],
            "group_id_sha256": manifest["roles"][role]["group_id_sha256"],
            "row_unit_pair_sha256": s4["roles"][role]["row_unit_pair_sha256"],
        } for role in ROLE_ORDER},
        "overlap": overlaps,
        "excluded_id_overlap_from_independent_s4_audit": s4["barred_id_overlap"],
        "scope": "25 role-only files: ZIP, byte hash and row/unit/name members; no X/task/SEX values opened",
        "scientific_fit_permission": False,
    }
