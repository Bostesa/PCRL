"""Fail-closed reader for the 25 admitted role-only repair exports.

Pass the predecessor repair `m1/DATA_ACCESS_MANIFEST.json` path. Its sibling
package's `.private/role_export_v1` holds the admitted archives. This module
never imports a historical all-role data loader or mixed-array exporter.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


class AccessRefused(PermissionError):
    pass


PROCEDURES = {
    "task_fit": {"OSF_DEFENSE_FIT": {"income", "occupation"}},
    "defense_fit": {"OSF_DEFENSE_FIT": {"income", "occupation", "sex"}},
    "calibration": {"CALIBRATION_HELDOUT": {"income", "occupation"}},
    "attack": {"ATTACK_FIT_NEW": {"sex"}},
    "screen": {"INNER_SELECTION": {"income", "occupation", "sex"}},
}
NONLABEL_FIELDS = {"features", "references_s0", "references_s1"}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode() + str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


class RoleOnlyStore:
    def __init__(self, manifest_path: str | Path):
        path = Path(manifest_path).resolve(strict=True)
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != "pcrl-utility-repair-role-admission-v1":
            raise AccessRefused("unexpected repair role manifest")
        if manifest.get("status") != "ROLE_IDENTITY_VERIFIED_FIT_CONDITIONAL":
            raise AccessRefused("repair role files were not admitted")
        self.manifest_path = path
        self.manifest = manifest
        self.package_root = path.parent.parent
        self.export_root = (self.package_root / ".private/role_export_v1").resolve(strict=True)

    def _authorize(self, role: str, field: str, procedure: str | None) -> Path:
        if role not in self.manifest["roles"]:
            raise AccessRefused("role is not admitted")
        if field in NONLABEL_FIELDS:
            if procedure is not None:
                raise AccessRefused("nonlabel field cannot name a label procedure")
        elif field not in PROCEDURES.get(procedure or "", {}).get(role, set()):
            raise AccessRefused("procedure, role and label are not admitted")
        key = f"{role}:{field}"
        relative = self.manifest["role_file_relative_path"].get(key)
        if relative is None:
            raise AccessRefused("role field has no admitted archive")
        path = (self.package_root / relative).resolve(strict=True)
        if (path.parent != self.export_root / role or path.name != f"{field}.npz"
                or path.suffix != ".npz" or not path.is_file()):
            raise AccessRefused("archive is outside exact role-only location")
        if sha_file(path) != self.manifest["role_file_sha256"][key]:
            raise AccessRefused("role-only archive byte hash mismatch")
        return path

    def read(self, role: str, field: str, procedure: str | None = None) -> dict:
        """Return one role/field archive after exact byte, member and schema checks."""
        path = self._authorize(role, field, procedure)
        required = ({"row_id", "unit", "X", "feature_names"} if field == "features"
                    else {"row_id", "unit", "U0_1", "U0_2", "Ucal_1", "Ucal_2", "d_1", "d_2"}
                    if field.startswith("references_s") else {"row_id", "unit", field})
        with ZipFile(path) as archive:
            if {entry.filename for entry in archive.infolist()} != {name + ".npy" for name in required}:
                raise AccessRefused("archive member set differs")
        with np.load(path, allow_pickle=False) as archive:
            out = {name: np.array(archive[name], copy=True) for name in required}
        ids, units = out["row_id"], out["unit"]
        meta = self.manifest["roles"][role]
        if (ids.dtype != np.int64 or units.dtype != np.int64
                or ids.shape != units.shape or ids.shape != (meta["rows"],)
                or len(np.unique(units)) != meta["groups"]
                or not np.all(np.diff(ids) > 0)
                or sha_array(np.sort(ids)) != meta["row_id_sha256"]
                or sha_array(np.unique(units)) != meta["group_id_sha256"]
                or sha_array(np.column_stack((ids, units))) != meta["row_unit_pair_sha256"]):
            raise AccessRefused("role row/group identity differs")
        if field == "features":
            x = out["X"]
            names = tuple(str(item) for item in out["feature_names"])
            if (x.dtype != np.float32 or x.shape != (len(ids), 83)
                    or not np.isfinite(x).all() or len(set(names)) != 83
                    or hashlib.sha256("\n".join(names).encode()).hexdigest()
                    != self.manifest["source_schema_sha256"]
                    or any(name.lower().split("=")[0] in
                           {"sex", "race", "income", "occupation", "fnlwgt"} for name in names)):
                raise AccessRefused("83 predictor schema differs")
        elif field.startswith("references_s"):
            for recipient, classes in ((1, 2), (2, 6)):
                decision = out[f"d_{recipient}"]
                if decision.dtype != np.int64 or decision.shape != ids.shape:
                    raise AccessRefused("reference decision schema differs")
                for kind in ("U0", "Ucal"):
                    p = out[f"{kind}_{recipient}"]
                    if (p.dtype != np.float64 or p.shape != (len(ids), classes)
                            or not np.isfinite(p).all() or np.any(p < 0)
                            or np.max(np.abs(p.sum(axis=1) - 1), initial=0) > 1e-8
                            or not np.array_equal(np.argmax(p, axis=1), decision)):
                        raise AccessRefused("reference probability schema differs")
        else:
            labels = out[field]
            classes = 6 if field == "occupation" else 2
            if (labels.dtype != np.int64 or labels.shape != ids.shape
                    or np.any((labels < 0) | (labels >= classes))):
                raise AccessRefused("label class order/range differs")
        return out


def assert_same_people(*records: dict) -> None:
    if not records:
        raise ValueError("at least one record required")
    first = records[0]
    if any(not np.array_equal(first[key], record[key])
           for record in records[1:] for key in ("row_id", "unit")):
        raise AccessRefused("ordered role members differ")
