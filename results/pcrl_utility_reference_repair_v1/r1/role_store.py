"""Exact role-only reader for the repair study; never opens mixed-role archives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


ALLOWED = {
    "OSF_DEFENSE_FIT": {"features", "income", "occupation", "references_s0", "references_s1"},
    "CALIBRATION_HELDOUT": {"features", "income", "occupation", "references_s0", "references_s1"},
    "INNER_SELECTION": {"features", "income", "occupation", "references_s0", "references_s1"},
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(a) -> str:
    x = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(x.dtype).encode() + str(x.shape).encode())
    h.update(x.tobytes())
    return h.hexdigest()


class RoleOnlyStore:
    def __init__(self, export_root: str | Path, manifest_path: str | Path):
        self.root = Path(export_root).resolve(strict=True)
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("schema") != "pcrl-utility-repair-role-admission-v1":
            raise PermissionError("repair role manifest schema differs")

    def read(self, role: str, field: str) -> dict:
        if field not in ALLOWED.get(role, set()):
            raise PermissionError("role/field is not admitted to the repair learner")
        path = self.root / role / f"{field}.npz"
        if path.resolve(strict=True) != path or not path.is_relative_to(self.root):
            raise PermissionError("role path escapes the admitted export root")
        if sha_file(path) != self.manifest["role_file_sha256"][f"{role}:{field}"]:
            raise PermissionError("role-only file hash differs")
        fields = ({"row_id", "unit", "X", "feature_names"} if field == "features"
                  else {"row_id", "unit", field} if field in ("income", "occupation")
                  else {"row_id", "unit", "U0_1", "U0_2", "Ucal_1", "Ucal_2", "d_1", "d_2"})
        with ZipFile(path) as archive:
            if {entry.filename for entry in archive.infolist()} != {name + ".npy" for name in fields}:
                raise PermissionError("role-only member list differs")
        with np.load(path, allow_pickle=False) as z:
            out = {name: np.array(z[name], copy=True) for name in fields}
        meta = self.manifest["roles"][role]
        rid, group = out["row_id"], out["unit"]
        if (rid.dtype != np.int64 or group.dtype != np.int64
                or rid.shape != (meta["rows"],) or group.shape != rid.shape
                or len(np.unique(group)) != meta["groups"]
                or not np.all(np.diff(rid) > 0)):
            raise PermissionError("role-only row/group geometry differs")
        if array_sha(np.column_stack((rid, group))) != meta["row_unit_pair_sha256"]:
            raise PermissionError("role-only row/group pairing differs")
        if field == "features":
            x, names = out["X"], tuple(str(v) for v in out["feature_names"])
            if (x.dtype != np.float32 or x.shape != (len(rid), 83)
                    or not np.isfinite(x).all() or len(names) != 83
                    or hashlib.sha256("\n".join(names).encode()).hexdigest()
                    != self.manifest["source_schema_sha256"]
                    or any(name.lower().split("=")[0] in
                           {"sex", "race", "income", "occupation", "fnlwgt"}
                           for name in names)):
                raise PermissionError("83 predictor columns differ or contain forbidden field")
        elif field in ("income", "occupation"):
            y = out[field]
            cap = 2 if field == "income" else 6
            if (y.dtype != np.int64 or y.shape != rid.shape
                    or np.any((y < 0) | (y >= cap))):
                raise PermissionError("task label schema differs")
        else:
            for i, cap in ((1, 2), (2, 6)):
                d = out[f"d_{i}"]
                for kind in ("U0", "Ucal"):
                    p = out[f"{kind}_{i}"]
                    if (p.dtype != np.float64 or p.shape != (len(rid), cap)
                            or not np.isfinite(p).all() or np.any(p < 0)
                            or np.max(np.abs(p.sum(1) - 1), initial=0) > 1e-8
                            or not np.array_equal(p.argmax(1), d)):
                        raise PermissionError("frozen reference schema or decision differs")
        return out


def assert_same_people(*records: dict) -> None:
    first = records[0]
    if any(not np.array_equal(first[key], other[key])
           for other in records[1:] for key in ("row_id", "unit")):
        raise PermissionError("role-only arrays differ in ordered people")
