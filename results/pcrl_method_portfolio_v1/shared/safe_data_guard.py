"""Fail-closed access to verified, role-only PCRL exports.

This scientific loader never imports the inherited loaders or the isolated
custodian exporter. It reads only files admitted in DATA_ACCESS_MANIFEST.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


class AccessRefused(PermissionError):
    pass


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_arr(value) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode() + str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


class SafeRoleStore:
    def __init__(self, manifest_path: str | Path | None = None):
        manifest_path = Path(manifest_path or Path(__file__).resolve().parent.parent / "DATA_ACCESS_MANIFEST.json")
        self.manifest_path = manifest_path.resolve()
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("schema") != "pcrl-method-portfolio-data-access-v1":
            raise AccessRefused("unrecognized data-access manifest")

    def _authorize(self, role: str, field: str, procedure: str | None) -> dict:
        if role not in self.manifest["roles"]:
            raise AccessRefused("role is not admitted")
        if field == "features" or field in {f"references_s{k}" for k in (0, 1, 2)}:
            if procedure is not None:
                raise AccessRefused("feature/reference request must not name a label procedure")
        else:
            allowed = self.manifest["label_allowlist"].get(procedure or "", {}).get(role, [])
            if field not in allowed:
                raise AccessRefused("procedure, role and label combination is not admitted")
        key = f"{role}:{field}"
        spec = self.manifest["real_file_specs"].get(key)
        if spec is None:
            raise AccessRefused("no verified role-only export admitted for this request")
        if spec.get("role") != role or spec.get("field") != field:
            raise AccessRefused("file specification has a different role or field")
        return spec

    def read(self, role: str, field: str, procedure: str | None = None) -> dict:
        """Return only the requested role and field; refusal occurs before file access."""
        spec = self._authorize(role, field, procedure)
        declared_path = Path(spec["path"])
        path = (declared_path if declared_path.is_absolute() else self.manifest_path.parent / declared_path).resolve()
        # Exact manifest entries are necessary but not sufficient: deny the known
        # all-row names even if accidentally entered in a proposed manifest.
        if path.name in {"adult_jcv.npz", "teacher.npz", "release.npz", "u.npz"}:
            raise AccessRefused("all-role or all-row archive cannot be admitted")
        if path.suffix != ".npz" or not path.is_file():
            raise AccessRefused("role-only archive is absent or has wrong format")
        if sha_file(path) != spec.get("sha256"):
            raise AccessRefused("role-only archive hash mismatch")
        if field == "features":
            required = {"row_id", "unit", "X", "feature_names"}
        elif field.startswith("references_s"):
            required = {"row_id", "unit", "U0_1", "U0_2", "Ucal_1", "Ucal_2", "d_1", "d_2"}
        else:
            required = {"row_id", "unit", field}
        if set(spec.get("members", [])) != required:
            raise AccessRefused("declared members differ from the exact role-only schema")
        # ZIP directory inspection does not decompress a member. Reject extra
        # arrays before numpy requests any member value.
        with ZipFile(path) as archive:
            actual = {entry.filename for entry in archive.infolist()}
        if actual != {name + ".npy" for name in required}:
            raise AccessRefused("archive contains missing or forbidden members")
        with np.load(path, allow_pickle=False) as archive:
            out = {name: np.array(archive[name], copy=True) for name in required}
        ids = np.asarray(out["row_id"])
        groups = np.asarray(out["unit"])
        expected = self.manifest["roles"][role]
        if ids.dtype != np.int64 or groups.dtype != np.int64:
            raise AccessRefused("row and group ID dtype mismatch")
        if ids.ndim != 1 or groups.shape != ids.shape or ids.size != expected["rows"]:
            raise AccessRefused("role row shape/count mismatch")
        if ids.size > 1 and not np.all(np.diff(ids) > 0):
            raise AccessRefused("row IDs are not in source order")
        if np.unique(groups).size != expected["groups"]:
            raise AccessRefused("role group count mismatch")
        if sha_arr(np.sort(ids)) != expected["row_id_sha256"]:
            raise AccessRefused("role row-ID hash mismatch")
        if sha_arr(np.unique(groups)) != expected["group_id_sha256"]:
            raise AccessRefused("role group-ID hash mismatch")
        pair_hash = sha_arr(np.column_stack((ids, groups)).astype(np.int64))
        if pair_hash != spec.get("row_unit_pair_sha256"):
            raise AccessRefused("role row-to-group pairing mismatch")
        if field == "features":
            x = out["X"]
            names = [str(v) for v in out["feature_names"]]
            if x.shape != (ids.size, self.manifest["source"]["columns"]):
                raise AccessRefused("feature shape mismatch")
            name_hash = hashlib.sha256("\n".join(names).encode()).hexdigest()
            if name_hash != self.manifest["source"]["feature_names_sha256_newline_join"]:
                raise AccessRefused("feature-column identity mismatch")
            if any(name.lower().split("=")[0] in {"sex", "race", "income", "occupation", "fnlwgt"}
                   for name in names):
                raise AccessRefused("forbidden predictor column")
        elif field.startswith("references_s"):
            for recipient, width in ((1, 2), (2, 6)):
                decision = out[f"d_{recipient}"]
                if decision.shape != ids.shape or not np.issubdtype(decision.dtype, np.integer):
                    raise AccessRefused("reference decision schema mismatch")
                for kind in ("U0", "Ucal"):
                    p = out[f"{kind}_{recipient}"]
                    if p.shape != (ids.size, width) or not np.all(np.isfinite(p)) or np.any(p < 0) or \
                            not np.allclose(p.sum(axis=1), 1, atol=1e-8):
                        raise AccessRefused("reference probability schema mismatch")
                    if not np.array_equal(np.argmax(p, axis=1), decision):
                        raise AccessRefused("reference decision mismatch")
        else:
            y = out[field]
            cap = {"income": 2, "occupation": 6, "sex": 2}[field]
            if y.shape != ids.shape or not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= cap)):
                raise AccessRefused("label shape/range mismatch")
        return out


if __name__ == "__main__":
    store = SafeRoleStore()
    print(json.dumps({"status": store.manifest["status"],
                      "admitted_real_files": len(store.manifest["real_file_specs"])}, sort_keys=True))
