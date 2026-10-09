"""Role-only protected-label reader admitted only by a sealed privacy stage.

Worker queue admission and the prospective lead/bank branch happen before
constructing this store. Each stage has a fixed, minimal role-field set.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from results.pcrl_utility_reference_repair_v1.r1.role_store import (
    RoleOnlyStore, array_sha, sha_file,
)


ALLOWED_BY_STAGE = {
    "r2_pair_select": frozenset({
        ("OSF_DEFENSE_FIT", "features"), ("OSF_DEFENSE_FIT", "income"),
        ("OSF_DEFENSE_FIT", "occupation"), ("OSF_DEFENSE_FIT", "sex"),
    }),
    "r2_release": frozenset({
        ("ATTACK_FIT_NEW", "features"), ("INNER_SELECTION", "features"),
    }),
    "r2_raw_union": frozenset({
        ("ATTACK_FIT_NEW", "features"), ("ATTACK_FIT_NEW", "sex"),
        ("INNER_SELECTION", "features"),
    }),
    "r2_attack_fit": frozenset({
        ("ATTACK_FIT_NEW", "features"), ("ATTACK_FIT_NEW", "sex"),
        ("INNER_SELECTION", "features"),
    }),
    "r2_privacy_score": frozenset({
        ("INNER_SELECTION", "sex"),
    }),
}


class PrivacyRoleStore(RoleOnlyStore):
    def __init__(self, export_root: str | Path, manifest_path: str | Path,
                 *, stage: str):
        if stage not in ALLOWED_BY_STAGE:
            raise PermissionError("unknown protected-label access stage")
        super().__init__(export_root, manifest_path)
        self.stage = stage

    def read(self, role: str, field: str) -> dict:
        if (role, field) not in ALLOWED_BY_STAGE[self.stage]:
            raise PermissionError("role/label is outside this sealed privacy stage")
        if role == "ATTACK_FIT_NEW" and field == "features":
            return self._read_att_features()
        if field != "sex":
            return super().read(role, field)
        # The inherited task store intentionally refuses SEX; its exact
        # role-only archive, byte and person checks are reproduced here.
        path = self.root / role / "sex.npz"
        if (path.resolve(strict=True) != path
                or not path.is_relative_to(self.root)
                or sha_file(path) != self.manifest["role_file_sha256"][f"{role}:sex"]):
            raise PermissionError("protected role-only file identity differs")
        fields = {"row_id", "unit", "sex"}
        with ZipFile(path) as archive:
            if {item.filename for item in archive.infolist()} != {
                    name + ".npy" for name in fields}:
                raise PermissionError("protected role-only members differ")
        with np.load(path, allow_pickle=False) as archive:
            out = {name: np.array(archive[name], copy=True) for name in fields}
        meta = self.manifest["roles"][role]
        row, unit, sex = out["row_id"], out["unit"], out["sex"]
        if (row.dtype != np.int64 or unit.dtype != np.int64
                or sex.dtype != np.int64
                or row.shape != (meta["rows"],)
                or unit.shape != row.shape or sex.shape != row.shape
                or len(np.unique(unit)) != meta["groups"]
                or not np.all(np.diff(row) > 0)
                or not np.all((sex == 0) | (sex == 1))
                or array_sha(np.column_stack((row, unit))) !=
                meta["row_unit_pair_sha256"]):
            raise PermissionError("protected role-only label/person geometry differs")
        return out

    def _read_att_features(self) -> dict:
        """Admit ATT predictors only for the stage-allowed privacy worker."""
        role, field = "ATTACK_FIT_NEW", "features"
        path = self.root / role / "features.npz"
        if (path.resolve(strict=True) != path
                or not path.is_relative_to(self.root)
                or sha_file(path) != self.manifest["role_file_sha256"][f"{role}:{field}"]):
            raise PermissionError("ATT feature role-only file identity differs")
        fields = {"row_id", "unit", "X", "feature_names"}
        with ZipFile(path) as archive:
            if {item.filename for item in archive.infolist()} != {
                    name + ".npy" for name in fields}:
                raise PermissionError("ATT feature role-only members differ")
        with np.load(path, allow_pickle=False) as archive:
            out = {name: np.array(archive[name], copy=True) for name in fields}
        meta = self.manifest["roles"][role]
        row, unit = out["row_id"], out["unit"]
        if (row.dtype != np.int64 or unit.dtype != np.int64
                or row.shape != (meta["rows"],) or unit.shape != row.shape
                or len(np.unique(unit)) != meta["groups"]
                or not np.all(np.diff(row) > 0)
                or array_sha(np.column_stack((row, unit))) !=
                meta["row_unit_pair_sha256"]):
            raise PermissionError("ATT feature role-only person geometry differs")
        x, names = out["X"], tuple(str(value) for value in out["feature_names"])
        if (x.dtype != np.float32 or x.shape != (len(row), 83)
                or not np.isfinite(x).all() or len(names) != 83
                or hashlib.sha256("\n".join(names).encode()).hexdigest() !=
                self.manifest["source_schema_sha256"]
                or any(name.lower().split("=")[0] in
                       {"sex", "race", "income", "occupation", "fnlwgt"}
                       for name in names)):
            raise PermissionError("ATT 83 predictors differ or contain forbidden field")
        return out
