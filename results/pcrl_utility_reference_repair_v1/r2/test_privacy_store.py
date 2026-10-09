"""Synthetic stage-specific SEX admission without any Adult file read."""

from __future__ import annotations

import json

import numpy as np
import pytest

from results.pcrl_utility_reference_repair_v1.r1.role_store import array_sha, sha_file
from results.pcrl_utility_reference_repair_v1.r2.privacy_store import PrivacyRoleStore


def _fixture(tmp_path, *, extra_member=False):
    root = tmp_path / "role_export_v1"
    path = root / "OSF_DEFENSE_FIT" / "sex.npz"
    path.parent.mkdir(parents=True)
    row = np.arange(10, 16, dtype=np.int64)
    unit = row.copy()
    values = {"row_id": row, "unit": unit,
              "sex": np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)}
    if extra_member:
        values["occupation"] = np.zeros(len(row), dtype=np.int64)
    np.savez_compressed(path, **values)
    manifest = {
        "schema": "pcrl-utility-repair-role-admission-v1",
        "role_file_sha256": {"OSF_DEFENSE_FIT:sex": sha_file(path)},
        "roles": {"OSF_DEFENSE_FIT": {
            "rows": len(row), "groups": len(row),
            "row_unit_pair_sha256": array_sha(np.column_stack((row, unit))),
        }},
    }
    manifest_path = tmp_path / "DATA_ACCESS_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))
    return root, manifest_path


def test_privacy_role_store_allows_only_the_sealed_stage_pair(tmp_path):
    root, manifest = _fixture(tmp_path)
    store = PrivacyRoleStore(root, manifest, stage="r2_pair_select")
    result = store.read("OSF_DEFENSE_FIT", "sex")
    assert result["sex"].tolist() == [0, 1, 0, 1, 0, 1]
    with pytest.raises(PermissionError, match="outside this sealed privacy stage"):
        store.read("INNER_SELECTION", "sex")
    with pytest.raises(PermissionError, match="outside this sealed privacy stage"):
        store.read("OSF_DEFENSE_FIT", "references_s0")


def test_privacy_role_store_rejects_extra_member_before_array_load(tmp_path):
    root, manifest = _fixture(tmp_path, extra_member=True)
    store = PrivacyRoleStore(root, manifest, stage="r2_pair_select")
    with pytest.raises(PermissionError, match="members differ"):
        store.read("OSF_DEFENSE_FIT", "sex")
