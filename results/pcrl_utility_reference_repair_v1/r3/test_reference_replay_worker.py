"""Data-free checks of the independent queued reference reader."""

import hashlib

import numpy as np
import pytest

from results.pcrl_utility_reference_repair_v1.r3.reference_replay_worker import (
    array_sha, role_arrays,
)
from results.pcrl_utility_reference_repair_v1.r3.source_reference import file_sha


def test_role_reader_checks_admission_and_alignment(tmp_path):
    role = "INNER_SELECTION"
    root = tmp_path / "exports"
    folder = root / role
    folder.mkdir(parents=True)
    features = folder / "features.npz"
    references = folder / "references_s0.npz"
    names = np.array([f"f{i}" for i in range(83)])
    row = np.array([11, 12, 13], dtype=np.int64)
    unit = np.array([5, 6, 7], dtype=np.int64)
    x = np.zeros((3, 83), dtype=np.float32)
    np.savez(features, X=x, row_id=row, unit=unit, feature_names=names)
    p2 = np.full((3, 2), 0.5)
    p6 = np.full((3, 6), 1 / 6)
    np.savez(references, row_id=row, unit=unit, U0_1=p2, U0_2=p6,
             Ucal_1=p2, Ucal_2=p6, d_1=np.zeros(3, dtype=np.int64),
             d_2=np.zeros(3, dtype=np.int64))
    manifest = {
        "roles": {role: {"rows": 3, "groups": 3}},
        "source_schema_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "role_file_sha256": {f"{role}:features": file_sha(features),
                             f"{role}:references_s0": file_sha(references)},
    }
    spec = {"inputs": {str(features): file_sha(features),
                       str(references): file_sha(references)}}
    values, refs, identity = role_arrays(root, manifest, role, 0, spec)
    assert np.array_equal(values, x)
    assert identity["row_sha256"] == array_sha(row)
    assert np.array_equal(refs["U0_2"], p6)

    spec["inputs"].pop(str(references))
    with pytest.raises(PermissionError, match="input absent"):
        role_arrays(root, manifest, role, 0, spec)
    spec["inputs"][str(references)] = file_sha(references)
    np.savez(references, row_id=row[::-1], unit=unit, U0_1=p2, U0_2=p6,
             Ucal_1=p2, Ucal_2=p6, d_1=np.zeros(3, dtype=np.int64),
             d_2=np.zeros(3, dtype=np.int64))
    changed = file_sha(references)
    manifest["role_file_sha256"][f"{role}:references_s0"] = changed
    spec["inputs"][str(references)] = changed
    with pytest.raises(PermissionError, match="people or order"):
        role_arrays(root, manifest, role, 0, spec)
