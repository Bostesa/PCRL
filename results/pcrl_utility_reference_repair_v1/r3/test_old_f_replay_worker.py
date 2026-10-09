"""Synthetic alignment and endpoint checks for historical F replay."""

import numpy as np
import pytest

from results.pcrl_utility_reference_repair_v1.r3.old_f_replay_worker import _compare


def _fixture():
    row = np.arange(12, dtype=np.int64)
    yi = row % 2
    yo = row % 6
    old = {
        "row_id": row, "unit": row,
        "income": yi, "occupation": yo,
        "old_F_income": np.full((12, 2), 0.5, dtype=np.float64),
        "old_F_occupation_group": np.full((12, 6), 1 / 6, dtype=np.float64),
        "U0_1": np.full((12, 2), 0.5, dtype=np.float64),
        "U0_2": np.full((12, 6), 1 / 6, dtype=np.float64),
        "Ucal_1": np.full((12, 2), 0.5, dtype=np.float64),
        "Ucal_2": np.full((12, 6), 1 / 6, dtype=np.float64),
    }
    new = {k: value.copy() for k, value in old.items()
           if not k.startswith("old_F_")}
    new["F_income"] = old["old_F_income"].copy()
    new["F_occupation_group"] = old["old_F_occupation_group"].copy()
    role = {k: old[k].copy() for k in ("row_id", "unit", "income", "occupation")}
    return old, new, role


def test_old_new_task_replay_uses_identical_people():
    old, new, role = _fixture()
    result = _compare(old, new, role)
    assert all(abs(value) < 1e-15 for task in result["old_minus_new"].values()
               for value in task.values())
    new["row_id"][0] += 1
    with pytest.raises(PermissionError, match="people or references differ"):
        _compare(old, new, role)


def test_old_new_task_replay_detects_reference_or_label_drift():
    old, new, role = _fixture()
    new["Ucal_1"][0] = [0.25, 0.75]
    with pytest.raises(PermissionError, match="people or references differ"):
        _compare(old, new, role)
    old, new, role = _fixture()
    role["income"][0] = 1 - role["income"][0]
    with pytest.raises(PermissionError, match="INNER role"):
        _compare(old, new, role)
