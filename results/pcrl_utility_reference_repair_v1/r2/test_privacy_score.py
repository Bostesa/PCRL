"""Data-free INNER-only protected-label geometry for later frozen scoring."""

import numpy as np
import pytest

from results.pcrl_utility_reference_repair_v1.r2.privacy_score import (
    _inner_labels,
)


def _record():
    row = np.arange(1, 2236, dtype=np.int64)
    unit = np.arange(1, 2236, dtype=np.int64)
    unit[-1] = unit[-2]
    sex = np.arange(2235, dtype=np.int64) % 2
    return {"row_id": row, "unit": unit, "sex": sex}


def test_inner_label_geometry_requires_pinned_people_and_two_classes():
    record = _record()
    y, digest = _inner_labels(record)
    assert len(y) == 2235 and len(digest) == 64
    record["sex"] = np.zeros(2235, dtype=np.int64)
    with pytest.raises(PermissionError, match="INNER SEX"):
        _inner_labels(record)
    record = _record()
    record["unit"] = np.arange(2235, dtype=np.int64)
    with pytest.raises(PermissionError, match="INNER SEX"):
        _inner_labels(record)
