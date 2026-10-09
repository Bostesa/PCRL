"""Data-free checks for the independent complete-release matrix contract."""

import numpy as np
import pytest

from results.pcrl_utility_reference_repair_v1.r3.release_replay_worker import (
    _blocks, _compare_float, _schema_hash, _view_hashes,
)
from results.pcrl_utility_reference_repair_v1.r3.task_replay import array_sha


def test_local_complete_view_includes_values_probabilities_and_decision():
    selected = np.array([[2.0], [4.0]], dtype=np.float32)
    probability = np.array([[.8, .2], [.3, .7]], dtype=np.float64)
    decision = np.array([0, 1], dtype=np.int64)
    raw, x = _blocks(selected, probability, decision, 2)
    expected = np.array([[2.0, .8, .2, 1.0, 0.0],
                         [4.0, .3, .7, 0.0, 1.0]])
    assert np.array_equal(x, expected)
    constants = {"feature_names": ["x"], "mask_id": "m"}
    schema = _schema_hash(("selected_values", "probabilities", "decisions"),
                          ("feature_names", "mask_id"), constants,
                          (raw, probability, np.eye(2)[decision]))
    aware = np.column_stack((x, probability))
    hashes = _view_hashes(x, aware, raw, schema)
    assert hashes["x"] == array_sha(expected)
    assert hashes["x_raw_only"] == array_sha(raw)
    assert hashes["x_defense_aware"] == array_sha(aware)
    assert hashes["wire_schema"] == schema


def test_independent_probability_replay_rejects_material_change():
    original = np.array([[.8, .2]], dtype=np.float64)
    near = original + np.array([[1e-8, -1e-8]])
    assert _compare_float(near, original, "probability") < 2e-7
    with pytest.raises(PermissionError, match="probability"):
        _compare_float(np.array([[.7, .3]]), original, "probability")
