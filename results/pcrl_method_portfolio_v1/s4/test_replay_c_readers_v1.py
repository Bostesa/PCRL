"""Synthetic, data-free checks for independent C reader receipt replay."""

import numpy as np
import pytest
import hashlib
import joblib

from replay_c_readers_v1 import (fixed_split, oriented_auc, reader_matrix,
                                 replay_model_predictions)


class ToyBinaryModel:
    def predict_proba(self, values):
        x = np.asarray(values, dtype=np.float64)[:, 0]
        p = 1.0 / (1.0 + np.exp(-x))
        return np.column_stack((1.0 - p, p))


def test_oriented_auc_ties_and_reverse_alias():
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    score = np.array([0.2, 0.8, 0.5, 0.5], dtype=np.float64)
    result = oriented_auc(labels, score)
    assert result["positive_alias_auc"] == 0.875
    assert result["reverse_alias_auc"] == 0.125
    assert result["fixed_alias_max_auc"] == 0.875
    with pytest.raises(ValueError, match="binary"):
        oriented_auc(np.array([0, 0, 0, 0]), score)


def test_c_six_reader_matrices_keep_complete_and_raw_views_distinct():
    view = {
        "x": np.ones((2, 3)),
        "x_defense_aware": np.full((2, 4), 2.0),
        "x_raw_only": np.full((2, 1), 3.0),
    }
    assert reader_matrix(view, "LR_C1").shape == (2, 3)
    assert reader_matrix(view, "MLP_64x32").shape == (2, 3)
    assert reader_matrix(view, "HGB_15x200").shape == (2, 3)
    assert np.array_equal(reader_matrix(view, "DA_MLP_64x32"), view["x_defense_aware"])
    assert np.array_equal(reader_matrix(view, "RAW_LR_C1"), view["x_raw_only"])
    assert np.array_equal(reader_matrix(view, "RAW_HGB_15x200"), view["x_raw_only"])
    with pytest.raises(ValueError, match="reader"):
        reader_matrix(view, "CELL_A1")


def test_fixed_split_refuses_wrong_group_count():
    with pytest.raises(ValueError, match="4,061"):
        fixed_split(np.array([1, 1, 2, 2], dtype=np.int64))


def test_hash_bound_reader_model_matches_saved_tune_and_inner(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(ToyBinaryModel(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    att = {"x": np.array([[0.], [1.], [2.]], dtype=np.float64)}
    inner = {"x": np.array([[3.], [4.]], dtype=np.float64)}
    model = ToyBinaryModel()
    expected_tune = model.predict_proba(att["x"][[0, 2]])[:, 1]
    expected_inner = model.predict_proba(inner["x"])[:, 1]
    replay_model_predictions(path, digest, "LR_C1", att, inner,
                             np.array([0, 2]), expected_tune, expected_inner)
    with pytest.raises(ValueError, match="model SHA"):
        replay_model_predictions(path, "0" * 64, "LR_C1", att, inner,
                                 np.array([0, 2]), expected_tune, expected_inner)
    with pytest.raises(ValueError, match="prediction"):
        replay_model_predictions(path, digest, "LR_C1", att, inner,
                                 np.array([0, 2]), expected_tune + 0.01,
                                 expected_inner)
