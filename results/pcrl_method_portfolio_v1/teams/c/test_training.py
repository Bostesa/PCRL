"""Synthetic numerical checks for Team C's retrained task head."""

import numpy as np
import pytest

from prototype import AdultGroupedSchema
from training import HeadSpec, fit_task_head, restore_head, save_head


def occupation_rows():
    groups = np.arange(6 * 80, dtype=np.int64)
    unit = np.repeat(groups, 2)
    y = np.repeat(groups % 6, 2)
    X = np.zeros((len(unit), 83), dtype=np.float32)
    X[:, 4] = y / 5
    return X, y, unit


def test_six_class_head_refits_on_all_allowed_rows_and_restores(tmp_path):
    schema = AdultGroupedSchema()
    X, y, unit = occupation_rows()
    head = fit_task_head(
        X, y, unit, schema.feature_names, ("hours-per-week",),
        purpose="occupation_group", model_id="synthetic-occ-19", seed=19,
        spec=HeadSpec(),
    )
    assert head.receipt["head_subfits"] == 2
    assert head.receipt["early_stop_fit_rows"] + head.receipt["early_stop_validation_rows"] == len(X)
    assert head.receipt["full_refit_rows"] == len(X)
    assert 1 <= head.receipt["best_epoch"] <= 30
    selected = X[:12, (4,)]
    probabilities = head.predict_proba(selected)
    assert probabilities.shape == (12, 6)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)
    checkpoint = tmp_path / "private" / "head.npz"
    sha = save_head(head, checkpoint, private_root=tmp_path / "private")
    restored = restore_head(
        checkpoint, expected_sha256=sha, private_root=tmp_path / "private"
    )
    np.testing.assert_allclose(restored.predict_proba(selected), probabilities, atol=1e-7)
    with pytest.raises(FileExistsError):
        save_head(head, checkpoint, private_root=tmp_path / "private")
    assert checkpoint.exists()


def test_head_refuses_task_label_as_extra_predictor_before_fit():
    schema = AdultGroupedSchema()
    X, y, unit = occupation_rows()
    with pytest.raises(ValueError, match="ordered 83-column schema"):
        fit_task_head(
            np.column_stack((X, y)), y, unit,
            (*schema.feature_names, "occupation_group"),
            ("hours-per-week",), purpose="occupation_group",
            model_id="should-refuse", seed=19, spec=HeadSpec(),
        )
