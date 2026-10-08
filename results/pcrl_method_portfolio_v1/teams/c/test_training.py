"""Synthetic numerical checks for Team C's retrained task head."""

import numpy as np
import pytest

from prototype import AdultGroupedSchema
from training import FittedTaskHead, HeadSpec, _network, fit_task_head, restore_head, save_head


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


def test_six_class_inference_meets_s2_strict_simplex_and_keeps_argmax():
    import torch

    torch.manual_seed(71)
    model = _network(1, 6, 64)
    head = FittedTaskHead(
        "occupation_group", "synthetic-simplex-71", ("age",), 1,
        HeadSpec(), {}, model,
    )
    selected = np.linspace(-5, 5, 2_000, dtype=np.float32).reshape(-1, 1)
    probabilities = head.predict_proba(selected)
    assert probabilities.shape == (2_000, 6)
    assert probabilities.dtype == np.float64
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.max(np.abs(probabilities.sum(axis=1) - 1.0)) <= 1e-12
    with torch.no_grad():
        logits = model(torch.as_tensor(selected))
    np.testing.assert_array_equal(
        probabilities.argmax(axis=1), logits.argmax(dim=1).numpy()
    )
