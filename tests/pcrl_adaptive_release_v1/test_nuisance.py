"""Frozen nuisance fitting stays on its globally assigned household role."""
import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import nuisance, roles
from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs


def _households(role, n):
    found = []
    for value in range(10000):
        candidate = f"fixture_{value}"
        if roles.role_of(candidate) == role:
            found.append(candidate)
            if len(found) == n:
                return np.asarray(found)
    raise AssertionError("fixture role search failed")


def _rows(role="nuisance_train"):
    n = 12
    x = np.zeros((n, 32))
    x[:, 0] = np.linspace(-1, 1, n)
    ha = np.zeros((n, 4))
    ha[:, 0] = np.linspace(0, 1, n)
    return {"x": x, "ha": ha, "households": _households(role, n),
            "weights": np.ones(n),
            "labels": {"SEX": np.arange(n) % 2,
                       "RAC1P": np.arange(n) % 3}}


def test_fitted_predictor_preserves_missing_full_schema_class_and_predicts_without_labels():
    model, receipt = nuisance.fit_frozen_nuisance(_rows(), seed=7)
    assert receipt["training_role"] == "nuisance_train"
    assert receipt["class_support"]["RAC1P"][3] == 0
    pred = model.probabilities(RuntimeInputs(np.zeros((2, 32)), np.zeros((2, 4))))
    assert pred["SEX"].shape == (2, 2)
    assert pred["RAC1P"].shape == (2, 9)
    assert np.all(pred["RAC1P"][:, 3] == pytest.approx(nuisance.PROBABILITY_FLOOR))
    assert np.allclose(pred["RAC1P"].sum(axis=1), 1)
    assert not hasattr(model, "labels")


def test_nuisance_fit_rejects_wrong_role_and_does_not_drop_unsupported_classes():
    with pytest.raises(ValueError, match="nuisance_train"):
        nuisance.fit_frozen_nuisance(_rows("coefficient_split"), seed=7)
    rows = _rows()
    rows["labels"]["RAC1P"][0] = 9
    with pytest.raises(ValueError, match="full RAC1P schema"):
        nuisance.fit_frozen_nuisance(rows, seed=7)


def test_frozen_nuisance_private_receipt_reloads_without_refitting(tmp_path):
    model, receipt = nuisance.fit_frozen_nuisance(_rows(), seed=7)
    nuisance.save_frozen_nuisance(tmp_path / "private" / "nuisance", model, receipt)
    restored, loaded = nuisance.load_frozen_nuisance(tmp_path / "private" / "nuisance")
    assert loaded["training_sha256"] == receipt["training_sha256"]
    local = RuntimeInputs(np.zeros((2, 32)), np.zeros((2, 4)))
    for target in ("SEX", "RAC1P"):
        assert np.array_equal(restored.probabilities(local)[target],
                              model.probabilities(local)[target])
    assert nuisance.save_frozen_nuisance(tmp_path / "private" / "nuisance",
                                         model, receipt) == loaded
