"""Additive pre-run checks for alternating and deterministic Team A arms."""

import numpy as np
import pytest

from finite_channel import FiniteLaw, evaluate, fit_alternating, solve_coordinate
from test_finite_channel import _common_law, _identity


def _loose_caps(law, q1, q2):
    baseline = evaluate(law, q1, q2)["utility"]
    return {r: {m: baseline[r][m] + 1.0 for m in ("nll", "brier")}
            for r in ("r1", "r2")}


def test_rejects_mass_row_and_bank_normalization_errors():
    law = _common_law("L1")
    bad_mass = law.mass.copy()
    bad_mass[0, 0, 0, 0, 0] += 1e-6
    with pytest.raises(ValueError):
        FiniteLaw(bad_mass, *law.bank, *law.input_class)
    bad_bank = law.bank[0].copy()
    bad_bank[0, 0] += 1e-6
    with pytest.raises(ValueError):
        FiniteLaw(law.mass, bad_bank, law.bank[1], *law.input_class)
    q1, q2 = _identity(law)
    q1[0, 0] += 1e-6
    with pytest.raises(ValueError):
        evaluate(law, q1, q2)


def test_deterministic_coordinate_has_one_hot_rows():
    law = _common_law("L2")
    q1, q2 = _identity(law)
    fit = solve_coordinate(law, q1, q2, recipient="r1",
                           caps=_loose_caps(law, q1, q2),
                           deterministic=True, time_limit=5)
    assert fit["success"]
    assert np.all((fit["q"] == 0) | (fit["q"] == 1))
    assert np.array_equal(fit["q"].sum(axis=1), np.ones(law.n_input[0]))


def test_alternating_receipt_replays_each_coordinate():
    law = _common_law("L2")
    q1, q2 = _identity(law)
    fit = fit_alternating(law, q1, q2, caps=_loose_caps(law, q1, q2),
                          max_rounds=2, time_limit_per_step=5)
    assert fit["success"]
    assert len(fit["history"]) >= 2
    for step in fit["history"]:
        assert step["success"]
        assert step["max_residual"] <= 1e-7
        assert abs(step["t"] - step["replayed_t"]) <= 1e-7
    result = evaluate(law, fit["q1"], fit["q2"])
    assert result["accuracy"]["pair"] >= max(
        result["accuracy"]["r1"], result["accuracy"]["r2"]) - 1e-10
