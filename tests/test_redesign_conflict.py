"""Population/setup and honest held-out scoring regressions."""
import numpy as np
import torch
from concept_erasure import LeaceEraser
from experiments.run_redesign_conflict import (
    CONFIG, composition_stress, fit_affine, make_data, mixing, score,
)


def test_fixed_mixing_is_invertible_and_splits_are_distinct():
    matrix, _ = mixing()
    assert np.linalg.matrix_rank(matrix) == 8
    data, seeds, _ = make_data(0)
    assert len(set(seeds.values())) == 4
    assert np.allclose(data["representation_fit"][0].mean(0), 0, atol=1e-14)
    assert not np.allclose(data["test"][0].mean(0), 0, atol=1e-8)


def test_affine_intercept_uses_fitting_data_and_negative_scores_survive():
    w, b = fit_affine(np.zeros((10, 2)), np.full(10, 3.))
    prediction = np.zeros((4, 2)) @ w + b
    assert np.all(prediction == 3.)
    assert score(np.array([8., 9., 10., 11.]), prediction)["r2"] < 0
    assert score(np.ones(4), prediction)["r2"] is None


def test_continuous_leace_separates_conflicting_purposes():
    data, _, _ = make_data(91)
    x, y = data["representation_fit"]
    xt, yt = data["test"]
    shared = LeaceEraser.fit(torch.from_numpy(x), torch.from_numpy(y), **CONFIG["leace"])
    for p, prohibited in enumerate(((1, 2), (0, 2))):
        own = LeaceEraser.fit(torch.from_numpy(x), torch.from_numpy(y[:, prohibited]), **CONFIG["leace"])
        for eraser, preserves in ((own, True), (shared, False)):
            h, ht = eraser(torch.from_numpy(x)).numpy(), eraser(torch.from_numpy(xt)).numpy()
            w, b = fit_affine(h, y[:, p])
            r2 = score(yt[:, p], ht @ w + b)["r2"]
            assert r2 > .95 if preserves else r2 < .05
        hc = own(torch.from_numpy(x)).numpy()
        cov = (hc-hc.mean(0)).T @ (y[:, prohibited]-y[:, prohibited].mean(0)) / len(x)
        assert np.max(np.abs(cov)) < 1e-10


def test_composition_population_formula_and_exact_recovery():
    result = composition_stress()
    for row in result["rows"]:
        delta = row["delta"]
        assert np.isclose(row["population_single_r2"], delta**2/(1+delta**2))
        assert np.isclose(row["population_combined_r2"], float(delta != 0))
        if delta:
            assert row["predictive_scores_views_then_concat"][2]["r2"] > 1-1e-12


def test_composition_exact_independent_unit_variance_sample():
    # All four equally likely outcomes of independent Rademacher N and S
    # realize the population covariance exactly, independently of the formula
    # implementation used by the Gaussian Monte Carlo stress test.
    n = np.array([-1., -1., 1., 1.])
    s = np.array([-1., 1., -1., 1.])
    for delta in (0., .01, .1, 1.):
        h = np.column_stack((n + delta*s, n - delta*s))
        for columns in ([0], [1], [0, 1]):
            w, b = fit_affine(h[:, columns], s)
            r2 = score(s, h[:, columns] @ w + b)["r2"]
            expected = float(delta != 0) if len(columns) == 2 else delta**2/(1+delta**2)
            assert np.isclose(r2, expected, atol=1e-12)
