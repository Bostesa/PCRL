"""Required controlled fixtures for the score decomposition (synthetic; diagnostics, not scientific results)."""
import os

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
from odx.surfaces import decompose, exactness, softmax, surfaces  # noqa: E402


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


def _fit_auc(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    n = len(y) // 2
    m = HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(X[:n], y[:n])
    return _auc(y[n:], m.predict_proba(X[n:])[:, 1])


def test_offset_carries_sensitive_bit_margin_carries_task():
    rng = np.random.default_rng(0)
    n = 6000
    s, t = rng.integers(0, 2, n), rng.integers(0, 2, n)
    d = 2.0 * (2 * t - 1) + rng.normal(size=n)          # margin: task only
    c = 1.5 * s + rng.normal(scale=0.3, size=n)         # offset: sensitive only
    L = np.stack([c - d / 2, c + d / 2], 1)
    sf = surfaces(L)
    assert _fit_auc(sf["full"], s) > 0.9 and _fit_auc(sf["offset"], s) > 0.9
    assert abs(_fit_auc(sf["centred"], s) - 0.5) < 0.05 and abs(_fit_auc(sf["prob"], s) - 0.5) < 0.05
    assert np.allclose(sf["prob"], softmax(sf["dc"][:, :1] * np.array([[-0.5, 0.5]])))


def test_margin_carries_all_sensitive_signal_offset_independent():
    rng = np.random.default_rng(1)
    n = 6000
    s = rng.integers(0, 2, n)
    d = 1.5 * (2 * s - 1) + rng.normal(size=n)
    c = rng.normal(size=n)
    L = np.stack([c - d / 2, c + d / 2], 1)
    sf = surfaces(L)
    assert abs(_fit_auc(sf["offset"], s) - 0.5) < 0.05
    assert abs(_fit_auc(sf["full"], s) - _fit_auc(sf["centred"], s)) < 0.03


def test_constant_and_near_constant_head():
    L = np.tile([[0.3, -0.2]], (500, 1))
    sf = surfaces(L)
    assert np.all(sf["hard"][:, 0] == 1) and np.ptp(sf["prob"][:, 1]) == 0
    e = exactness(L)
    assert e["offset_sd"] < 1e-15 and e["argmax_equals_margin_positive"]


def test_exact_invertible_recoding_reproduces_predictions():
    rng = np.random.default_rng(2)
    L = rng.normal(size=(1000, 2)) * 3
    D = decompose(L)
    d, c = D["margin"][:, 0], D["offset"][:, 0]
    back = np.stack([c - d / 2, c + d / 2], 1)
    assert np.max(np.abs(back - L)) < 1e-12
    # an attacker on (d, c) composed with the linear map equals the same attacker on full logits
    w = np.array([0.7, -1.3])
    A = np.array([[-1.0, 1.0], [0.5, 0.5]])          # (l0,l1) -> (d,c)
    assert np.allclose((L @ A.T) @ w, L @ (A.T @ w))
    assert np.array_equal(L.argmax(1), (d > 0).astype(int)) or np.any(d == 0)


def test_multiclass_centering():
    rng = np.random.default_rng(3)
    L = rng.normal(size=(800, 6)) + rng.normal(size=(800, 1)) * 5
    D = decompose(L)
    assert np.allclose(D["centred_vec"].sum(1), 0, atol=1e-12)
    assert np.max(np.abs(softmax(D["centred_vec"]) - softmax(L))) < 1e-14
    assert np.max(np.abs(D["centred_vec"] + D["offset"] - L)) < 1e-12
    sf = surfaces(L)
    assert sf["centred"].shape[1] == 6 and np.array_equal(sf["hard"].argmax(1), L.argmax(1))


def test_saturated_probabilities_documented():
    L = np.array([[0.0, 40.0], [0.0, -40.0], [0.0, 1.0]])
    e = exactness(L)
    assert e["p1_saturated_exact_0_or_1"] >= 1          # sigma(40) == 1.0 in float64
    P = softmax(L)
    assert P[0, 1] == 1.0 and 0 < P[2, 1] < 1           # margin 40 not recoverable from p; margin 1 is


def test_family_counts():
    from odx import family as F
    assert len(F.PRIMARY) == 30 and F.S3_SIZE == 34 and F.S4_SIZE == 6 and F.S5_SIZE == 4
    assert abs(F.Z_PRIMARY - 3.14398) < 1e-4      # Phi^-1(1 - 0.05/60)
