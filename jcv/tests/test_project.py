"""Projection vs an independent QP solver (scipy SLSQP) on random and degenerate vectors."""
import numpy as np
from scipy.optimize import minimize

from jcv.project import project


def qp(p, A):
    cons = [{"type": "ineq", "fun": (lambda d, a=a: -a @ d), "jac": (lambda d, a=a: -a)} for a in A]
    r = minimize(lambda d: 0.5 * np.sum((d - p) ** 2), np.zeros_like(p), jac=lambda d: d - p,
                 constraints=cons, method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    return r.x


def test_random_against_slsqp():
    rng = np.random.default_rng(0)
    for trial in range(400):
        n = int(rng.integers(2, 12))
        m = int(rng.integers(0, 3))
        p = rng.normal(size=n)
        A = rng.normal(size=(m, n))
        d, info = project(p, A)
        assert info["certified"]
        ref = qp(p, A) if m else p
        assert np.allclose(d, ref, atol=1e-6), (trial, d, ref)
        if m:
            assert (A @ d <= 1e-8).all()


def test_degenerate_cases():
    p = np.array([1.0, 2.0, -0.5])
    a = np.array([0.0, 1.0, 0.0])
    # zero guard is vacuous
    d, info = project(p, np.vstack([np.zeros(3), a]))
    assert info["zero_guards"] == 1 and np.allclose(d, [1.0, 0.0, -0.5])
    # parallel guards -> one constraint
    d2, info2 = project(p, np.vstack([a, 3 * a]))
    assert info2["case"] == "parallel" and np.allclose(d2, d)
    # anti-parallel guards -> hyperplane a.d = 0
    d3, info3 = project(p, np.vstack([a, -2 * a]))
    assert info3["case"] == "antiparallel" and abs(a @ d3) < 1e-12 and np.allclose(d3, qp(p, np.vstack([a, -2 * a])), atol=1e-6)
    # already feasible -> unchanged
    d4, _ = project(np.array([1.0, -1.0, 0.0]), a[None])
    assert np.allclose(d4, [1.0, -1.0, 0.0])
    # privacy direction exactly opposing a task gradient is removed entirely
    d5, _ = project(a.copy(), a[None])
    assert np.allclose(d5, 0)


def test_first_order_non_increase():
    rng = np.random.default_rng(1)
    for _ in range(100):
        p = rng.normal(size=8)
        A = rng.normal(size=(2, 8))
        d, _ = project(p, A)
        assert (A @ d <= 1e-9).all()     # task-loss directional derivative grad.d <= 0
