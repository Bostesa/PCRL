"""Mathematical acceptance tests for the Track E projection (PROTOCOL §9).

Written and run BEFORE any ACS outcome. Synthetic data only; no survey row is read.
Each test states the exact claim it checks and nothing broader.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

from experiments.pcrl_competitive_method_v1 import projection as pj

TOL = 1e-9


def _channel(n=4000, d=6, seed=0, degenerate=False):
    rng = np.random.default_rng(seed)
    if degenerate:
        scales = np.array([3.0, 3.0, 3.0, 1.0, 0.5, 0.5])[:d]
    else:
        scales = np.linspace(3.0, 0.3, d)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    z = rng.normal(size=(n, d)) * scales @ q.T + rng.normal(size=d)
    return z


def _policy_projector(z, h, s, classes, k, scale=1.0):
    w = pj.fit_whitening(z)
    v = w.forward(z)
    kept = pj.nonredundant_columns(h)
    basis, _ = pj.polynomial_basis(h, kept)
    p = np.full((len(s), classes), 1.0 / classes)
    e = np.eye(classes)[s] - p
    G = pj.cross_moment(v, basis, e, np.ones(len(s), bool))
    K, _ = pj.role_kernel(G, 1e-12)
    lead = pj.leading_directions(scale * K, k)
    return w, v, pj.projector(lead['U'], w.rank), G


# ------------------------------------------------------------------ algebra
def test_whitening_round_trip_and_identity_at_k0():
    z = _channel()
    w = pj.fit_whitening(z)
    assert w.rank == z.shape[1]
    v = w.forward(z)
    assert np.allclose(np.cov(v.T, bias=True), np.eye(w.rank), atol=1e-8)
    assert np.allclose(pj.release(w, z, np.eye(w.rank)), z, atol=1e-10)


def test_row_convention_matches_column_vector_reference():
    z = _channel()
    rng = np.random.default_rng(3)
    s = rng.integers(0, 2, len(z))
    h = rng.random((len(z), 2))
    w, v, P, _ = _policy_projector(z, h, s, 2, 2)
    out = pj.release(w, z, P)
    # column-vector reference: z_out' = mu' + S' P' R' (z - mu)'
    ref = (w.mu[:, None] + w.S.T @ P.T @ w.R.T @ (z - w.mu).T).T
    assert np.allclose(out, ref, atol=1e-10)


def test_projector_idempotent_and_realised_rank():
    z = _channel()
    rng = np.random.default_rng(4)
    s = rng.integers(0, 3, len(z))
    h = rng.random((len(z), 2))
    for k in (1, 2, 4):
        w, v, P, _ = _policy_projector(z, h, s, 3, k)
        assert np.allclose(P @ P, P, atol=TOL) and np.allclose(P, P.T, atol=TOL)
        out = pj.release(w, z, P)
        centred = out - out.mean(0)
        rank = np.linalg.matrix_rank(centred, tol=1e-8 * np.linalg.norm(centred, 2))
        assert rank == w.rank - k
        # affine mean behaviour: the training mean is preserved exactly
        assert np.allclose(out.mean(0), w.mu, atol=1e-10)
        # applying the release twice changes nothing
        assert np.allclose(pj.release(w, out, P), out, atol=1e-8)


def test_scaled_alias_gives_identical_projector():
    """L2 = 2*K_local is an exact eigenspace alias of L at fixed k."""
    z = _channel()
    rng = np.random.default_rng(5)
    s = rng.integers(0, 2, len(z))
    h = rng.random((len(z), 2))
    for k in (1, 2, 4):
        _, _, P1, _ = _policy_projector(z, h, s, 2, k, 1.0)
        _, _, P2, _ = _policy_projector(z, h, s, 2, k, 2.0)
        assert np.allclose(P1, P2, atol=1e-10)


@pytest.mark.parametrize('degenerate', [False, True])
def test_orthogonal_reparameterisation_equivariance(degenerate):
    """Rotating the input channel rotates the release by the same matrix.

    Holds even with repeated covariance eigenvalues: the whitening's internal basis
    ambiguity cancels between v, P and S.
    """
    z = _channel(degenerate=degenerate)
    rng = np.random.default_rng(6)
    s = rng.integers(0, 2, len(z))
    h = rng.random((len(z), 2))
    O, _ = np.linalg.qr(rng.normal(size=(z.shape[1], z.shape[1])))
    for k in (1, 2):
        w1, _, P1, _ = _policy_projector(z, h, s, 2, k)
        w2, _, P2, _ = _policy_projector(z @ O, h, s, 2, k)
        assert np.allclose(pj.release(w2, z @ O, P2), pj.release(w1, z, P1) @ O, atol=1e-8)


def test_full_span_zeroes_training_moment_partial_does_not():
    z = _channel()
    rng = np.random.default_rng(7)
    h = rng.random((len(z), 2))
    s = (z[:, 0] + z[:, 1] * h[:, 0] + rng.normal(size=len(z)) > 0).astype(int)
    w = pj.fit_whitening(z)
    v = w.forward(z)
    kept = pj.nonredundant_columns(h)
    basis, _ = pj.polynomial_basis(h, kept)
    e = np.eye(2)[s] - np.array([1 - s.mean(), s.mean()])
    G = pj.cross_moment(v, basis, e, np.ones(len(s), bool))
    rank = np.linalg.matrix_rank(G, tol=1e-10 * np.abs(G).max())
    K, _ = pj.role_kernel(G, 1e-12)
    full = pj.leading_directions(K, rank)
    P = pj.projector(full['U'], w.rank)
    assert np.abs(P @ G).max() < 1e-10 * np.abs(G).max()
    partial = pj.projector(pj.leading_directions(K, 1)['U'], w.rank)
    if rank > 1:
        assert np.abs(partial @ G).max() > 1e-3 * np.abs(G).max()


def test_masked_moment_uses_global_centering_and_valid_denominator():
    rng = np.random.default_rng(8)
    v = rng.normal(size=(100, 3)) + 1.0
    basis = np.ones((100, 1))
    e = rng.normal(size=(100, 2))
    valid = np.zeros(100, bool)
    valid[:40] = True
    G = pj.cross_moment(v, basis, e, valid)
    assert G.shape == (3, 2)
    assert np.allclose(G, v[:40].T @ e[:40] / 40)            # no re-centring on the mask


def test_basis_dimensions_binary_services():
    rng = np.random.default_rng(9)
    p = rng.random((500, 3))
    h_a = np.column_stack([1 - p[:, 0], p[:, 0], 1 - p[:, 1], p[:, 1]])
    h_ab = np.column_stack([h_a, 1 - p[:, 2], p[:, 2]])
    ka, kab = pj.nonredundant_columns(h_a), pj.nonredundant_columns(h_ab)
    assert len(ka) == 2 and len(kab) == 3
    assert pj.polynomial_basis(h_a, ka)[0].shape[1] == 6
    assert pj.polynomial_basis(h_ab, kab)[0].shape[1] == 10


# ------------------------------------------------------------------ stationarity theorem
def test_zero_correction_gradient_equals_negative_cross_moment():
    """Fixed-offset multinomial logit with logits l0(h) + Theta . (b(h) kron v).

    Gradient at Theta = 0 is exactly -G computed with e = onehot(s) - softmax(l0(h)).
    The objective is convex in Theta, so G = 0 makes zero correction globally optimal
    IN THIS FAMILY, with THIS offset, on THESE rows. A free per-class intercept is NOT
    in the family; its gradient is -mean(e), which need not vanish.
    """
    rng = np.random.default_rng(10)
    n, r, K = 3000, 3, 3
    h = rng.random((n, 2))
    v = rng.normal(size=(n, r))
    l0 = np.column_stack([np.zeros(n), h[:, 0], -h[:, 1]])
    s = np.array([rng.choice(K, p=softmax(l0[i] + 0.8 * v[i, 0] * np.eye(K)[0]))
                  for i in range(n)])
    basis, _ = pj.polynomial_basis(h, [0, 1])
    e = np.eye(K)[s] - softmax(l0, axis=1)
    X = pj.kron_rows(basis, np.ones((n, 1)))                 # b(h)

    def features(vv):
        return (X[:, :, None] * vv[:, None, :]).reshape(n, -1)      # b kron v

    def objective(theta, F):
        W = theta.reshape(F.shape[1], K)
        logits = l0 + F @ W
        loss = -log_softmax(logits, axis=1)[np.arange(n), s].mean()
        grad = F.T @ (softmax(logits, axis=1) - np.eye(K)[s]) / n
        return loss, grad.ravel()

    F = features(v)
    _, grad0 = objective(np.zeros(F.shape[1] * K), F)
    G = (F.T @ e) / n
    assert np.allclose(grad0, -G.ravel(), atol=1e-12)

    # after full-span removal of the declared moments the gradient vanishes
    Gv = pj.cross_moment(v, basis, e, np.ones(n, bool))
    rank = np.linalg.matrix_rank(Gv, tol=1e-10 * np.abs(Gv).max())
    P = pj.projector(pj.leading_directions(Gv @ Gv.T, rank)['U'], r)
    F2 = features(v @ P)
    loss0, g2 = objective(np.zeros(F2.shape[1] * K), F2)
    assert np.abs(g2).max() < 1e-10
    fitted = minimize(objective, np.zeros(F2.shape[1] * K), args=(F2,), jac=True,
                      method='L-BFGS-B')
    assert fitted.fun >= loss0 - 1e-9                          # zero correction is optimal
    # the intercept caveat: mean residual is not zero in general
    assert np.abs(e.mean(0)).max() > 1e-4


# ------------------------------------------------------------------ counterexamples
def test_counterexample_information_survives_vanishing_moment():
    """s depends on |v|; every linear-in-v moment vanishes; a quadratic probe recovers s."""
    rng = np.random.default_rng(11)
    n = 6000
    v = rng.normal(size=(n, 1))
    s = (np.abs(v[:, 0]) > 0.674).astype(int)
    e = np.eye(2)[s] - np.array([1 - s.mean(), s.mean()])
    G = pj.cross_moment(v, np.ones((n, 1)), e, np.ones(n, bool))
    assert np.abs(G).max() < 0.03
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    base = log_loss(s, np.tile([1 - s.mean(), s.mean()], (n, 1)))
    probe = LogisticRegression().fit(v ** 2, s)
    assert log_loss(s, probe.predict_proba(v ** 2)) < base - 0.2


def test_synthetic_interaction_local_misses_joint_detects():
    """Sign of the z-s relation flips with H_B. Local (H_A-only) moments ~0; joint ones do not."""
    rng = np.random.default_rng(12)
    n = 20000
    hA = rng.random((n, 2))
    hB = rng.integers(0, 2, n).astype(float)
    z = rng.normal(size=(n, 4))
    sign = 2 * hB - 1
    s = (sign * z[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    w = pj.fit_whitening(z)
    v = w.forward(z)
    e = np.eye(2)[s] - np.array([1 - s.mean(), s.mean()])
    bA, _ = pj.polynomial_basis(hA, pj.nonredundant_columns(hA))
    hAB = np.column_stack([hA, hB])
    bAB, _ = pj.polynomial_basis(hAB, pj.nonredundant_columns(hAB))
    GA = pj.cross_moment(v, bA, e, np.ones(n, bool))
    GAB = pj.cross_moment(v, bAB, e, np.ones(n, bool))
    assert np.linalg.norm(GAB) > 10 * np.linalg.norm(GA)
    lead = pj.leading_directions(GAB @ GAB.T, 1)['U'][:, 0]
    # the joint-view leading direction aligns with z_0
    corr = abs(np.corrcoef(v @ lead, z[:, 0])[0, 1])
    assert corr > 0.95


def test_correlated_authorised_and_sensitive_protection_costs_utility():
    """One direction predicts both a source task and s; removing it costs source capability."""
    rng = np.random.default_rng(13)
    n = 8000
    z = rng.normal(size=(n, 4))
    s = (z[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    y = (z[:, 0] + z[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(int)
    h = rng.random((n, 2))
    w, v, P, _ = _policy_projector(z, h, s, 2, 1)
    out = pj.release(w, z, P)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    before = log_loss(y, LogisticRegression().fit(z, y).predict_proba(z))
    after = log_loss(y, LogisticRegression().fit(out, y).predict_proba(out))
    s_before = log_loss(s, LogisticRegression().fit(z, s).predict_proba(z))
    s_after = log_loss(s, LogisticRegression().fit(out, s).predict_proba(out))
    assert s_after > s_before + 0.3 and after > before + 0.05


def test_controls_rank_and_random_determinism():
    z = _channel()
    pca = pj.pca_retaining(z, 4)
    assert np.linalg.matrix_rank(pca['projector']) == 4
    r1, r2 = pj.random_subspace(6, 4, 99), pj.random_subspace(6, 4, 99)
    assert np.array_equal(r1['projector'], r2['projector'])
    assert np.allclose(r1['projector'] @ r1['projector'], r1['projector'], atol=1e-12)


def test_basis_redundancy_survives_float32_rounding():
    """The ACS service wires are float32 probabilities cast to float64: pairs sum to 1
    only to ~1e-7. The redundancy rule must still find one coordinate per binary task."""
    rng = np.random.default_rng(14)
    p = rng.random((3000, 3)).astype(np.float32)
    h_a = np.column_stack([1 - p[:, 0], p[:, 0], 1 - p[:, 1], p[:, 1]]).astype(np.float32)
    h_ab = np.column_stack([h_a, 1 - p[:, 2], p[:, 2]]).astype(np.float32).astype(np.float64)
    h_a = h_a.astype(np.float64)
    assert len(pj.nonredundant_columns(h_a)) == 2
    assert len(pj.nonredundant_columns(h_ab)) == 3
