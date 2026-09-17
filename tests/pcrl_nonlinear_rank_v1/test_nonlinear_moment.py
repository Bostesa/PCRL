"""Falsification fixtures for the nonlinear moment refinement.

These establish that the code measures the stated finite objective. They do NOT
certify privacy. No fixture parameter is tuned to make a measure look successful:
each fixture states in advance what a correct implementation must show, including
the two fixtures whose whole point is that the penalty can be WRONG
(conditional-null concentration and nuisance misspecification).
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.pcrl_nonlinear_rank_v1.nonlinear_moment import (BANDS, RFF_PER_BAND, RFF_TOTAL,
                                                                 ChiFeatures, ProtectedRole,
                                                                 build_features, quadratic_pairs,
                                                                 role_moments,
                                                                 role_moment_gradient)
from experiments.pcrl_nonlinear_rank_v1.objective import (PolicyObjective, RolePenalty, qr_retract,
                                                          spectral_solution, stiefel_descent,
                                                          tangent_project)

RNG = np.random.default_rng(20260917)


def whiten(x):
    x = x - x.mean(0)
    cov = x.T @ x / len(x)
    values, vectors = np.linalg.eigh((cov + cov.T) / 2)
    keep = values > max(1e-12, 1e-10 * values[-1])
    return x @ (vectors[:, keep] / np.sqrt(values[keep]))


def make_role(name, view, basis, residual, gram_dim, n_valid=None, classes=None):
    classes = classes if classes is not None else residual.shape[1]
    return ProtectedRole(name=name, view=view, basis=basis, residual=residual,
                         linear_gram=np.zeros((gram_dim, gram_dim)),
                         n_valid=n_valid if n_valid is not None else len(basis),
                         classes=classes, support=np.full(classes, len(basis) // classes),
                         unsupported=[], raw_trace=1.0, zero_trace=False)


# --------------------------------------------------------------- 1. feature family
def test_feature_family_shape_and_frozen_scaling():
    z = RNG.normal(size=(500, 5))
    chi = ChiFeatures.fit(z, seed=0)
    assert chi.n_quadratic == len(quadratic_pairs(5)) == 15
    assert chi.width == 15 + RFF_TOTAL == 111
    assert chi.omega_scaled.shape == (5, RFF_TOTAL)
    assert len(BANDS) * RFF_PER_BAND == RFF_TOTAL
    # three distinct bandwidths, ratios exactly 0.5 : 1 : 2 on the drawn directions
    norms = [np.linalg.norm(chi.omega_scaled[:, i * 32:(i + 1) * 32], axis=0).mean() for i in range(3)]
    assert norms[0] / norms[1] == pytest.approx(1 / 0.5, rel=.25)
    assert norms[2] / norms[1] == pytest.approx(1 / 2.0, rel=.25)
    # the frozen scaling does not move when a DIFFERENT projection is transformed
    other = RNG.normal(size=(300, 5)) * 7.0
    before = chi.mean.copy(), chi.scale.copy()
    chi.transform(other)
    assert np.array_equal(chi.mean, before[0]) and np.array_equal(chi.scale, before[1])


def test_bandwidth_failure_is_explicit_not_silent():
    with pytest.raises(ValueError, match='bandwidth diagnostic failure'):
        ChiFeatures.fit(np.ones((20, 3)), seed=0)


def test_chunk_size_invariance_of_moments_and_gradient():
    """Fixture 8: two evaluations, fixed objects, different chunk sizes."""
    v = whiten(RNG.normal(size=(700, 9)))
    w = qr_retract(RNG.normal(size=(v.shape[1], 4)))
    chi = ChiFeatures.fit(v @ w, seed=1)
    basis = np.column_stack((np.ones(len(v)), RNG.uniform(size=(len(v), 2))))
    residual = RNG.normal(size=(len(v), 3)) * .3
    role = make_role('A/x', 'A', basis, residual, v.shape[1])
    scale = np.ones(chi.width)
    small = build_features(chi, v, w, chunk=64)
    large = build_features(chi, v, w, chunk=4096)
    assert np.max(abs(small.chi - large.chi)) < 1e-12
    a = role_moments(role, small, None)
    b = role_moments(role, large, None)
    assert np.max(abs(a - b)) < 1e-12
    va, ga, _ = role_moment_gradient(role, chi, small, None, scale)
    vb, gb, _ = role_moment_gradient(role, chi, large, None, scale)
    assert abs(va - vb) < 1e-12 and np.max(abs(ga - gb)) < 1e-12


# --------------------------------------------------------------- 2. magnitude fixture
def test_magnitude_fixture_first_moment_zero_quadratic_detects():
    """(Z,S) equally likely (-1,0),(1,0),(-2,1),(2,1): |Z| reveals S, first moment does not."""
    z = np.array([-1., 1., -2., 2.])
    s = np.array([0, 0, 1, 1])
    onehot = np.eye(2)[s]
    prior = onehot.mean(0)
    e = onehot - prior                                   # marginal residual, constant basis
    first_moment = (z[:, None] * e).mean(0)
    assert np.max(abs(first_moment)) < 1e-15             # the old penalty sees nothing

    quadratic = (z ** 2)[:, None]
    quadratic_moment = ((quadratic - quadratic.mean(0)) * e).mean(0)
    assert np.max(abs(quadratic_moment)) > .4            # the quadratic block sees it clearly
    # and thresholding |Z| recovers S perfectly
    assert np.mean((abs(z) > 1.5) == s) == 1.0


def test_magnitude_fixture_through_the_real_feature_family():
    n = 4000
    pick = RNG.integers(0, 4, n)
    z = np.array([-1., 1., -2., 2.])[pick]
    s = np.array([0, 0, 1, 1])[pick]
    v = whiten(np.column_stack((z, RNG.normal(size=(n, 3)))))
    w = np.zeros((v.shape[1], 1))
    # a direction that reproduces z up to scale
    w[:, 0] = np.linalg.lstsq(v, z, rcond=None)[0]
    w /= np.linalg.norm(w)
    chi = ChiFeatures.fit(v @ w, seed=2)
    onehot = np.eye(2)[s]
    e = onehot - onehot.mean(0)
    basis = np.ones((n, 1))
    role = make_role('A/S', 'A', basis, e, v.shape[1])
    m = role_moments(role, build_features(chi, v, w, chunk=512), None)
    linear_like = abs((v @ w * e).mean(0)).max()
    quadratic_block = np.sqrt(np.sum(m[:chi.n_quadratic] ** 2))
    assert linear_like < .02                             # first moment is ~0
    assert quadratic_block > 10 * linear_like            # the new penalty is not blind


# --------------------------------------------------------------- 3. XOR fixture
def test_xor_fixture_marginal_blind_interaction_detects():
    """S,H independent fair signs, Z = S*H: marginal moment 0, H-interaction moment 1."""
    s = np.array([-1., -1., 1., 1.])
    h = np.array([-1., 1., -1., 1.])
    z = s * h
    assert np.mean(z * s) == 0.0                         # marginal protection misses it
    assert abs(np.mean(z * s * h)) == 1.0                # H interaction detects it
    assert np.array_equal(z * h, s)                      # S is exactly recoverable


def test_xor_fixture_requires_the_intercept_and_interaction_basis():
    n = 4000
    s = RNG.choice([-1., 1.], n)
    h = RNG.choice([-1., 1.], n)
    z = s * h
    v = whiten(np.column_stack((z, RNG.normal(size=(n, 3)))))
    w = np.linalg.lstsq(v, z, rcond=None)[0].reshape(-1, 1)
    w /= np.linalg.norm(w)
    label = (s > 0).astype(int)
    e = np.eye(2)[label] - np.eye(2)[label].mean(0)
    chi = ChiFeatures.fit(v @ w, seed=3)
    constant = make_role('A/S', 'A', np.ones((n, 1)), e, v.shape[1])
    interacted = make_role('A/S', 'A', np.column_stack((np.ones(n), h)), e, v.shape[1])
    bundle = build_features(chi, v, w, chunk=512)
    m_const = role_moments(constant, bundle, None)
    m_inter = role_moments(interacted, bundle, None)
    # the constant basis sees almost nothing; adding the H column exposes the disclosure
    assert np.sqrt(np.sum(m_const ** 2)) < .05
    assert np.sqrt(np.sum(m_inter[:, 1, :] ** 2)) > .3


# --------------------------------------------------------------- 4. conditional null
def test_conditional_null_moments_concentrate_but_need_not_be_exactly_zero():
    """S and Z share only H; with oracle conditionals the POPULATION moments vanish."""
    def moment_norm(n):
        h = RNG.uniform(-1, 1, n)
        p = 1 / (1 + np.exp(-2 * h))                     # oracle conditional of S given H
        s = (RNG.uniform(size=n) < p).astype(int)
        z = h + RNG.normal(scale=.5, size=n)             # independent residual given H
        v = whiten(np.column_stack((z, RNG.normal(size=(n, 2)))))
        w = np.linalg.lstsq(v, z, rcond=None)[0].reshape(-1, 1)
        w /= np.linalg.norm(w)
        chi = ChiFeatures.fit(v @ w, seed=4)
        onehot = np.eye(2)[s]
        oracle = np.column_stack((1 - p, p))             # ORACLE nuisance, not a fitted one
        role = make_role('A/S', 'A', np.column_stack((np.ones(n), h, h ** 2)),
                         onehot - oracle, v.shape[1])
        return np.sqrt(np.sum(role_moments(role, build_features(chi, v, w, 1024), None) ** 2))

    small, large = moment_norm(2000), moment_norm(32000)
    assert large < small                                  # concentrating towards zero
    assert small > 0.0                                    # finite-sample moments are NOT exactly 0
    assert large < .05


# --------------------------------------------------------------- 5. nuisance error
def test_misspecified_nuisance_produces_moments_without_incremental_leakage():
    """A wrong m(H) makes the penalty fire even though Z adds nothing beyond H."""
    n = 20000
    h = RNG.uniform(-1, 1, n)
    p = 1 / (1 + np.exp(-3 * h))
    s = (RNG.uniform(size=n) < p).astype(int)
    z = h + RNG.normal(scale=.5, size=n)                  # Z ⊥ S | H by construction
    v = whiten(np.column_stack((z, RNG.normal(size=(n, 2)))))
    w = np.linalg.lstsq(v, z, rcond=None)[0].reshape(-1, 1)
    w /= np.linalg.norm(w)
    chi = ChiFeatures.fit(v @ w, seed=5)
    onehot = np.eye(2)[s]
    basis = np.column_stack((np.ones(n), h, h ** 2))
    oracle = make_role('A/S', 'A', basis, onehot - np.column_stack((1 - p, p)), v.shape[1])
    prior = onehot.mean(0)
    wrong = make_role('A/S', 'A', basis, onehot - prior, v.shape[1])   # deliberately misspecified
    bundle = build_features(chi, v, w, chunk=2048)
    oracle_norm = np.sqrt(np.sum(role_moments(oracle, bundle, None) ** 2))
    wrong_norm = np.sqrt(np.sum(role_moments(wrong, bundle, None) ** 2))
    # Same data, same channel, no incremental leakage: only the nuisance changed.
    assert wrong_norm > 10 * oracle_norm
    # Therefore a nonzero penalty is NOT evidence of incremental disclosure.


# --------------------------------------------------------------- 6. dimension fixture
def test_fixed_rank_optimum_versus_variable_rank_optimum():
    values = np.array([3., 2., 1., -.5, -2.])
    basis = qr_retract(RNG.normal(size=(5, 5)))
    a = basis @ np.diag(values) @ basis.T
    for r in (0, 1, 3, 5):
        w, spectrum = spectral_solution(a, r)
        assert w.shape == (5, r)
        assert float(np.trace(w.T @ a @ w)) == pytest.approx(values[np.argsort(-values)][:r].sum(), abs=1e-10)
        if r:
            assert np.max(abs(w.T @ w - np.eye(r))) < 1e-10
    # Fixed rank 5 is FORCED to take the two negative directions; the variable-rank
    # optimum stops at 3. These are different statements about the same matrix.
    w5, _ = spectral_solution(a, 5)
    w3, _ = spectral_solution(a, 3)
    assert float(np.trace(w5.T @ a @ w5)) < float(np.trace(w3.T @ a @ w3))
    assert int((values > max(1e-12, 1e-10 * abs(values).max())).sum()) == 3
    # zero rank is a valid collapse, not an error
    w0, _ = spectral_solution(a, 0)
    assert w0.shape == (5, 0) and float(np.trace(w0.T @ a @ w0)) == 0.0


def test_repeated_eigenvalues_give_an_optimal_but_not_unique_basis():
    a = np.diag([2., 2., 1.])
    w, _ = spectral_solution(a, 2)
    assert float(np.trace(w.T @ a @ w)) == pytest.approx(4.0, abs=1e-12)
    rotated = w @ qr_retract(RNG.normal(size=(2, 2)))
    assert float(np.trace(rotated.T @ a @ rotated)) == pytest.approx(4.0, abs=1e-12)
    # equal objective, different coordinates: signs do not resolve the rotation


# --------------------------------------------------------------- 7. gradient / retraction
def _toy_objective(seed=11, n=600, q=7, r=3):
    v = whiten(RNG.normal(size=(n, q)))
    q = v.shape[1]
    basis = np.column_stack((np.ones(n), RNG.uniform(size=(n, 2))))
    labels = RNG.integers(0, 3, n)
    onehot = np.eye(3)[labels]
    residual = onehot - onehot.mean(0)
    gram = np.eye(q) * .01
    role = ProtectedRole(name='A/SEX', view='A', basis=basis, residual=residual, linear_gram=gram,
                         n_valid=n, classes=3, support=np.bincount(labels, minlength=3),
                         unsupported=[], raw_trace=1., zero_trace=False)
    other = ProtectedRole(**{**role.__dict__, 'name': 'AB/SEX', 'view': 'AB'})
    w_ref = qr_retract(RNG.normal(size=(q, r)))
    chi = ChiFeatures.fit(v @ w_ref, seed=seed)
    p_local = RolePenalty.build(role, chi, v, None, w_ref, chunk=256)
    p_ab = RolePenalty.build(other, chi, v, None, w_ref, chunk=256)
    cov = RNG.normal(size=(q, q))
    utility = cov @ cov.T
    utility /= np.trace(utility)
    obj = PolicyObjective(utility=utility, local={'A/SEX': p_local}, coalition={'AB/SEX': p_ab},
                          policy='C1', family='nonlinear', features=v, chunk=256)
    return obj, w_ref


def test_analytic_gradient_matches_central_finite_differences():
    obj, w = _toy_objective()
    loss, grad, _ = obj.loss_and_grad(w)
    eps = 1e-6
    errors = []
    for _ in range(12):
        direction = RNG.normal(size=w.shape)
        direction /= np.linalg.norm(direction)
        numeric = (obj.loss(w + eps * direction) - obj.loss(w - eps * direction)) / (2 * eps)
        analytic = float(np.sum(grad * direction))
        errors.append(abs(numeric - analytic) / max(1.0, abs(numeric)))
    assert max(errors) < 2e-5, max(errors)


def test_nonlinear_term_actually_depends_on_W():
    """Guards against a penalty that is secretly constant in W (a silent no-op)."""
    obj, w = _toy_objective()
    other = qr_retract(w + .5 * RNG.normal(size=w.shape))
    _, _, parts_a = obj.loss_and_grad(w)
    _, _, parts_b = obj.loss_and_grad(other)
    a = parts_a['A/SEX']['nonlinear_normalized']
    b = parts_b['A/SEX']['nonlinear_normalized']
    assert abs(a - b) > 1e-6
    assert a == pytest.approx(1.0, abs=1e-9)      # equals 1 at the reference by construction


def test_retraction_and_tangent_projection_are_correct():
    for shape in ((7, 3), (5, 5), (9, 1)):
        m = RNG.normal(size=shape)
        q = qr_retract(m)
        assert np.max(abs(q.T @ q - np.eye(shape[1]))) < 1e-12
        # retraction of an already-orthonormal matrix is that matrix (positive-R convention)
        assert np.max(abs(qr_retract(q) - q)) < 1e-12
        g = RNG.normal(size=shape)
        t = tangent_project(q, g)
        sym = q.T @ t + t.T @ q
        assert np.max(abs(sym)) < 1e-12           # tangency: W'T + T'W = 0


def test_descent_decreases_the_objective_and_stays_feasible():
    obj, w = _toy_objective()
    result = stiefel_descent(obj, w, max_updates=25)
    assert result['loss'] <= result['initial_loss'] + 1e-15
    assert result['feasibility_max_abs'] < 1e-10
    assert result['stop_reason'] in ('budget_exhausted', 'line_search_failed_no_improvement',
                                     'riemannian_gradient_below_tolerance')
    accepted = [h for h in result['history'][1:] if h['accepted']]
    for a, b in zip(accepted, accepted[1:]):
        assert b['loss'] <= a['loss'] + 1e-15     # monotone on accepted steps


def test_original_family_closed_form_beats_or_matches_descent():
    obj, w = _toy_objective()
    linear = PolicyObjective(utility=obj.utility, local=obj.local, coalition=obj.coalition,
                             policy='C1', family='original', features=obj.features, chunk=256)
    closed, values = spectral_solution(linear.spectral_matrix(), w.shape[1])
    descent = stiefel_descent(linear, w, max_updates=200)
    assert linear.loss(closed) <= descent['loss'] + 1e-9
    assert -linear.loss(closed) == pytest.approx(values[:w.shape[1]].sum(), abs=1e-10)


def test_utility_term_is_covariance_normalised_so_collapse_cannot_pay():
    """Any orthonormal W gives Z'Z/n = I: no rescaling of Z is available to the optimiser."""
    v = whiten(RNG.normal(size=(800, 6)))
    for r in (1, 3, 6):
        w = qr_retract(RNG.normal(size=(v.shape[1], r)))
        z = v @ w
        assert np.max(abs(z.T @ z / len(z) - np.eye(r))) < 1e-9


# --------------------------------------------------------------- 9. rotation invariance
def test_nonlinear_penalty_is_not_a_trace_form():
    """Terminal B finding A2, verified numerically.

    A trace form ``tr(W' A W)`` with W-independent A is invariant under W -> W Q
    for orthogonal Q. Utility and the original linear penalty are invariant; the
    nonlinear penalty is not. So no W-independent matrix represents it and no
    closed-form eigensolution exists.
    """
    from experiments.pcrl_nonlinear_rank_v1.diagnostics import rotation_invariance_check
    obj, w = _toy_objective()
    check = rotation_invariance_check(obj, w)
    assert check['utility_change_abs'] < 1e-12
    assert check['original_penalty_change_abs'] < 1e-12
    assert check['nonlinear_penalty_change_abs'] > 1e-6


def test_released_channel_information_is_rotation_invariant():
    """Z and ZQ are related by an invertible map, so no attacker can distinguish them.

    This is what makes the rotation share a measure of surrogate movement rather
    than of disclosure reduction: a logistic attacker's achievable log loss is
    identical on Z and ZQ, because the rotation can be absorbed into its weights.
    """
    from sklearn.linear_model import LogisticRegression
    v = whiten(RNG.normal(size=(1500, 8)))
    w = qr_retract(RNG.normal(size=(v.shape[1], 4)))
    q = qr_retract(RNG.normal(size=(4, 4)))
    z = v @ w
    labels = (z[:, 0] + .5 * z[:, 1] ** 2 + RNG.normal(scale=.5, size=len(z)) > 0).astype(int)
    losses = []
    for features in (z, z @ q):
        model = LogisticRegression(C=1., max_iter=5000).fit(features, labels)
        p = np.clip(model.predict_proba(features), 1e-12, 1.)
        losses.append(float(-np.mean(np.log(p[np.arange(len(labels)), labels]))))
    assert abs(losses[0] - losses[1]) < 1e-6


def test_rotation_share_is_between_zero_and_one_on_a_real_descent():
    from experiments.pcrl_nonlinear_rank_v1.diagnostics import rotation_share
    obj, w = _toy_objective()
    refined = stiefel_descent(obj, w, max_updates=40)
    share = rotation_share(obj, w, refined['W'], max_updates=40)
    assert share['applicable']
    assert share['total_training_gain'] >= -1e-12
    assert share['rotation_only_training_gain'] >= -1e-12
    # rotation is a strict subset of the search space, so it cannot beat the free search
    assert share['loss_at_best_rotation_of_original_subspace'] >= share['loss_at_nonlinear_solution'] - 1e-9
    # The projector distance is the well-conditioned measure of "same subspace";
    # arccos of a singular value near 1 amplifies 1e-16 into ~1e-8, so the angle
    # gets the looser tolerance.
    assert share['subspace_original_vs_rotation']['projector_frobenius_distance'] < 1e-10
    assert share['subspace_original_vs_rotation']['max_principal_angle_rad'] < 1e-6
    assert abs(share['utility_at_best_rotation'] - share['utility_at_original']) < 1e-10


# --------------------------------------------------------------- 10. why the slack exists
# The rotation diagnostic found that most of the refinement's training-objective gain
# is reachable by rotation alone. These tests localise the cause, which turns out to be
# a specification error in the feature family rather than anything about ACS data.
#
# Disclosure from the released channel is invariant under Z -> Z Q (Q orthogonal),
# because Z and ZQ determine each other. A penalty meant to measure that disclosure
# should therefore be invariant too. The quadratic block as specified is not, for an
# exactly identifiable reason: it sums squared moments over the monomials a <= b with
# equal weight, while the squared Frobenius norm of the symmetric moment matrix
# M_ab = E[z_a z_b g] counts each off-diagonal entry twice. Since M -> Q' M Q under
# rotation, only the Frobenius-weighted sum is preserved.

def _quadratic_block(z, g, offdiag_weight, pairs):
    total = 0.0
    for a, b in pairs:
        m = float(np.mean(z[:, a] * z[:, b] * g))
        total += (1.0 if a == b else offdiag_weight) * m * m
    return total


def test_equal_weighted_quadratic_block_is_not_rotation_invariant():
    """This is the shipped convention, and it leaves slack a rotation can harvest."""
    z = RNG.normal(size=(4000, 5))
    g = RNG.normal(size=4000)
    q = qr_retract(RNG.normal(size=(5, 5)))
    pairs = quadratic_pairs(5)
    before = _quadratic_block(z, g, 1.0, pairs)
    after = _quadratic_block(z @ q, g, 1.0, pairs)
    assert abs(after / before - 1.0) > .02          # a rotation moves it materially


def test_frobenius_weighted_quadratic_block_is_exactly_rotation_invariant():
    """Weighting off-diagonal monomials by 2 makes the block ||M||_F^2, which M -> Q'MQ preserves."""
    z = RNG.normal(size=(4000, 5))
    g = RNG.normal(size=4000)
    q = qr_retract(RNG.normal(size=(5, 5)))
    pairs = quadratic_pairs(5)
    before = _quadratic_block(z, g, 2.0, pairs)
    after = _quadratic_block(z @ q, g, 2.0, pairs)
    assert abs(after - before) < 1e-12 * max(1.0, before)
    # and it equals the squared Frobenius norm of the full symmetric moment matrix
    m = np.einsum('na,nb,n->ab', z, z, g) / len(z)
    assert before == pytest.approx(float(np.sum(m * m)), rel=1e-12)


def test_fourier_block_slack_shrinks_as_features_are_added():
    """The RFF block's slack is Monte-Carlo error, not a specification error.

    omega ~ N(0, I) is rotationally symmetric in distribution, so the feature SET is
    distributionally invariant and the block converges to a rotation-invariant limit;
    at a finite feature count it is only approximately invariant.
    """
    z = RNG.normal(size=(6000, 5))
    g = RNG.normal(size=6000)
    q = qr_retract(RNG.normal(size=(5, 5)))

    def slack(m, seed):
        rng = np.random.default_rng(seed)
        omega = rng.normal(size=(5, m))
        phase = rng.uniform(0, 2 * np.pi, m)
        def block(zz):
            f = np.sqrt(2 / m) * np.cos(zz @ omega + phase)
            return float(np.sum((f.T @ g / len(zz)) ** 2))
        return abs(block(z @ q) / block(z) - 1.0)

    few = float(np.mean([slack(16, 100 + i) for i in range(12)]))
    many = float(np.mean([slack(1024, 200 + i) for i in range(12)]))
    assert many < few
