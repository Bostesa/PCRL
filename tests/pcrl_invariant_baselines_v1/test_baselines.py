"""Verification fixtures for the three external baseline adaptations.

These check that each implementation computes the published object it claims to, and
that the properties each paper actually guarantees hold on a fixture. They do **not**
establish that any baseline protects anything on ACS.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

RNG = np.random.default_rng(20260930)


# =================================================================== LEACE
def test_generalised_leace_reproduces_the_verified_two_attribute_wrapper():
    """The schema is generalised to three attributes; the RECIPE must be unchanged.

    ``experiments/acs_protection_maps.py:fit_joint_leace`` is the repository's verified,
    regression-tested LEACE path but whitelists exactly SEX and RAC1P. This study needs
    coverage as well, so the recipe is generalised -- and must reproduce the verified
    wrapper bitwise-to-tolerance on the two-attribute case it covers.
    """
    from experiments.acs_protection_maps import fit_joint_leace
    from experiments.pcrl_invariant_baselines_v1 import erasure_baselines as eb

    n, d = 4000, 14
    x = RNG.normal(size=(n, d))
    sex = RNG.integers(0, 2, n)
    race = RNG.integers(0, 9, n)
    x += np.eye(2)[sex] @ RNG.normal(size=(2, d)) * 0.7
    x += np.eye(9)[race] @ RNG.normal(size=(9, d)) * 0.4

    verified = fit_joint_leace(x, {'SEX': sex, 'RAC1P': race})
    reference_projection = np.asarray(verified.matrix)

    original_schema = eb.PROTECTED_SCHEMA
    try:
        eb.PROTECTED_SCHEMA = {'SEX': 2, 'RAC1P': 9}
        z, complete, _ = eb.joint_onehot({'SEX': sex, 'RAC1P': race}, n)
        mine = eb._stabilized_leace(x[complete], z)
    finally:
        eb.PROTECTED_SCHEMA = original_schema

    assert np.allclose(mine['projection'], reference_projection, rtol=2e-11, atol=2e-11)
    assert np.allclose(mine['mean'], np.asarray(verified.mean), rtol=2e-11, atol=2e-11)
    # 14 coordinates minus a centred SEX(2)+RAC1P(9) joint one-hot of rank 9 leaves 5,
    # which is the rank the verified wrapper's own regression test asserts.
    assert mine['projection_retained_rank'] == 5


def test_leace_erases_linear_information_and_costs_exactly_the_predicted_rank():
    """Rank loss is sum(k-1) over the joint one-hot; cross-covariance goes to zero."""
    from experiments.pcrl_invariant_baselines_v1 import erasure_baselines as eb

    n, d = 6000, 16
    x = RNG.normal(size=(n, d))
    labels = {'SEX': RNG.integers(0, 2, n), 'RAC1P': RNG.integers(0, 9, n),
              'public_coverage': RNG.integers(0, 2, n)}
    for name, size in eb.PROTECTED_SCHEMA.items():
        x += np.eye(size)[labels[name]] @ RNG.normal(size=(size, d)) * 0.6

    z, complete, _ = eb.joint_onehot(labels, n)
    fitted = eb._stabilized_leace(x[complete], z)
    released = eb.apply_affine(x[complete], fitted['projection'], fitted['mean'])

    expected_loss = sum(k - 1 for k in eb.PROTECTED_SCHEMA.values())      # 1 + 8 + 1 = 10
    assert fitted['projection_retained_rank'] == d - expected_loss == 6
    before = eb.cross_covariance_max_abs(x[complete], z)
    after = eb.cross_covariance_max_abs(released, z)
    assert after < 1e-10 * max(1.0, before), (before, after)
    # mean preserved exactly, and P is idempotent
    assert np.max(abs(released.mean(0) - x[complete].mean(0))) < 1e-9
    p = fitted['projection']
    assert np.max(abs(p @ p - p)) < 1e-9


def test_leace_guarantee_is_linear_only_a_nonlinear_probe_still_recovers():
    """The guarantee is about AFFINE predictors. A quadratic feature still recovers.

    Declared in advance: this is what LEACE's own scoping says, and it is why the study
    scores the erasure baselines against the same nonlinear attacker slate as every
    other arm rather than against a linear probe.
    """
    from experiments.pcrl_invariant_baselines_v1 import erasure_baselines as eb

    n, d = 6000, 8
    s = RNG.integers(0, 2, n)
    base = RNG.normal(size=(n, d))
    base[:, 0] = np.where(s == 1, RNG.normal(1.5, 0.3, n), RNG.normal(-1.5, 0.3, n))
    base[:, 1] = np.where(s == 1, RNG.normal(0, 2.0, n), RNG.normal(0, 0.5, n))  # variance cue
    labels = {'SEX': s, 'RAC1P': np.zeros(n, dtype=int), 'public_coverage': np.zeros(n, dtype=int)}
    z, complete, _ = eb.joint_onehot(labels, n)
    fitted = eb._stabilized_leace(base[complete], z)
    released = eb.apply_affine(base[complete], fitted['projection'], fitted['mean'])

    target = s[complete] - s[complete].mean()
    def r2(features):
        f = np.column_stack((np.ones(len(features)), features))
        coefficient = np.linalg.lstsq(f, target, rcond=None)[0]
        residual = target - f @ coefficient
        return 1.0 - float(residual @ residual) / float(target @ target)

    linear = r2(released)
    quadratic = r2(np.column_stack((released, released ** 2)))
    assert linear < 1e-8, f'affine recovery should be erased, got {linear:.3e}'
    assert quadratic > 10 * max(linear, 1e-12), (
        f'a quadratic feature should still recover: linear {linear:.3e} vs {quadratic:.3e}')


# =================================================================== SPLINCE
def test_splince_satisfies_both_published_constraints_and_keeps_leace_rank():
    """Thm 1: P Sigma_xz = 0 AND P Sigma_xy = Sigma_xy, with LEACE's kernel and rank."""
    from pcrl.baselines.splince import fit_splince

    n, d = 5000, 16
    concept = RNG.integers(0, 2, n)
    target = RNG.normal(size=(n, 2))
    x = RNG.normal(size=(n, d))
    x[:, 0] += 1.2 * concept
    x[:, 3] += 0.8 * target[:, 0]
    x[:, 5] += 0.5 * target[:, 1]

    projection, mean, info = fit_splince(x, concept, target, cond_max=1e6)
    record = info.to_dict() if hasattr(info, 'to_dict') else dict(info.__dict__)
    assert not record.get('fallback_to_leace'), record

    xc = x - x.mean(0)
    sigma_xz = xc.T @ (np.eye(2)[concept] - np.eye(2)[concept].mean(0)) / (n - 1)
    sigma_xy = xc.T @ (target - target.mean(0)) / (n - 1)
    assert np.max(abs(projection @ sigma_xz)) < 1e-8          # kernel constraint
    assert np.max(abs(projection @ sigma_xy - sigma_xy)) < 1e-8   # range constraint
    assert np.max(abs(projection @ projection - projection)) < 1e-8   # idempotent
    # same kernel as LEACE => same rank cost: one centred binary one-hot costs 1
    assert np.linalg.matrix_rank(projection, tol=1e-10) == d - 1


def test_splince_infeasibility_is_detected_and_not_silently_relaxed():
    """When the whitened concept and target spans coincide the constraints conflict.

    The published method states no infeasibility theorem and offers no fallback. The
    repository implementation falls back to LEACE; this study must DETECT that and
    report SCOPED INFEASIBLE instead of shipping a LEACE fit under the SPLINCE name.
    """
    from pcrl.baselines.splince import fit_splince

    n, d = 3000, 8
    concept = RNG.integers(0, 2, n)
    x = RNG.normal(size=(n, d))
    x[:, 0] += 2.0 * concept
    # target is the concept itself, so colsp(W Sigma_xy) == colsp(W Sigma_xz)
    target = concept.astype(float).reshape(-1, 1)
    _, _, info = fit_splince(x, concept, target, cond_max=1e6)
    record = info.to_dict() if hasattr(info, 'to_dict') else dict(info.__dict__)
    infeasible = bool(record.get('fallback_to_leace')) or not record.get('feasible', True)
    assert infeasible, f'perfectly overlapping spans must be flagged, got {record}'


# =================================================================== OptNet-ARL
def test_closed_form_kernel_ridge_player_matches_an_explicit_solve():
    """Tiny reference problem: Lemma 1's expanded form vs an explicit ridge solution."""
    from experiments.pcrl_invariant_baselines_v1.optnet_arl import (centre, explained_fraction,
                                                                    gaussian_gram)

    n, r = 40, 3
    z = torch.from_numpy(RNG.normal(size=(n, r)))
    y = torch.from_numpy(RNG.normal(size=(n, 2)))
    gamma = 1e-3

    value = float(explained_fraction(z, y, gamma))

    # Explicit: Lambda_hat = (K~^2 + n gamma I)^{-1} K~ u, explained = ||K~ L||^2 + n gamma ||L||^2
    m = centre(gaussian_gram(z)).numpy()
    u = centre(y).numpy()
    reg = n * gamma
    lam = np.linalg.solve(m.T @ m + reg * np.eye(n), m.T @ u)
    explicit = (np.sum((m @ lam) ** 2) + reg * np.sum(lam ** 2)) / np.sum(u ** 2)
    assert abs(value - explicit) <= 1e-10 * max(1.0, abs(explicit))


def test_player_solution_is_differentiable_and_matches_finite_differences():
    """Differentiation through the closed-form solve, on a tiny reference problem."""
    from experiments.pcrl_invariant_baselines_v1.optnet_arl import explained_fraction

    n, r = 30, 3
    z = torch.from_numpy(RNG.normal(size=(n, r))).requires_grad_(True)
    y = torch.from_numpy(RNG.normal(size=(n, 2)))
    value = explained_fraction(z, y, 1e-3)
    value.backward()
    analytic = z.grad.detach().numpy().copy()

    base = z.detach().numpy().copy()
    direction = RNG.normal(size=base.shape)
    direction /= np.linalg.norm(direction)
    step = 1e-6
    with torch.no_grad():
        plus = float(explained_fraction(torch.from_numpy(base + step * direction), y, 1e-3))
        minus = float(explained_fraction(torch.from_numpy(base - step * direction), y, 1e-3))
    numeric = (plus - minus) / (2 * step)
    assert abs(numeric - float(np.sum(analytic * direction))) <= 1e-5 * max(1.0, abs(numeric))


def test_theorem_4_1_count_is_computed_without_forming_the_n_by_n_matrix():
    """The low-rank route must agree with the dense one on a small fixture."""
    from experiments.pcrl_invariant_baselines_v1.optnet_arl import theorem_4_1_rank

    n = 300
    protected = {'A/SEX': np.eye(2)[RNG.integers(0, 2, n)],
                 'A/RAC1P': np.eye(9)[RNG.integers(0, 9, n)]}
    teacher = RNG.normal(size=(n, 5))
    weights = {'A/SEX': 0.5, 'A/RAC1P': 0.5}

    fast = theorem_4_1_rank(protected, teacher, weights)
    b = -( (teacher - teacher.mean(0)) @ (teacher - teacher.mean(0)).T )
    for role, onehot in protected.items():
        s = onehot - onehot.mean(0)
        b = b + weights[role] * (s @ s.T)
    dense = np.linalg.eigvalsh((b + b.T) / 2)
    tolerance = 1e-10 * max(1.0, float(abs(dense).max()))
    assert fast['negative_eigenvalue_count'] == int((dense < -tolerance).sum())
    assert fast['low_rank_factor_width'] == 2 + 9 + 5


def test_encoder_architecture_is_frozen_and_deterministic():
    from experiments.pcrl_invariant_baselines_v1.optnet_arl import (ENCODER_SEED_BASE, HIDDEN,
                                                                    RANK, Encoder)
    torch.manual_seed(ENCODER_SEED_BASE)
    a = Encoder(128).double()
    torch.manual_seed(ENCODER_SEED_BASE)
    b = Encoder(128).double()
    x = torch.from_numpy(RNG.normal(size=(20, 128)))
    assert torch.equal(a(x), b(x))
    assert a(x).shape == (20, RANK)
    assert HIDDEN == (64,)
