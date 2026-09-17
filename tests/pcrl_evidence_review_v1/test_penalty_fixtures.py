"""Small deterministic fixtures behind METHOD_REVIEW.md.

Each test pins one claim about what a finite conditional-moment penalty can and
cannot see. They are population or exact-enumeration constructions, not
simulations: every expectation below is computed on a fully enumerated
distribution, so the numbers are exact up to floating point.

Run: python -m pytest tests/pcrl_evidence_review_v1/test_penalty_fixtures.py
"""
from __future__ import annotations

import itertools
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

TOL = 1e-12


def moment(z_features, s_centred, probs):
    """E[ g(Z,H) (1{S=c} - P(S=c|H)) ] for a stack of output-side features g."""
    return (probs[:, None, None] * z_features[:, :, None] * s_centred[:, None, :]).sum(axis=0)


# --------------------------------------------------------------------- XOR
def xor_distribution():
    """S, H independent fair signs; Z = S*H. Support enumerated with exact masses."""
    rows = []
    for s, h in itertools.product((-1, 1), (-1, 1)):
        rows.append((s, h, s * h, 0.25))
    s = np.array([r[0] for r in rows], float)
    h = np.array([r[1] for r in rows], float)
    z = np.array([r[2] for r in rows], float)
    p = np.array([r[3] for r in rows], float)
    return s, h, z, p


def test_xor_marginal_moment_is_blind_interaction_basis_sees_it():
    s, h, z, p = xor_distribution()
    # S is a fair sign and is independent of H, so P(S=1|H) = 1/2 for both H values.
    s_centred = np.column_stack([(s == 1).astype(float) - 0.5])
    # Output-side feature is linear in Z; basis functions of H are {1, H}.
    marginal = moment(z[:, None], s_centred, p)              # basis b(H) = 1
    interaction = moment((z * h)[:, None], s_centred, p)     # basis b(H) = H
    assert abs(marginal[0, 0]) < TOL, marginal
    assert abs(interaction[0, 0]) > 0.4, interaction
    # and S is exactly recoverable from (Z, H)
    assert np.all(z * h == s)


# --------------------------------------------------------------- magnitude
def magnitude_distribution():
    """(Z,S) uniform on {(-1,0),(1,0),(-2,1),(2,1)}; H is constant (degenerate)."""
    z = np.array([-1.0, 1.0, -2.0, 2.0])
    s = np.array([0, 0, 1, 1])
    p = np.full(4, 0.25)
    return z, s, p


def test_magnitude_needs_a_nonlinear_output_feature():
    z, s, p = magnitude_distribution()
    s_centred = (s == 1).astype(float)[:, None] - 0.5
    # H is constant, so the richest possible basis in H is {1}: enriching the
    # nuisance/basis side cannot help here, only enriching the Z side can.
    linear = moment(z[:, None], s_centred, p)
    assert abs(linear[0, 0]) < TOL, linear
    squared = moment((z ** 2)[:, None], s_centred, p)
    assert abs(squared[0, 0]) == 0.75, squared  # exact: E[Z^2 (S - 1/2)] = 3/4
    # thresholding recovers S exactly
    assert np.all((np.abs(z) > 1.5).astype(int) == s)


def test_polynomial_basis_in_H_alone_cannot_fix_magnitude():
    """Any b(H) with H constant is a scalar multiple of 1, so every moment vanishes."""
    z, s, p = magnitude_distribution()
    s_centred = (s == 1).astype(float)[:, None] - 0.5
    for coefficient in (1.0, -3.5, 17.0):
        feats = (coefficient * z)[:, None]
        assert abs(moment(feats, s_centred, p)[0, 0]) < TOL


# ----------------------------------------------------- conditional null via H
def conditional_null_distribution(grid=9):
    """H uniform on a grid; S | H Bernoulli(sigmoid(H)); Z = H exactly.

    All dependence between Z and S flows through H, so the *conditional* moments
    vanish for every feature of Z while the *marginal* ones do not.
    """
    h = np.linspace(-2.0, 2.0, grid)
    pi = 1.0 / (1.0 + np.exp(-h))
    rows_h, rows_s, rows_p = [], [], []
    for hv, pv in zip(h, pi):
        rows_h += [hv, hv]
        rows_s += [1, 0]
        rows_p += [pv / grid, (1 - pv) / grid]
    return np.array(rows_h), np.array(rows_s), np.array(rows_p), pi


def test_conditional_penalty_is_zero_when_all_dependence_flows_through_H():
    h, s, p, _ = conditional_null_distribution()
    z = h.copy()                      # Z carries nothing beyond H
    m = 1.0 / (1.0 + np.exp(-h))      # correct nuisance P(S=1|H)
    s_centred = ((s == 1).astype(float) - m)[:, None]
    for feats in (z[:, None], (z ** 2)[:, None], np.column_stack([z, z ** 2, np.tanh(z)])):
        assert np.abs(moment(feats, s_centred, p)).max() < 1e-12
    # the marginal (unconditional) version is emphatically non-zero
    s_marg = ((s == 1).astype(float) - (p * (s == 1)).sum())[:, None]
    assert abs(moment(z[:, None], s_marg, p)[0, 0]) > 0.05


def test_misspecified_nuisance_manufactures_a_penalty_under_conditional_independence():
    """A wrong m(H) makes the penalty positive although S _||_ Z | H holds exactly."""
    h, s, p, _ = conditional_null_distribution()
    z = h.copy()
    wrong = np.full_like(h, 0.5)      # constant nuisance: misspecified
    s_centred = ((s == 1).astype(float) - wrong)[:, None]
    value = abs(moment(z[:, None], s_centred, p)[0, 0])
    assert value > 0.05, value


# ------------------------------------------------- rank selection vs Ky Fan
def test_free_rank_optimum_keeps_exactly_the_positive_eigenvalues():
    """max over all W with orthonormal columns and free width of tr(W'AW)
    equals the sum of the positive eigenvalues of A."""
    rng = np.random.default_rng(11)
    B = rng.normal(size=(8, 8))
    A = (B + B.T) / 2
    vals = np.linalg.eigvalsh(A)
    free = vals[vals > 0].sum()
    best_fixed = {r: vals[::-1][:r].sum() for r in range(1, 9)}
    assert abs(max(best_fixed.values()) - free) < 1e-10
    argmax_r = max(best_fixed, key=best_fixed.get)
    assert argmax_r == int((vals > 0).sum())
    # the fixed-rank optimum at r = 8 is strictly worse whenever A has a negative eigenvalue
    assert (vals < 0).any()
    assert best_fixed[8] < free - 1e-10


def test_sign_selection_on_the_old_matrix_does_not_optimise_a_new_matrix():
    """The positive-eigenvalue set of A is not the positive-eigenvalue set of A'."""
    rng = np.random.default_rng(23)
    B = rng.normal(size=(10, 10))
    A = (B + B.T) / 2
    C = rng.normal(size=(10, 10))
    penalty = C @ C.T                                  # a new, different penalty
    A2 = A - 0.7 * penalty / np.trace(penalty) * np.trace(A)
    w1, V1 = np.linalg.eigh(A)
    w2, V2 = np.linalg.eigh(A2)
    keep1 = V1[:, w1 > 0]
    value_transplanted = float(np.trace(keep1.T @ A2 @ keep1))
    value_native = float(w2[w2 > 0].sum())
    assert value_native > value_transplanted + 1e-8, (value_native, value_transplanted)


def test_kernelised_output_features_break_the_trace_form():
    """With Z = W'V and a nonlinear map psi on Z, the penalty is not tr(W'AW)
    for any W-independent A: it changes if W is replaced by W*Q, Q orthogonal,
    although tr(W'AW) does not."""
    rng = np.random.default_rng(31)
    n, d, r = 400, 6, 2
    V = rng.normal(size=(n, d))
    V -= V.mean(0)
    V = np.linalg.qr(V)[0] * np.sqrt(n)               # whitened columns
    resid = rng.normal(size=n)
    resid -= resid.mean()
    W, _ = np.linalg.qr(rng.normal(size=(d, r)))
    Q, _ = np.linalg.qr(rng.normal(size=(r, r)))

    def linear_penalty(Wm):
        Z = V @ Wm
        return float((Z * resid[:, None]).mean(0) @ (Z * resid[:, None]).mean(0))

    def squared_penalty(Wm):
        Z = V @ Wm
        psi = Z ** 2                                   # a nonlinear output feature
        return float((psi * resid[:, None]).mean(0) @ (psi * resid[:, None]).mean(0))

    assert abs(linear_penalty(W) - linear_penalty(W @ Q)) < 1e-9
    assert abs(squared_penalty(W) - squared_penalty(W @ Q)) > 1e-6
