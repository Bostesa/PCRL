"""Independent fixtures for the approximate-composition lemma (original Proposition 6).

Lemma. Views h_1..h_k with covariances Sigma_p > 0, target A with tr(Sigma_A) > 0, block-whitened
correlation R (R_pq = Sigma_p^{-1/2} Sigma_pq Sigma_q^{-1/2}) positive definite. With
R^2(h;A) = tr(Sigma_Ah Sigma_h^{-1} Sigma_hA) / tr(Sigma_A):
    R^2([h_1..h_k]; A) <= sum_p R^2(h_p; A) / lambda_min(R).
The singular case uses pseudo-inverses and lambda_+(R) (smallest positive eigenvalue).
The same bound holds per target column (max-single-class) and for every contrast a'A, hence for the
top squared canonical correlation (all-class-contrast).

Numerical coverage: random Gaussian population covariances (exact linear algebra in float64).
This is evidence for, not a proof of, the lemma; the proof is in COMPOSITION_LEMMA.tex.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def inv_sqrt(S):
    w, V = np.linalg.eigh(S)
    return V @ np.diag(w ** -0.5) @ V.T


def r2(Shh, ShA, SA, ridge=0.0):
    M = np.linalg.solve(Shh + ridge * np.eye(len(Shh)), ShA)
    return float(np.trace(ShA.T @ M) / np.trace(SA))


def instance(rng, k, d, c, collinear=0.0):
    n = k * d + c
    L = rng.normal(size=(n, n))
    if collinear:
        L[d:2 * d] = L[:d] + collinear * rng.normal(size=(d, n))  # view 2 nearly copies view 1
    S = L @ L.T / n
    blocks = [slice(p * d, (p + 1) * d) for p in range(k)]
    A = slice(k * d, n)
    return S, blocks, A


def whitened_R(S, blocks):
    D = np.zeros((sum(b.stop - b.start for b in blocks),) * 2)
    Wi = [inv_sqrt(S[b, b]) for b in blocks]
    idx = np.concatenate([np.arange(b.start, b.stop) for b in blocks])
    Sh = S[np.ix_(idx, idx)]
    off = 0
    for p, b in enumerate(blocks):
        w = b.stop - b.start
        D[off:off + w, off:off + w] = Wi[p]
        off += w
    return D @ Sh @ D


def check(trials=2000, seed=0):
    rng = np.random.default_rng(seed)
    worst_util, viol, viol_col, viol_cca = 0.0, 0, 0, 0
    for i in range(trials):
        k, d, c = rng.integers(2, 4), rng.integers(1, 4), rng.integers(1, 4)
        S, blocks, A = instance(rng, k, d, c, collinear=(0.05 if i % 3 == 0 else 0.0))
        idx = np.concatenate([np.arange(b.start, b.stop) for b in blocks])
        Sh, ShA, SA = S[np.ix_(idx, idx)], S[idx, A], S[A, A]
        R = whitened_R(S, blocks)
        lam = np.linalg.eigvalsh(R).min()
        per = [r2(S[b, b], S[b, A], SA) for b in blocks]
        joint = r2(Sh, ShA, SA)
        bound = sum(per) / lam
        viol += joint > bound * (1 + 1e-9) + 1e-12
        worst_util = max(worst_util, joint / bound)
        # per target column (max-single-class form)
        for j in range(A.stop - A.start):
            col = slice(A.start + j, A.start + j + 1)
            pj = sum(r2(S[b, b], S[b, col], S[col, col]) for b in blocks)
            viol_col += r2(Sh, S[idx, col], S[col, col]) > pj / lam * (1 + 1e-9) + 1e-12
        # all-contrast form: top squared canonical correlation
        def cca(Shh, ShA_):
            Wa = inv_sqrt(SA)
            M = inv_sqrt(Shh) @ ShA_ @ Wa
            return float(np.linalg.svd(M, compute_uv=False)[0] ** 2)
        per_cca = sum(cca(S[b, b], S[b, A]) for b in blocks)
        viol_cca += cca(Sh, ShA) > per_cca / lam * (1 + 1e-9) + 1e-12
    return {"trials": int(trials), "violations_pooled": int(viol), "violations_per_column": int(viol_col),
            "violations_top_cca": int(viol_cca), "max_utilisation": worst_util}


def ridge_form_check(seed=1, trials=20000):
    """Ridge form with a RIDGE premise at the same tau: R2_tau(concat) <= sum_p R2_tau(h_p)/lambda_min(R).
    Proof (COMPOSITION_LEMMA.tex, part iii): lambda_min(R) <= 1, so R + tau D^-2 >= lambda (I + tau D^-2).
    This replaces an earlier over-cautious note that the premise had to be unregularised."""
    rng = np.random.default_rng(seed)
    viol, worst = 0, 0.0
    for _ in range(trials):
        tau = float(10 ** rng.uniform(-2, 1.5))
        S, blocks, A = instance(rng, int(rng.integers(2, 4)), int(rng.integers(1, 3)), int(rng.integers(1, 3)),
                                collinear=float(rng.uniform(0.0, 0.3)))
        idx = np.concatenate([np.arange(b.start, b.stop) for b in blocks])
        lam = np.linalg.eigvalsh(whitened_R(S, blocks)).min()
        per_ridge = sum(r2(S[b, b], S[b, A], S[A, A], ridge=tau) for b in blocks)
        joint_ridge = r2(S[np.ix_(idx, idx)], S[idx, A], S[A, A], ridge=tau)
        viol += joint_ridge > per_ridge / lam * (1 + 1e-9) + 1e-12
        worst = max(worst, joint_ridge / (per_ridge / lam))
        assert lam <= 1 + 1e-12
    return {"trials": trials, "violations": int(viol), "max_utilisation": worst, "lambda_min_le_1": True}


if __name__ == "__main__":
    out = {"lemma_checks": check(), "ridge_form": ridge_form_check()}
    json.dump(out, sys.stdout, indent=1)
