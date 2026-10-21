"""SUPERSEDED first sketch (Designer B, 2026-10-21): the Fisher-shaped propensity shift. Kept only so that the two early label-free
records FEASIBILITY_V0_run1.json and EXPLORE_V0.json (rows PCC_* / SHUF_* / GROUPS_*) are traceable to their formulas; those
records were produced by the first versions of mech_sketch.py / feasibility.py, which were then rewritten. Not part of the proposal.

  z      = [H2 log p_I ; H5 log p_O] (centred logits, R^5);  F(x) = block-diagonal Fisher of the task log loss in z
  shift  d(x) = clip((q(x) - qbar) (F(x) + 0.05 I)^-1 lam, +-2.5);  release t = z - alpha d (softmax back)
  lam    solves mean[(q - qbar)(s - sbar) G] lam = Cov(z, s) with G = (F + 0.05 I)^-1, plus 5 fixed-point corrections for the cap
Finding (plug-in, V0): person-specific Fisher-shaped directions push most of the shift into the logits of small probabilities, so
the shift itself re-encodes the propensity: pair-view LR on the log-ratios alone fell to 0.4987 at alpha = 1, yet LR on the rich
features stayed at 0.6204 (SHUF 0.6226). The global (flat) direction, later stratified by the occupation decision, does not do this.
"""
import numpy as np

RHO, CAP, FIX_ITERS = 0.05, 2.5, 5


def helmert(k):
    H = np.zeros((k - 1, k))
    for i in range(1, k):
        H[i - 1, :i] = 1.0
        H[i - 1, i] = -float(i)
        H[i - 1] /= np.sqrt(i * (i + 1.0))
    return H


H2, H5 = helmert(2), helmert(5)


def to_z(p_inc, p_occ):
    return np.column_stack([np.log(np.maximum(p_inc, 1e-300)) @ H2.T, np.log(np.maximum(p_occ, 1e-300)) @ H5.T])


def ginv(p_inc, p_occ, rho=RHO):
    G = np.zeros((len(p_inc), 5, 5))
    for sl, H, p in ((slice(0, 1), H2, p_inc), (slice(1, 5), H5, p_occ)):
        hp = p @ H.T
        F = np.einsum('ak,nk,bk->nab', H, p, H) - hp[:, :, None] * hp[:, None, :] + rho * np.eye(H.shape[0])[None]
        G[:, sl, sl] = np.linalg.inv(F)
    return G


def shift(G, amp, lam):
    return np.clip(amp[:, None] * np.einsum('nab,b->na', G, lam), -CAP, CAP)


def fit_lam(p_inc, p_occ, q, s):
    z, qbar, sc = to_z(p_inc, p_occ), float(q.mean()), s - s.mean()
    G = ginv(p_inc, p_occ)
    c = (z * sc[:, None]).mean(0)
    A = np.einsum('n,nab->ab', (q - qbar) * sc, G) / len(q)
    lam = np.linalg.solve(A, c)
    for _ in range(FIX_ITERS):
        lam = lam + np.linalg.solve(A, ((z - shift(G, q - qbar, lam)) * sc[:, None]).mean(0))
    return lam, qbar
