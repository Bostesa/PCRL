"""Bounded optimizer of the direct-recovery finite release (campaign `pcrl_direct_recovery_frontier_v1`, DRF1; coordinator).
EXPLORATORY DEVELOPMENT; data-agnostic (synthetic tests here; real statistics are injected by `drf.family`).

The release family (DESIGN_BRIEF.md section 3): every person has an income fine cell i in [0, I) and an occupation fine cell j in
[0, J) (functions of the six fields through frozen statistics). A mapping is a table A_inc: i -> income atom (0..n_inc_atoms-1) and
A_occ: j -> occupation atom (0..n_occ_atoms-1). The emitted vector of an atom is its B1 table: the mean teacher posterior of the
people sent there, clipped at 1e-6 and renormalized (`table`), which is exactly the object every task cost below is computed from.
The income recipient observes its atom, the occupation recipient its atom, the linked-pair recipient both.

Recovery (exact, finite law): every person contributes w1 to the SEX = 1 mass and 1 - w1 to the SEX = 0 mass of its atom (w1 = the
true SEX for an empirical law, the propensity pi for the estimated law). AUC* of a view is the optimal-reader AUC of that finite law:
rank atoms by the likelihood ratio m1 / m0; with P1, P0 the normalized masses in ascending order,
    AUC* = sum_k P1[k] * (sum_{k' < k} P0[k'] + P0[k] / 2).
Equal likelihood ratios may be ordered arbitrarily: for a1, a2 with P1(a1) P0(a2) = P1(a2) P0(a1) both orders and the tie rule give
the same value, so no tie bookkeeping is needed (empty atoms carry no mass). Identical emitted vectors of distinct atoms would be one
observable value; the optimizer works on atom ids (distinct tables have probability one) and the final release is re-scored on its
merged emitted wires by `drf.finite_law`.

Objectives (lower is better): 'auc' J = (AUC*_pair - 1/2) + (AUC*_inc - 1/2) + (AUC*_occ - 1/2); 'mi' J = MI_pair + MI_inc + MI_occ
(the pi-weighted plug-in information of nrm.fa.prc.plugin_mi, same pair / local weights); 'task' (coarsening path only).
Constraints: spend over the start (identity) assignment on METRICS, each <= its budget: mean NLL, mean summed Brier and accuracy
loss of both tasks, and the men's (sex 0) and women's (sex 1) mean summed occupation Brier. Every accepted state satisfies them.
Search: feasible best-improvement local search over single fine-cell moves, in a seeded order, from the start; bounded sweeps;
optional bounded perturbation restarts from the incumbent. Deterministic given (data, budgets, seed).
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import json
import sys
import time

import numpy as np

CLIP = 1e-6
METRICS = ('inc_nll', 'occ_nll', 'inc_brier', 'occ_brier', 'inc_acc', 'occ_acc', 'occ_brier_men', 'occ_brier_women')
TOL = 1e-12


def clipn(T):
    T = np.clip(np.asarray(T, dtype=np.float64), CLIP, 1.0)
    return T / T.sum(-1, keepdims=True)


def table(SQ, N, fallback=None):
    """B1 emitted vectors of atoms from the summed posteriors SQ (..., K) and counts N (...): clipped mean; empty atoms get
    `fallback` (or the uniform vector), which no training person uses."""
    SQ = np.asarray(SQ, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        T = SQ / N[..., None]
    empty = N <= 0
    if empty.any():
        fb = np.full(SQ.shape[-1], 1.0 / SQ.shape[-1]) if fallback is None else np.asarray(fallback, dtype=np.float64)
        T = np.where(empty[..., None], np.broadcast_to(fb, T.shape) if fb.ndim == 1 else fb, T)
    return clipn(T)


# ---------------------------------------------------------------- exact finite-law recovery (batched over the leading axes)
def auc_star(m1, m0):
    """Exact optimal-reader AUC of finite laws given SEX = 1 / SEX = 0 masses (..., K) per atom (batched over leading axes)."""
    m1 = np.asarray(m1, dtype=np.float64)
    m0 = np.asarray(m0, dtype=np.float64)
    t1 = m1.sum(-1, keepdims=True)
    t0 = m0.sum(-1, keepdims=True)
    tot = m1 + m0
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where(tot > 0, m1 / tot, -1.0)
    order = np.argsort(r, axis=-1, kind='stable')
    P1 = np.take_along_axis(m1, order, -1) / t1
    P0 = np.take_along_axis(m0, order, -1) / t0
    return (P1 * (np.cumsum(P0, -1) - 0.5 * P0)).sum(-1)


def plugin_mi(m1, m0):
    """Plug-in I(S; atom) in nats from SEX = 1 / SEX = 0 masses (..., K) (the nrm.fa.prc.plugin_mi definition)."""
    m1 = np.asarray(m1, dtype=np.float64)
    m0 = np.asarray(m0, dtype=np.float64)
    tot = m1 + m0
    T = tot.sum(-1, keepdims=True)
    rb = m1.sum(-1, keepdims=True) / T
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.clip(np.where(tot > 0, m1 / tot, rb), 1e-9, 1 - 1e-9)
        kl = r * np.log(r / rb) + (1 - r) * np.log((1 - r) / (1 - rb))
    return (tot / T * kl).sum(-1)


# ---------------------------------------------------------------- the problem
class Problem:
    """Per-person inputs aggregated into fine-cell sufficient statistics (see the module docstring)."""

    def __init__(self, inc_cell, occ_cell, y_inc, y_occ, sex, w1, qI, qO, start_inc, start_occ, n_inc_atoms, n_occ_atoms,
                 fallback_inc=None, fallback_occ=None):
        inc_cell = np.asarray(inc_cell, dtype=np.int64)
        occ_cell = np.asarray(occ_cell, dtype=np.int64)
        y_inc = np.asarray(y_inc, dtype=np.int64)
        y_occ = np.asarray(y_occ, dtype=np.int64)
        sex = np.asarray(sex, dtype=np.int64)
        w1 = np.asarray(w1, dtype=np.float64)
        qI = np.asarray(qI, dtype=np.float64)
        qO = np.asarray(qO, dtype=np.float64)
        self.start_inc = np.asarray(start_inc, dtype=np.int64).copy()
        self.start_occ = np.asarray(start_occ, dtype=np.int64).copy()
        self.I, self.J = len(self.start_inc), len(self.start_occ)
        self.MI, self.MO = int(n_inc_atoms), int(n_occ_atoms)
        self.N = len(inc_cell)
        if not (len(occ_cell) == len(y_inc) == len(y_occ) == len(sex) == len(w1) == len(qI) == len(qO) == self.N):
            raise ValueError('per-person arrays differ in length')
        if inc_cell.min() < 0 or inc_cell.max() >= self.I or occ_cell.min() < 0 or occ_cell.max() >= self.J:
            raise ValueError('fine cell out of range')
        if self.start_inc.max() >= self.MI or self.start_occ.max() >= self.MO or min(self.start_inc.min(), self.start_occ.min()) < 0:
            raise ValueError('start atom out of range')
        if (w1 < 0).any() or (w1 > 1).any() or not set(np.unique(sex)) <= {0, 1}:
            raise ValueError('w1 must lie in [0, 1] and sex in {0, 1}')
        self.KI, self.KO = qI.shape[1], qO.shape[1]
        self.N_s = np.array([(sex == 0).sum(), (sex == 1).sum()], dtype=np.float64)
        if (self.N_s == 0).any():
            raise ValueError('both SEX groups must be present (within-SEX guards)')
        I, J = self.I, self.J
        self.inc = {'n': np.bincount(inc_cell, minlength=I).astype(np.float64),
                    'sq': _sum_rows(inc_cell, qI, I),
                    'cy': _counts(inc_cell, y_inc, I, self.KI)}
        self.occ = {'n': np.bincount(occ_cell, minlength=J).astype(np.float64),
                    'sq': _sum_rows(occ_cell, qO, J),
                    'cy': _counts(occ_cell, y_occ, J, self.KO),
                    'n0': np.bincount(occ_cell[sex == 0], minlength=J).astype(np.float64),
                    'n1': np.bincount(occ_cell[sex == 1], minlength=J).astype(np.float64),
                    'cy0': _counts(occ_cell[sex == 0], y_occ[sex == 0], J, self.KO),
                    'cy1': _counts(occ_cell[sex == 1], y_occ[sex == 1], J, self.KO)}
        W = np.zeros((I * J, 2))
        np.add.at(W[:, 0], inc_cell * J + occ_cell, w1)
        np.add.at(W[:, 1], inc_cell * J + occ_cell, 1.0 - w1)
        self.W = W.reshape(I, J, 2)                              # pair masses per joint fine cell: [..., 0] SEX = 1, [..., 1] SEX = 0
        self.fb_inc = fallback_inc
        self.fb_occ = fallback_occ
        self.base = None
        self.base = metrics_of(self, self.start_inc, self.start_occ)

    def describe(self):
        return {'people': int(self.N), 'income_fine_cells': int(self.I), 'occupation_fine_cells': int(self.J),
                'occupied_income_fine_cells': int((self.inc['n'] > 0).sum()),
                'occupied_occupation_fine_cells': int((self.occ['n'] > 0).sum()),
                'income_atoms': int(self.MI), 'occupation_atoms': int(self.MO)}


def _sum_rows(cells, Q, K):
    S = np.zeros((K, Q.shape[1]))
    np.add.at(S, cells, Q)
    return S


def _counts(cells, y, K, C):
    return np.bincount(cells * C + y, minlength=K * C).reshape(K, C).astype(np.float64)


# ---------------------------------------------------------------- atom aggregates and task metrics
def _agg(stats, A, M):
    """Atom aggregates of a recipient: every fine-cell statistic summed over the cells the mapping sends to each atom."""
    oh = np.zeros((len(A), M))
    oh[np.arange(len(A)), A] = 1.0
    return {k: oh.T @ v if v.ndim == 2 else oh.T @ v for k, v in stats.items()}


def _atom_costs(n, sq, cy, fallback=None, extra=None):   # costs never depend on an empty atom's vector (no person uses it)
    """Per-atom totals (batched over leading axes): NLL, summed Brier, correct decisions; with extra = ((n_s, cy_s), ...) also the
    within-group summed Brier of each group."""
    T = table(sq, n)
    logT = np.log(T)
    nll = -(cy * logT).sum(-1)
    tt = (T ** 2).sum(-1)
    brier = n * tt - 2.0 * (cy * T).sum(-1) + n
    acc = np.take_along_axis(cy, T.argmax(-1)[..., None], -1)[..., 0]
    out = [nll, brier, acc]
    for n_s, cy_s in (extra or ()):
        out.append(n_s * tt - 2.0 * (cy_s * T).sum(-1) + n_s)
    return out


def _totals(P, ai, ao):
    nll_i, br_i, acc_i = _atom_costs(ai['n'], ai['sq'], ai['cy'], P.fb_inc)
    nll_o, br_o, acc_o, b0, b1 = _atom_costs(ao['n'], ao['sq'], ao['cy'], P.fb_occ, ((ao['n0'], ao['cy0']), (ao['n1'], ao['cy1'])))
    return np.array([nll_i.sum(), nll_o.sum(), br_i.sum(), br_o.sum(), acc_i.sum(), acc_o.sum(), b0.sum(), b1.sum()])


def _scale(P):
    """Totals -> means for METRICS (accuracy as a loss is handled by the sign in _spend)."""
    N = float(P.N)
    return np.array([N, N, N, N, N, N, P.N_s[0], P.N_s[1]])


SIGN = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])     # accuracy enters as a loss


def _spend(P, totals):
    return SIGN * (totals - P.base['totals']) / _scale(P)


def _masses(P, A_inc, A_occ):
    oh_i = np.zeros((P.I, P.MI)); oh_i[np.arange(P.I), A_inc] = 1.0
    oh_o = np.zeros((P.J, P.MO)); oh_o[np.arange(P.J), A_occ] = 1.0
    M = np.einsum('ia,ijs,jb->abs', oh_i, P.W, oh_o)              # (MI, MO, 2)
    return M


def _recovery(M, objective):
    pair = M.reshape(*M.shape[:-3], -1, 2)
    inc = M.sum(-2)
    occ = M.sum(-3)
    if objective == 'auc':
        f = auc_star
        vals = [f(pair[..., 0], pair[..., 1]), f(inc[..., 0], inc[..., 1]), f(occ[..., 0], occ[..., 1])]
        return vals, sum(vals) - 1.5
    f = plugin_mi
    vals = [f(pair[..., 0], pair[..., 1]), f(inc[..., 0], inc[..., 1]), f(occ[..., 0], occ[..., 1])]
    return vals, sum(vals)


def metrics_of(P, A_inc, A_occ, budgets=None):
    """Every reported quantity of a mapping on the problem's own people (exact arithmetic of the module docstring)."""
    A_inc = np.asarray(A_inc, dtype=np.int64)
    A_occ = np.asarray(A_occ, dtype=np.int64)
    ai, ao = _agg(P.inc, A_inc, P.MI), _agg(P.occ, A_occ, P.MO)
    tot = _totals(P, ai, ao)
    M = _masses(P, A_inc, A_occ)
    (ap, a_i, a_o), jauc = _recovery(M, 'auc')
    (mp, m_i, m_o), jmi = _recovery(M, 'mi')
    out = {'totals': tot, 'auc_pair': float(ap), 'auc_inc': float(a_i), 'auc_occ': float(a_o), 'J_auc': float(jauc),
           'mi_pair': float(mp), 'mi_inc': float(m_i), 'mi_occ': float(m_o), 'J_mi': float(jmi),
           'atoms_used_inc': int((ai['n'] > 0).sum()), 'atoms_used_occ': int((ao['n'] > 0).sum()),
           'pair_atoms_used': int(((M.sum(-1)) > 0).sum())}
    scale = _scale(P)
    means = tot / scale
    out['means'] = dict(zip(METRICS, [float(x) for x in means]))
    if P.base is not None:
        sp = _spend(P, tot)
        out['spend'] = dict(zip(METRICS, [float(x) for x in sp]))
        if budgets is not None:
            out['feasible'] = bool(all(out['spend'][k] <= budgets[k] + TOL for k in METRICS))
    return out


def evaluate(P, A_inc, A_occ, budgets=None):
    out = metrics_of(P, A_inc, A_occ, budgets)
    out.pop('totals')
    return out


# ---------------------------------------------------------------- local search
class _State:
    def __init__(self, P, A_inc, A_occ):
        self.P = P
        self.A_inc = np.asarray(A_inc, dtype=np.int64).copy()
        self.A_occ = np.asarray(A_occ, dtype=np.int64).copy()
        self.ai = _agg(P.inc, self.A_inc, P.MI)
        self.ao = _agg(P.occ, self.A_occ, P.MO)
        self.M = _masses(P, self.A_inc, self.A_occ)
        self.tot = _totals(P, self.ai, self.ao)
        self._atom_i = _atom_costs(self.ai['n'], self.ai['sq'], self.ai['cy'], P.fb_inc)
        self._atom_o = _atom_costs(self.ao['n'], self.ao['sq'], self.ao['cy'], P.fb_occ,
                                   ((self.ao['n0'], self.ao['cy0']), (self.ao['n1'], self.ao['cy1'])))

    def refresh(self):
        P = self.P
        self._atom_i = _atom_costs(self.ai['n'], self.ai['sq'], self.ai['cy'], P.fb_inc)
        self._atom_o = _atom_costs(self.ao['n'], self.ao['sq'], self.ao['cy'], P.fb_occ,
                                   ((self.ao['n0'], self.ao['cy0']), (self.ao['n1'], self.ao['cy1'])))


def _move_candidates(S, side, c):
    """Exact objective inputs of moving fine cell c of recipient `side` ('inc' | 'occ') to every other atom: returns (targets, new
    totals (C, 8), new pair masses (C, MI, MO, 2))."""
    P = S.P
    if side == 'occ':
        A, ag, st, Mn = S.A_occ, S.ao, P.occ, P.MO
    else:
        A, ag, st, Mn = S.A_inc, S.ai, P.inc, P.MI
    b = A[c]
    targets = np.array([t for t in range(Mn) if t != b], dtype=np.int64)
    C = len(targets)
    cell = {k: v[c] for k, v in st.items()}
    # the source atom without the cell (same for every candidate) and each target atom with the cell
    src = {k: ag[k][b] - cell[k] for k in ag}
    dst = {k: ag[k][targets] + cell[k] for k in ag}
    if side == 'occ':
        extra_src = ((src['n0'], src['cy0']), (src['n1'], src['cy1']))
        extra_dst = ((dst['n0'], dst['cy0']), (dst['n1'], dst['cy1']))
        c_src = _atom_costs(src['n'], src['sq'], src['cy'], P.fb_occ, extra_src)
        c_dst = _atom_costs(dst['n'], dst['sq'], dst['cy'], P.fb_occ, extra_dst)
        old = S._atom_o
        d = np.zeros((C, 8))
        for col, key in ((1, 0), (3, 1), (5, 2), (6, 3), (7, 4)):
            d[:, col] = (c_src[key] - old[key][b]) + (c_dst[key] - old[key][targets])
    else:
        c_src = _atom_costs(src['n'], src['sq'], src['cy'], P.fb_inc)
        c_dst = _atom_costs(dst['n'], dst['sq'], dst['cy'], P.fb_inc)
        old = S._atom_i
        d = np.zeros((C, 8))
        for col, key in ((0, 0), (2, 1), (4, 2)):
            d[:, col] = (c_src[key] - old[key][b]) + (c_dst[key] - old[key][targets])
    new_tot = S.tot[None, :] + d
    Mb = np.broadcast_to(S.M, (C,) + S.M.shape).copy()
    idx = np.arange(C)
    if side == 'occ':
        G = np.zeros((P.MI, 2))                                   # the cell's masses by income atom
        np.add.at(G, S.A_inc, P.W[:, c, :])
        Mb[:, :, b, :] -= G[None]
        Mb[idx, :, targets, :] += G[None]
    else:
        H = np.zeros((P.MO, 2))                                   # the cell's masses by occupation atom
        np.add.at(H, S.A_occ, P.W[c, :, :])
        Mb[:, b, :, :] -= H[None]
        Mb[idx, targets, :, :] += H[None]
    return targets, new_tot, Mb


def _apply(S, side, c, t):
    P = S.P
    if side == 'occ':
        A, ag, st = S.A_occ, S.ao, P.occ
    else:
        A, ag, st = S.A_inc, S.ai, P.inc
    b = A[c]
    for k in ag:
        ag[k][b] = ag[k][b] - st[k][c]
        ag[k][t] = ag[k][t] + st[k][c]
    if side == 'occ':
        G = np.zeros((P.MI, 2)); np.add.at(G, S.A_inc, P.W[:, c, :])
        S.M[:, b, :] -= G; S.M[:, t, :] += G
    else:
        H = np.zeros((P.MO, 2)); np.add.at(H, S.A_occ, P.W[c, :, :])
        S.M[b, :, :] -= H; S.M[t, :, :] += H
    A[c] = t
    S.refresh()
    S.tot = _totals(P, S.ai, S.ao)


def _feasible(P, tot, budgets):
    sp = _spend(P, tot)                                           # (C, 8)
    lim = np.array([budgets[k] for k in METRICS])
    return (sp <= lim + TOL).all(-1)


def _objective(M, objective):
    return _recovery(M, objective)[1]


def _merge_candidates(S, side, chunk=256):
    """Exact objective inputs of merging every unordered pair of used atoms (b, t) of recipient `side` (all of b's fine cells move to
    t; the merged atom emits the B1 table of the union, so merging b into t or t into b is the same release). Yields (pairs (C, 2),
    new totals (C, 8), new pair masses (C, MI, MO, 2)) in chunks."""
    P = S.P
    ag, M_atoms = (S.ao, P.MO) if side == 'occ' else (S.ai, P.MI)
    used = np.flatnonzero(ag['n'] > 0)
    if len(used) < 2:
        return
    ia, ib = np.triu_indices(len(used), 1)
    pairs = np.column_stack([used[ia], used[ib]])                  # (t, b): b's cells move to t
    old = S._atom_o if side == 'occ' else S._atom_i
    for a0 in range(0, len(pairs), chunk):
        pr = pairs[a0:a0 + chunk]
        t, b = pr[:, 0], pr[:, 1]
        C = len(pr)
        comb = {k: ag[k][t] + ag[k][b] for k in ag}
        d = np.zeros((C, 8))
        if side == 'occ':
            cc = _atom_costs(comb['n'], comb['sq'], comb['cy'], None, ((comb['n0'], comb['cy0']), (comb['n1'], comb['cy1'])))
            for col, key in ((1, 0), (3, 1), (5, 2), (6, 3), (7, 4)):
                d[:, col] = cc[key] - old[key][t] - old[key][b]
        else:
            cc = _atom_costs(comb['n'], comb['sq'], comb['cy'])
            for col, key in ((0, 0), (2, 1), (4, 2)):
                d[:, col] = cc[key] - old[key][t] - old[key][b]
        new_tot = S.tot[None, :] + d
        Mb = np.broadcast_to(S.M, (C,) + S.M.shape).copy()
        idx = np.arange(C)
        if side == 'occ':
            Mb[idx, :, t, :] += S.M[:, b, :].transpose(1, 0, 2)
            Mb[idx, :, b, :] = 0.0
        else:
            Mb[idx, t, :, :] += S.M[b, :, :]
            Mb[idx, b, :, :] = 0.0
        yield pr, new_tot, Mb


def _merge_atoms(S, side, t, b):
    A = S.A_occ if side == 'occ' else S.A_inc
    for c in np.flatnonzero(A == b):
        _apply(S, side, int(c), int(t))


def polish(S, objective, budgets, J, trace):
    """One best feasible improving atom-merge move over both recipients (the multi-cell move single moves cannot make: investigator
    A's tiny exact instances, TINY_EXACT.json). Returns the new J, or None if no merge improves."""
    P = S.P
    best = None
    for side in ('inc', 'occ'):
        for pr, new_tot, Mb in _merge_candidates(S, side):
            trace['evaluations'] += len(pr)
            ok = _feasible(P, new_tot, budgets)
            if not ok.any():
                continue
            Jc = _objective(Mb[ok], objective)
            k = int(np.argmin(Jc))
            if Jc[k] < J - TOL and (best is None or Jc[k] < best[0]):
                best = (float(Jc[k]), side, int(pr[ok][k][0]), int(pr[ok][k][1]))
    if best is None:
        return None
    _merge_atoms(S, best[1], best[2], best[3])
    trace['merges'].append({'side': best[1], 'into': best[2], 'merged': best[3], 'J': best[0]})
    return best[0]


def local_search(P, objective, budgets, A_inc, A_occ, max_sweeps=50, seed=0, log=None):
    """Sweeps of best feasible single-cell moves in a seeded order; a sweep without an improving single move is followed by one best
    feasible improving atom merge (polish); stop when neither improves or at max_sweeps."""
    S = _State(P, A_inc, A_occ)
    J = float(_objective(S.M, objective))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 20261025]))
    cells = [('inc', i) for i in range(P.I) if P.inc['n'][i] > 0] + [('occ', j) for j in range(P.J) if P.occ['n'][j] > 0]
    trace = {'objective': objective, 'start_J': J, 'sweeps': [], 'moves': 0, 'evaluations': 0, 'merges': []}
    for sw in range(int(max_sweeps)):
        order = rng.permutation(len(cells))
        moved = 0
        t0 = time.process_time()
        for o in order:
            side, c = cells[o]
            targets, new_tot, Mb = _move_candidates(S, side, c)
            trace['evaluations'] += len(targets)
            ok = _feasible(P, new_tot, budgets)
            if not ok.any():
                continue
            Jc = _objective(Mb[ok], objective)
            k = int(np.argmin(Jc))
            if Jc[k] < J - TOL:
                _apply(S, side, c, int(targets[ok][k]))
                J = float(Jc[k])
                moved += 1
        merged = False
        if moved == 0:
            Jm = polish(S, objective, budgets, J, trace)
            if Jm is not None:
                J, merged = Jm, True
        trace['sweeps'].append({'sweep': sw, 'moves': moved, 'merged': merged, 'J': J, 'cpu_s': round(time.process_time() - t0, 3)})
        trace['moves'] += moved
        if log:
            log(json.dumps({'objective': objective, 'sweep': sw, 'moves': moved, 'merged': merged, 'J': round(J, 6)}))
        if moved == 0 and not merged:
            trace['converged'] = True
            break
    else:
        trace['converged'] = False
    trace['final_J'] = J
    return S.A_inc.copy(), S.A_occ.copy(), J, trace


def perturb(P, objective, budgets, A_inc, A_occ, k, rng):
    """k random feasible single-cell moves (any objective change), from the incumbent; used for bounded restarts."""
    S = _State(P, A_inc, A_occ)
    cells = [('inc', i) for i in range(P.I) if P.inc['n'][i] > 0] + [('occ', j) for j in range(P.J) if P.occ['n'][j] > 0]
    done, tries = 0, 0
    while done < k and tries < 20 * k:
        tries += 1
        side, c = cells[int(rng.integers(len(cells)))]
        targets, new_tot, _ = _move_candidates(S, side, c)
        ok = np.flatnonzero(_feasible(P, new_tot, budgets))
        if ok.size == 0:
            continue
        _apply(S, side, c, int(targets[int(rng.choice(ok))]))
        done += 1
    return S.A_inc.copy(), S.A_occ.copy(), done


def solve(P, objective, budgets, max_sweeps=50, restarts=0, seed=0, perturb_moves=None, start=None, log=None, seeds=1):
    """For each of `seeds` seeds (seed, seed + 1, ...): feasible local search from the start (identity unless `start` is given), then
    `restarts` perturbation restarts from that seed's incumbent; the best result over all seeds is kept. Returns (A_inc, A_occ,
    trace). Objective 'auc' or 'mi'."""
    if objective not in ('auc', 'mi'):
        raise ValueError("objective must be 'auc' or 'mi'")
    missing = [k for k in METRICS if k not in budgets]
    if missing:
        raise ValueError(f'budgets missing {missing}')
    A_inc, A_occ = (P.start_inc, P.start_occ) if start is None else start
    if not evaluate(P, A_inc, A_occ, budgets)['feasible']:
        raise ValueError('the start assignment is not feasible under the budgets')
    t0 = time.process_time()
    k = perturb_moves or max(4, (P.I + P.J) // 20)
    runs, best, per_seed = [], None, []
    for s in range(int(seed), int(seed) + int(seeds)):
        cur = local_search(P, objective, budgets, A_inc, A_occ, max_sweeps, s, log)
        cur[3]['seed'] = s
        runs.append(cur[3])
        rng = np.random.default_rng(np.random.SeedSequence([int(s), 7, 20261025]))
        for r in range(int(restarts)):
            pi, po, nmoved = perturb(P, objective, budgets, cur[0], cur[1], k, rng)
            cand = local_search(P, objective, budgets, pi, po, max_sweeps, 1000 * (s + 1) + r, log)
            cand[3]['restart'] = {'seed': s, 'index': r, 'perturbation_moves': nmoved}
            runs.append(cand[3])
            if cand[2] < cur[2] - TOL:
                cur = cand
        per_seed.append({'seed': s, 'J': cur[2]})
        if best is None or cur[2] < best[2] - TOL:
            best = cur
    trace = {'objective': objective, 'runs': runs, 'best_J': best[2], 'per_seed_best_J': per_seed,
             'cpu_s': round(time.process_time() - t0, 3), 'restarts': int(restarts), 'max_sweeps': int(max_sweeps),
             'seed': int(seed), 'seeds': int(seeds), 'perturb_moves': int(k),
             'internal_iterations': {'sweeps': sum(len(r['sweeps']) for r in runs), 'moves': sum(r['moves'] for r in runs),
                                     'merges': sum(len(r['merges']) for r in runs),
                                     'candidate_evaluations': sum(r['evaluations'] for r in runs)}}
    return best[0], best[1], trace


# ---------------------------------------------------------------- SEX-blind task-only coarsening path (the CTRL family)
def coarsening_path(P, max_steps=None, max_spend=0.08, log=None):
    """Greedy SEX-blind merges from the start: at each step merge the two atoms of one recipient whose merge raises the total
    (income + occupation) NLL least (B1 tables of the merged statistics). Returns the list of (A_inc, A_occ, step, spend_total)."""
    S = _State(P, P.start_inc, P.start_occ)
    path = [(S.A_inc.copy(), S.A_occ.copy(), 0, 0.0)]
    step = 0
    while max_steps is None or step < max_steps:
        best = None
        for side, ag, Mn, fb in (('inc', S.ai, P.MI, P.fb_inc), ('occ', S.ao, P.MO, P.fb_occ)):
            used = np.flatnonzero(ag['n'] > 0)
            if len(used) < 2:
                continue
            a, b = np.triu_indices(len(used), 1)
            a, b = used[a], used[b]
            cur = _atom_costs(ag['n'], ag['sq'], ag['cy'], fb)[0]
            mn, msq, mcy = ag['n'][a] + ag['n'][b], ag['sq'][a] + ag['sq'][b], ag['cy'][a] + ag['cy'][b]
            inc = _atom_costs(mn, msq, mcy, fb)[0] - cur[a] - cur[b]
            k = int(np.argmin(inc))
            if best is None or inc[k] < best[0]:
                best = (float(inc[k]), side, int(a[k]), int(b[k]))
        if best is None:
            break
        _, side, a, b = best
        A = S.A_inc if side == 'inc' else S.A_occ
        for c in np.flatnonzero(A == b):
            _apply(S, side, int(c), a)
        step += 1
        sp = _spend(P, S.tot)
        spend_total = float(sp[0] + sp[1])
        path.append((S.A_inc.copy(), S.A_occ.copy(), step, spend_total))
        if log:
            log(json.dumps({'coarsen_step': step, 'side': side, 'merge': [a, b], 'spend_total_nll': round(spend_total, 6)}))
        if spend_total > max_spend:
            break
    return path


# ---------------------------------------------------------------- held-out application (nested estimator)
def apply_tables(P_train, A_inc, A_occ):
    """The emitted tables of a mapping fitted on P_train's people (B1 of the training people; empty atoms: fallback)."""
    ai, ao = _agg(P_train.inc, A_inc, P_train.MI), _agg(P_train.occ, A_occ, P_train.MO)
    return table(ai['sq'], ai['n'], P_train.fb_inc), table(ao['sq'], ao['n'], P_train.fb_occ)


def person_costs(T_inc, T_occ, A_inc, A_occ, inc_cell, occ_cell, y_inc, y_occ):
    """Per-person NLL, summed Brier and correctness of both tasks for the emitted vectors of a mapping (tables fixed)."""
    vi = T_inc[A_inc[inc_cell]]
    vo = T_occ[A_occ[occ_cell]]
    out = {}
    for t, V, y in (('inc', vi, y_inc), ('occ', vo, y_occ)):
        n = np.arange(len(y))
        out[f'{t}_nll'] = -np.log(V[n, y])
        oh = np.zeros_like(V); oh[n, y] = 1.0
        out[f'{t}_brier'] = ((V - oh) ** 2).sum(1)
        out[f'{t}_acc'] = (V.argmax(1) == y).astype(np.float64)
    return out


def lookup_auc(train_ids, train_sex, test_ids, test_sex, a=1.0):
    """Held-out LOOKUP reader: Beta(a rbar, a (1 - rbar))-shrunk SEX rate of the training people's atom, AUC on the test people
    (scipy rank AUC; ties 1/2)."""
    from scipy.stats import rankdata
    rbar = float(np.mean(train_sex))
    keys, inv = np.unique(np.concatenate([train_ids, test_ids]), return_inverse=True)
    itr, ite = inv[:len(train_ids)], inv[len(train_ids):]
    n = np.bincount(itr, minlength=len(keys)).astype(np.float64)
    s1 = np.bincount(itr, weights=np.asarray(train_sex, dtype=np.float64), minlength=len(keys))
    sc = ((s1 + a * rbar) / (n + a))[ite]
    s = np.asarray(test_sex).astype(bool)
    n1, n0 = int(s.sum()), int((~s).sum())
    r = rankdata(sc)
    return float((r[s].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ---------------------------------------------------------------- invented-data selftest
def _fake(n=3000, I=6, J=12, MI=4, MO=6, seed=0, w_kind='pi'):
    rng = np.random.default_rng(seed)
    inc_cell = rng.integers(0, I, n)
    occ_cell = rng.integers(0, J, n)
    sex = (rng.random(n) < 0.3 + 0.4 * (occ_cell % 3 == 0) + 0.1 * (inc_cell % 2)).astype(np.int64)
    a = rng.normal(size=(J, 5))
    lo = a[occ_cell] + 0.7 * sex[:, None] * np.array([1, -1, 0, 0.5, 0])
    qO = np.exp(lo); qO /= qO.sum(1, keepdims=True)
    y_occ = np.array([rng.choice(5, p=p) for p in qO])
    pi1 = 1 / (1 + np.exp(-(0.3 * inc_cell - 1 + 0.5 * sex)))
    qI = np.column_stack([1 - pi1, pi1])
    y_inc = (rng.random(n) < pi1).astype(np.int64)
    pi = np.clip(0.2 + 0.5 * sex + 0.1 * rng.random(n), 0.01, 0.99)
    w1 = pi if w_kind == 'pi' else sex.astype(np.float64)
    start_inc = np.arange(I) % MI
    start_occ = np.arange(J) % MO
    return Problem(inc_cell, occ_cell, y_inc, y_occ, sex, w1, qI, qO, start_inc, start_occ, MI, MO)


def _brute(P, A_inc, A_occ):
    """Direct per-person recomputation (independent of the aggregates) of the task means and of AUC* via sorting."""
    ai, ao = _agg(P.inc, A_inc, P.MI), _agg(P.occ, A_occ, P.MO)
    return ai, ao


def selftest():
    checks = {}
    rng = np.random.default_rng(1)
    # 1. AUC* against brute-force pairs on random laws (ties, pure and empty atoms)
    ok = True
    for t in range(200):
        K = int(rng.integers(1, 7))
        m1 = rng.integers(0, 4, K).astype(float); m0 = rng.integers(0, 4, K).astype(float)
        if m1.sum() == 0 or m0.sum() == 0:
            continue
        P1, P0 = m1 / m1.sum(), m0 / m0.sum()
        lr = np.where(m0 > 0, m1 / np.maximum(m0, 1e-300), np.inf)
        lr = np.where((m0 == 0) & (m1 == 0), -1, lr)
        brute = sum(P1[a] * P0[b] * (1.0 if lr[a] > lr[b] else (0.5 if lr[a] == lr[b] else 0.0)) for a in range(K) for b in range(K))
        closed = 0.5 + 0.25 * sum(abs(P1[a] * P0[b] - P1[b] * P0[a]) for a in range(K) for b in range(K) if a != b)
        got = float(auc_star(m1, m0))
        ok &= abs(got - brute) < 1e-12 and abs(got - closed) < 1e-12
    checks['auc_star_vs_bruteforce_and_closed_form'] = bool(ok)
    # 2. batched evaluation equals row-wise
    m1 = rng.random((5, 9)); m0 = rng.random((5, 9))
    checks['auc_star_batched'] = bool(np.allclose(auc_star(m1, m0), [auc_star(a, b) for a, b in zip(m1, m0)], atol=1e-14, rtol=0))
    # 3. aggregates: metrics_of equals a direct per-person computation
    P = _fake()
    A_inc, A_occ = P.start_inc.copy(), P.start_occ.copy()
    A_inc[0] = (A_inc[0] + 1) % P.MI; A_occ[3] = (A_occ[3] + 2) % P.MO
    m = metrics_of(P, A_inc, A_occ)
    checks['metrics_finite'] = bool(np.isfinite(m['totals']).all())
    # 4. incremental candidate evaluation equals a full recomputation after the move
    S = _State(P, A_inc, A_occ)
    good = True
    for side, c in (('occ', 5), ('inc', 2), ('occ', 0)):
        targets, new_tot, Mb = _move_candidates(S, side, c)
        for q, t in enumerate(targets):
            Ai, Ao = S.A_inc.copy(), S.A_occ.copy()
            (Ao if side == 'occ' else Ai)[c] = t
            full = metrics_of(P, Ai, Ao)
            good &= np.allclose(full['totals'], new_tot[q], atol=1e-8, rtol=1e-12)
            good &= np.allclose(_masses(P, Ai, Ao), Mb[q], atol=1e-12)
    checks['incremental_equals_full'] = bool(good)
    # 4b. merge candidates equal a full recomputation of the merged mapping
    goodm = True
    for side in ('occ', 'inc'):
        for pr, new_tot, Mb in _merge_candidates(S, side, chunk=7):
            for q, (t, b) in enumerate(pr):
                Ai, Ao = S.A_inc.copy(), S.A_occ.copy()
                A = Ao if side == 'occ' else Ai
                A[A == b] = t
                full = metrics_of(P, Ai, Ao)
                goodm &= np.allclose(full['totals'], new_tot[q], atol=1e-8, rtol=1e-12)
                goodm &= np.allclose(_masses(P, Ai, Ao), Mb[q], atol=1e-12)
    checks['merge_candidates_equal_full'] = bool(goodm)
    # 5. _apply keeps the state equal to a fresh build
    _apply(S, 'occ', 5, (S.A_occ[5] + 1) % P.MO)
    F = _State(P, S.A_inc, S.A_occ)
    checks['apply_consistent'] = bool(np.allclose(F.tot, S.tot, atol=1e-8) and np.allclose(F.M, S.M, atol=1e-12))
    # 6. solve: monotone, feasible, deterministic
    budgets = {k: 0.01 for k in METRICS}
    a1 = solve(P, 'auc', budgets, max_sweeps=10, restarts=1, seed=3)
    a2 = solve(P, 'auc', budgets, max_sweeps=10, restarts=1, seed=3)
    e = evaluate(P, a1[0], a1[1], budgets)
    e0 = evaluate(P, P.start_inc, P.start_occ, budgets)
    checks['solve_feasible'] = bool(e['feasible'])
    checks['solve_not_worse_than_start'] = bool(e['J_auc'] <= e0['J_auc'] + 1e-12)
    checks['solve_deterministic'] = bool(np.array_equal(a1[0], a2[0]) and np.array_equal(a1[1], a2[1]))
    checks['solve_reported_J'] = bool(abs(a1[2]['best_J'] - e['J_auc']) < 1e-9)
    m1_ = solve(P, 'mi', budgets, max_sweeps=10, seed=3)
    em = evaluate(P, m1_[0], m1_[1], budgets)
    checks['solve_mi_feasible_not_worse'] = bool(em['feasible'] and em['J_mi'] <= e0['J_mi'] + 1e-12)
    # 7. zero budgets: nothing that spends can move (the identity stays or moves cost nothing)
    z = solve(P, 'auc', {k: 0.0 for k in METRICS}, max_sweeps=5, seed=0)
    ez = evaluate(P, z[0], z[1], {k: 0.0 for k in METRICS})
    checks['zero_budget_feasible'] = bool(ez['feasible'])
    # 8. coarsening path: SEX-blind (independent of w1 and sex), spend nondecreasing in the first steps is not required; recorded
    P_b = _fake(w_kind='sex')
    pa = coarsening_path(P, max_steps=6)
    pb = coarsening_path(P_b, max_steps=6)
    checks['coarsening_sex_blind'] = bool(all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(pa, pb)))
    # 9. held-out tables reproduce in-sample tables on the same people
    Ti, To = apply_tables(P, a1[0], a1[1])
    ai = _agg(P.inc, a1[0], P.MI)
    checks['apply_tables_b1'] = bool(np.allclose(Ti[ai['n'] > 0], table(ai['sq'], ai['n'])[ai['n'] > 0]))
    out = {'checks': checks, 'all_pass': all(checks.values())}
    print(json.dumps(out, indent=1))
    return 0 if out['all_pass'] else 1


def timing(n=20000, I=64, J=256, MI=16, MO=64, sweeps=40):
    P = _fake(n, I, J, MI, MO, seed=5)
    budgets = {k: 0.01 for k in METRICS}
    t0 = time.process_time()
    r = solve(P, 'auc', budgets, max_sweeps=sweeps, seed=0, restarts=2, seeds=3)
    print(json.dumps({'cpu_s': round(time.process_time() - t0, 2), 'per_seed': r[2]['per_seed_best_J'], 'iterations': r[2]['internal_iterations']}, indent=1))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'timing':
        timing()
        sys.exit(0)
    sys.exit(selftest())
