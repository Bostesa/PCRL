"""SCM merge engine for the engineering benchmark (data-free core; PROTOCOL.md sections 3-5).

Exact all-pairs greedy merging over cells with per-cell sufficient statistics. At every step every live pair is scored;
candidates are tried in priority order (ties to the lowest (a, b)); a merge is accepted when every cumulative cost stays
within its budget. Occupation and guard costs are cell-level merge identities; income costs are computed exactly on the
32-group coarsened release (the released income object). All costs are plug-in under the fitted teacher law.
"""
from __future__ import annotations

import numpy as np

KAPPA = 20.0
N_GROUPS = 32
TINY = 1e-300
CAP = {'nll': 0.010, 'brier': 0.005, 'acc': 0.010}


def xlogy(x, y):
    return np.where(x > 0, x * np.log(np.maximum(y, TINY)), 0.0)


# ------------------------------------------------------------------ KL-Bregman k-means
def kl_assign(T1, T2, mu1, mu2, chunk=4096):
    """argmin_c KL(t1||mu1_c) + KL(t2||mu2_c)  <=>  argmax_c sum t log mu_c; ties to the lowest index."""
    L1, L2 = np.log(np.maximum(mu1, TINY)), np.log(np.maximum(mu2, TINY))
    out = np.empty(len(T1), dtype=np.int64)
    for s in range(0, len(T1), chunk):
        sc = T1[s:s + chunk] @ L1.T + T2[s:s + chunk] @ L2.T
        out[s:s + chunk] = sc.argmax(1)
    return out


def kl_kmeans(T1, T2, k, seed, iters=50):
    """k-means++ seeding with KL distortion, then Lloyd steps (centroid = member mean). Empty cells are dropped at the end."""
    rng = np.random.default_rng(seed); n = len(T1)
    idx = [int(rng.integers(n))]
    def dist_to(c):
        m1, m2 = T1[c], T2[c]
        return (xlogy(T1, T1) - xlogy(T1, m1[None, :])).sum(1) + (xlogy(T2, T2) - xlogy(T2, m2[None, :])).sum(1)
    d = dist_to(idx[0])
    for _ in range(1, k):
        p = np.maximum(d, 0); tot = p.sum()
        j = int(rng.choice(n, p=p / tot)) if tot > 0 else int(rng.integers(n))
        idx.append(j); d = np.minimum(d, dist_to(j))
    mu1, mu2 = T1[idx].copy(), T2[idx].copy()
    assign = kl_assign(T1, T2, mu1, mu2); it = 0
    for it in range(1, iters + 1):
        cnt = np.bincount(assign, minlength=k).astype(np.float64); keep = cnt > 0
        s1 = np.zeros_like(mu1); np.add.at(s1, assign, T1); s2 = np.zeros_like(mu2); np.add.at(s2, assign, T2)
        mu1[keep] = s1[keep] / cnt[keep, None]; mu2[keep] = s2[keep] / cnt[keep, None]
        new = kl_assign(T1, T2, mu1, mu2)
        if np.array_equal(new, assign):
            break
        assign = new
    occ = np.flatnonzero(np.bincount(assign, minlength=k) > 0)
    mu1, mu2 = mu1[occ], mu2[occ]
    assign = kl_assign(T1, T2, mu1, mu2)
    return mu1, mu2, assign, {'nominal_cells': int(k), 'occupied_cells': int(len(occ)), 'lloyd_iterations': int(it)}


# ------------------------------------------------------------------ cell statistics
class Stats:
    """Per-cell sufficient statistics on MODEL_FIT (teacher OOF law)."""

    def __init__(self, assign, k, T1, T2, pi, sex, occ):
        a = np.asarray(assign); self.N = float(len(a)); self.k = k
        self.n = np.bincount(a, minlength=k).astype(np.float64)
        self.S1 = np.zeros((k, T1.shape[1])); np.add.at(self.S1, a, T1)
        self.S2 = np.zeros((k, T2.shape[1])); np.add.at(self.S2, a, T2)
        self.SP = np.bincount(a, weights=pi, minlength=k).astype(np.float64)
        self.Ns = {g: float((sex == g).sum()) for g in (0, 1)}
        self.ns = {g: np.bincount(a[sex == g], minlength=k).astype(np.float64) for g in (0, 1)}
        Y = np.eye(T2.shape[1])[occ]
        mu_f = self.S2 / np.maximum(self.n, 1e-300)[:, None]
        self.Y = {}   # kappa-shrunk SEX-specific label sums, fixed once at the fine cells: n_fs * (Y_fs + kappa mu_f) / (n_fs + kappa)
        for g in (0, 1):
            raw = np.zeros((k, T2.shape[1])); np.add.at(raw, a[sex == g], Y[sex == g])
            ns = self.ns[g][:, None]
            self.Y[g] = np.where(ns > 0, ns * (raw + KAPPA * mu_f) / (ns + KAPPA), 0.0)
        # constants of the plug-in law for absolute income costs
        self.c_brier1 = float((T1 ** 2).sum()); self.c_ent1 = float(xlogy(T1, T1).sum()); self.c_max1 = float(T1.max(1).sum())
        self.c_brier2 = float((T2 ** 2).sum()); self.c_ent2 = float(xlogy(T2, T2).sum()); self.c_max2 = float(T2.max(1).sum())

    def copy(self):
        s = Stats.__new__(Stats)
        for key, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                v = v.copy()
            elif key in ('ns', 'Y'):
                v = {g: x.copy() for g, x in v.items()}
            setattr(s, key, v)
        return s


def income_groups(n, S1, alive):
    """32 contiguous groups of live cells sorted by pooled P(income = 1) (ties by index), midpoint mass rule."""
    idx = np.flatnonzero(alive)
    p1 = S1[idx, 1] / n[idx]
    order = idx[np.lexsort((idx, p1))]
    cum = np.cumsum(n[order]) - n[order] / 2.0
    g = np.minimum(N_GROUPS - 1, np.floor(N_GROUPS * cum / n[alive].sum()).astype(np.int64))
    grp = np.full(len(n), -1, dtype=np.int64); grp[order] = g
    return grp


def grouped_income_costs(st, alive, grp=None):
    """Absolute plug-in income costs of the group-pooled release relative to the teacher: (brier, nll, acc), per person."""
    if grp is None:
        grp = income_groups(st.n, st.S1, alive)
    G = np.bincount(grp[alive], weights=st.n[alive], minlength=N_GROUPS)
    SG = np.zeros((N_GROUPS, st.S1.shape[1])); np.add.at(SG, grp[alive], st.S1[alive])
    used = G > 0
    v = SG[used] / G[used, None]
    brier = (st.c_brier1 - (G[used] * (v ** 2).sum(1)).sum()) / st.N
    nll = (st.c_ent1 - xlogy(SG[used], v).sum()) / st.N
    acc = (st.c_max1 - SG[used][np.arange(used.sum()), v.argmax(1)].sum()) / st.N
    return np.array([brier, nll, acc]), grp


def occupation_abs_costs(st, alive):
    n, S2 = st.n[alive], st.S2[alive]; mu = S2 / n[:, None]
    brier = (st.c_brier2 - (n * (mu ** 2).sum(1)).sum()) / st.N
    nll = (st.c_ent2 - xlogy(S2, mu).sum()) / st.N
    acc = (st.c_max2 - S2[np.arange(len(n)), mu.argmax(1)].sum()) / st.N
    return np.array([brier, nll, acc])


def guard_abs(st, alive):
    """Plug-in within-SEX occupation Brier of the cell-pooled release under the fixed fine-cell shrunk label law:
    (1/N_s) sum_c [n_cs (||mu_c||^2 + 1) - 2 mu_c . Ytil_cs]."""
    out = []
    mu = st.S2[alive] / st.n[alive, None]
    for g in (0, 1):
        ns = st.ns[g][alive]
        out.append(float((ns * ((mu ** 2).sum(1) + 1.0) - 2 * (mu * st.Y[g][alive]).sum(1)).sum() / st.Ns[g]))
    return np.array(out)


def pair_deltas(st, I, J):
    """Cell-level exact changes for merging I[k] with J[k] (vectorised)."""
    na, nb = st.n[I], st.n[J]; nm = na + nb; N = st.N
    out = {}
    for name, S in (('1', st.S1), ('2', st.S2)):
        ma, mb = S[I] / na[:, None], S[J] / nb[:, None]; mm = (S[I] + S[J]) / nm[:, None]
        out['brier' + name] = na * nb / nm * ((ma - mb) ** 2).sum(1) / N
        out['nll' + name] = (na * (xlogy(ma, ma) - xlogy(ma, mm)).sum(1) + nb * (xlogy(mb, mb) - xlogy(mb, mm)).sum(1)) / N
        out['acc' + name] = (na * ma.max(1) + nb * mb.max(1) - nm * mm.max(1)) / N
        if name == '2':
            mu_a, mu_b, mu_m = ma, mb, mm
    for g in (0, 1):   # exact and additive under the fixed fine-cell shrunk label law
        nsa, nsb = st.ns[g][I], st.ns[g][J]; Ya, Yb = st.Y[g][I], st.Y[g][J]
        out[f'guard{g}'] = ((nsa + nsb) * (mu_m ** 2).sum(1) - nsa * (mu_a ** 2).sum(1) - nsb * (mu_b ** 2).sum(1)
                            - 2 * ((mu_m * (Ya + Yb)).sum(1) - (mu_a * Ya).sum(1) - (mu_b * Yb).sum(1))) / st.Ns[g]
    pa, pb = st.SP[I] / na, st.SP[J] / nb; pm = (st.SP[I] + st.SP[J]) / nm
    out['sexvar'] = na * nb / nm * (pa - pb) ** 2 / N
    def klb(x, y):
        y = np.clip(y, 1e-15, 1 - 1e-15)
        return xlogy(x, x) - xlogy(x, y) + xlogy(1 - x, 1 - x) - xlogy(1 - x, 1 - y)
    out['sexinfo'] = (na * klb(pa, pm) + nb * klb(pb, pm)) / N
    return out


def merge_into(st, a, b):
    st.n[a] += st.n[b]; st.S1[a] += st.S1[b]; st.S2[a] += st.S2[b]; st.SP[a] += st.SP[b]
    for g in (0, 1):
        st.ns[g][a] += st.ns[g][b]; st.Y[g][a] += st.Y[g][b]
        st.ns[g][b] = 0; st.Y[g][b] = 0
    st.n[b] = 0; st.S1[b] = 0; st.S2[b] = 0; st.SP[b] = 0


BUDGET_KEYS = ('brier1', 'nll1', 'acc1', 'brier2', 'nll2', 'acc2', 'guard0', 'guard1')


def budgets(beta):
    return {'brier1': beta * 0.005, 'nll1': beta * 0.010, 'acc1': 0.005, 'brier2': beta * 0.005, 'nll2': beta * 0.010, 'acc2': 0.005,
            'guard0': beta * 0.005, 'guard1': beta * 0.005}


def run_path(st0, kind, beta=None, gamma=None, target=None, max_steps=None):
    """One merge path from the fine cells. kind in {'SCM', 'TWIN', 'IBSI', 'SHUF'} (SHUF = SCM on permuted SEX rates; the caller
    permutes st.SP). Returns (merge_map, record)."""
    st = st0.copy(); k = st.k; alive = st.n > 0; parent = np.arange(k)
    capped = kind in ('SCM', 'TWIN', 'SHUF')
    B = budgets(beta) if capped else None
    inc0, _ = grouped_income_costs(st, alive); occ0 = occupation_abs_costs(st, alive); g0 = guard_abs(st, alive)
    spent = {key: 0.0 for key in BUDGET_KEYS}
    steps, reason = 0, 'exhausted'
    while True:
        live = np.flatnonzero(alive)
        if target is not None and len(live) <= target:
            reason = 'target_reached'; break
        if max_steps is not None and steps >= max_steps:
            reason = 'max_steps'; break
        if len(live) < 2:
            reason = 'single_cell'; break
        I, J = np.triu_indices(len(live), 1); I, J = live[I], live[J]
        d = pair_deltas(st, I, J)
        if capped:
            R = {key: max(B[key] - spent[key], 1e-15) for key in BUDGET_KEYS}
            norm = sum(np.maximum(d[key], 0.0) / R[key] for key in BUDGET_KEYS)
            pri = (d['sexvar'] / (norm + 1e-12)) if kind in ('SCM', 'SHUF') else -norm
            order = np.lexsort((J, I, -pri))
        else:   # IBSI: minimise information-loss Lagrangian
            score = d['nll1'] + d['nll2'] - gamma * d['sexinfo']
            order = np.lexsort((J, I, score))
        accepted = None
        if capped:   # vectorised prefilter on the cell-level occupation and guard budgets (exact; order preserved)
            ok = np.ones(len(I), dtype=bool)
            for key in ('brier2', 'nll2', 'acc2', 'guard0', 'guard1'):
                ok &= spent[key] + d[key] <= B[key] + 1e-15
            order = order[ok[order]]
        for o in order:
            if capped:
                a, b = int(I[o]), int(J[o])
                trial = st.copy(); merge_into(trial, a, b); al = alive.copy(); al[b] = False
                inc, _ = grouped_income_costs(trial, al)
                dinc = inc - inc0
                if dinc[0] > B['brier1'] + 1e-15 or dinc[1] > B['nll1'] + 1e-15 or dinc[2] > B['acc1'] + 1e-15:
                    continue
                accepted = (o, a, b, dinc); break
            else:
                accepted = (o, int(I[o]), int(J[o]), None); break
        if accepted is None:
            reason = 'no_admissible_pair'; break
        o, a, b, dinc = accepted
        for key in ('brier2', 'nll2', 'acc2', 'guard0', 'guard1'):
            spent[key] += float(d[key][o])
        merge_into(st, a, b); alive[b] = False; parent[b] = a; steps += 1
        if capped:
            spent['brier1'], spent['nll1'], spent['acc1'] = (float(x) for x in dinc)
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    final = np.array([find(i) for i in range(k)], dtype=np.int64)
    inc, grp = grouped_income_costs(st, alive); occ = occupation_abs_costs(st, alive); gd = guard_abs(st, alive)
    rec = {'kind': kind, 'beta': beta, 'gamma': gamma, 'target': target, 'steps': steps, 'stop_reason': reason,
           'final_cells': int(alive.sum()), 'spent_plugin_vs_C0': {**{k_: float(v) for k_, v in spent.items()},
                                                                 'income_grouped_brier_nll_acc': [float(x) for x in inc - inc0],
                                                                 'occupation_brier_nll_acc': [float(x) for x in occ - occ0],
                                                                 'guard0_guard1': [float(x) for x in gd - g0]}}
    return final, grp, st, alive, rec


def tables(st, alive, final, grp):
    """Release tables: occupation vector per final cell, income vector per group (stored once per group)."""
    v2 = np.zeros_like(st.S2); v2[alive] = st.S2[alive] / st.n[alive, None]
    G = np.bincount(grp[alive], weights=st.n[alive], minlength=N_GROUPS)
    SG = np.zeros((N_GROUPS, st.S1.shape[1])); np.add.at(SG, grp[alive], st.S1[alive])
    v1 = np.zeros_like(SG); used = G > 0; v1[used] = SG[used] / G[used, None]
    cell_of_initial = final
    group_of_initial = grp[final]
    return {'v1': v1, 'v2': v2, 'cell_of_initial': cell_of_initial, 'group_of_initial': group_of_initial}


def release(tab, initial_cells):
    c = tab['cell_of_initial'][initial_cells]; g = tab['group_of_initial'][initial_cells]
    return {'income': tab['v1'][g].copy(), 'occupation': tab['v2'][c].copy()}
