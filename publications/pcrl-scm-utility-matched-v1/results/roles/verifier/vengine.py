"""Verifier's own merge-cost implementation, written from the source PROTOCOL.md sections 3-5 (not from engine.py).

State: per-cell MODEL_FIT counts and sums of the out-of-fold teacher vectors and SEX propensity; per SEX the counts and the
kappa-shrunk label sums fixed once at the fine cells (kappa = 20). Every cost is plug-in under the teacher OOF law.
Two independent routes are provided for occupation and guard costs: the exact per-merge change ("delta" route, the protocol
identities) and the difference of absolute plug-in costs ("absolute" route); the verifier compares them along each path.
"""
from __future__ import annotations

import numpy as np

KAPPA = 20.0
NG = 32
TINY = 1e-300
BUDGET = {'brier1': 0.005, 'nll1': 0.010, 'acc1': 0.005, 'brier2': 0.005, 'nll2': 0.010, 'acc2': 0.005, 'guard0': 0.005, 'guard1': 0.005}
KEYS = ('brier1', 'nll1', 'acc1', 'brier2', 'nll2', 'acc2', 'guard0', 'guard1')
CELL_KEYS = ('brier2', 'nll2', 'acc2', 'guard0', 'guard1')
SLACK = 1e-15


def plogq(p, q):
    """p log q with the convention 0 log q = 0 and log clipped at 1e-300."""
    out = np.zeros(np.broadcast(p, q).shape)
    pos = np.broadcast_to(p, out.shape) > 0
    pp = np.broadcast_to(p, out.shape); qq = np.broadcast_to(q, out.shape)
    out[pos] = pp[pos] * np.log(np.maximum(qq[pos], TINY))
    return out


class St:
    def __init__(self, assign, t1, t2, pi, sex, occ, k):
        a = np.asarray(assign)
        self.k = k; self.N = float(len(a))
        self.n = np.zeros(k); np.add.at(self.n, a, 1.0)
        self.S1 = np.zeros((k, t1.shape[1])); np.add.at(self.S1, a, t1)
        self.S2 = np.zeros((k, t2.shape[1])); np.add.at(self.S2, a, t2)
        self.SP = np.zeros(k); np.add.at(self.SP, a, pi)
        self.Ns = [float((sex == g).sum()) for g in (0, 1)]
        self.ns = []; self.Y = []
        mu_f = self.S2 / np.maximum(self.n, TINY)[:, None]
        for g in (0, 1):
            sel = sex == g
            cnt = np.zeros(k); np.add.at(cnt, a[sel], 1.0)
            lab = np.zeros((k, t2.shape[1])); np.add.at(lab, (a[sel], occ[sel]), 1.0)
            shr = np.zeros((k, t2.shape[1]))
            m = cnt > 0
            shr[m] = (lab[m] + KAPPA * mu_f[m]) / (cnt[m] + KAPPA)[:, None]          # y-tilde_{f,s}
            self.ns.append(cnt); self.Y.append(cnt[:, None] * shr)                   # stored sums n_{f,s} y-tilde
        self.c_sq1 = float((t1 ** 2).sum()); self.c_ent1 = float(plogq(t1, t1).sum()); self.c_max1 = float(t1.max(1).sum())
        self.c_sq2 = float((t2 ** 2).sum()); self.c_ent2 = float(plogq(t2, t2).sum()); self.c_max2 = float(t2.max(1).sum())
        self.alive = self.n > 0
        self.parent = np.arange(k)

    def clone(self):
        o = St.__new__(St)
        for key, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                v = v.copy()
            elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                v = [x.copy() for x in v]
            setattr(o, key, v)
        return o

    def merge(self, a, b):
        assert self.alive[a] and self.alive[b] and a != b
        self.n[a] += self.n[b]; self.S1[a] += self.S1[b]; self.S2[a] += self.S2[b]; self.SP[a] += self.SP[b]
        for g in (0, 1):
            self.ns[g][a] += self.ns[g][b]; self.Y[g][a] += self.Y[g][b]
            self.ns[g][b] = 0.0; self.Y[g][b] = 0.0
        self.n[b] = 0.0; self.S1[b] = 0.0; self.S2[b] = 0.0; self.SP[b] = 0.0
        self.alive[b] = False; self.parent[b] = a

    def roots(self):
        r = np.arange(self.k)
        for _ in range(self.k):
            nxt = self.parent[r]
            if np.array_equal(nxt, r):
                break
            r = nxt
        return r


def pair_costs(st, I, J):
    """Exact per-merge plug-in changes (section 4 identities), vectorised over pairs."""
    na, nb = st.n[I], st.n[J]; nm = na + nb; N = st.N
    d = {}
    for tag, S in (('1', st.S1), ('2', st.S2)):
        ma = S[I] / na[:, None]; mb = S[J] / nb[:, None]; mm = (S[I] + S[J]) / nm[:, None]
        d['brier' + tag] = (na * nb / nm) * ((ma - mb) ** 2).sum(1) / N
        d['nll' + tag] = (na * (plogq(ma, ma) - plogq(ma, mm)).sum(1) + nb * (plogq(mb, mb) - plogq(mb, mm)).sum(1)) / N
        d['acc' + tag] = (na * ma.max(1) + nb * mb.max(1) - nm * mm.max(1)) / N
        if tag == '2':
            oa, ob, om = ma, mb, mm
    for g in (0, 1):
        sa, sb = st.ns[g][I], st.ns[g][J]; Ya, Yb = st.Y[g][I], st.Y[g][J]
        d[f'guard{g}'] = ((sa + sb) * (om ** 2).sum(1) - sa * (oa ** 2).sum(1) - sb * (ob ** 2).sum(1)
                          - 2.0 * ((om * (Ya + Yb)).sum(1) - (oa * Ya).sum(1) - (ob * Yb).sum(1))) / st.Ns[g]
    pa, pb = st.SP[I] / na, st.SP[J] / nb; pm = (st.SP[I] + st.SP[J]) / nm
    d['sexvar'] = (na * nb / nm) * (pa - pb) ** 2 / N
    pmc = np.clip(pm, 1e-15, 1 - 1e-15)

    def kb(x, y):
        return plogq(x, x) - plogq(x, y) + plogq(1 - x, 1 - x) - plogq(1 - x, 1 - y)
    d['sexinfo'] = (na * kb(pa, pmc) + nb * kb(pb, pmc)) / N
    return d


def income_groups(n, S1, alive):
    """Section 4: live cells sorted by pooled P(income = 1) ascending, ties by index; g = min(31, floor(32 (M_before + n_c/2)/N))."""
    idx = np.flatnonzero(alive)
    p1 = S1[idx, 1] / n[idx]
    order = idx[np.lexsort((idx, p1))]
    mass = n[order]
    before = np.cumsum(mass) - mass
    g = np.minimum(NG - 1, np.floor(NG * (before + mass / 2.0) / n[alive].sum()).astype(np.int64))
    grp = np.full(len(n), -1, dtype=np.int64); grp[order] = g
    return grp


def grouped_income_abs(st, n=None, S1=None, alive=None):
    """Absolute plug-in income (Brier, NLL, accuracy loss) of the 32-group release, relative to the teacher, per person."""
    n = st.n if n is None else n; S1 = st.S1 if S1 is None else S1; alive = st.alive if alive is None else alive
    grp = income_groups(n, S1, alive)
    G = np.zeros(NG); SG = np.zeros((NG, S1.shape[1]))
    live = np.flatnonzero(alive)
    np.add.at(G, grp[live], n[live]); np.add.at(SG, grp[live], S1[live])
    u = G > 0
    v = SG[u] / G[u, None]
    brier = (st.c_sq1 - (G[u] * (v ** 2).sum(1)).sum()) / st.N
    nll = (st.c_ent1 - plogq(SG[u], v).sum()) / st.N
    acc = (st.c_max1 - SG[u][np.arange(u.sum()), v.argmax(1)].sum()) / st.N
    return np.array([brier, nll, acc]), grp


def occupation_abs(st):
    a = st.alive; n = st.n[a]; S2 = st.S2[a]; mu = S2 / n[:, None]
    return np.array([(st.c_sq2 - (n * (mu ** 2).sum(1)).sum()) / st.N, (st.c_ent2 - plogq(S2, mu).sum()) / st.N,
                     (st.c_max2 - S2[np.arange(len(n)), mu.argmax(1)].sum()) / st.N])


def guard_abs(st):
    a = st.alive; mu = st.S2[a] / st.n[a, None]
    return np.array([float((st.ns[g][a] * ((mu ** 2).sum(1) + 1.0) - 2.0 * (mu * st.Y[g][a]).sum(1)).sum() / st.Ns[g]) for g in (0, 1)])


def tables(st):
    """Release tables: occupation vector per final cell, income vector per group (stored once per group)."""
    a = st.alive
    v2 = np.zeros_like(st.S2); v2[a] = st.S2[a] / st.n[a, None]
    grp = income_groups(st.n, st.S1, a)
    live = np.flatnonzero(a)
    G = np.zeros(NG); SG = np.zeros((NG, st.S1.shape[1]))
    np.add.at(G, grp[live], st.n[live]); np.add.at(SG, grp[live], st.S1[live])
    v1 = np.zeros_like(SG); u = G > 0; v1[u] = SG[u] / G[u, None]
    roots = st.roots()
    return {'v1': v1, 'v2': v2, 'cell_of_initial': roots, 'group_of_initial': grp[roots]}
