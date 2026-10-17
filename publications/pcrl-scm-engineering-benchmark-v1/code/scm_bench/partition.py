"""Shared-cell merging engine (data-free core; the pilot's mechanism module calls it on MODEL_FIT statistics).

Cells carry sufficient statistics over their members: count n, sums of the teacher income and occupation posteriors, the
sum of the out-of-fold SEX propensity, and (optionally) SEX-specific occupation-label sums for the within-SEX guard.
For a release that gives each cell its pooled teacher posterior, merging cells a and b into m changes the plug-in task
metrics (teacher posteriors taken as truth) exactly by:
  Brier_i : n_a n_b / n_m * ||mu_ia - mu_ib||^2 / N                         (Ward)
  NLL_i   : [n_a KL(mu_ia || mu_im) + n_b KL(mu_ib || mu_im)] / N
  Acc_i   : [n_a max mu_ia + n_b max mu_ib - n_m max mu_im] / N
and the coalition SEX criteria by
  variance: n_a n_b / n_m * (pi_a - pi_b)^2 / N                            (between-cell variance of the SEX rate)
  info    : [n_a KL_B(pi_a || pi_m) + n_b KL_B(pi_b || pi_m)] / N         (plug-in mutual information, nats)
All identities are checked against brute-force recomputation in test_partition_synthetic.py.
"""
from __future__ import annotations
import heapq

import numpy as np

EPS = 1e-300


def kl_rows(p, q):
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(p > 0, p * (np.log(np.maximum(p, EPS)) - np.log(np.maximum(q, EPS))), 0.0)
    return t.sum(-1)


def kl_bern(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.clip(np.asarray(b, dtype=np.float64), 1e-15, 1 - 1e-15)
    return kl_rows(np.stack((1 - a, a), -1), np.stack((1 - b, b), -1))


class Cells:
    """Sufficient statistics per cell. t1: (N, 2) income posteriors; t2: (N, 5); pi: (N,) out-of-fold SEX propensity;
    guard: optional dict with 'sex' (N,) labels and 'occ' (N,) occupation labels for SEX-specific occupation means."""

    def __init__(self, assign, n_cells, t1, t2, pi=None, guard=None):
        a = np.asarray(assign, dtype=np.int64); self.N = len(a); self.k0 = n_cells
        self.n = np.bincount(a, minlength=n_cells).astype(np.float64)
        self.s1 = np.zeros((n_cells, t1.shape[1])); np.add.at(self.s1, a, t1)
        self.s2 = np.zeros((n_cells, t2.shape[1])); np.add.at(self.s2, a, t2)
        self.spi = np.bincount(a, weights=pi, minlength=n_cells) if pi is not None else None
        self.g = None; self.Ns = None
        if guard is not None:
            occ1h = np.eye(t2.shape[1])[guard['occ']]
            self.g = {}; self.Ns = {s: float((np.asarray(guard['sex']) == s).sum()) for s in (0, 1)}
            for s in (0, 1):
                m = guard['sex'] == s
                self.g[s] = {'n': np.bincount(a[m], minlength=n_cells).astype(np.float64)}
                self.g[s]['y'] = np.zeros((n_cells, t2.shape[1])); np.add.at(self.g[s]['y'], a[m], occ1h[m])
                self.g[s]['t'] = np.zeros((n_cells, t2.shape[1])); np.add.at(self.g[s]['t'], a[m], t2[m])
        self.alive = self.n > 0

    def mean(self, i):
        n = np.maximum(self.n[i], 1e-300)[..., None]
        return self.s1[i] / n, self.s2[i] / n


def merge_delta(C, a, b):
    """Exact plug-in changes for merging cells a and b (arrays of equal length allowed)."""
    a = np.atleast_1d(a); b = np.atleast_1d(b)
    na, nb = C.n[a], C.n[b]; nm = na + nb
    out = {}
    for i, s in ((1, C.s1), (2, C.s2)):
        ma, mb = s[a] / na[:, None], s[b] / nb[:, None]; mm = (s[a] + s[b]) / nm[:, None]
        out[f'brier{i}'] = na * nb / nm * ((ma - mb) ** 2).sum(1) / C.N
        out[f'nll{i}'] = (na * kl_rows(ma, mm) + nb * kl_rows(mb, mm)) / C.N
        out[f'acc{i}'] = (na * ma.max(1) + nb * mb.max(1) - nm * mm.max(1)) / C.N
    if C.spi is not None:
        pa, pb = C.spi[a] / na, C.spi[b] / nb; pm = (C.spi[a] + C.spi[b]) / nm
        out['sexvar'] = na * nb / nm * (pa - pb) ** 2 / C.N
        out['sexinfo'] = (na * kl_bern(pa, pm) + nb * kl_bern(pb, pm)) / C.N
    if C.g is not None:
        # within-SEX occupation summed-Brier increase when both groups' members get the pooled cell posterior instead of
        # their own cell's pooled posterior (plug-in with occupation labels), per SEX group, divided by group size
        ma2, mb2 = C.s2[a] / na[:, None], C.s2[b] / nb[:, None]; mm2 = (C.s2[a] + C.s2[b]) / nm[:, None]
        for s in (0, 1):
            G = C.g[s]; Ns = C.Ns[s]

            def brier_sum(nn, ysum, mu):
                # sum over members of ||mu - e_y||^2 = nn*||mu||^2 - 2 mu.ysum + nn
                return nn * (mu ** 2).sum(1) - 2 * (mu * ysum).sum(1) + nn
            before = brier_sum(G['n'][a], G['y'][a], ma2) + brier_sum(G['n'][b], G['y'][b], mb2)
            after = brier_sum(G['n'][a] + G['n'][b], G['y'][a] + G['y'][b], mm2)
            out[f'guard{s}'] = (after - before) / Ns
    return out


def apply_merge(C, a, b):
    C.n[a] += C.n[b]; C.s1[a] += C.s1[b]; C.s2[a] += C.s2[b]
    if C.spi is not None:
        C.spi[a] += C.spi[b]
    if C.g is not None:
        for s in (0, 1):
            for k in ('n', 'y', 't'):
                C.g[s][k][a] += C.g[s][k][b]
    C.n[b] = 0; C.s1[b] = 0; C.s2[b] = 0
    if C.spi is not None:
        C.spi[b] = 0
    if C.g is not None:
        for s in (0, 1):
            for k in ('n', 'y', 't'):
                C.g[s][k][b] = 0
    C.alive[b] = False


def greedy(C, edges, score_fn, admissible_fn, stop_count=None, max_steps=None):
    """Lazy-heap greedy agglomeration on a candidate graph.

    edges: iterable of (a, b) with a < b over initial cells (merged cells inherit their members' edges).
    score_fn(delta) -> float (smaller is merged first); admissible_fn(delta, spent) -> bool.
    Stops at stop_count live cells, at max_steps merges, or when no admissible edge remains.
    Returns (merge_map from initial cell to final cell, spent cost dict, steps, stop_reason)."""
    parent = np.arange(C.k0)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    nbr = {}
    for a, b in edges:
        nbr.setdefault(a, set()).add(b); nbr.setdefault(b, set()).add(a)
    version = np.zeros(C.k0, dtype=np.int64)
    heap = []

    def push(a, b):
        if a == b or not (C.alive[a] and C.alive[b]):
            return
        a, b = (a, b) if a < b else (b, a)
        d = {k: float(v[0]) for k, v in merge_delta(C, a, b).items()}
        heapq.heappush(heap, (score_fn(d), a, b, int(version[a]), int(version[b]), d))
    for a in list(nbr):
        for b in nbr[a]:
            if a < b:
                push(a, b)
    spent = {}
    steps, live = 0, int(C.alive.sum())
    reason = 'exhausted'
    while heap:
        if stop_count is not None and live <= stop_count:
            reason = 'stop_count'; break
        if max_steps is not None and steps >= max_steps:
            reason = 'max_steps'; break
        sc, a, b, va, vb, d = heapq.heappop(heap)
        if not (C.alive[a] and C.alive[b]) or version[a] != va or version[b] != vb:
            continue
        if not admissible_fn(d, spent):
            continue
        apply_merge(C, a, b); parent[b] = a; version[a] += 1; live -= 1; steps += 1
        for k, v in d.items():
            spent[k] = spent.get(k, 0.0) + v
        nb = (nbr.pop(a, set()) | nbr.pop(b, set())) - {a, b}
        nb = {find(x) for x in nb} - {a}
        nbr[a] = nb
        for x in nb:
            nbr.setdefault(x, set()).discard(b); nbr[x].add(a)
            push(a, x)
    else:
        reason = 'exhausted'
    final = np.array([find(i) for i in range(C.k0)], dtype=np.int64)
    return final, spent, steps, reason


def kmeans(Q, k, seed, iters=50, chunk=2048):
    """Euclidean k-means with k-means++ initialisation (deterministic given seed); returns centroids and assignment."""
    Q = np.ascontiguousarray(Q, dtype=np.float64); n = len(Q); rng = np.random.default_rng(seed)
    cent = np.empty((k, Q.shape[1])); cent[0] = Q[rng.integers(n)]
    d2 = ((Q - cent[0]) ** 2).sum(1)
    for j in range(1, k):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        cent[j] = Q[rng.choice(n, p=p)]
        d2 = np.minimum(d2, ((Q - cent[j]) ** 2).sum(1))
    assign = nearest(Q, cent, chunk)
    for _ in range(iters):
        sums = np.zeros_like(cent); np.add.at(sums, assign, Q); cnt = np.bincount(assign, minlength=k)
        keep = cnt > 0
        cent[keep] = sums[keep] / cnt[keep, None]
        new = nearest(Q, cent, chunk)
        if np.array_equal(new, assign):
            break
        assign = new
    return cent, assign


def nearest(Q, cent, chunk=2048):
    out = np.empty(len(Q), dtype=np.int64); c2 = (cent ** 2).sum(1)
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk]
        d = c2[None, :] - 2 * q @ cent.T
        out[s:s + chunk] = d.argmin(1)          # ties go to the lowest index
    return out
