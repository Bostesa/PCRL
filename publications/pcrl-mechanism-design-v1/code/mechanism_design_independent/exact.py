"""Minimal exact finite-law toolkit, written independently of mechanism_design.prototype.finite_law.

A law is a list of cells. Each cell carries mass w, pi = P(S=1 | x) and, for task t in {1, 2}, the within-SEX task
posteriors q[t][s] = P(Y_t | x, S=s) (tuples). The SEX-blind task posterior is eta_t = (1-pi) q[t][0] + pi q[t][1].
Works with fractions.Fraction (exact) or floats.

Bayes AUC of S from a discrete view: cells are grouped by their released symbol; groups are sorted by the posterior
P(S=1 | symbol) (equivalently the likelihood ratio) and the ROC area is accumulated with ties counted 1/2. This is a
sort-based implementation (the coordinator's toolkit uses a pairwise sum), so agreement is an independent check.
"""
from __future__ import annotations

import math
from fractions import Fraction


class Cell:
    __slots__ = ('w', 'pi', 'q', 'name')

    def __init__(self, w, pi, q1, q2, name=None):
        self.w, self.pi, self.name = w, pi, name
        self.q = {1: (tuple(q1[0]), tuple(q1[1])), 2: (tuple(q2[0]), tuple(q2[1]))}

    def eta(self, t):
        a, b = self.q[t]
        return tuple((1 - self.pi) * x + self.pi * y for x, y in zip(a, b))


def check_law(cells, tol=1e-12):
    tot = sum(c.w for c in cells)
    assert abs(float(tot) - 1) < tol, tot
    for c in cells:
        assert 0 <= float(c.pi) <= 1
        for t in (1, 2):
            for s in (0, 1):
                assert abs(float(sum(c.q[t][s])) - 1) < tol


def pbar(cells):
    return sum(c.w * c.pi for c in cells)


def _norm(k):
    """Round every numeric leaf of a (nested) symbol to 11 decimals so float-equal releases share a symbol."""
    if isinstance(k, tuple):
        return tuple(_norm(x) for x in k)
    if isinstance(k, (float, Fraction)):
        return round(float(k), 11)
    return k


def bayes_auc(cells, symbol):
    """symbol(i) -> hashable released symbol of cell i. Exact for Fractions; floats are grouped at 1e-11."""
    exact = isinstance(cells[0].pi, Fraction)
    groups = {}
    for i, c in enumerate(cells):
        k = symbol(i) if exact else _norm(symbol(i))
        m1, m0 = groups.get(k, (0, 0))
        groups[k] = (m1 + c.w * c.pi, m0 + c.w * (1 - c.pi))
    P1 = sum(g[0] for g in groups.values())
    P0 = sum(g[1] for g in groups.values())
    rows = []
    for m1, m0 in groups.values():
        post = m1 / (m1 + m0) if (m1 + m0) > 0 else 0
        rows.append((post if exact else round(float(post), 12), m1 / P1, m0 / P0))
    rows.sort(key=lambda r: r[0])
    # accumulate: AUC = sum_i a1_i * (cum0 below i + 1/2 tie0)
    auc = 0
    cum0 = 0
    i = 0
    half = Fraction(1, 2) if exact else 0.5
    while i < len(rows):
        j = i
        a1 = 0
        a0 = 0
        while j < len(rows) and rows[j][0] == rows[i][0]:
            a1 += rows[j][1]
            a0 += rows[j][2]
            j += 1
        auc += a1 * (cum0 + half * a0)
        cum0 += a0
        i = j
    return auc


def auc_from_x(cells):
    return bayes_auc(cells, lambda i: ('cell', i))


def argmax_first(v):
    b = 0
    for k in range(1, len(v)):
        if v[k] > v[b]:
            b = k
    return b


def metrics(cells, t, rel, floor=1e-12):
    """rel[i] = released vector for cell i. Returns accuracy, NLL (float) and summed Brier vs the SEX-blind truth."""
    acc = 0
    nll = 0.0
    brier = 0
    for i, c in enumerate(cells):
        e = c.eta(t)
        r = rel[i]
        acc += c.w * e[argmax_first(r)]
        nll += float(c.w) * sum(float(e[y]) * -math.log(max(float(r[y]), floor)) for y in range(len(e)) if e[y] > 0)
        brier += c.w * (sum(x * x for x in r) - 2 * sum(x * y for x, y in zip(r, e)) + 1)
    return {'accuracy': acc, 'nll': nll, 'brier': brier}


def excess(cells, t, rel, ref):
    a, b = metrics(cells, t, rel), metrics(cells, t, ref)
    return {'accuracy_loss': b['accuracy'] - a['accuracy'], 'nll_excess': a['nll'] - b['nll'],
            'brier_excess': a['brier'] - b['brier']}


def within_sex_brier(cells, t, rel, s):
    num = 0
    den = 0
    for i, c in enumerate(cells):
        ps = c.pi if s == 1 else 1 - c.pi
        qs = c.q[t][s]
        r = rel[i]
        num += c.w * ps * (sum(x * x for x in r) - 2 * sum(x * y for x, y in zip(r, qs)) + 1)
        den += c.w * ps
    return num / den


def mutual_info_sx_nats(cells):
    pb = float(pbar(cells))
    tot = 0.0
    for c in cells:
        p = float(c.pi)
        for ps, pm in ((p, pb), (1 - p, 1 - pb)):
            if ps > 0:
                tot += float(c.w) * ps * math.log(ps / pm)
    return tot


def partitions(n):
    """All set partitions of range(n) via restricted growth strings (independent of the coordinator's recursion)."""
    a = [0] * n

    def rec(i, m):
        if i == n:
            blocks = {}
            for idx, b in enumerate(a):
                blocks.setdefault(b, []).append(idx)
            yield tuple(tuple(v) for v in blocks.values())
            return
        for b in range(m + 1):
            a[i] = b
            yield from rec(i + 1, max(m, b + 1))
    if n == 0:
        yield ()
        return
    yield from rec(0, 0)


def block_posterior(cells, t, part):
    out = [None] * len(cells)
    for blk in part:
        m = sum(cells[i].w for i in blk)
        k = len(cells[blk[0]].eta(t))
        v = tuple(sum(cells[i].w * cells[i].eta(t)[y] for i in blk) / m for y in range(k))
        for i in blk:
            out[i] = v
    return out
