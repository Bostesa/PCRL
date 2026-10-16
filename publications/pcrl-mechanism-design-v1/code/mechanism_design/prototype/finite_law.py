"""Exact finite-law quantities for data-free mechanism checks (no real data).

A finite law is a list of input cells c with probability mass w[c] and a joint conditional table
P(S, Y1, Y2 | c) of shape (2, K1, K2). A deterministic two-recipient release assigns each cell a probability vector
per recipient; the released symbol is the vector itself together with its argmax decision, so cells with identical
released vectors are indistinguishable to a reader.

Everything here works with fractions.Fraction (exact) or floats. The Bayes AUC of S from a discrete view is computed
from the likelihood-ratio ordering with ties counted 1/2; by the Neyman-Pearson lemma it is the largest AUC any
scoring function of that view can achieve, so it upper-bounds every fitted reader of the same view.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Hashable, Sequence

Number = Fraction | float


@dataclass(frozen=True)
class FiniteLaw:
    """w[c] = P(X = c); joint[c][s][y1][y2] = P(S = s, Y1 = y1, Y2 = y2 | X = c)."""
    w: tuple
    joint: tuple
    k1: int
    k2: int
    names: tuple = ()

    def __post_init__(self):
        if len(self.w) != len(self.joint):
            raise ValueError('mass and conditional tables differ in length')
        tot = sum(self.w)
        if abs(float(tot) - 1.0) > 1e-12:
            raise ValueError(f'cell masses sum to {tot}, not 1')
        for c, tab in enumerate(self.joint):
            if len(tab) != 2 or any(len(tab[s]) != self.k1 for s in range(2)) or any(len(r) != self.k2 for s in range(2) for r in tab[s]):
                raise ValueError(f'cell {c}: table shape is not (2, K1, K2)')
            if abs(float(sum(tab[s][a][b] for s in range(2) for a in range(self.k1) for b in range(self.k2))) - 1.0) > 1e-12:
                raise ValueError(f'cell {c}: conditional table does not sum to 1')

    @property
    def cells(self) -> range:
        return range(len(self.w))

    def p_s(self, c: int, s: int = 1) -> Number:
        return sum(self.joint[c][s][a][b] for a in range(self.k1) for b in range(self.k2))

    def p_y(self, c: int, task: int) -> list:
        """Task posterior T_task(c) = P(Y_task | X = c) as a list."""
        if task == 1:
            return [sum(self.joint[c][s][a][b] for s in range(2) for b in range(self.k2)) for a in range(self.k1)]
        return [sum(self.joint[c][s][a][b] for s in range(2) for a in range(self.k1)) for b in range(self.k2)]

    def p_y_given_s(self, c: int, task: int, s: int) -> list:
        """Unnormalised P(Y_task, S = s | X = c)."""
        if task == 1:
            return [sum(self.joint[c][s][a][b] for b in range(self.k2)) for a in range(self.k1)]
        return [sum(self.joint[c][s][a][b] for a in range(self.k1)) for b in range(self.k2)]


def argmax_first(v: Sequence[Number]) -> int:
    """Deterministic decision: the lowest index among the maxima."""
    best = 0
    for i in range(1, len(v)):
        if v[i] > v[best]:
            best = i
    return best


def _key(v: Sequence[Number]) -> tuple:
    return tuple(v)


def view_symbols(release: Callable[[int], Hashable], law: FiniteLaw) -> dict:
    """Map each released symbol to the list of cells that emit it."""
    groups: dict = {}
    for c in law.cells:
        groups.setdefault(release(c), []).append(c)
    return groups


def bayes_auc(law: FiniteLaw, release: Callable[[int], Hashable]) -> Number:
    """Exact Bayes AUC for S (positive class S = 1) from the discrete view `release(c)`.

    AUC = sum over symbol pairs (u, v) of P(u | S=1) P(v | S=0) [1(LR(u) > LR(v)) + 1/2 1(LR(u) = LR(v))],
    with symbols ordered by the likelihood ratio, compared by cross-multiplication (exact for Fractions).
    """
    groups = view_symbols(release, law)
    p1 = sum(law.w[c] * law.p_s(c, 1) for c in law.cells)
    p0 = 1 - p1
    if p1 == 0 or p0 == 0:
        raise ValueError('S is degenerate under this law')
    mass = []
    for cells in groups.values():
        m1 = sum(law.w[c] * law.p_s(c, 1) for c in cells) / p1
        m0 = sum(law.w[c] * law.p_s(c, 0) for c in cells) / p0
        mass.append((m1, m0))
    auc = 0 * p1
    half = Fraction(1, 2) if isinstance(p1, Fraction) else 0.5
    for (a1, a0) in mass:
        for (b1, b0) in mass:
            # LR(a) > LR(b)  <=>  a1 * b0 > b1 * a0  (cross-multiplied; zero masses handled consistently)
            lhs, rhs = a1 * b0, b1 * a0
            if lhs > rhs:
                auc += a1 * b0
            elif lhs == rhs:
                auc += half * a1 * b0
    return auc


def mutual_information_bits(law: FiniteLaw, release: Callable[[int], Hashable]) -> float:
    """I(S; view) in bits (float)."""
    groups = view_symbols(release, law)
    p1 = float(sum(law.w[c] * law.p_s(c, 1) for c in law.cells))
    ps = (1 - p1, p1)
    mi = 0.0
    for cells in groups.values():
        pv = float(sum(law.w[c] for c in cells))
        for s in (0, 1):
            pjs = float(sum(law.w[c] * law.p_s(c, s) for c in cells))
            if pjs > 0:
                mi += pjs * math.log2(pjs / (pv * ps[s]))
    return mi


def task_metrics(law: FiniteLaw, task: int, q: Callable[[int], Sequence[Number]], floor: float = 1e-12) -> dict:
    """Expected accuracy, NLL (nats, float) and summed Brier of released vectors q(c) for one task.

    Matches acs_replication/data.py: accuracy uses argmax (lowest index on ties); NLL = -log max(p_y, 1e-12);
    Brier = sum_k (p_k - 1[y = k])^2.
    """
    acc = 0
    nll = 0.0
    brier = 0
    for c in law.cells:
        t = law.p_y(c, task)
        p = q(c)
        d = argmax_first(p)
        acc += law.w[c] * t[d]
        nll += float(law.w[c]) * sum(float(t[y]) * -math.log(max(float(p[y]), floor)) for y in range(len(t)) if t[y] > 0)
        brier += law.w[c] * sum(t[y] * sum((p[k] - (1 if k == y else 0)) ** 2 for k in range(len(p))) for y in range(len(t)))
    return {'accuracy': acc, 'nll': nll, 'brier': brier}


def clause_excess(law: FiniteLaw, task: int, q, ref) -> dict:
    """Accuracy loss, NLL excess and summed Brier excess of release q against reference ref (both cell maps)."""
    a, r = task_metrics(law, task, q), task_metrics(law, task, ref)
    return {'accuracy_loss': r['accuracy'] - a['accuracy'], 'nll_excess': a['nll'] - r['nll'], 'brier_excess': a['brier'] - r['brier']}


def posterior_release(law: FiniteLaw, task: int, partition: Sequence[Sequence[int]]) -> dict:
    """The calibrated release that is constant on each block: q(c) = P(Y_task | block(c)) (exact mass-weighted mean)."""
    out = {}
    for block in partition:
        m = sum(law.w[c] for c in block)
        k = law.k1 if task == 1 else law.k2
        v = tuple(sum(law.w[c] * law.p_y(c, task)[y] for c in block) / m for y in range(k))
        for c in block:
            out[c] = v
    return out


def set_partitions(items: Sequence[int]):
    """All set partitions of `items` (Bell-number many), as tuples of tuples, in a fixed order."""
    items = list(items)
    if not items:
        yield ()
        return
    first, rest = items[0], items[1:]
    for part in set_partitions(rest):
        yield ((first,),) + part
        for i in range(len(part)):
            yield part[:i] + ((first,) + part[i],) + part[i + 1:]
