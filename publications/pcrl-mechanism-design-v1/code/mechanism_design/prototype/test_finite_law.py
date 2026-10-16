"""Self-tests of the exact finite-law toolkit (data-free)."""
from fractions import Fraction as F
import itertools
import random

from mechanism_design import no_real_data
no_real_data.install()
from mechanism_design.prototype.finite_law import FiniteLaw, bayes_auc, mutual_information_bits, task_metrics, posterior_release, set_partitions, clause_excess


def law_from(cells, k1=2, k2=2):
    """cells: list of (mass, {(s, y1, y2): prob})."""
    w, joint = [], []
    for m, tab in cells:
        t = [[[F(0)] * k2 for _ in range(k1)] for _ in range(2)]
        for (s, a, b), p in tab.items():
            t[s][a][b] = F(p)
        w.append(F(m)); joint.append(tuple(tuple(tuple(r) for r in t[s]) for s in range(2)))
    return FiniteLaw(tuple(w), tuple(joint), k1, k2)


# 1. perfectly separating view -> AUC 1; constant view -> AUC 1/2
L = law_from([(F(1, 2), {(1, 0, 0): 1}), (F(1, 2), {(0, 0, 0): 1})])
assert bayes_auc(L, lambda c: c) == 1
assert bayes_auc(L, lambda c: 0) == F(1, 2)
# 2. AUC equals P(score_pos > score_neg) + 1/2 P(tie) by brute force on a random law with the posterior as score
random.seed(3)
for trial in range(50):
    C = random.randint(2, 7)
    cells = []
    ms = [F(random.randint(1, 9)) for _ in range(C)]; tot = sum(ms)
    for c in range(C):
        a = F(random.randint(0, 10), 10)
        cells.append((ms[c] / tot, {(1, 0, 0): a, (0, 0, 0): 1 - a}))
    L = law_from(cells)
    view = [random.randint(0, 3) for _ in range(C)]
    # brute force: score = posterior of the symbol; pairs of independent draws
    p1 = sum(L.w[c] * L.p_s(c, 1) for c in L.cells)
    post = {}
    for v in set(view):
        cs = [c for c in L.cells if view[c] == v]
        mv = sum(L.w[c] for c in cs)
        post[v] = sum(L.w[c] * L.p_s(c, 1) for c in cs) / mv
    brute = F(0)
    for i in L.cells:
        for j in L.cells:
            wi = L.w[i] * L.p_s(i, 1) / p1; wj = L.w[j] * L.p_s(j, 0) / (1 - p1)
            si, sj = post[view[i]], post[view[j]]
            brute += wi * wj * (1 if si > sj else (F(1, 2) if si == sj else 0))
    assert bayes_auc(L, lambda c: view[c]) == brute, trial
    # 3. data processing: merging symbols cannot increase the Bayes AUC; the full cell view is maximal
    merged = [v // 2 for v in view]
    assert bayes_auc(L, lambda c: merged[c]) <= bayes_auc(L, lambda c: view[c]) <= bayes_auc(L, lambda c: c)
    assert mutual_information_bits(L, lambda c: merged[c]) <= mutual_information_bits(L, lambda c: view[c]) + 1e-12
# 4. utility: the Bayes posterior release minimises NLL and Brier among partition releases; Brier excess decomposes
L = law_from([(F(1, 4), {(0, 1, 0): F(3, 4), (1, 0, 0): F(1, 4)}), (F(1, 4), {(1, 1, 0): F(1, 2), (0, 0, 0): F(1, 2)}),
              (F(1, 2), {(1, 0, 0): F(9, 10), (0, 1, 0): F(1, 10)})])
best = posterior_release(L, 1, [(0,), (1,), (2,)])
for part in set_partitions([0, 1, 2]):
    q = posterior_release(L, 1, part)
    ex = clause_excess(L, 1, lambda c: q[c], lambda c: best[c])
    assert ex['nll_excess'] >= -1e-12 and ex['brier_excess'] >= 0
    # Brier excess = E ||q - T||^2 exactly
    d2 = sum(L.w[c] * sum((q[c][k] - L.p_y(c, 1)[k]) ** 2 for k in range(2)) for c in L.cells)
    assert ex['brier_excess'] == d2
assert sum(1 for _ in set_partitions(range(5))) == 52
print('finite_law self-test: PASS')
