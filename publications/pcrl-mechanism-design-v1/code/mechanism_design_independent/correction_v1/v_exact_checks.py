"""Independent exact recomputation of the correction pass's pre-registered checks (verifier role, correction_v1).

Written by the independent verifier on 2026-10-16 without reading the coordinator's check code or outputs. Every law
is re-entered by hand from results/pcrl_mechanism_design_v1/corrections/CHECKS_PREREGISTRATION.md (this program does
not read that file, or any other file), and every metric is implemented here from its definition with
fractions.Fraction. math.log is used only for NLL, mutual information, KL and the C2 asymptotic approximation. Nothing
from mechanism_design/prototype or any other campaign module is imported; only the no_real_data guard is installed.
The only file written is the --out JSON.

Pre-registered checks: A1-A2, B1-B4, C1a-C1c, C2a-C2b, D1-D2, E1-E3 (expectations as in the pre-registration).

Supplementary checks (NOT pre-registered; their expected outcomes are written here before the single queue run):
  S1  C2 cross-method agreement beyond the registered cases: closed form == sequential DP over draws (state = number
      of touched pairs) == Stirling-number sum == DP over pairs, exactly, for (M, n) in {(3,4), (2,5), (1000,20),
      (1000,60), (1000,200)}; brute-force enumeration agrees with all four at extra small cases (including n = 0).
      Expected: all equal.
  S2  On law E: partitions whose Bayes accuracy for S equals that of X while AUC* is strictly lower. Expected exactly 8
      (the partitions that never merge across pi = 1/2 number 5 x 2 = 10; minus the 2 pi-constant ones). This shows
      that equality of a single operating-point summary does not imply sufficiency, while AUC* equality does (E2).
  S3  Proper-scoring / MMP identities on a hand-made 3-cell, 3-class law with S:
      (i)   Brier(m) - Brier(eta) == E||eta - m||^2 == E[(pi - w)^2 ||q1 - q0||^2] exactly;
      (ii)  |NLL(m) - NLL(eta) - E KL(eta || m)| < 1e-12;
      (iii) Brier(f) - Brier(r) == E||f - eta||^2 - E||r - eta||^2 exactly, for two X-only predictors f, r;
      (iv)  with perturbed ("fitted") q-hat the general identity still holds exactly, but the closed form evaluated
            with q-hat differs from Brier(m-hat) - Brier(eta); the exact decomposition
            E||m-hat - eta||^2 = E[(pi-w)^2||q1-q0||^2] + E||m-hat - m||^2 + 2 E<m - eta, m-hat - m> holds;
      (v)   for the S-using predictor q^S, Brier(q^S) - Brier(eta) == -E||q^S - eta||^2 < 0, so the X-only identity
            does not extend to predictors that use S.
      Expected: (i)-(iii) hold, (iv) the q-hat closed form differs while the decomposition holds, (v) negative.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import itertools
import json
import math
import operator
import os
import platform
import sys
import time
import traceback
from fractions import Fraction as F

from mechanism_design import no_real_data

no_real_data.install()

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(200000)

HALF = F(1, 2)
CHECKS: list[dict] = []
ERRORS: list[dict] = []
EXTRA: dict = {}


# ----------------------------------------------------------------------------------------------------------------------
# Small exact helpers (all written from the definitions in the pre-registration)
# ----------------------------------------------------------------------------------------------------------------------

def fs(x) -> str:
    """Exact rational as 'p/q' (or 'p')."""
    x = F(x)
    return f'{x.numerator}/{x.denominator}' if x.denominator != 1 else f'{x.numerator}'


def jsonable(obj):
    if isinstance(obj, F):
        return {'exact': fs(obj), 'float': float(obj)}
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return str(obj)
    return obj


def ln(x) -> float:
    """Natural log of a positive rational (difference of integer logs; exact 0 for x == 1)."""
    x = F(x)
    if x <= 0:
        raise ValueError(f'log of non-positive value {x}')
    if x == 1:
        return 0.0
    return math.log(x.numerator) - math.log(x.denominator)


def make_view(cells: dict, release) -> dict:
    """cells: x -> (P(X=x, S=0), P(X=x, S=1)); release: x -> hashable view value. Returns v -> [P(v,0), P(v,1)]."""
    view: dict = {}
    for x, (p0, p1) in cells.items():
        acc = view.setdefault(release(x), [F(0), F(0)])
        acc[0] += p0
        acc[1] += p1
    return view


def auc_star(view: dict) -> F:
    """AUC*(S;V) = sum_{v,w} P(v,1)P(w,0)[1{P(v,1)P(w,0) > P(w,1)P(v,0)} + 1/2 1{=}] / (P(S=1)P(S=0)).

    All ordered pairs, including v == w (always a tie), likelihood ratios compared by cross-multiplication."""
    items = list(view.values())
    p0 = sum((a for a, _ in items), F(0))
    p1 = sum((b for _, b in items), F(0))
    num = F(0)
    for v0, v1 in items:
        for w0, w1 in items:
            weight = v1 * w0          # P(v,1) P(w,0)
            other = w1 * v0           # P(w,1) P(v,0)
            if weight > other:
                num += weight
            elif weight == other:
                num += HALF * weight
    return num / (p1 * p0)


def auc_fixed(view: dict, score) -> F:
    """AUC of a fixed score h(V): sum_{v,w} P(v,1)P(w,0)[1{h(v) > h(w)} + 1/2 1{h(v) = h(w)}] / (P(S=1)P(S=0))."""
    items = list(view.items())
    p0 = sum((a for _, (a, _b) in items), F(0))
    p1 = sum((b for _, (_a, b) in items), F(0))
    num = F(0)
    for v, (_v0, v1) in items:
        for w, (w0, _w1) in items:
            weight = v1 * w0
            hv, hw = score(v), score(w)
            if hv > hw:
                num += weight
            elif hv == hw:
                num += HALF * weight
    return num / (p1 * p0)


def bayes_acc_S(view: dict) -> F:
    """Bayes accuracy for S from the view: sum_v max(P(v,0), P(v,1))."""
    return sum((max(a, b) for a, b in view.values()), F(0))


def mutual_info_bits(view: dict):
    """I(S;V) in bits and whether S and V are exactly independent (P(v,s) == P(v)P(s) for every cell)."""
    p0 = sum((a for a, _ in view.values()), F(0))
    p1 = sum((b for _, b in view.values()), F(0))
    terms = []
    independent = True
    for v0, v1 in view.values():
        pv = v0 + v1
        for pvs, ps in ((v0, p0), (v1, p1)):
            if pvs != pv * ps:
                independent = False
            if pvs == 0:
                continue
            terms.append(float(pvs) * ln(pvs / (pv * ps)))
    return math.fsum(terms) / math.log(2), independent


def argmax_first(p) -> int:
    best = 0
    for k in range(1, len(p)):
        if p[k] > p[best]:
            best = k
    return best


def in_simplex(p) -> bool:
    return all(q >= 0 for q in p) and sum(p, F(0)) == 1


def task_metrics(pxy: dict, pred) -> dict:
    """Expected first-index-argmax accuracy, summed Brier and NLL (nats, no floor) under the law pxy[(x, y)]."""
    acc, brier, nll_terms, min_p = F(0), F(0), [], None
    for (x, y), p in pxy.items():
        if p == 0:
            continue
        q = pred(x)
        if argmax_first(q) == y:
            acc += p
        brier += p * sum(((q[k] - (1 if k == y else 0)) ** 2 for k in range(len(q))), F(0))
        if q[y] <= 0:
            raise ValueError('zero predicted probability on a positive-mass label')
        nll_terms.append(float(p) * -ln(q[y]))
        min_p = q[y] if min_p is None else min(min_p, q[y])
    return {'accuracy': acc, 'brier_summed': brier, 'nll_nats': math.fsum(nll_terms),
            'min_prob_on_positive_mass_label': min_p}


def record(cid, statement, expected, value, passed, hand=None, hand_agrees=None, note=None):
    CHECKS.append({'id': cid, 'statement': statement, 'expected': expected, 'value': value, 'pass': bool(passed),
                   'hand_value': hand, 'hand_value_agrees': (None if hand_agrees is None else bool(hand_agrees)),
                   'note': note})


def approx_ok(x, five_dp: float) -> bool:
    """The pre-registration's decimal approximations are 5-decimal roundings."""
    return abs(round(float(x), 5) - five_dp) < 1e-12


# ----------------------------------------------------------------------------------------------------------------------
# Law A
# ----------------------------------------------------------------------------------------------------------------------

def law_A():
    PX = [F(1, 8), F(3, 8), F(1, 4), F(1, 4)]
    PI = [F(1, 4), F(1, 3), F(2, 3), F(3, 4)]
    ETA = [F(2, 5), F(4, 5), F(1, 5), F(3, 5)]
    assert sum(PX) == 1
    cells = {x: (PX[x] * (1 - PI[x]), PX[x] * PI[x]) for x in range(4)}

    def release(x):
        vec = (1 - ETA[x], ETA[x])
        return (vec, argmax_first(vec))

    vX, vR = make_view(cells, lambda x: x), make_view(cells, release)
    aX, aR = auc_star(vX), auc_star(vR)
    p1 = sum((c[1] for c in cells.values()), F(0))
    EXTRA['A'] = {'P(S=1)': p1, 'P(S=0)': 1 - p1, 'release_values': {str(x): release(x)[0] for x in range(4)}}
    hand = F(3299, 4606)
    record('A1', 'Injective release R = ((1-eta, eta), argmax): AUC*(S;R) == AUC*(S;X)',
           'equal; R has 4 distinct values', {'auc_star_X': aX, 'auc_star_R': aR, 'distinct_release_values': len(vR)},
           aX == aR and len(vR) == 4,
           hand={'auc_star': hand, 'approx': 0.71624}, hand_agrees=(aX == hand and aR == hand and approx_ok(hand, 0.71624)))
    s_eta = auc_fixed(vR, lambda r: r[0][1])
    s_rev = auc_fixed(vR, lambda r: -r[0][1])
    h_eta, h_rev = F(1763, 4606), F(2843, 4606)
    record('A2', 'Restricted readers: score = released eta, and its reversal; both strictly below AUC*',
           'AUC(eta) = 1763/4606, AUC(reversal) = 2843/4606, both < AUC*',
           {'auc_score_eta': s_eta, 'auc_score_reversed': s_rev, 'auc_star': aX},
           s_eta < aX and s_rev < aX,
           hand={'auc_score_eta': h_eta, 'approx_eta': 0.38276, 'auc_score_reversed': h_rev, 'approx_reversed': 0.61724},
           hand_agrees=(s_eta == h_eta and s_rev == h_rev and approx_ok(h_eta, 0.38276) and approx_ok(h_rev, 0.61724)))


# ----------------------------------------------------------------------------------------------------------------------
# Law B
# ----------------------------------------------------------------------------------------------------------------------

def law_B():
    PT = [F(1, 4), F(1, 2), F(1, 4)]
    PZ = [F(3, 5), F(2, 5)]                     # P(Z = 0), P(Z = 1); S = Z
    PY1 = [F(1, 5), F(3, 5), F(9, 10)]          # P(Y1 = 1 | T = t)
    PY2 = [(F(7, 10), F(1, 5), F(1, 10)), (F(1, 5), F(3, 5), F(1, 5)), (F(1, 10), F(3, 10), F(3, 5))]
    assert all(sum(r) == 1 for r in PY2) and sum(PT) == 1
    joint = {}
    for t, z, y1, y2 in itertools.product(range(3), range(2), range(2), range(3)):
        joint[(t, z, y1, y2)] = PT[t] * PZ[z] * (PY1[t] if y1 == 1 else 1 - PY1[t]) * PY2[t][y2]
    assert sum(joint.values()) == 1
    X = [(t, z) for t in range(3) for z in range(2)]
    px = {x: sum((p for k, p in joint.items() if (k[0], k[1]) == x), F(0)) for x in X}
    # S = Z: P(X = x, S = s) from the joint
    cells = {x: (sum((p for k, p in joint.items() if (k[0], k[1]) == x and k[1] == 0), F(0)),
                 sum((p for k, p in joint.items() if (k[0], k[1]) == x and k[1] == 1), F(0))) for x in X}
    pxy1 = {(x, c): sum((joint[(x[0], x[1], c, y2)] for y2 in range(3)), F(0)) for x in X for c in range(2)}
    pxy2 = {(x, c): sum((joint[(x[0], x[1], y1, c)] for y1 in range(2)), F(0)) for x in X for c in range(3)}
    bayes1 = {x: tuple(pxy1[(x, c)] / px[x] for c in range(2)) for x in X}       # full-input Bayes P(Y1 | X = x)
    bayes2 = {x: tuple(pxy2[(x, c)] / px[x] for c in range(3)) for x in X}
    rel1 = {x: (1 - PY1[x[0]], PY1[x[0]]) for x in X}                             # release R_1(t, z) = P(Y1 | T = t)
    rel2 = {x: PY2[x[0]] for x in X}
    equal_cells = {str(x): (rel1[x] == bayes1[x], rel2[x] == bayes2[x]) for x in X}
    all_equal = all(a and b for a, b in equal_cells.values())
    contract_ok = all(in_simplex(rel1[x]) and in_simplex(rel2[x]) for x in X)
    m1r, m1b = task_metrics(pxy1, lambda x: rel1[x]), task_metrics(pxy1, lambda x: bayes1[x])
    m2r, m2b = task_metrics(pxy2, lambda x: rel2[x]), task_metrics(pxy2, lambda x: bayes2[x])
    metrics_equal = (m1r['accuracy'] == m1b['accuracy'] and m1r['brier_summed'] == m1b['brier_summed']
                     and m2r['accuracy'] == m2b['accuracy'] and m2r['brier_summed'] == m2b['brier_summed']
                     and m1r['nll_nats'] == m1b['nll_nats'] and m2r['nll_nats'] == m2b['nll_nats'])
    hand = {'income_accuracy': F(29, 40), 'income_brier_summed': F(73, 200),
            'occupation_accuracy': F(5, 8), 'occupation_brier_summed': F(53, 100)}
    mine = {'income_accuracy': m1r['accuracy'], 'income_brier_summed': m1r['brier_summed'],
            'occupation_accuracy': m2r['accuracy'], 'occupation_brier_summed': m2r['brier_summed']}
    record('B1', 'Release R_i(x) equals the full-input Bayes predictor P(Y_i | X = x) in every cell; metrics equal Bayes',
           'exact equality in all 6 cells for both tasks; accuracy, summed Brier and NLL equal the Bayes values',
           {'release_equals_bayes_per_cell(task1,task2)': equal_cells, 'release_in_simplex': contract_ok,
            'income_release': m1r, 'income_bayes': m1b, 'occupation_release': m2r, 'occupation_bayes': m2b},
           all_equal and metrics_equal and contract_ok,
           hand=hand, hand_agrees=all(mine[k] == hand[k] for k in hand),
           note='NLL equality is established by exact equality of the predicted distributions in every cell.')
    vX = make_view(cells, lambda x: x)
    aX = auc_star(vX)
    record('B2', 'AUC*(S;X) with S = Z a coordinate of X', '1', {'auc_star_X': aX}, aX == 1,
           hand={'auc_star_X': F(1)}, hand_agrees=(aX == 1))
    r1 = lambda x: (rel1[x], argmax_first(rel1[x]))
    r2 = lambda x: (rel2[x], argmax_first(rel2[x]))
    v1, v2 = make_view(cells, r1), make_view(cells, r2)
    v12 = make_view(cells, lambda x: (r1(x), r2(x)))
    a1, a2, a12 = auc_star(v1), auc_star(v2), auc_star(v12)
    record('B3', 'AUC*(S;R_1) = AUC*(S;R_2) = AUC*(S;(R_1,R_2))', 'all 1/2',
           {'auc_star_R1': a1, 'auc_star_R2': a2, 'auc_star_R1R2': a12}, a1 == a2 == a12 == HALF,
           hand={'auc_star': HALF}, hand_agrees=(a1 == a2 == a12 == HALF))
    iX, indep_X = mutual_info_bits(vX)
    iR, indep_R = mutual_info_bits(v12)
    h25 = -(0.4 * math.log2(0.4) + 0.6 * math.log2(0.6))
    record('B4', 'Distinct release pairs; I(S;X) = h(2/5) bits; I(S;(R_1,R_2)) = 0',
           '3 distinct pairs for 6 inputs; I(S;X) = h(2/5) ~ 0.97095 bits; I(S;R) = 0 (exact independence)',
           {'inputs': len(X), 'distinct_release_pairs': len(v12), 'I_S_X_bits': iX, 'h(2/5)_bits': h25,
            'I_S_R_bits': iR, 'S_R_exactly_independent': indep_R, 'S_X_exactly_independent': indep_X},
           len(v12) == 3 and len(X) == 6 and abs(iX - h25) < 1e-12 and indep_R and iR == 0.0,
           hand={'distinct_pairs': 3, 'I_S_X_bits_approx': 0.97095, 'I_S_R_bits': 0},
           hand_agrees=(len(v12) == 3 and approx_ok(iX, 0.97095) and iR == 0.0))
    EXTRA['B'] = {'P(S=1)': sum((c[1] for c in cells.values()), F(0)),
                  'release_pairs': [[list(map(fs, rel1[x])), list(map(fs, rel2[x]))] for x in X]}


# ----------------------------------------------------------------------------------------------------------------------
# Law C1
# ----------------------------------------------------------------------------------------------------------------------

def law_C1():
    PI = [F(1, 10), F(1, 2), F(3, 10), F(7, 10), F(1, 5), F(9, 10)]
    f = [F(1, 4), F(1, 4), F(1, 2), F(1, 2), F(3, 4), F(3, 4)]
    cells = {x: (F(1, 6) * (1 - PI[x]), F(1, 6) * PI[x]) for x in range(6)}
    release = lambda x: (1 - f[x], f[x])
    O = [0, 2, 4]
    on_O = len({release(x) for x in O})
    identity_on_O = len(set(O))
    record('C1a', 'Sample-uniqueness check on observed O = {0,2,4}: f and the identity both pass',
           'f: 3 distinct outputs for 3 distinct inputs; identity: 3 for 3',
           {'f_distinct_outputs_on_O': on_O, 'identity_distinct_outputs_on_O': identity_on_O, 'inputs_in_O': len(O)},
           on_O == len(O) == 3 and identity_on_O == 3,
           hand={'f_distinct_on_O': 3, 'inputs': 3}, hand_agrees=(on_O == 3))
    pre = collections.Counter(release(x) for x in range(6))
    record('C1b', 'On the declared support f is 2-to-1', '3 distinct outputs for 6 cells, each with exactly 2 preimages',
           {'distinct_outputs_on_support': len(pre), 'preimage_sizes': sorted(pre.values())},
           len(pre) == 3 and all(c == 2 for c in pre.values()),
           hand={'distinct_outputs': 3, 'cells': 6}, hand_agrees=(len(pre) == 3))
    aX = auc_star(make_view(cells, lambda x: x))
    aF = auc_star(make_view(cells, release))
    hF, hX = F(1091, 1782), F(1461, 1782)
    record('C1c', 'AUC*(S; f(X)) < AUC*(S; X) under the population law', 'strictly below',
           {'auc_star_fX': aF, 'auc_star_X': aX}, aF < aX,
           hand={'auc_star_fX': hF, 'approx_fX': 0.61223, 'auc_star_X': hX, 'approx_X': 0.81987},
           hand_agrees=(aF == hF and aX == hX and approx_ok(hF, 0.61223) and approx_ok(hX, 0.81987)))


# ----------------------------------------------------------------------------------------------------------------------
# Law C2: four exact methods
# ----------------------------------------------------------------------------------------------------------------------

def c2_closed(M: int, n: int) -> F:
    """P(M, n) = sum_k C(M,k) (-1)^(M-k) 2^k k^n / (2M)^n, with 0^0 = 1."""
    num = sum(math.comb(M, k) * (-1) ** (M - k) * 2 ** k * (1 if (k == 0 and n == 0) else k ** n) for k in range(M + 1))
    return F(num, (2 * M) ** n)


def c2_brute(M: int, n: int):
    """Enumerate all (2M)^n samples; cells 2i and 2i+1 form pair i; no collision = no pair with both members drawn."""
    N = 2 * M
    good = 0
    total = 0
    for seq in itertools.product(range(N), repeat=n):
        total += 1
        drawn = set(seq)
        if not any((2 * i in drawn) and (2 * i + 1 in drawn) for i in range(M)):
            good += 1
    assert total == N ** n
    return F(good, total), good, total


def c2_dp_draws(M: int, n: int) -> F:
    """Sequential DP over the n draws; state j = number of pairs with exactly one member drawn so far (no collision).

    From state j: redraw one of the j drawn cells (j ways, stay); draw a member of an untouched pair (2(M-j) ways,
    j -> j+1); drawing a partner of a drawn cell (j ways) is a collision and is dropped."""
    a = [0] * (n + 1)
    a[0] = 1
    for _ in range(n):
        new = [0] * (n + 1)
        for j, cnt in enumerate(a):
            if cnt == 0:
                continue
            new[j] += cnt * j
            if j < M:
                new[j + 1] += cnt * 2 * (M - j)
        a = new
    return F(sum(a), (2 * M) ** n)


def c2_stirling(M: int, n: int) -> F:
    """count = sum_j C(M,j) 2^j j! S2(n,j): choose j cells with no two partners, then a surjection of the n draws."""
    S = [[0] * (n + 1) for _ in range(n + 1)]
    S[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            S[i][j] = j * S[i - 1][j] + S[i - 1][j - 1]
    count = sum(math.comb(M, j) * 2 ** j * math.factorial(j) * S[n][j] for j in range(0, min(n, M) + 1))
    return F(count, (2 * M) ** n)


def c2_dp_pairs(M: int, n: int) -> F:
    """DP over pairs: A_i(r) = number of ways pairs 1..i fill a fixed set of r labelled draw positions, each pair's
    positions all showing the same member. A_i(r) = A_{i-1}(r) + 2 sum_{t<r} C(r,t) A_{i-1}(t). count = A_M(n)."""
    C = [[math.comb(r, t) for t in range(r + 1)] for r in range(n + 1)]
    A = [0] * (n + 1)
    A[0] = 1
    for _ in range(M):
        A = [A[r] + 2 * sum(map(operator.mul, C[r][:r], A[:r])) for r in range(n + 1)]
    return F(A[n], (2 * M) ** n)


def law_C2():
    rows = []
    # C2a: registered brute-force comparisons
    agree_a = {}
    for (M, n) in [(3, 4), (2, 5)]:
        pc = c2_closed(M, n)
        pb, good, total = c2_brute(M, n)
        agree_a[f'({M},{n})'] = pc == pb
        rows.append({'M': M, 'n': n, 'closed_form': pc, 'brute_force': pb, 'brute_force_counts': [good, total],
                     'registered': 'C2a'})
    record('C2a', 'Closed form equals brute-force enumeration at (M,n) = (3,4) and (2,5)', 'exact equality',
           {f"P{r['M'], r['n']}": {'closed': r['closed_form'], 'brute': r['brute_force'],
                                   'counts_good_total': r['brute_force_counts']} for r in rows},
           all(agree_a.values()), hand=None, hand_agrees=None,
           note='No hand value is registered for C2a (only the equality).')
    # C2b: registered intervals on the exact values
    large = {}
    t0 = time.process_time()
    for n in (20, 60, 200):
        pc = c2_closed(1000, n)
        approx = math.exp(-n * (n - 1) / (4 * 1000))
        large[n] = (pc, approx)
    c2_cpu_closed = time.process_time() - t0
    in20 = F(88, 100) <= large[20][0] <= F(94, 100)
    in60 = F(35, 100) <= large[60][0] <= F(47, 100)
    in200 = large[200][0] < F(1, 1000)
    ap20 = 0.88 <= large[20][1] <= 0.94
    ap60 = 0.35 <= large[60][1] <= 0.47
    ap200 = large[200][1] < 1e-3
    record('C2b', 'Exact P(1000, n) inside the registered intervals',
           'P(1000,20) in [0.88,0.94]; P(1000,60) in [0.35,0.47]; P(1000,200) < 1e-3',
           {'P(1000,20)': float(large[20][0]), 'P(1000,60)': float(large[60][0]), 'P(1000,200)': float(large[200][0]),
            'approx_exp(-n(n-1)/4M)': {'20': large[20][1], '60': large[60][1], '200': large[200][1]},
            'approx_inside_same_intervals': [ap20, ap60, ap200],
            'exact_fractions_in_cross_method_table': True},
           in20 and in60 and in200,
           hand={'P(1000,20)': '[0.88, 0.94]', 'P(1000,60)': '[0.35, 0.47]', 'P(1000,200)': '< 1e-3'},
           hand_agrees=(in20 and in60 and in200 and ap20 and ap60 and ap200),
           note='The hand values are intervals derived from exp(-n(n-1)/(4M)); agreement means both the exact value '
                'and the approximation fall inside them.')
    # S1: cross-method agreement table
    cross = []
    timing = {'closed_form_large_cpu_s': c2_cpu_closed}
    for (M, n) in [(3, 4), (2, 5), (1000, 20), (1000, 60), (1000, 200)]:
        t = time.process_time()
        pc = c2_closed(M, n)
        t_c = time.process_time() - t
        t = time.process_time()
        pd = c2_dp_draws(M, n)
        t_d = time.process_time() - t
        t = time.process_time()
        pst = c2_stirling(M, n)
        t_s = time.process_time() - t
        t = time.process_time()
        pp = c2_dp_pairs(M, n)
        t_p = time.process_time() - t
        entry = {'M': M, 'n': n, 'closed_form': fs(pc), 'float': float(pc),
                 'dp_draws_equal': pd == pc, 'stirling_equal': pst == pc, 'dp_pairs_equal': pp == pc,
                 'cpu_s': {'closed': round(t_c, 4), 'dp_draws': round(t_d, 4), 'stirling': round(t_s, 4),
                           'dp_pairs': round(t_p, 4)}}
        if M <= 3:
            pb, good, total = c2_brute(M, n)
            entry['brute_force_equal'] = pb == pc
            entry['brute_force_counts'] = [good, total]
        cross.append(entry)
    extra_small = []
    for (M, n) in [(1, 1), (1, 3), (2, 2), (2, 6), (3, 3), (3, 5), (4, 4), (5, 3), (3, 0)]:
        pb, good, total = c2_brute(M, n)
        vals = {'closed': c2_closed(M, n), 'dp_draws': c2_dp_draws(M, n), 'stirling': c2_stirling(M, n),
                'dp_pairs': c2_dp_pairs(M, n)}
        extra_small.append({'M': M, 'n': n, 'brute_force': fs(pb), 'counts_good_total': [good, total],
                            'all_methods_equal_brute': all(v == pb for v in vals.values())})
    all_cross = all(e['dp_draws_equal'] and e['stirling_equal'] and e['dp_pairs_equal']
                    and e.get('brute_force_equal', True) for e in cross)
    all_small = all(e['all_methods_equal_brute'] for e in extra_small)
    EXTRA['C2_cross_method'] = {'registered_and_large_cases': cross, 'extra_small_brute_force_cases': extra_small,
                                'all_agree': all_cross and all_small, 'timing': timing,
                                'supplementary_expectation': 'all equal (S1, not pre-registered)',
                                'supplementary_pass': all_cross and all_small}


# ----------------------------------------------------------------------------------------------------------------------
# Law D
# ----------------------------------------------------------------------------------------------------------------------

def law_D():
    PI = [F(1, 5), F(4, 5), F(1, 5), F(4, 5)]
    ETA = [F(1, 3), F(2, 3), F(1, 3), F(2, 3)]
    cells = {x: (F(1, 4) * (1 - PI[x]), F(1, 4) * PI[x]) for x in range(4)}
    release = lambda x: (1 - ETA[x], ETA[x])
    pre = collections.Counter(release(x) for x in range(4))
    record('D1', 'The release eta merges {0,2} and {1,3}: 2-to-1', '2 distinct outputs, each with 2 preimages',
           {'distinct_outputs': len(pre), 'preimage_sizes': sorted(pre.values()),
            'blocks': sorted(sorted(x for x in range(4) if release(x) == r) for r in pre)},
           len(pre) == 2 and all(c == 2 for c in pre.values()),
           hand={'distinct_outputs': 2, 'cells': 4}, hand_agrees=(len(pre) == 2))
    aX = auc_star(make_view(cells, lambda x: x))
    aR = auc_star(make_view(cells, release))
    record('D2', 'Non-injective release keeps all S-information: AUC*(S;R) = AUC*(S;X)', 'both exactly 4/5',
           {'auc_star_R': aR, 'auc_star_X': aX}, aR == aX == F(4, 5),
           hand={'auc_star': F(4, 5)}, hand_agrees=(aR == aX == F(4, 5)))


# ----------------------------------------------------------------------------------------------------------------------
# Law E: exhaustive 52-partition enumeration
# ----------------------------------------------------------------------------------------------------------------------

def set_partitions(n: int):
    """All set partitions of {0..n-1} as restricted growth strings (rgs[0] = 0, rgs[i] <= max(rgs[:i]) + 1)."""
    out = []

    def rec(rgs, mx):
        if len(rgs) == n:
            out.append(tuple(rgs))
            return
        for b in range(mx + 2):
            rec(rgs + [b], max(mx, b))

    rec([0], 0)
    return out


def bell(n: int) -> int:
    """Bell number by B_{m+1} = sum_k C(m,k) B_k (independent of the enumeration)."""
    B = [1]
    for m in range(n):
        B.append(sum(math.comb(m, k) * B[k] for k in range(m + 1)))
    return B[n]


def blocks_of(rgs):
    d = collections.defaultdict(list)
    for x, b in enumerate(rgs):
        d[b].append(x)
    return [d[b] for b in sorted(d)]


def law_E():
    PX = [F(1, 10), F(1, 5), F(3, 10), F(1, 4), F(3, 20)]
    PI = [F(1, 5), F(2, 5), F(2, 5), F(3, 5), F(4, 5)]
    assert sum(PX) == 1
    cells = {x: (PX[x] * (1 - PI[x]), PX[x] * PI[x]) for x in range(5)}
    vX = make_view(cells, lambda x: x)
    aX = auc_star(vX)
    accX = bayes_acc_S(vX)
    parts = set_partitions(5)
    canon = {frozenset(frozenset(b) for b in blocks_of(r)) for r in parts}
    rows = []
    for rgs in parts:
        blocks = blocks_of(rgs)
        v = make_view(cells, lambda x, rgs=rgs: rgs[x])
        a = auc_star(v)
        rows.append({'blocks': blocks, 'auc_star': a, 'le_X': a <= aX, 'equal_X': a == aX, 'lt_X': a < aX,
                     'pi_constant_blocks': all(len({PI[x] for x in b}) == 1 for b in blocks),
                     'injective': len(blocks) == 5, 'bayes_acc_S': bayes_acc_S(v)})
    n_parts = len(parts)
    eq = [r['blocks'] for r in rows if r['equal_X']]
    expected_eq = [[[0], [1], [2], [3], [4]], [[0], [1, 2], [3], [4]]]
    E1 = n_parts == 52 and len(canon) == 52 and bell(5) == 52 and all(r['le_X'] for r in rows)
    record('E1', 'AUC*(S; block) <= AUC*(S; X) for all set partitions of the 5 cells',
           'all 52 partitions satisfy <=',
           {'partitions_enumerated': n_parts, 'distinct_partitions': len(canon), 'bell_5_recurrence': bell(5),
            'auc_star_X': aX, 'all_le': all(r['le_X'] for r in rows),
            'max_auc_over_non_equal': max((r['auc_star'] for r in rows if not r['equal_X']), default=None)},
           E1, hand={'partitions': 52}, hand_agrees=(n_parts == 52))
    E2 = sorted(eq) == sorted(expected_eq)
    record('E2', 'Equality holds exactly for the partitions whose blocks each have constant pi',
           'exactly 2: finest and {{0},{1,2},{3},{4}} (first injective, second not)',
           {'equality_partitions': eq, 'count': len(eq),
            'injective_flags': [len(b) == 5 for b in eq],
            'equality_iff_pi_constant_all_52': all(r['equal_X'] == r['pi_constant_blocks'] for r in rows)},
           E2 and all(r['equal_X'] == r['pi_constant_blocks'] for r in rows),
           hand={'count': 2, 'partitions': expected_eq}, hand_agrees=E2)
    n_lt = sum(r['lt_X'] for r in rows)
    record('E3', 'The other partitions are strictly below', '50 strictly below',
           {'strictly_below': n_lt, 'equal': len(eq), 'total': n_parts}, n_lt == 50 and n_lt + len(eq) == 52,
           hand={'strictly_below': 50}, hand_agrees=(n_lt == 50))
    s2 = [r['blocks'] for r in rows if r['bayes_acc_S'] == accX and r['lt_X']]
    EXTRA['E_partitions'] = {'auc_star_X': aX, 'bayes_acc_S_X': accX,
                             'rows': [{'blocks': r['blocks'], 'auc_star': r['auc_star'], 'equal_X': r['equal_X'],
                                       'pi_constant_blocks': r['pi_constant_blocks'], 'bayes_acc_S': r['bayes_acc_S']}
                                      for r in sorted(rows, key=lambda r: (-r['auc_star'], str(r['blocks'])))]}
    EXTRA['S2_bayes_accuracy_vs_auc'] = {
        'statement': 'partitions with Bayes accuracy for S equal to that of X but AUC* strictly lower',
        'expected_supplementary': 8, 'count': len(s2), 'partitions': s2, 'pass': len(s2) == 8,
        'keeps_bayes_accuracy_total': sum(r['bayes_acc_S'] == accX for r in rows)}


# ----------------------------------------------------------------------------------------------------------------------
# Supplementary S3: proper-scoring / MMP identities on a hand-made law
# ----------------------------------------------------------------------------------------------------------------------

def supp_identities():
    K = 3
    PX = [F(1, 3), F(1, 2), F(1, 6)]
    PI = [F(1, 4), F(1, 2), F(4, 5)]
    Q0 = [(F(1, 2), F(1, 3), F(1, 6)), (F(1, 5), F(3, 5), F(1, 5)), (F(1, 10), F(1, 10), F(4, 5))]
    Q1 = [(F(1, 6), F(1, 3), F(1, 2)), (F(2, 5), F(2, 5), F(1, 5)), (F(3, 10), F(1, 2), F(1, 5))]
    W = F(1, 3)
    assert sum(PX) == 1 and all(in_simplex(q) for q in Q0 + Q1)
    Q = (Q0, Q1)
    joint = {(x, s, y): PX[x] * (PI[x] if s else 1 - PI[x]) * Q[s][x][y]
             for x in range(3) for s in range(2) for y in range(K)}
    assert sum(joint.values()) == 1
    pxy = {(x, y): joint[(x, 0, y)] + joint[(x, 1, y)] for x in range(3) for y in range(K)}
    px = {x: sum((pxy[(x, y)] for y in range(K)), F(0)) for x in range(3)}
    eta = {x: tuple(pxy[(x, y)] / px[x] for y in range(K)) for x in range(3)}
    eta_formula_ok = all(eta[x] == tuple((1 - PI[x]) * Q0[x][k] + PI[x] * Q1[x][k] for k in range(K)) for x in range(3))

    def mix(q0, q1, x):
        return tuple((1 - W) * q0[x][k] + W * q1[x][k] for k in range(K))

    def brier(pred):
        return sum((p * sum(((pred[x][k] - (1 if k == y else 0)) ** 2 for k in range(K)), F(0))
                    for (x, y), p in pxy.items()), F(0))

    def nll(pred):
        return math.fsum(float(p) * -ln(pred[x][y]) for (x, y), p in pxy.items() if p != 0)

    def sq(a, b):
        return sum(((a[k] - b[k]) ** 2 for k in range(K)), F(0))

    def dot(a, b):
        return sum((a[k] * b[k] for k in range(K)), F(0))

    def E(fn):
        return sum((px[x] * fn(x) for x in range(3)), F(0))

    def kl(a, b):
        return math.fsum(float(a[k]) * ln(a[k] / b[k]) for k in range(K) if a[k] != 0)

    m = {x: mix(Q0, Q1, x) for x in range(3)}
    i_lhs = brier(m) - brier(eta)
    i_mid = E(lambda x: sq(eta[x], m[x]))
    i_rhs = E(lambda x: (PI[x] - W) ** 2 * sq(Q1[x], Q0[x]))
    ii_lhs = nll(m) - nll(eta)
    ii_rhs = math.fsum(float(px[x]) * kl(eta[x], m[x]) for x in range(3))
    f = {0: (F(1, 3), F(1, 3), F(1, 3)), 1: (F(1, 2), F(1, 4), F(1, 4)), 2: (F(1, 5), F(1, 5), F(3, 5))}
    r = {0: (F(1, 4), F(1, 2), F(1, 4)), 1: (F(1, 10), F(7, 10), F(1, 5)), 2: (F(1, 6), F(1, 6), F(2, 3))}
    iii_lhs = brier(f) - brier(r)
    iii_rhs = E(lambda x: sq(f[x], eta[x])) - E(lambda x: sq(r[x], eta[x]))
    Q0h = [(F(1, 2) + F(1, 20), F(1, 3) - F(1, 20), F(1, 6)), Q0[1], Q0[2]]
    Q1h = [Q1[0], (F(2, 5) - F(1, 10), F(2, 5) + F(1, 10), F(1, 5)), Q1[2]]
    assert all(in_simplex(q) for q in Q0h + Q1h)
    mh = {x: mix(Q0h, Q1h, x) for x in range(3)}
    iv_lhs = brier(mh) - brier(eta)
    iv_general = E(lambda x: sq(mh[x], eta[x]))
    iv_closed_with_qhat = E(lambda x: (PI[x] - W) ** 2 * sq(Q1h[x], Q0h[x]))
    iv_decomp = i_rhs + E(lambda x: sq(mh[x], m[x])) + 2 * E(
        lambda x: dot(tuple(m[x][k] - eta[x][k] for k in range(K)), tuple(mh[x][k] - m[x][k] for k in range(K))))
    brier_qS = sum((p * sum(((Q[s][x][k] - (1 if k == y else 0)) ** 2 for k in range(K)), F(0))
                    for (x, s, y), p in joint.items()), F(0))
    v_lhs = brier_qS - brier(eta)
    v_sq = sum((PX[x] * (PI[x] if s else 1 - PI[x]) * sq(Q[s][x], eta[x]) for x in range(3) for s in range(2)), F(0))
    res = {
        'law': {'P(X)': PX, 'pi': PI, 'q0': Q0, 'q1': Q1, 'w': W},
        'eta_equals_total_probability_formula': eta_formula_ok,
        'i': {'brier_m_minus_brier_eta': i_lhs, 'E_sq_eta_minus_m': i_mid, 'E_(pi-w)^2_sq_q1_minus_q0': i_rhs,
              'holds_exactly': i_lhs == i_mid == i_rhs},
        'ii': {'nll_m_minus_nll_eta': ii_lhs, 'E_KL_eta_m': ii_rhs, 'abs_diff': abs(ii_lhs - ii_rhs),
               'holds_1e-12': abs(ii_lhs - ii_rhs) < 1e-12},
        'iii': {'brier_f_minus_brier_r': iii_lhs, 'E_sq_f_eta_minus_E_sq_r_eta': iii_rhs,
                'holds_exactly': iii_lhs == iii_rhs},
        'iv': {'brier_mhat_minus_brier_eta': iv_lhs, 'E_sq_mhat_minus_eta': iv_general,
               'closed_form_with_qhat': iv_closed_with_qhat, 'decomposition_value': iv_decomp,
               'general_identity_holds': iv_lhs == iv_general, 'closed_form_with_qhat_differs': iv_lhs != iv_closed_with_qhat,
               'decomposition_holds': iv_decomp == iv_general},
        'v': {'brier_qS_minus_brier_eta': v_lhs, 'E_sq_qS_minus_eta': v_sq,
              'equals_minus_E_sq': v_lhs == -v_sq, 'negative': v_lhs < 0},
    }
    res['pass'] = (eta_formula_ok and res['i']['holds_exactly'] and res['ii']['holds_1e-12'] and res['iii']['holds_exactly']
                   and res['iv']['general_identity_holds'] and res['iv']['closed_form_with_qhat_differs']
                   and res['iv']['decomposition_holds'] and res['v']['equals_minus_E_sq'] and res['v']['negative'])
    EXTRA['S3_scoring_identities'] = res


# ----------------------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    t0, w0 = time.process_time(), time.time()
    for name, fn in [('A', law_A), ('B', law_B), ('C1', law_C1), ('C2', law_C2), ('D', law_D), ('E', law_E),
                     ('S3', supp_identities)]:
        try:
            fn()
        except Exception as exc:     # a bug in this verifier code: recorded, and the unit exits non-zero
            ERRORS.append({'section': name, 'error': repr(exc), 'traceback': traceback.format_exc()})
    registered = [c for c in CHECKS]
    summary = {
        'registered_checks': len(registered),
        'registered_ids': [c['id'] for c in registered],
        'registered_pass': sum(c['pass'] for c in registered),
        'registered_failed_ids': [c['id'] for c in registered if not c['pass']],
        'hand_values_compared': sum(c['hand_value_agrees'] is not None for c in registered),
        'hand_value_disagreements': [c['id'] for c in registered if c['hand_value_agrees'] is False],
        'supplementary_pass': {'S1_C2_cross_method': EXTRA.get('C2_cross_method', {}).get('supplementary_pass'),
                               'S2_bayes_accuracy_vs_auc': EXTRA.get('S2_bayes_accuracy_vs_auc', {}).get('pass'),
                               'S3_scoring_identities': EXTRA.get('S3_scoring_identities', {}).get('pass')},
        'errors': [e['section'] for e in ERRORS],
    }
    out = {
        'schema': 'V_EXACT_CHECKS/1',
        'role': 'independent verifier, correction pass correction_v1 (2026-10-16)',
        'unit_id': os.environ.get('MD_UNIT_ID'),
        'generated_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'python': platform.python_version(),
        'module': 'mechanism_design_independent.correction_v1.v_exact_checks',
        'laws_source': 'entered by hand from results/pcrl_mechanism_design_v1/corrections/CHECKS_PREREGISTRATION.md '
                       '(the program reads no file)',
        'independence': 'no import of mechanism_design.correction.* or campaign metric modules; only the '
                        'no_real_data guard is imported and installed',
        'summary': summary,
        'checks': registered,
        'supplementary': {k: v for k, v in EXTRA.items()},
        'errors': ERRORS,
        'process_cpu_s': round(time.process_time() - t0, 3),
        'wall_s': round(time.time() - w0, 3),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump(jsonable(out), fh, indent=1, sort_keys=False)
        fh.write('\n')
    print(json.dumps(jsonable(summary)))
    return 2 if ERRORS else 0


if __name__ == '__main__':
    sys.exit(main())
