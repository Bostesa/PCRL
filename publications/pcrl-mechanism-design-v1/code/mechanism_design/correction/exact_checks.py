"""Exact data-free checks for the correction pass (laws and expected outcomes: corrections/CHECKS_PREREGISTRATION.md).

Every law is a finite joint law in exact fractions. AUC* compares likelihood ratios by cross-multiplication and counts
ties as 1/2, including the pairs inside one view value, so it is the area under the optimal ROC. Logarithms appear only
in NLL and mutual information. No real-person data is read; the data guard is installed before anything else runs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import os
import sys
from fractions import Fraction as F
from math import comb

from mechanism_design import no_real_data

no_real_data.install()


# ---------- generic exact tools ----------

def view_law(cells):
    """cells: iterable of (mass, pi, view). Returns {view: [P(view, S=0), P(view, S=1)]}."""
    agg = {}
    for mass, pi, v in cells:
        a = agg.setdefault(v, [F(0), F(0)])
        a[0] += mass * (1 - pi)
        a[1] += mass * pi
    return agg


def auc_star(agg):
    p0 = sum(a[0] for a in agg.values())
    p1 = sum(a[1] for a in agg.values())
    tot = F(0)
    for a in agg.values():
        for b in agg.values():
            lhs, rhs = a[1] * b[0], b[1] * a[0]
            if lhs > rhs:
                tot += lhs
            elif lhs == rhs:
                tot += lhs / 2
    return tot / (p1 * p0)


def score_auc(cells, score):
    """AUC of a fixed score h(cell); cells: list of (mass, pi, key)."""
    p1 = sum(m * pi for m, pi, _ in cells)
    p0 = sum(m * (1 - pi) for m, pi, _ in cells)
    tot = F(0)
    for m1, pi1, k1 in cells:
        for m0, pi0, k0 in cells:
            w = m1 * pi1 * m0 * (1 - pi0)
            if score(k1) > score(k0):
                tot += w
            elif score(k1) == score(k0):
                tot += w / 2
    return tot / (p1 * p0)


def mi_bits(agg):
    p0 = sum(a[0] for a in agg.values())
    p1 = sum(a[1] for a in agg.values())
    out = 0.0
    for a in agg.values():
        pv = a[0] + a[1]
        for s, ps in ((0, p0), (1, p1)):
            if a[s] > 0:
                out += float(a[s]) * math.log2(float(a[s] / (pv * ps)))
    return out


def argmax_first(v):
    best = 0
    for k in range(1, len(v)):
        if v[k] > v[best]:
            best = k
    return best


def task_metrics(masses, truth, pred):
    """masses[x], truth[x] = P(Y | x), pred[x] = released vector. Accuracy and summed Brier exact; NLL float (nats)."""
    acc, brier, nll = F(0), F(0), 0.0
    for x, mx in enumerate(masses):
        p, q = pred[x], truth[x]
        acc += mx * q[argmax_first(p)]
        for y, qy in enumerate(q):
            brier += mx * qy * sum((p[k] - (1 if k == y else 0)) ** 2 for k in range(len(p)))
            if qy > 0:
                nll += float(mx * qy) * -math.log(float(p[y]))
    return {'accuracy': acc, 'summed_brier': brier, 'nll_nats': nll}


def set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in set_partitions(rest):
        yield [[first]] + part
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]


def fr(x):
    return {'fraction': f'{x.numerator}/{x.denominator}', 'float': float(x)} if isinstance(x, F) else x


# ---------- the pre-registered laws ----------

def check_A():
    masses = [F(1, 8), F(3, 8), F(1, 4), F(1, 4)]
    pi = [F(1, 4), F(1, 3), F(2, 3), F(3, 4)]
    eta = [F(2, 5), F(4, 5), F(1, 5), F(3, 5)]
    release = {x: (1 - eta[x], eta[x], argmax_first((1 - eta[x], eta[x]))) for x in range(4)}
    a_x = auc_star(view_law((masses[x], pi[x], x) for x in range(4)))
    a_r = auc_star(view_law((masses[x], pi[x], release[x]) for x in range(4)))
    cells = [(masses[x], pi[x], x) for x in range(4)]
    naive = score_auc(cells, lambda x: eta[x])
    rev = score_auc(cells, lambda x: -eta[x])
    res = {'injective': len(set(release.values())) == 4, 'auc_star_X': a_x, 'auc_star_R': a_r, 'naive_score_auc': naive,
           'reversed_score_auc': rev}
    checks = {
        'A0_release_injective': res['injective'],
        'A1_auc_star_R_equals_X': a_r == a_x,
        'A1_hand_value_3299_4606': a_x == F(3299, 4606),
        'A2_naive_below_auc_star': naive < a_x and naive == F(1763, 4606),
        'A2_reversed_below_auc_star': rev < a_x and rev == F(2843, 4606),
    }
    return res, checks


def check_B():
    pt = [F(1, 4), F(1, 2), F(1, 4)]
    pz1 = F(2, 5)
    y1 = [[1 - p, p] for p in (F(1, 5), F(3, 5), F(9, 10))]
    y2 = [[F(7, 10), F(1, 5), F(1, 10)], [F(1, 5), F(3, 5), F(1, 5)], [F(1, 10), F(3, 10), F(3, 5)]]
    cells = [(t, z) for t in range(3) for z in range(2)]
    mass = {(t, z): pt[t] * (pz1 if z == 1 else 1 - pz1) for t, z in cells}
    # full-input Bayes predictor P(Y_i | T=t, Z=z), computed from the joint law P(t)P(z)P(y1|t)P(y2|t)
    joint = {}
    for (t, z) in cells:
        for a in range(2):
            for b in range(3):
                joint[(t, z, a, b)] = mass[(t, z)] * y1[t][a] * y2[t][b]
    bayes1 = {c: [sum(joint[(c[0], c[1], a, b)] for b in range(3)) / mass[c] for a in range(2)] for c in cells}
    bayes2 = {c: [sum(joint[(c[0], c[1], a, b)] for a in range(2)) / mass[c] for b in range(3)] for c in cells}
    rel1 = {c: list(y1[c[0]]) for c in cells}          # R_1(t, z) = P(Y1 | T = t)
    rel2 = {c: list(y2[c[0]]) for c in cells}          # R_2(t, z) = P(Y2 | T = t)
    key1 = {c: tuple(rel1[c]) + (argmax_first(rel1[c]),) for c in cells}
    key2 = {c: tuple(rel2[c]) + (argmax_first(rel2[c]),) for c in cells}
    pi = {c: F(c[1]) for c in cells}                    # S = Z
    masses = [mass[c] for c in cells]
    m_rel1 = task_metrics(masses, [bayes1[c] for c in cells], [rel1[c] for c in cells])
    m_bay1 = task_metrics(masses, [bayes1[c] for c in cells], [bayes1[c] for c in cells])
    m_rel2 = task_metrics(masses, [bayes2[c] for c in cells], [rel2[c] for c in cells])
    m_bay2 = task_metrics(masses, [bayes2[c] for c in cells], [bayes2[c] for c in cells])
    v_x = view_law((mass[c], pi[c], c) for c in cells)
    v_1 = view_law((mass[c], pi[c], key1[c]) for c in cells)
    v_2 = view_law((mass[c], pi[c], key2[c]) for c in cells)
    v_p = view_law((mass[c], pi[c], (key1[c], key2[c])) for c in cells)
    res = {'release_equals_bayes_task1': all(rel1[c] == bayes1[c] for c in cells),
           'release_equals_bayes_task2': all(rel2[c] == bayes2[c] for c in cells),
           'task1_release': {k: fr(v) for k, v in m_rel1.items()}, 'task1_bayes': {k: fr(v) for k, v in m_bay1.items()},
           'task2_release': {k: fr(v) for k, v in m_rel2.items()}, 'task2_bayes': {k: fr(v) for k, v in m_bay2.items()},
           'auc_star_X': auc_star(v_x), 'auc_star_R1': auc_star(v_1), 'auc_star_R2': auc_star(v_2), 'auc_star_pair': auc_star(v_p),
           'distinct_inputs': len(cells), 'distinct_release_pairs': len(set((key1[c], key2[c]) for c in cells)),
           'mi_bits_S_X': mi_bits(v_x), 'mi_bits_S_pair': mi_bits(v_p)}
    checks = {
        'B1_release_equals_full_input_bayes_both_tasks': res['release_equals_bayes_task1'] and res['release_equals_bayes_task2'],
        'B1_metrics_equal_bayes': m_rel1 == m_bay1 and m_rel2 == m_bay2,
        'B1_hand_values': (m_rel1['accuracy'], m_rel1['summed_brier'], m_rel2['accuracy'], m_rel2['summed_brier'])
                          == (F(29, 40), F(73, 200), F(5, 8), F(53, 100)),
        'B2_auc_star_X_is_1': res['auc_star_X'] == 1,
        'B3_auc_star_release_views_half': res['auc_star_R1'] == res['auc_star_R2'] == res['auc_star_pair'] == F(1, 2),
        'B4_many_to_one_6_to_3': res['distinct_inputs'] == 6 and res['distinct_release_pairs'] == 3,
        'B4_mi': abs(res['mi_bits_S_X'] - 0.97095) < 5e-6 and abs(res['mi_bits_S_pair']) < 1e-15,
    }
    return res, checks


def check_C1():
    masses = [F(1, 6)] * 6
    pi = [F(1, 10), F(1, 2), F(3, 10), F(7, 10), F(1, 5), F(9, 10)]
    f = [F(1, 4), F(1, 4), F(1, 2), F(1, 2), F(3, 4), F(3, 4)]
    observed = [0, 2, 4]
    a_x = auc_star(view_law((masses[x], pi[x], x) for x in range(6)))
    a_f = auc_star(view_law((masses[x], pi[x], f[x]) for x in range(6)))
    res = {'distinct_outputs_on_observed': len({f[x] for x in observed}), 'distinct_observed_inputs': len(observed),
           'identity_distinct_on_observed': len(set(observed)), 'distinct_outputs_on_support': len(set(f)), 'support_size': 6,
           'auc_star_X': a_x, 'auc_star_f': a_f}
    checks = {
        'C1a_sample_unique_on_observed': res['distinct_outputs_on_observed'] == 3 == res['distinct_observed_inputs'],
        'C1a_identity_also_passes': res['identity_distinct_on_observed'] == 3,
        'C1b_two_to_one_on_support': res['distinct_outputs_on_support'] == 3,
        'C1c_strict_auc_loss': a_f < a_x,
        'C1c_hand_values': a_f == F(1091, 1782) and a_x == F(1461, 1782),
    }
    return res, checks


def c2_formula(M, n):
    return F(sum(comb(M, k) * (-1) ** (M - k) * 2 ** k * k ** n for k in range(M + 1)), (2 * M) ** n)


def c2_brute(M, n):
    ok = 0
    for seq in itertools.product(range(2 * M), repeat=n):
        s = set(seq)
        if not any((2 * j in s) and (2 * j + 1 in s) for j in range(M)):
            ok += 1
    return F(ok, (2 * M) ** n)


def check_C2():
    small = {f'{M},{n}': {'formula': c2_formula(M, n), 'brute': c2_brute(M, n)} for M, n in ((3, 4), (2, 5))}
    large = {n: c2_formula(1000, n) for n in (20, 60, 200)}
    res = {'small': {k: {kk: fr(vv) for kk, vv in v.items()} for k, v in small.items()},
           'large_M1000': {str(n): float(v) for n, v in large.items()},
           'poisson_approx_M1000': {str(n): math.exp(-n * (n - 1) / 4000) for n in (20, 60, 200)}}
    checks = {
        'C2a_formula_equals_brute_force': all(v['formula'] == v['brute'] for v in small.values()),
        'C2b_n20_in_0.88_0.94': F(88, 100) <= large[20] <= F(94, 100),
        'C2b_n60_in_0.35_0.47': F(35, 100) <= large[60] <= F(47, 100),
        'C2b_n200_below_1e-3': large[200] < F(1, 1000),
    }
    return res, checks


def check_D():
    masses = [F(1, 4)] * 4
    pi = [F(1, 5), F(4, 5), F(1, 5), F(4, 5)]
    eta = [F(1, 3), F(2, 3), F(1, 3), F(2, 3)]
    a_x = auc_star(view_law((masses[x], pi[x], x) for x in range(4)))
    a_r = auc_star(view_law((masses[x], pi[x], eta[x]) for x in range(4)))
    res = {'distinct_outputs': len(set(eta)), 'auc_star_X': a_x, 'auc_star_R': a_r}
    checks = {'D1_two_to_one': res['distinct_outputs'] == 2, 'D2_equal_and_four_fifths': a_r == a_x == F(4, 5)}
    return res, checks


def check_E():
    masses = [F(1, 10), F(1, 5), F(3, 10), F(1, 4), F(3, 20)]
    pi = [F(1, 5), F(2, 5), F(2, 5), F(3, 5), F(4, 5)]
    a_x = auc_star(view_law((masses[x], pi[x], x) for x in range(5)))
    rows = []
    for part in set_partitions(list(range(5))):
        label = {x: i for i, block in enumerate(part) for x in block}
        a = auc_star(view_law((masses[x], pi[x], label[x]) for x in range(5)))
        constant = all(len({pi[x] for x in block}) == 1 for block in part)
        rows.append({'partition': sorted(sorted(b) for b in part), 'auc_star': a, 'pi_constant_blocks': constant})
    equal = [r for r in rows if r['auc_star'] == a_x]
    res = {'n_partitions': len(rows), 'auc_star_X': a_x, 'n_equal': len(equal),
           'equal_partitions': [r['partition'] for r in equal], 'max_auc_star_among_unequal':
           max(r['auc_star'] for r in rows if r['auc_star'] != a_x),
           'partitions': [{'partition': r['partition'], 'auc_star': fr(r['auc_star']), 'pi_constant_blocks': r['pi_constant_blocks']}
                          for r in rows]}
    checks = {
        'E0_52_partitions': len(rows) == 52,
        'E1_all_at_most_auc_star_X': all(r['auc_star'] <= a_x for r in rows),
        'E2_equality_iff_pi_constant_blocks': all((r['auc_star'] == a_x) == r['pi_constant_blocks'] for r in rows),
        'E2_exactly_two_equal': sorted(map(str, res['equal_partitions'])) == sorted(map(str, [[[0], [1], [2], [3], [4]], [[0], [1, 2], [3], [4]]])),
        'E3_fifty_strictly_below': sum(r['auc_star'] < a_x for r in rows) == 50,
    }
    return res, checks


def run():
    out = {'results': {}, 'checks': {}}
    for name, fn in (('A', check_A), ('B', check_B), ('C1', check_C1), ('C2', check_C2), ('D', check_D), ('E', check_E)):
        res, checks = fn()
        out['results'][name] = {k: fr(v) for k, v in res.items()}
        out['checks'].update(checks)
    out['n_checks'] = len(out['checks'])
    out['n_pass'] = sum(bool(v) for v in out['checks'].values())
    out['all_pass'] = out['n_pass'] == out['n_checks']
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    out = run()
    out['provenance'] = {'preregistration': 'results/pcrl_mechanism_design_v1/corrections/CHECKS_PREREGISTRATION.md (commit b2958c76d)',
                         'unit': os.environ.get('MD_UNIT_ID'), 'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                         'python': sys.version.split()[0]}
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({'n_checks': out['n_checks'], 'n_pass': out['n_pass'], 'all_pass': out['all_pass'],
                      'failed': [k for k, v in out['checks'].items() if not v]}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
