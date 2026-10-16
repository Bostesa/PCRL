"""Mechanism-independent exact demonstrations required of any prototype (data-free).

Each demonstration states its law, access and expected outcome in EXPECTED (written before the first run) and
then computes the quantities exactly with mechanism_design.prototype.finite_law.

D1 decision-preserving bound. For any release R = (p, d) whose decision map d equals a fixed reference decision map
   d_ref, AUC*(S; R) >= AUC*(S; d_ref): the Bayes attacker may ignore p. Checked over every calibrated partition
   release that refines the decision partition, on a law where probabilities add information (strict inequality
   exists) and one where they cannot. A release allowed to change decisions can fall below AUC*(S; d_ref).
D2 true labels are not a disclosure floor. (a) S depends on Y only through label noise and S is independent of X:
   every release f(X) has AUC 1/2 while the label-only attacker has AUC > 1/2. (b) X carries an S-proxy that the
   labels do not: the full-information release exceeds the label-only AUC.
D3 fitted readers are not population floors. On a finite law, an injective release has Bayes AUC = AUC*(S; X); a
   logistic reader fitted on a finite auxiliary sample of releases reaches a lower expected AUC, and a reader can
   gain AUC from more auxiliary data without the release changing.
"""
from __future__ import annotations

import json
import math
import sys
from fractions import Fraction as F

import numpy as np

from mechanism_design import no_real_data
from mechanism_design.prototype.finite_law import FiniteLaw, argmax_first, bayes_auc, posterior_release, set_partitions

EXPECTED = {
    'D1': 'Every decision-preserving release has AUC* >= AUC*(S; d_ref); on law D1a some release is strictly above it; '
          'on law D1b (S constant within each decision block given the task posterior) every one equals it; a '
          'decision-changing release on D1a goes strictly below AUC*(S; d_ref).',
    'D2': 'Law D2a: AUC*(S; X) = 1/2 exactly (so every release has AUC* = 1/2) while label-only AUC*(S; Y1) > 1/2. '
          'Law D2b: AUC*(S; full release) > AUC*(S; Y1, Y2).',
    'D3': 'Injective release: AUC* = AUC*(S; X). Fitted logistic reader (on the release, n_aux = 200 and 5000, '
          '200 repetitions each): mean held-out AUC below AUC*, and higher at n_aux = 5000 than at 200.',
}


def table(cells, k1=2, k2=2):
    w, joint = [], []
    for m, tab in cells:
        t = [[[F(0)] * k2 for _ in range(k1)] for _ in range(2)]
        for (s, a, b), p in tab.items():
            t[s][a][b] = F(p)
        w.append(F(m))
        joint.append(tuple(tuple(tuple(r) for r in t[s]) for s in range(2)))
    return FiniteLaw(tuple(w), tuple(joint), k1, k2)


def cell(m, p_s1, p_y1_given_s, k2_dist=None):
    """One cell: P(S=1)=p_s1; P(Y1=1 | S=s) = p_y1_given_s[s]; Y2 degenerate at 0 unless k2_dist given (independent)."""
    tab = {}
    for s, ps in ((0, 1 - p_s1), (1, p_s1)):
        for a, pa in ((0, 1 - p_y1_given_s[s]), (1, p_y1_given_s[s])):
            if k2_dist is None:
                tab[(s, a, 0)] = tab.get((s, a, 0), F(0)) + ps * pa
            else:
                for b, pb in enumerate(k2_dist[s]):
                    tab[(s, a, b)] = tab.get((s, a, b), F(0)) + ps * pa * pb
    return (m, tab)


def d1():
    out = {}
    # D1a: four cells; decision blocks {0,1} (income decision 0) and {2,3} (decision 1); within a block the cells
    # differ in P(S) and slightly in the task posterior, so probabilities can add S information.
    lawa = table([cell(F(1, 4), F(2, 10), (F(3, 10), F(3, 10))), cell(F(1, 4), F(7, 10), (F(4, 10), F(4, 10))),
                  cell(F(1, 4), F(3, 10), (F(7, 10), F(7, 10))), cell(F(1, 4), F(8, 10), (F(6, 10), F(6, 10)))])
    # D1b: within each decision block the cells have equal P(S): probabilities cannot add S information.
    lawb = table([cell(F(1, 4), F(3, 10), (F(3, 10), F(3, 10))), cell(F(1, 4), F(3, 10), (F(4, 10), F(4, 10))),
                  cell(F(1, 4), F(6, 10), (F(7, 10), F(7, 10))), cell(F(1, 4), F(6, 10), (F(6, 10), F(6, 10)))])
    for name, law in (('D1a', lawa), ('D1b', lawb)):
        ref = posterior_release(law, 1, [(c,) for c in law.cells])
        dref = {c: argmax_first(ref[c]) for c in law.cells}
        auc_d = bayes_auc(law, lambda c: dref[c])
        rows = []
        for part in set_partitions(list(law.cells)):
            q = posterior_release(law, 1, part)
            dec = {c: argmax_first(q[c]) for c in law.cells}
            preserving = all(dec[c] == dref[c] for c in law.cells)
            auc = bayes_auc(law, lambda c: (q[c], dec[c]))
            rows.append({'partition': [list(b) for b in part], 'decision_preserving': preserving, 'auc': str(auc), 'auc_float': float(auc)})
        pres = [r for r in rows if r['decision_preserving']]
        chg = [r for r in rows if not r['decision_preserving']]
        out[name] = {'auc_decision_only': str(auc_d), 'auc_decision_only_float': float(auc_d),
                     'n_partitions': len(rows), 'n_decision_preserving': len(pres),
                     'min_auc_decision_preserving': min(r['auc_float'] for r in pres),
                     'max_auc_decision_preserving': max(r['auc_float'] for r in pres),
                     'all_preserving_at_least_decision_only': all(F(r['auc']) >= auc_d for r in pres),
                     'some_preserving_strictly_above': any(F(r['auc']) > auc_d for r in pres),
                     'min_auc_decision_changing': min((r['auc_float'] for r in chg), default=None),
                     'some_changing_strictly_below_decision_only': any(F(r['auc']) < auc_d for r in chg)}
    return out


def d2():
    out = {}
    # D2a: S independent of X (every cell has P(S=1)=1/2) but Y1's noise depends on S: P(Y1=1|S=0)=0.2+t, |S=1)=0.6+t.
    lawa = table([cell(F(1, 2), F(1, 2), (F(2, 10), F(6, 10))), cell(F(1, 2), F(1, 2), (F(3, 10), F(7, 10)))])
    auc_x = bayes_auc(lawa, lambda c: c)
    # label-only AUC: view = Y1 (not a function of X); computed from P(S, Y1)
    p = {(s, a): sum(lawa.w[c] * lawa.joint[c][s][a][0] for c in lawa.cells) for s in (0, 1) for a in (0, 1)}
    auc_y = label_only_auc(p)
    out['D2a'] = {'auc_star_X': str(auc_x), 'auc_label_only_Y1': str(auc_y), 'auc_label_only_float': float(auc_y),
                  'every_release_auc': str(auc_x), 'label_exceeds_every_release': auc_y > auc_x}
    # D2b: X = (task feature t, proxy z); Y1 depends on t only; S depends on z (and weakly on t).
    cells = []
    for t in (0, 1):
        for z in (0, 1):
            ps = F(2, 10) + F(6, 10) * z + F(1, 10) * t - F(1, 20)
            py = F(3, 10) + F(4, 10) * t
            cells.append(cell(F(1, 4), ps, (py, py)))
    lawb = table(cells)
    full = bayes_auc(lawb, lambda c: c)
    p = {(s, a): sum(lawb.w[c] * lawb.joint[c][s][a][0] for c in lawb.cells) for s in (0, 1) for a in (0, 1)}
    out['D2b'] = {'auc_star_full_release': str(full), 'auc_label_only': str(label_only_auc(p)),
                  'release_exceeds_label_only': full > label_only_auc(p)}
    return out


def label_only_auc(p):
    """Bayes AUC of S from a binary label with joint P(S=s, Y=a) = p[(s, a)]."""
    p1 = p[(1, 0)] + p[(1, 1)]
    p0 = p[(0, 0)] + p[(0, 1)]
    m = [(p[(1, a)] / p1, p[(0, a)] / p0) for a in (0, 1)]
    auc = F(0)
    for a1, a0 in m:
        for b1, b0 in m:
            if a1 * b0 > b1 * a0:
                auc += a1 * b0
            elif a1 * b0 == b1 * a0:
                auc += F(1, 2) * a1 * b0
    return auc


def empirical_auc(score, s):
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = s.sum(); n0 = len(s) - n1
    return (ranks[s == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def d3(seed=20261016):
    """Injective release of a 12-cell law: Bayes AUC equals AUC*(S;X); a fitted logistic reader on the release is lower."""
    rng = np.random.default_rng(seed)
    C = 12
    ps = [F(1, 10) + F(8, 10) * F((7 * c) % C, C - 1) for c in range(C)]   # S-propensity not monotone in the release
    py = [F(2, 10) + F(6, 10) * F(c, C - 1) for c in range(C)]
    law = table([cell(F(1, C), ps[c], (py[c], py[c])) for c in range(C)])
    rel = posterior_release(law, 1, [(c,) for c in law.cells])            # injective: one value per cell
    auc_star = bayes_auc(law, lambda c: rel[c])
    auc_x = bayes_auc(law, lambda c: c)
    from sklearn.linear_model import LogisticRegression
    w = np.array([float(x) for x in law.w]); psf = np.array([float(x) for x in ps])
    feat = np.array([[float(rel[c][1]), math.log(float(rel[c][1])) - math.log(float(rel[c][0]))] for c in range(C)])
    res = {}
    for n_aux in (200, 5000):
        aucs = []
        for rep in range(200):
            cs = rng.choice(C, size=n_aux, p=w); s = (rng.random(n_aux) < psf[cs]).astype(int)
            ct = rng.choice(C, size=20000, p=w); st = (rng.random(20000) < psf[ct]).astype(int)
            m = LogisticRegression(C=1.0, max_iter=1000).fit(feat[cs], s)
            aucs.append(empirical_auc(m.decision_function(feat[ct]), st))
        res[str(n_aux)] = {'mean_auc': float(np.mean(aucs)), 'sd': float(np.std(aucs))}
    # an oracle reader that maps each released value to its true posterior attains the Bayes AUC
    return {'auc_star_release': str(auc_star), 'auc_star_release_float': float(auc_star), 'auc_star_X_float': float(auc_x),
            'release_is_injective': len({rel[c] for c in law.cells}) == C, 'fitted_logistic': res,
            'fitted_below_bayes_both': all(v['mean_auc'] < float(auc_star) for v in res.values()),
            'more_aux_data_helps': res['5000']['mean_auc'] > res['200']['mean_auc']}


def main(argv=None):
    no_real_data.install()
    out = {'expected_before_run': EXPECTED, 'D1_decision_preserving': d1(), 'D2_true_label_not_floor': d2(), 'D3_fitted_vs_bayes': d3()}
    d1r, d2r, d3r = out['D1_decision_preserving'], out['D2_true_label_not_floor'], out['D3_fitted_vs_bayes']
    out['checks'] = {
        'D1a_bound_holds': d1r['D1a']['all_preserving_at_least_decision_only'],
        'D1a_strict_exists': d1r['D1a']['some_preserving_strictly_above'],
        'D1a_changing_goes_below': d1r['D1a']['some_changing_strictly_below_decision_only'],
        'D1b_bound_holds': d1r['D1b']['all_preserving_at_least_decision_only'],
        'D1b_all_equal': not d1r['D1b']['some_preserving_strictly_above'],
        'D2a_label_exceeds_release': d2r['D2a']['label_exceeds_every_release'],
        'D2b_release_exceeds_label': d2r['D2b']['release_exceeds_label_only'],
        'D3_injective_equals_X': d3r['release_is_injective'] and abs(d3r['auc_star_release_float'] - d3r['auc_star_X_float']) < 1e-12,
        'D3_fitted_below_bayes': d3r['fitted_below_bayes_both'],
        'D3_more_data_helps': d3r['more_aux_data_helps'],
    }
    path = argv[0] if argv else None
    text = json.dumps(out, indent=1)
    if path:
        open(path, 'w').write(text)
    print(json.dumps(out['checks'], indent=1))
    return 0 if all(out['checks'].values()) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
