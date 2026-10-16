"""Designer 1 data-free check: S-mixture marginalisation (MMP) on exact finite laws.

No real data. Laws are written down here. For each law:
  - eta_i(c)  = P(Y_i | X=c)                       (S-blind Bayes posterior; the reference, analogue of a perfect U0)
  - MMP_i(c)  = sum_s pibar_s P(Y_i | X=c, S=s)      (pibar = P(S) under the law)
  - identities: Brier(MMP) - Brier(eta) = E[(pi(X)-pibar)^2 ||q^1(X)-q^0(X)||^2]  (exact)
                NLL(MMP) - NLL(eta) <= I(S;X)                                      (joint convexity of KL)
  - exact Bayes AUC of S from each recipient alone and from the linked pair (probabilities + decisions)
  - exact frontier over all deterministic releases (every pair of set partitions of the cells, calibrated on each
    block -- calibrated block posteriors weakly dominate any other release that induces the same partition), and the
    local and sequential (S12, S21) design analogues, under two utility budgets measured against eta.
"""
from mechanism_design import no_real_data
no_real_data.install()

import itertools
import json
import math
import sys
from pathlib import Path

from mechanism_design.prototype.finite_law import (FiniteLaw, bayes_auc, mutual_information_bits, task_metrics,
                                                   set_partitions, argmax_first)

OUT = Path(__file__).resolve().parent / 'D1_MMP_FINITE_LAWS_RESULTS_V2.json'
BUDGETS = {'cap': {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005},
           'cap_plus_headroom': {'accuracy_loss': 0.016, 'nll_excess': 0.020, 'brier_excess': 0.009}}
TOL = 1e-12


def make_law(cells, k1, k2):
    """cells: list of (mass, pi1, q1 = {s: vec}, q2 = {s: vec}); Y1 indep Y2 given (X, S)."""
    w, joint = [], []
    for m, pi1, q1, q2 in cells:
        ps = (1 - pi1, pi1)
        tab = tuple(tuple(tuple(ps[s] * q1[s][a] * q2[s][b] for b in range(k2)) for a in range(k1)) for s in range(2))
        w.append(m); joint.append(tab)
    return FiniteLaw(tuple(w), tuple(joint), k1, k2), cells


def qs(law_cells, c, task, s):
    return law_cells[c][2][s] if task == 1 else law_cells[c][3][s]


def releases(law, cells):
    pibar = sum(law.w[c] * law.p_s(c, 1) for c in law.cells)
    eta = {t: {c: tuple(law.p_y(c, t)) for c in law.cells} for t in (1, 2)}
    mmp = {t: {c: tuple((1 - pibar) * qs(cells, c, t, 0)[y] + pibar * qs(cells, c, t, 1)[y]
                        for y in range(law.k1 if t == 1 else law.k2)) for c in law.cells} for t in (1, 2)}
    return pibar, eta, mmp


def key(v):
    return tuple(round(float(x), 12) for x in v)


def pair_auc(law, r1, r2):
    return float(bayes_auc(law, lambda c: (key(r1[c]), key(r2[c]))))


def single_auc(law, r):
    return float(bayes_auc(law, lambda c: key(r[c])))


def excess(law, task, q, ref):
    a, r = task_metrics(law, task, lambda c: q[c]), task_metrics(law, task, lambda c: ref[c])
    return {'accuracy_loss': float(r['accuracy'] - a['accuracy']), 'nll_excess': float(a['nll'] - r['nll']),
            'brier_excess': float(a['brier'] - r['brier'])}


def feasible(ex, budget):
    return all(ex[m] <= budget[m] + TOL for m in budget)


def calibrated(law, task, part):
    k = law.k1 if task == 1 else law.k2
    out = {}
    for block in part:
        m = sum(law.w[c] for c in block)
        v = tuple(sum(law.w[c] * law.p_y(c, task)[y] for c in block) / m for y in range(k))
        for c in block:
            out[c] = v
    return out


def identities(law, cells, pibar, eta, mmp):
    res = {}
    ixs = 0.0  # I(S;X) in nats
    for c in law.cells:
        p1 = float(law.p_s(c, 1))
        for s, ps in ((0, 1 - p1), (1, p1)):
            pb = float(pibar) if s == 1 else 1 - float(pibar)
            if ps > 0:
                ixs += float(law.w[c]) * ps * math.log(ps / pb)
    res['I_S_X_nats'] = ixs
    for t in (1, 2):
        ex = excess(law, t, mmp[t], eta[t])
        k = law.k1 if t == 1 else law.k2
        brier_formula = sum(float(law.w[c]) * (float(law.p_s(c, 1)) - float(pibar)) ** 2 *
                            sum((qs(cells, c, t, 1)[y] - qs(cells, c, t, 0)[y]) ** 2 for y in range(k)) for c in law.cells)
        res[f'task{t}'] = {'mmp_excess_vs_eta': ex, 'brier_identity_rhs': brier_formula,
                           'brier_identity_abs_error': abs(ex['brier_excess'] - brier_formula),
                           'nll_bound_holds': ex['nll_excess'] <= ixs + 1e-12}
    return res


def frontier(law, eta, mmp, name):
    cells = list(law.cells)
    parts = list(set_partitions(cells))
    per = {}
    for t in (1, 2):
        rows = []
        for p in parts:
            r = calibrated(law, t, p)
            ex = excess(law, t, r, eta[t])
            rows.append({'part': p, 'rel': r, 'ex': ex, 'auc': single_auc(law, r)})
        per[t] = rows
    out = {'n_partitions_per_task': len(parts)}
    for bname, B in BUDGETS.items():
        f1 = [r for r in per[1] if feasible(r['ex'], B)]
        f2 = [r for r in per[2] if feasible(r['ex'], B)]
        best = None
        for a in f1:
            for b in f2:
                v = pair_auc(law, a['rel'], b['rel'])
                if best is None or v < best[0] - 1e-15:
                    best = (v, a, b)
        # local design: each recipient minimises its own AUC (ties: smaller summed normalised cost)
        cost = lambda r: sum(r['ex'][m] / B[m] for m in B)
        l1 = min(f1, key=lambda r: (round(r['auc'], 12), cost(r)))
        l2 = min(f2, key=lambda r: (round(r['auc'], 12), cost(r)))
        s12 = min(f2, key=lambda b: (round(pair_auc(law, l1['rel'], b['rel']), 12), cost(b)))   # recipient 1 first (local), then 2 sees the pair
        s21 = min(f1, key=lambda a: (round(pair_auc(law, a['rel'], l2['rel']), 12), cost(a)))
        out[bname] = {
            'feasible_partitions': [len(f1), len(f2)],
            'global_min_pair_auc': best[0], 'global_parts': [best[1]['part'], best[2]['part']],
            'local_pair_auc': pair_auc(law, l1['rel'], l2['rel']), 'local_single_aucs': [l1['auc'], l2['auc']],
            'S12_pair_auc': pair_auc(law, l1['rel'], s12['rel']), 'S21_pair_auc': pair_auc(law, s21['rel'], l2['rel']),
        }
    return out


def summarize(name, law, cells, with_frontier=True):
    pibar, eta, mmp = releases(law, cells)
    rec = {'law': name, 'pibar': float(pibar),
           'AUC_S_X': float(bayes_auc(law, lambda c: c)),
           'eta': {'pair_auc': pair_auc(law, eta[1], eta[2]), 'single_auc': [single_auc(law, eta[1]), single_auc(law, eta[2])],
                   'metrics': [{k: float(v) for k, v in task_metrics(law, t, lambda c: eta[t][c]).items()} for t in (1, 2)]},
           'mmp': {'pair_auc': pair_auc(law, mmp[1], mmp[2]), 'single_auc': [single_auc(law, mmp[1]), single_auc(law, mmp[2])],
                   'excess_vs_eta': [excess(law, t, mmp[t], eta[t]) for t in (1, 2)],
                   'feasible': {b: all(feasible(excess(law, t, mmp[t], eta[t]), B) for t in (1, 2)) for b, B in BUDGETS.items()}},
           'identities': identities(law, cells, pibar, eta, mmp)}
    # calibrated MMP: same partition as MMP, calibrated on its blocks (weakly better utility, same or lower AUC)
    cal = {}
    for t in (1, 2):
        groups = {}
        for c in law.cells:
            groups.setdefault(key(mmp[t][c]), []).append(c)
        cal[t] = calibrated(law, t, list(groups.values()))
    rec['mmp_calibrated'] = {'pair_auc': pair_auc(law, cal[1], cal[2]),
                             'excess_vs_eta': [excess(law, t, cal[t], eta[t]) for t in (1, 2)]}
    if with_frontier:
        rec['frontier'] = frontier(law, eta, mmp, name)
    return rec


def law_N():
    """Negative: the tasks depend on X only through S (chain X -> S -> Y). Required disclosure."""
    q1 = {0: (0.35, 0.65), 1: (0.70, 0.30)}
    q2 = {0: (0.35, 0.10, 0.55), 1: (0.40, 0.55, 0.05)}
    cells = [(0.25, p, q1, q2) for p in (0.10, 0.37, 0.63, 0.90)]
    return make_law(cells, 2, 3)


def law_F():
    """Feasible: Z is a pure SEX proxy (moves P(S|X), no task effect given S); W is a task feature independent of S."""
    q1 = {0: {0: (0.70, 0.30), 1: (0.80, 0.20)}, 1: {0: (0.25, 0.75), 1: (0.38, 0.62)}}
    q2 = {0: {0: (0.20, 0.20, 0.60), 1: (0.25, 0.55, 0.20)}, 1: {0: (0.60, 0.10, 0.30), 1: (0.65, 0.25, 0.10)}}
    cells = []
    for z, piz in enumerate((0.30, 0.47, 0.64)):
        for w in (0, 1):
            cells.append((1 / 6, piz, q1[w], q2[w]))
    return make_law(cells, 2, 3)


def law_R():
    """Residual: Z pure proxy; H ('hours') moves P(S|X) AND the tasks within sex. MMP removes Z, keeps H."""
    q1 = {0: {0: (0.80, 0.20), 1: (0.88, 0.12)}, 1: {0: (0.35, 0.65), 1: (0.50, 0.50)}}
    q2 = {0: {0: (0.25, 0.30, 0.45), 1: (0.25, 0.60, 0.15)}, 1: {0: (0.45, 0.10, 0.45), 1: (0.60, 0.30, 0.10)}}
    cells = []
    ph = (0.35, 0.65)
    for z, a in enumerate((-0.6, 0.0, 0.6)):
        for h, b in enumerate((0.5, -0.5)):
            pi = 1 / (1 + math.exp(-(a + b - 0.12)))
            cells.append((ph[h] / 3, pi, q1[h], q2[h]))
    return make_law(cells, 2, 3)


def main():
    out = {}
    for name, fn in (('N_required_disclosure', law_N), ('F_pure_proxy', law_F), ('R_task_relevant_proxy', law_R)):
        law, cells = fn()
        out[name] = summarize(name, law, cells)
        print(name, json.dumps({k: out[name][k] for k in ('AUC_S_X', 'eta', 'mmp', 'mmp_calibrated')}, default=str)[:3000])
        print(name, 'frontier', json.dumps(out[name]['frontier'], default=str)[:3000])
        sys.stdout.flush()
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
