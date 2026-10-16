"""Verifier checks for designer 1 (MMP: SEX-mixture marginalisation). Synthetic / data-free.

1. Recompute laws N, F, R (law definitions re-entered from the proposal's prototype) with independent code:
   AUC*(S;X), eta and MMP pair/single Bayes AUC, excess vs eta, the Brier identity, NLL = E KL(eta||m) <= I(S;X),
   block-calibrated MMP, and the exact frontier over all pairs of set partitions with local / S12 / S21 controls.
2. Law R: AUC*(S;H) and AUC*(S;Z) (is the MMP residual exactly the H-channel?).
3. Within-SEX Brier formula (constant gap) on random laws; search for a non-constant-gap law where removing Delta
   LOWERS one group's within-SEX Brier (the 'both groups rise' signature is condition-dependent).
4. Counterexample: a law where MMP RAISES the pair Bayes AUC relative to eta (no general monotonicity).
5. Brier-optimal constant mixing weight w* = E[pi ||g||^2] / E[||g||^2] vs pibar (utility-only improvement).
"""
from mechanism_design import no_real_data
no_real_data.install()

import json
import math
import random
from fractions import Fraction as Fr
from pathlib import Path

from mechanism_design_independent.exact import (Cell, check_law, pbar, bayes_auc, auc_from_x, metrics, excess,
                                                within_sex_brier, mutual_info_sx_nats, partitions, block_posterior)

OUT = Path(__file__).resolve().parent.parent / 'results' / 'pcrl_mechanism_design_v1' / 'roles' / 'math_verifier' / 'V_MMP_EXACT_V1.json'
BUDGETS = {'cap': {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005},
           'cap_plus_headroom': {'accuracy_loss': 0.016, 'nll_excess': 0.020, 'brier_excess': 0.009}}


def f(x):
    return float(x)


def law_N():
    q1 = ((Fr(35, 100), Fr(65, 100)), (Fr(70, 100), Fr(30, 100)))
    q2 = ((Fr(35, 100), Fr(10, 100), Fr(55, 100)), (Fr(40, 100), Fr(55, 100), Fr(5, 100)))
    return [Cell(Fr(1, 4), p, q1, q2, name=f'pi={p}') for p in (Fr(10, 100), Fr(37, 100), Fr(63, 100), Fr(90, 100))]


def law_F():
    # outer index w (task feature, independent of S), inner index s
    q1 = {0: ((Fr(70, 100), Fr(30, 100)), (Fr(80, 100), Fr(20, 100))),
          1: ((Fr(25, 100), Fr(75, 100)), (Fr(38, 100), Fr(62, 100)))}
    q2 = {0: ((Fr(20, 100), Fr(20, 100), Fr(60, 100)), (Fr(25, 100), Fr(55, 100), Fr(20, 100))),
          1: ((Fr(60, 100), Fr(10, 100), Fr(30, 100)), (Fr(65, 100), Fr(25, 100), Fr(10, 100)))}
    cells = []
    for z, pz in enumerate((Fr(30, 100), Fr(47, 100), Fr(64, 100))):
        for w in (0, 1):
            cells.append(Cell(Fr(1, 6), pz, q1[w], q2[w], name=('z', z, 'w', w)))
    return cells


def law_R():
    q1 = {0: ((0.80, 0.20), (0.88, 0.12)), 1: ((0.35, 0.65), (0.50, 0.50))}
    q2 = {0: ((0.25, 0.30, 0.45), (0.25, 0.60, 0.15)), 1: ((0.45, 0.10, 0.45), (0.60, 0.30, 0.10))}
    ph = (0.35, 0.65)
    cells = []
    for z, a in enumerate((-0.6, 0.0, 0.6)):
        for h, b in enumerate((0.5, -0.5)):
            pi = 1 / (1 + math.exp(-(a + b - 0.12)))
            cells.append(Cell(ph[h] / 3, pi, q1[h], q2[h], name=('z', z, 'h', h)))
    return cells


def mmp(cells, t, weight=None):
    w = pbar(cells) if weight is None else weight
    return [tuple((1 - w) * a + w * b for a, b in zip(*c.q[t])) for c in cells]


def eta(cells, t):
    return [c.eta(t) for c in cells]


def pair_auc(cells, r1, r2):
    return bayes_auc(cells, lambda i: (r1[i], r2[i]))


def single_auc(cells, r):
    return bayes_auc(cells, lambda i: r[i])


def feasible(ex, B):
    return all(f(ex[m]) <= B[m] + 1e-12 for m in B)


def kl(p, q):
    return sum(f(a) * math.log(f(a) / f(b)) for a, b in zip(p, q) if a > 0)


def analyse(name, cells):
    check_law(cells)
    pb = pbar(cells)
    E = {t: eta(cells, t) for t in (1, 2)}
    M = {t: mmp(cells, t) for t in (1, 2)}
    rec = {'law': name, 'pibar': f(pb), 'AUC_S_X': f(auc_from_x(cells)), 'I_S_X_nats': mutual_info_sx_nats(cells),
           'eta_pair_auc': f(pair_auc(cells, E[1], E[2])),
           'eta_single_auc': [f(single_auc(cells, E[t])) for t in (1, 2)],
           'mmp_pair_auc': f(pair_auc(cells, M[1], M[2])),
           'mmp_single_auc': [f(single_auc(cells, M[t])) for t in (1, 2)], 'tasks': {}}
    for t in (1, 2):
        ex = excess(cells, t, M[t], E[t])
        rhs = sum(c.w * (c.pi - pb) ** 2 * sum((b - a) ** 2 for a, b in zip(*c.q[t])) for c in cells)
        ekl = sum(f(c.w) * kl(c.eta(t), M[t][i]) for i, c in enumerate(cells))
        # within-SEX Brier change of MMP vs eta
        ws = {s: f(within_sex_brier(cells, t, M[t], s) - within_sex_brier(cells, t, E[t], s)) for s in (0, 1)}
        # Brier-optimal constant weight
        num = sum(c.w * c.pi * sum((b - a) ** 2 for a, b in zip(*c.q[t])) for c in cells)
        den = sum(c.w * sum((b - a) ** 2 for a, b in zip(*c.q[t])) for c in cells)
        wstar = num / den if den else pb
        price_wstar = sum(c.w * (c.pi - wstar) ** 2 * sum((b - a) ** 2 for a, b in zip(*c.q[t])) for c in cells)
        rec['tasks'][t] = {'mmp_excess_vs_eta': {k: f(v) for k, v in ex.items()},
                           'brier_identity_rhs': f(rhs), 'brier_identity_abs_err': abs(f(ex['brier_excess']) - f(rhs)),
                           'nll_excess_minus_E_KL': abs(f(ex['nll_excess']) - ekl),
                           'nll_bound_I_S_X_holds': f(ex['nll_excess']) <= rec['I_S_X_nats'] + 1e-12,
                           'within_sex_brier_change_mmp_vs_eta': {'men_s0': ws[0], 'women_s1': ws[1]},
                           'brier_optimal_constant_weight': f(wstar), 'brier_price_at_wstar': f(price_wstar)}
    rec['mmp_feasible'] = {b: all(feasible(rec['tasks'][t]['mmp_excess_vs_eta'], B) for t in (1, 2)) for b, B in BUDGETS.items()}
    # block-calibrated MMP
    cal = {}
    for t in (1, 2):
        groups = {}
        for i in range(len(cells)):
            groups.setdefault(tuple(round(f(x), 12) for x in M[t][i]), []).append(i)
        cal[t] = block_posterior(cells, t, list(groups.values()))
    rec['mmp_block_calibrated'] = {'pair_auc': f(pair_auc(cells, cal[1], cal[2])),
                                   'excess_vs_eta': [{k: f(v) for k, v in excess(cells, t, cal[t], E[t]).items()} for t in (1, 2)]}
    # exact frontier
    parts = list(partitions(len(cells)))
    per = {}
    for t in (1, 2):
        rows = []
        for p in parts:
            r = block_posterior(cells, t, p)
            rows.append({'part': p, 'rel': r, 'ex': excess(cells, t, r, E[t]), 'auc': f(single_auc(cells, r))})
        per[t] = rows
    fr = {'n_partitions': len(parts)}
    for bname, B in BUDGETS.items():
        f1 = [r for r in per[1] if feasible(r['ex'], B)]
        f2 = [r for r in per[2] if feasible(r['ex'], B)]
        cost = lambda r: sum(f(r['ex'][m]) / B[m] for m in B)
        best = min(((f(pair_auc(cells, a['rel'], b['rel'])), a['part'], b['part']) for a in f1 for b in f2), key=lambda x: x[0])
        l1 = min(f1, key=lambda r: (round(r['auc'], 12), cost(r)))
        l2 = min(f2, key=lambda r: (round(r['auc'], 12), cost(r)))
        s12 = min(f2, key=lambda b: (round(f(pair_auc(cells, l1['rel'], b['rel'])), 12), cost(b)))
        s21 = min(f1, key=lambda a: (round(f(pair_auc(cells, a['rel'], l2['rel'])), 12), cost(a)))
        fr[bname] = {'n_feasible': [len(f1), len(f2)], 'global_min_pair_auc': best[0], 'global_parts': [best[1], best[2]],
                     'local_pair_auc': f(pair_auc(cells, l1['rel'], l2['rel'])), 'local_single': [l1['auc'], l2['auc']],
                     'S12_pair_auc': f(pair_auc(cells, l1['rel'], s12['rel'])),
                     'S21_pair_auc': f(pair_auc(cells, s21['rel'], l2['rel']))}
    rec['frontier'] = fr
    return rec


def within_sex_formula_check(n_laws=300, seed=7):
    """Constant gap g: men's change = ||g||^2 [v(3pb-1)+k3]/(1-pb); women's = ||g||^2 [v(2-3pb)-k3]/pb."""
    rng = random.Random(seed)
    worst = 0.0
    for _ in range(n_laws):
        n = rng.randint(3, 8)
        w = [rng.random() for _ in range(n)]
        s = sum(w)
        w = [x / s for x in w]
        pis = [rng.uniform(0.05, 0.95) for _ in range(n)]
        g = rng.uniform(-0.2, 0.2)
        cells = []
        for i in range(n):
            q0 = rng.uniform(max(0.0, -g) + 0.01, min(1.0, 1 - g) - 0.01)
            cells.append(Cell(w[i], pis[i], ((1 - q0, q0), (1 - q0 - g, q0 + g)), ((0.5, 0.5), (0.5, 0.5))))
        pb = sum(c.w * c.pi for c in cells)
        v = sum(c.w * (c.pi - pb) ** 2 for c in cells)
        k3 = sum(c.w * (c.pi - pb) ** 3 for c in cells)
        g2 = 2 * g * g
        E = eta(cells, 1)
        M = mmp(cells, 1)
        men = within_sex_brier(cells, 1, M, 0) - within_sex_brier(cells, 1, E, 0)
        wom = within_sex_brier(cells, 1, M, 1) - within_sex_brier(cells, 1, E, 1)
        fm = g2 * (v * (3 * pb - 1) + k3) / (1 - pb)
        fw = g2 * (v * (2 - 3 * pb) - k3) / pb
        worst = max(worst, abs(men - fm), abs(wom - fw))
    return {'n_laws': n_laws, 'max_abs_error': worst, 'pass': worst < 1e-12}


def nonconstant_gap_counterexample(seed=11, tries=20000):
    """Find a law (pibar near 0.47) where removing Delta LOWERS one SEX group's within-SEX Brier."""
    rng = random.Random(seed)
    for _ in range(tries):
        n = 4
        w = [0.25] * n
        pis = [rng.uniform(0.1, 0.9) for _ in range(n)]
        pb = sum(x * y for x, y in zip(w, pis))
        if abs(pb - 0.47) > 0.03:
            continue
        cells = []
        for i in range(n):
            q0 = rng.uniform(0.05, 0.95)
            q1 = rng.uniform(0.05, 0.95)
            cells.append(Cell(w[i], pis[i], ((1 - q0, q0), (1 - q1, q1)), ((0.5, 0.5), (0.5, 0.5))))
        E = eta(cells, 1)
        M = mmp(cells, 1)
        men = within_sex_brier(cells, 1, M, 0) - within_sex_brier(cells, 1, E, 0)
        wom = within_sex_brier(cells, 1, M, 1) - within_sex_brier(cells, 1, E, 1)
        if men < -1e-4 or wom < -1e-4:
            return {'found': True, 'pibar': pb, 'pis': pis,
                    'q_female_minus_male': [c.q[1][1][1] - c.q[1][0][1] for c in cells],
                    'men_change': men, 'women_change': wom,
                    'population_change': sum(c.w * (c.pi - pb) ** 2 * 2 * (c.q[1][1][1] - c.q[1][0][1]) ** 2 for c in cells)}
    return {'found': False}


def mmp_raises_recovery_counterexample():
    """eta merges two cells (the pi-term cancels the within-SEX difference); MMP separates them."""
    qA = ((Fr(1, 2), Fr(1, 2)), (Fr(3, 4), Fr(1, 4)))        # P(Y=1|A,s=0)=1/2, P(Y=1|A,s=1)=1/4
    qB = ((Fr(3, 20), Fr(17, 20)), (Fr(13, 20), Fr(7, 20)))  # P(Y=1|B,s=0)=17/20, P(Y=1|B,s=1)=7/20
    cells = [Cell(Fr(1, 2), Fr(1, 5), qA, qA, name='A'), Cell(Fr(1, 2), Fr(4, 5), qB, qB, name='B')]
    check_law(cells)
    E = {t: eta(cells, t) for t in (1, 2)}
    M = {t: mmp(cells, t) for t in (1, 2)}
    return {'eta_A_B_task1': [str(E[1][0][1]), str(E[1][1][1])], 'mmp_A_B_task1': [str(M[1][0][1]), str(M[1][1][1])],
            'eta_pair_auc': f(pair_auc(cells, E[1], E[2])), 'mmp_pair_auc': f(pair_auc(cells, M[1], M[2])),
            'AUC_S_X': f(auc_from_x(cells)),
            'note': 'MMP is not monotone in Bayes recovery: here it raises the pair Bayes AUC from 0.5 to AUC*(S;X).'}


def main():
    out = {}
    for name, fn in (('N_required_disclosure', law_N), ('F_pure_proxy', law_F), ('R_task_relevant_proxy', law_R)):
        out[name] = analyse(name, fn())
        print(name, json.dumps({k: v for k, v in out[name].items() if k not in ('frontier', 'tasks')}, default=str))
        print(name, 'frontier', json.dumps({b: {k: v for k, v in out[name]['frontier'][b].items() if k != 'global_parts'}
                                            for b in BUDGETS}, default=str), flush=True)
    R = law_R()
    out['R_task_relevant_proxy']['AUC_S_H'] = f(bayes_auc(R, lambda i: R[i].name[3]))
    out['R_task_relevant_proxy']['AUC_S_Z'] = f(bayes_auc(R, lambda i: R[i].name[1]))
    out['within_sex_formula_constant_gap'] = within_sex_formula_check()
    out['nonconstant_gap_counterexample'] = nonconstant_gap_counterexample()
    out['mmp_raises_recovery_counterexample'] = mmp_raises_recovery_counterexample()
    print(json.dumps({k: out[k] for k in ('within_sex_formula_constant_gap', 'nonconstant_gap_counterexample',
                                          'mmp_raises_recovery_counterexample')}, default=str))
    print('law R AUC*(S;H) =', out['R_task_relevant_proxy']['AUC_S_H'], 'AUC*(S;Z) =', out['R_task_relevant_proxy']['AUC_S_Z'])
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
