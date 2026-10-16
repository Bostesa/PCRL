"""Phase 2: independent exact recomputation of the six finite laws of the MMP prototype (data-free).

Law definitions re-entered from mechanism_design/prototype/mmp.py (read, not imported). All quantities computed with
this package's own toolkit (exact Fractions): AUC*(S;X), I(S;X), twin and MMP pair/single Bayes AUCs, MMP excess vs
eta (accuracy, NLL, summed Brier), the Brier identity (exact equality) and NLL = E KL(eta||m) <= I(S;X), within-SEX
increases, w*, and the exhaustive frontier (global minimum pair Bayes AUC over all calibrated partition pairs feasible
at a budget) with local / S12 / S21 designs. Local designs: minimum own AUC, ties broken by lower summed-Brier excess;
any remaining ties are reported as the [min, max] range of the resulting pair AUC. Finally the numbers are compared
with the coordinator's PROTOTYPE_RESULTS.json (read as data).
"""
from mechanism_design import no_real_data
no_real_data.install()

import json
import math
from fractions import Fraction as F
from pathlib import Path

from mechanism_design_independent.exact import (Cell, check_law, pbar, bayes_auc, auc_from_x, excess, within_sex_brier,
                                                mutual_info_sx_nats, partitions, block_posterior)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'pcrl_mechanism_design_v1' / 'roles' / 'math_verifier' / 'V_PROTO_FINITE_V1.json'
COORD = ROOT / 'results' / 'pcrl_mechanism_design_v1' / 'prototype' / 'PROTOTYPE_RESULTS.json'
BUDGETS = {'cap': (F(1, 100), 0.010, F(5, 1000)), 'cap_plus_headroom': (F(16, 1000), 0.020, F(9, 1000))}   # (acc, nll, brier)


def d(x):
    return F(x)


def cell(m, pi, y1, q20, q21):
    """y1 = (P(Y1=1|S=0), P(Y1=1|S=1)); q2 vectors per SEX."""
    q1 = ((1 - d(y1[0]), d(y1[0])), (1 - d(y1[1]), d(y1[1])))
    return Cell(d(m), d(pi), q1, (tuple(map(d, q20)), tuple(map(d, q21))))


def laws():
    L = {}
    L['L_NEG'] = [cell(F(1, 6), pi, ('0.55', '0.30'), ('0.40', '0.15', '0.45'), ('0.40', '0.50', '0.10'))
                  for pi in ('0.15', '0.5', '0.85') for _w in (0, 1)]
    cs = []
    for pi in ('0.2', '0.5', '0.8'):
        for w in (0, 1):
            y1 = (d('0.30') + d('0.35') * w, d('0.20') + d('0.35') * w)
            q0 = ('0.15', '0.25', '0.60') if w == 0 else ('0.60', '0.10', '0.30')
            q1 = ('0.15', '0.40', '0.45') if w == 0 else ('0.60', '0.30', '0.10')
            cs.append(cell(F(1, 6), pi, y1, q0, q1))
    L['L_PROXY'] = cs
    cs = []
    for pi in ('0.2', '0.5', '0.8'):
        for w in (0, 1):
            y1 = (d('0.30') + d('0.35') * w, d('0.20') + d('0.35') * w)
            q0 = ('0.30', '0.20', '0.50') if w == 0 else ('0.60', '0.15', '0.25')
            q1 = ('0.25', '0.55', '0.20') if w == 0 else ('0.55', '0.35', '0.10')
            cs.append(cell(F(1, 6), pi, y1, q0, q1))
    L['L_PROXY_FLIP'] = cs
    cs = []
    for h in (0, 1):
        for z in (0, 1, 2):
            pi = d('0.20') + d('0.30') * h + d('0.15') * z
            y1 = (d('0.70') - d('0.30') * h, d('0.60') - d('0.30') * h)
            q0 = ('0.45', '0.15', '0.40') if h == 0 else ('0.25', '0.40', '0.35')
            q1 = ('0.45', '0.30', '0.25') if h == 0 else ('0.25', '0.60', '0.15')
            cs.append(cell(F(1, 6), pi, y1, q0, q1))
    L['L_MIXED'] = cs
    cs = []
    for h in (0, 1, 2):
        for e in (0, 1):
            pi = d('0.25') + d('0.25') * h + d('0.05') * e
            y1 = (d('0.65') - d('0.15') * h + d('0.15') * e, d('0.55') - d('0.15') * h + d('0.15') * e)
            c0 = d('0.45') + d('0.10') * e - d('0.05') * h
            m0 = d('0.15') + d('0.05') * h
            m1 = d('0.35') + d('0.05') * h
            cs.append(cell(F(1, 6), pi, y1, (c0, m0, 1 - c0 - m0), (c0, m1, 1 - c0 - m1)))
    L['L_NOPROXY'] = cs
    cs = []
    for a in (0, 1):
        for b in (0, 1):
            pi = d('0.15') if (a ^ b) == 0 else d('0.85')
            y1 = (d('0.30') + d('0.40') * a, d('0.25') + d('0.40') * a)
            q0 = ('0.60', '0.15', '0.25') if b == 0 else ('0.20', '0.30', '0.50')
            q1 = ('0.55', '0.25', '0.20') if b == 0 else ('0.15', '0.40', '0.45')
            cs.append(cell(F(1, 4), pi, y1, q0, q1))
    L['L_XOR'] = cs
    for v in L.values():
        check_law(v)
    return L


def mix(c, t, w):
    return tuple((1 - w) * a + w * b for a, b in zip(*c.q[t]))


def feasible(ex, B):
    return ex['accuracy_loss'] <= B[0] and ex['nll_excess'] <= B[1] + 1e-15 and ex['brier_excess'] <= B[2]


def kl(p, q):
    return sum(float(a) * math.log(float(a) / float(b)) for a, b in zip(p, q) if a > 0)


def pair(cells, r1, r2):
    return bayes_auc(cells, lambda i: (r1[i], r2[i]))


def single(cells, r):
    return bayes_auc(cells, lambda i: r[i])


def frontier(cells, eta):
    parts = list(partitions(len(cells)))
    tab = {}
    for t in (1, 2):
        rows = []
        for p in parts:
            r = block_posterior(cells, t, p)
            rows.append({'part': p, 'rel': r, 'ex': excess(cells, t, r, eta[t]), 'own': single(cells, r)})
        tab[t] = rows
    out = {'n_partitions': len(parts)}
    for bn, B in BUDGETS.items():
        f1 = [r for r in tab[1] if feasible(r['ex'], B)]
        f2 = [r for r in tab[2] if feasible(r['ex'], B)]
        res = {'n_feasible': [len(f1), len(f2)]}
        if not f1 or not f2:
            res['feasible'] = False
            out[bn] = res
            continue
        best = min(pair(cells, a['rel'], b['rel']) for a in f1 for b in f2)

        def local_set(rows):
            m = min(r['own'] for r in rows)
            tied = [r for r in rows if r['own'] == m]
            mb = min(r['ex']['brier_excess'] for r in tied)
            return [r for r in tied if r['ex']['brier_excess'] == mb]
        L1, L2 = local_set(f1), local_set(f2)
        loc = [pair(cells, a['rel'], b['rel']) for a in L1 for b in L2]
        s12 = [min(pair(cells, a['rel'], b['rel']) for b in f2) for a in L1]
        s21 = [min(pair(cells, a['rel'], b['rel']) for a in f1) for b in L2]
        res.update({'feasible': True, 'global_min_pair_auc': float(best),
                    'local_pair_auc_range': [float(min(loc)), float(max(loc))],
                    'local_single_aucs': [float(L1[0]['own']), float(L2[0]['own'])],
                    'local_tie_set_sizes': [len(L1), len(L2)],
                    'S12_pair_auc_range': [float(min(s12)), float(max(s12))],
                    'S21_pair_auc_range': [float(min(s21)), float(max(s21))]})
        out[bn] = res
    return out


def analyse(name, cells):
    pb = pbar(cells)
    eta = {t: [c.eta(t) for c in cells] for t in (1, 2)}
    m = {t: [mix(c, t, pb) for c in cells] for t in (1, 2)}
    isx = mutual_info_sx_nats(cells)
    rec = {'pibar': str(pb), 'auc_star_X': float(auc_from_x(cells)), 'I_S_X_nats': isx,
           'twin': {'pair': float(pair(cells, eta[1], eta[2])), 'income': float(single(cells, eta[1])),
                    'occupation': float(single(cells, eta[2]))},
           'mmp': {'pair': float(pair(cells, m[1], m[2])), 'pair_exact': str(pair(cells, m[1], m[2])),
                   'income': float(single(cells, m[1])), 'occupation': float(single(cells, m[2]))},
           'tasks': {}}
    for t in (1, 2):
        ex = excess(cells, t, m[t], eta[t])
        rhs = sum(c.w * (c.pi - pb) ** 2 * sum((b - a) ** 2 for a, b in zip(*c.q[t])) for c in cells)
        ekl = sum(float(c.w) * kl(eta[t][i], m[t][i]) for i, c in enumerate(cells))
        ws = {s: within_sex_brier(cells, t, m[t], s) - within_sex_brier(cells, t, eta[t], s) for s in (0, 1)}
        g2 = [sum((b - a) ** 2 for a, b in zip(*c.q[t])) for c in cells]
        den = sum(c.w * g for c, g in zip(cells, g2))
        wstar = (sum(c.w * c.pi * g for c, g in zip(cells, g2)) / den) if den else None
        price = (lambda w_: sum(c.w * (c.pi - w_) ** 2 * g for c, g in zip(cells, g2)))
        rec['tasks'][f'task{t}'] = {
            'excess_vs_eta': {'accuracy_loss': float(ex['accuracy_loss']), 'nll_excess': ex['nll_excess'], 'brier_excess': float(ex['brier_excess'])},
            'brier_identity_exact': ex['brier_excess'] == rhs, 'nll_minus_E_KL': abs(ex['nll_excess'] - ekl),
            'nll_bound_holds': ex['nll_excess'] <= isx + 1e-12,
            'within_sex_increase': {'male': float(ws[0]), 'female': float(ws[1]), 'population': float(ex['brier_excess'])},
            'w_star': None if wstar is None else float(wstar),
            'price_at_pibar': float(price(pb)), 'price_at_w_star': None if wstar is None else float(price(wstar))}
    rec['mmp']['feasible'] = {bn: all(feasible(excess(cells, t, m[t], eta[t]), B) for t in (1, 2)) for bn, B in BUDGETS.items()}
    rec['frontier'] = frontier(cells, eta)
    return rec


def compare(mine, coord):
    """Max absolute difference between my numbers and the coordinator's, per law (data comparison only)."""
    out = {}
    for name, r in mine.items():
        c = coord['finite'][name]
        diffs = {
            'auc_star_X': abs(r['auc_star_X'] - c['auc_star_X']),
            'twin_pair': abs(r['twin']['pair'] - c['twin_eta']['pair_auc']),
            'twin_income': abs(r['twin']['income'] - c['twin_eta']['income_auc']),
            'twin_occupation': abs(r['twin']['occupation'] - c['twin_eta']['occupation_auc']),
            'mmp_pair': abs(r['mmp']['pair'] - c['mmp']['pair_auc']),
            'mmp_income': abs(r['mmp']['income'] - c['mmp']['income_auc']),
            'mmp_occupation': abs(r['mmp']['occupation'] - c['mmp']['occupation_auc'])}
        for t in ('task1', 'task2'):
            for k in ('accuracy_loss', 'nll_excess', 'brier_excess'):
                diffs[f'{t}_{k}'] = abs(r['tasks'][t]['excess_vs_eta'][k] - c['mmp']['excess_vs_eta'][t][k])
            for g in ('male', 'female'):
                diffs[f'{t}_within_{g}'] = abs(r['tasks'][t]['within_sex_increase'][g] - c['subgroup_brier_increase_mmp_vs_eta'][t][g])
            if r['tasks'][t]['w_star'] is not None and c['w_star'][t].get('w_star') is not None:
                diffs[f'{t}_w_star'] = abs(r['tasks'][t]['w_star'] - c['w_star'][t]['w_star'])
        feas_match = r['mmp']['feasible'] == c['mmp']['feasible']
        fr = {}
        for bn in BUDGETS:
            a, b = r['frontier'][bn], c['frontier'][bn]
            if not a.get('feasible') or not b.get('feasible'):
                fr[bn] = {'both_infeasible': a.get('feasible') == b.get('feasible')}
                continue
            fr[bn] = {'global_diff': abs(a['global_min_pair_auc'] - b['global_min_pair_auc']),
                      'coord_local_in_my_range': a['local_pair_auc_range'][0] - 1e-9 <= b['local_pair_auc'] <= a['local_pair_auc_range'][1] + 1e-9,
                      'coord_s12_in_my_range': a['S12_pair_auc_range'][0] - 1e-9 <= b['s12_pair_auc'] <= a['S12_pair_auc_range'][1] + 1e-9,
                      'coord_s21_in_my_range': a['S21_pair_auc_range'][0] - 1e-9 <= b['s21_pair_auc'] <= a['S21_pair_auc_range'][1] + 1e-9,
                      'n_feasible_match': a['n_feasible'] == [b['n_feasible_income_partitions'], b['n_feasible_occupation_partitions']]}
        out[name] = {'max_abs_diff_scalars': max(diffs.values()), 'argmax': max(diffs, key=diffs.get),
                     'mmp_feasibility_match': feas_match, 'frontier': fr}
    return out


def main():
    res = {name: analyse(name, cells) for name, cells in laws().items()}
    for name, r in res.items():
        print(name, 'auc*X %.6f' % r['auc_star_X'], 'twin', {k: round(v, 6) for k, v in r['twin'].items()},
              'mmp', {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r['mmp'].items() if k != 'pair_exact'})
        for t, v in r['tasks'].items():
            print('  ', t, v)
        for bn in BUDGETS:
            print('  frontier', bn, r['frontier'][bn])
    out = {'laws': res}
    if COORD.exists():
        out['comparison_with_coordinator'] = compare(res, json.loads(COORD.read_text()))
        print(json.dumps(out['comparison_with_coordinator'], indent=1))
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
