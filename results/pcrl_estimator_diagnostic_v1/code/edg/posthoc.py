"""POST HOC summaries requested by the statistical reviewer's design review (roles/statistical_reviewer/REVIEW_DESIGN.md), written
after the outcomes and labelled post hoc everywhere. Reads only this study's synthetic outputs (out/*.jsonl.gz) and SUMMARY.json;
writes POST_HOC.json. Nothing here changes a registered quantity.

  integrity        unique (cell, rep) / (bank cell, rep) keys and the declared counts (e4)
  dbank_target     R2SE for D_bank judged against its own target dpop = pop(T_x) - pop(T_y) per repetition: coverage of dpop by
                   d +- 2 se, and the exact-calibration nulls IND and COMP occupation - income where dpop = dA = 0 (a2)
  b_coverage       coverage of dA by d +- 2 se for B_indep and B_same in the non-null comparisons (a4)
  cross_fit        mean +- MCSE of D_nested - pop_pooled per Figure-2 view (e2)
  m20_direction    P(d >= +0.020) and P(d <= -0.020) per estimator in the null comparisons (b3)
  bank             MCSE of every bank component (d2); per-recipe class / finite-training split for the six refitted recipes (d3);
                   paired bank differences on the same people (d4); bank LOOKUP == grid D_bank check (d4)
  drf1_agreement   D_nested at n = 20,000 against the bank winner's SCORE AUC at n = 8,000 for the same law (f3)"""
from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict

import numpy as np

from . import OUT, RESULTS
from . import laws as L
from . import simulate as SIM


def _load(name):
    with gzip.open(OUT / name, 'rt') as fh:
        return [json.loads(x) for x in fh]


def _ms(x):
    x = np.asarray(x, dtype=np.float64)
    return {'mean': float(x.mean()), 'mcse': float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else None, 'R': int(len(x))}


def main(argv=None):
    sim, bank = _load('sim_all.jsonl.gz'), _load('bank_all.jsonl.gz')
    scs = {s['cell']: s for s in SIM.scenarios()}
    laws = {}

    def law_of(cell):
        sc = scs[cell]
        k = (sc['family'], sc['pi'], sc['occ'])
        if k not in laws:
            laws[k] = L.build(*k)
        return laws[k]
    P = {'schema': 'EDG1-post-hoc-v1', 'label': 'POST HOC (after outcomes; requested by the statistical review)'}
    # integrity (e4)
    kc = Counter((r['cell'], r['rep']) for r in sim)
    kb = Counter((r['bank_cell'], r['rep']) for r in bank)
    per_cell = Counter(r['cell'] for r in sim)
    per_bank = Counter(r['bank_cell'] for r in bank)
    P['integrity'] = {'grid_rows': len(sim), 'grid_unique_keys': len(kc), 'grid_duplicates': sum(v > 1 for v in kc.values()),
                      'grid_cells': len(per_cell), 'grid_reps_per_cell': sorted(set(per_cell.values())),
                      'bank_rows': len(bank), 'bank_unique_keys': len(kb), 'bank_duplicates': sum(v > 1 for v in kb.values()),
                      'bank_cells': len(per_bank), 'bank_reps_per_cell': sorted(set(per_bank.values())),
                      'ok': len(kc) == len(sim) == 49 * 500 and len(kb) == len(bank) == 5 * 10}
    # D_bank against its own target, B coverage, cross-fit term, M20 by direction
    dbt, bcov, m20 = defaultdict(list), defaultdict(list), defaultdict(lambda: [0, 0, 0])
    for r in sim:
        law = law_of(r['cell'])
        sc = scs[r['cell']]
        for x, y in law['comparisons']:
            name = f'{x}-{y}'
            dA = law['views'][x]['A'] - law['views'][y]['A']
            d, se, _ = r['cmp'][name]['D_bank']
            dpop = r['views'][x]['D_bank_pop'] - r['views'][y]['D_bank_pop']
            key = (sc['family'], name)
            dbt[key].append((abs(d - dpop) <= 2 * se, dpop, (d - 2 * se > 0) or (d + 2 * se < 0)))
            if abs(dA) > 1e-12:
                for e in ('B_indep', 'B_same'):
                    d2, se2, _ = r['cmp'][name][e]
                    bcov[(e, sc['family'], name)].append(abs(d2 - dA) <= 2 * se2)
            else:
                for e in SIM.ESTIMATORS:
                    d3 = r['cmp'][name][e][0]
                    c = m20[(e, sc['family'], name)]
                    c[0] += d3 >= 0.020
                    c[1] += d3 <= -0.020
                    c[2] += 1
    P['dbank_target'] = {f'{k[0]}|{k[1]}': {'coverage_of_dpop': float(np.mean([v[0] for v in vals])),
                                            'mean_dpop': float(np.mean([v[1] for v in vals])),
                                            'resolved_rate': float(np.mean([v[2] for v in vals])), 'R': len(vals)}
                         for k, vals in sorted(dbt.items())}
    P['b_coverage_non_null'] = {f'{k[0]}|{k[1]}|{k[2]}': {'coverage_of_dA': float(np.mean(v)), 'R': len(v)} for k, v in sorted(bcov.items())}
    P['m20_direction_null'] = {f'{k[0]}|{k[1]}|{k[2]}': {'P_d_ge_+0.020': v[0] / v[2], 'P_d_le_-0.020': v[1] / v[2], 'R': v[2]}
                               for k, v in sorted(m20.items())}
    cf = {}
    for cell in (5, 17, 29, 41):
        rs = [r for r in sim if r['cell'] == cell]
        for v in rs[0]['views']:
            x = [r['views'][v]['D_nested'] - r['views'][v]['D_nested_pop_pooled'] for r in rs]
            m = _ms(x)
            m['within_2_mcse_of_0'] = abs(m['mean']) <= 2 * m['mcse']
            cf[f"{scs[cell]['family']}|{v}"] = m
    P['cross_fit_term_figure2_cells'] = cf
    # bank
    S = json.load(open(RESULTS / 'SUMMARY.json'))
    comp = {}
    for key, b in S['bank'].items():
        for v, vr in b['views'].items():
            comp[f'{key}|{v}'] = {k: {'mean': s['mean'], 'mcse': s['mcse'], 'resolved_2mcse': abs(s['mean']) > 2 * s['mcse']}
                                  for k, s in vr['decomposition'].items()}
    P['bank_component_mcse'] = comp
    cl = S['bank_class_limits']
    split = {}
    for k, v in cl.items():
        key, view = k.rsplit('|', 1)
        rows = [r for r in bank if r['bank_cell'] == key]
        for rec, pl in v['pop_auc_large_fit'].items():
            at4k = _ms([r['views'][view]['recipes'][rec]['pop_auc'] for r in rows])
            split[f'{k}|{rec}'] = {'A': v['A'], 'pop_large_30000': pl, 'pop_fit_4000_mean': at4k['mean'], 'pop_fit_4000_mcse': at4k['mcse'],
                                   'class_loss_A_minus_large': v['A'] - pl, 'finite_training_loss_large_minus_4000': pl - at4k['mean'],
                                   'kind': 'saturated (large-sample value, not a class limit)' if rec in ('LOOKUP', 'CATI01', 'CATI1') else 'class limit'}
    P['class_split'] = split
    paired = {}
    pairs = {'REF|0.5|skewed|8000': [('L256', 'L16')], 'RED|0.5|skewed|8000': [('pair', 'income'), ('income', 'occupation')],
             'RED|0.2|skewed|8000': [('pair', 'income'), ('income', 'occupation')], 'IND|0.5|skewed|8000': [('pair', 'income'), ('pair', 'occupation')],
             'COMP|0.5|uniform|8000': [('pair', 'income')]}
    for key, lst in pairs.items():
        rows = [r for r in bank if r['bank_cell'] == key]
        for x, y in lst:
            paired[f'{key}|{x}-{y}'] = {
                'winner_score_diff': _ms([r['views'][x]['winner_score_auc'] - r['views'][y]['winner_score_auc'] for r in rows]),
                'winner_pop_diff': _ms([r['views'][x]['winner_pop_auc'] - r['views'][y]['winner_pop_auc'] for r in rows]),
                'dA': rows[0]['views'][x]['A'] - rows[0]['views'][y]['A']}
    P['bank_paired_differences'] = paired
    # bank LOOKUP recipe == grid D_bank for the same people (pop and VALIDATION AUC)
    grid = {(r['cell'], r['rep']): r for r in sim}
    eq, n_eq = [], 0
    for r in bank:
        g = grid[(r['cell'], r['rep'])]
        for v, e in r['views'].items():
            gv = g['views'][v]
            rec = e['recipes']['LOOKUP']
            ok = abs(rec['pop_auc'] - gv['D_bank_pop']) <= 1e-12 and abs(rec['val_auc'] - gv['D_bank_val_auc']) <= 1e-12
            eq.append(ok)
            n_eq += 1
    P['bank_lookup_equals_grid_dbank'] = {'checked_views': n_eq, 'all_equal': bool(all(eq)), 'equal_count': int(sum(eq))}
    # DRF1-type agreement: D_nested at 20,000 vs bank winner SCORE at 8,000 (same law)
    agree = {}
    for key in pairs:
        fam, pi, occ, _ = key.split('|')
        c20 = [s['cell'] for s in SIM.scenarios() if (s['family'], s['pi'], s['occ'], s['n'], s['clustered']) == (fam, float(pi), occ, 20000, False)][0]
        rows20 = [r for r in sim if r['cell'] == c20]
        rowsb = [r for r in bank if r['bank_cell'] == key]
        for v in rowsb[0]['views']:
            dn = _ms([r['views'][v]['D_nested'] for r in rows20])
            wb = _ms([r['views'][v]['winner_score_auc'] for r in rowsb])
            agree[f'{key}|{v}'] = {'A': rowsb[0]['views'][v]['A'], 'D_nested_n20000': dn, 'bank_winner_score_n8000': wb,
                                   'difference': wb['mean'] - dn['mean']}
    P['drf1_type_agreement'] = agree
    with open(RESULTS / 'POST_HOC.json', 'w') as fh:
        json.dump(P, fh, indent=1)
    print(json.dumps({'integrity': P['integrity'], 'bank_lookup_equals_grid_dbank': P['bank_lookup_equals_grid_dbank']}))
