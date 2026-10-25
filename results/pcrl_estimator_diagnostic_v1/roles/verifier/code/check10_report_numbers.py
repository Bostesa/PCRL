"""ROUND 2, CHECK10: numbers that ESTIMATOR_DIAGNOSTIC.md attributes to the results, against SUMMARY.json / POST_HOC.json
(own extraction code). A claimed value with d decimals matches if |computed - claimed| <= 0.5 * 10^-d (+1e-9), i.e. the computed
value rounds to it (half-way cases accepted either way). Ranges: both ends. Each item records the report line."""
import json

import numpy as np

import vlib as V

T0 = V.utc_now()
S = json.load(open(V.D / 'SUMMARY.json'))
P = json.load(open(V.D / 'POST_HOC.json'))
SIM = S['sim']
items = []


def cells(fam=None, pi=None, occ=None, n=None, clustered=False):
    out = []
    for c, v in SIM.items():
        s = v['scenario']
        if (fam is None or s['family'] == fam) and (pi is None or s['pi'] == pi) and (occ is None or s['occ'] == occ) \
                and (n is None or s['n'] == n) and (clustered is None or s['clustered'] == clustered):
            out.append(c)
    return out


def close(x, claim, d):
    return abs(x - claim) <= 0.5 * 10 ** (-d) + 1e-9


def add(ident, line, text, computed, claimed, d):
    if isinstance(claimed, (list, tuple)):
        ok = all(close(a, b, d) for a, b in zip(computed, claimed))
    else:
        ok = close(computed, claimed, d)
    items.append({'id': ident, 'report_line': line, 'claim': text, 'claimed': claimed, 'computed': computed, 'decimals': d,
                  'match': bool(ok)})


def cmp_vals(cs, comp, est, path):
    out = []
    for c in cs:
        x = SIM[c]['comparisons'][comp][est]
        for k in path:
            x = x[k]
        out.append(x)
    return out


REFI = cells('REF')
REFCMP = ('L64-L16', 'L256-L16', 'L1024-L16')
# ---------------- answer 1
v = [r for cm in REFCMP for e in ('C_same', 'C_nested') for r in cmp_vals(REFI, cm, e, ('P_d_pos', 'rate'))]
add('A1.1', 26, 'REF: C_same and C_nested scored the refined output higher in 100% of repetitions', min(v), 1.0, 3)
v = [r for cm in REFCMP for r in cmp_vals(REFI, cm, 'C_same', ('R2SE', 'false_positive', 'rate'))]
add('A1.2a', 28, 'R2SE resolved 99.6-100% (C_same, independent REF cells)', [min(v) * 100, max(v) * 100], [99.6, 100.0], 1)
v = [r for cm in REFCMP for r in cmp_vals(['48'], cm, 'C_same', ('R2SE', 'false_positive', 'rate'))]
add('A1.2b', 28, '99.2% in the clustered cell (C_same, minimum over its comparisons)', min(v) * 100, 99.2, 1)
v = [r for cm in REFCMP for r in cmp_vals(REFI + ['48'], cm, 'C_nested', ('R2SE', 'false_positive', 'rate'))]
add('A1.2c', 29, '100% (C_nested)', min(v) * 100, 100.0, 1)
RED = cells('RED')
add('A1.3a', 31, 'RED C_same pair > income by +0.011 to +0.111', [min(cmp_vals(RED, 'pair-income', 'C_same', ('mean',))),
                                                                 max(cmp_vals(RED, 'pair-income', 'C_same', ('mean',)))], [0.011, 0.111], 3)
add('A1.3b', 31, 'RED C_nested pair > income by +0.045 to +0.229', [min(cmp_vals(RED, 'pair-income', 'C_nested', ('mean',))),
                                                                   max(cmp_vals(RED, 'pair-income', 'C_nested', ('mean',)))], [0.045, 0.229], 3)
v = cmp_vals(RED, 'pair-income', 'C_same', ('R2SE', 'false_positive', 'rate')) + cmp_vals(RED, 'pair-income', 'C_nested', ('R2SE', 'false_positive', 'rate'))
add('A1.4', 32, 'RED R2SE false-positive rate 100% in every RED cell (C_same, C_nested)', min(v) * 100, 100.0, 1)
IND = cells('IND')
v = cmp_vals(IND, 'pair-income', 'C_same', ('mean',))
add('A1.5a', 33, 'IND C_same pair > income by +0.093 to +0.355', [min(v), max(v)], [0.093, 0.355], 3)
add('A1.5b', 33, 'IND C_same pair-income R2SE false positives 100%', min(cmp_vals(IND, 'pair-income', 'C_same', ('R2SE', 'false_positive', 'rate'))) * 100, 100.0, 1)
COMP = cells('COMP')
v = cmp_vals(COMP, 'occupation-income', 'C_same', ('R2SE', 'false_positive', 'rate'))
add('A1.6', 35, 'COMP occupation vs income: C_same false positives 0-1.0%', [min(v) * 100, max(v) * 100], [0.0, 1.0], 1)
for est, pmax, lo in (('D_nested', 0.29, 14), ('D_bank', 0.11, 26)):
    pp = [r for cm in REFCMP for r in cmp_vals(REFI, cm, est, ('P_d_pos', 'rate'))]
    fp = [r for cm in REFCMP for r in cmp_vals(REFI, cm, est, ('R2SE', 'false_positive', 'rate'))]
    add(f'A1.7 {est}', 37, f'{est} on REF: P(d > 0) at most {pmax}; R2SE false positives {lo}-100%', [max(pp), min(fp) * 100, max(fp) * 100],
        [pmax, lo, 100], 2 if est == 'D_nested' else 2)
pos = [r for cm in REFCMP for e in ('D_nested', 'D_bank') for r in cmp_vals(REFI, cm, e, ('R2SE', 'resolved_pos', 'rate'))]
add('A1.8', 37, 'all but at most 0.8 percentage points in the "refined is less recoverable" direction (max resolved_pos)', max(pos) * 100, 0.8, 1)
cov = [v['coverage_of_dpop'] for v in P['dbank_target'].values()]
add('A1.9', 41, 'D_bank R2SE covered its own target in 94.8-95.9% (post hoc)', [min(cov) * 100, max(cov) * 100], [94.8, 95.9], 1)
nulls = [P['dbank_target'][k]['resolved_rate'] for k in ('IND|occupation-income', 'IND|pair-income', 'IND|pair-occupation', 'COMP|occupation-income')]
add('A1.10', 43, 'exact nulls (IND, COMP occupation - income): D_bank resolved 4.2-5.2% (post hoc pooled)', [min(nulls) * 100, max(nulls) * 100], [4.2, 5.2], 1)
dn = [r for cm in ('pair-income', 'pair-occupation', 'occupation-income') for r in cmp_vals(IND, cm, 'D_nested', ('R2SE', 'false_positive', 'rate'))] \
    + cmp_vals(COMP, 'occupation-income', 'D_nested', ('R2SE', 'false_positive', 'rate'))
add('A1.11', 45, 'D_nested 13-20% resolved under the exact nulls (per cell)', [min(dn) * 100, max(dn) * 100], [13, 20], 0)
bb = [abs(SIM[c]['views'][v]['B_indep']['bias']) for c in SIM for v in SIM[c]['views']]
add('A1.12', 46, "B_indep's bias within +-0.0014 of A everywhere", max(bb), 0.0014, 4)
bc = [v['coverage_of_dA'] for v in P['b_coverage_non_null'].values()]
add('A1.13', 47, 'B coverage of dA 95.2-95.8% in non-null comparisons (post hoc)', [min(bc) * 100, max(bc) * 100], [95.2, 95.8], 1)
bnull = [SIM[c]['comparisons'][cm][e]['R2SE']['false_positive']['rate'] for c in SIM for cm in SIM[c]['comparisons']
         for e in ('B_same', 'B_indep') if SIM[c]['comparisons'][cm]['null']]
add('A1.14', 48, "B's null false-positive rates 0-7.0%", [min(bnull) * 100, max(bnull) * 100], [0.0, 7.0], 1)
# ---------------- answer 2 table
c17, c5, c15, c3 = cells('REF', 0.5, 'skewed', 20000)[0], cells('IND', 0.5, 'skewed', 20000)[0], cells('REF', 0.5, 'skewed', 2000)[0], cells('IND', 0.5, 'skewed', 2000)[0]
tab = {('C_same', c17, c5): ([0.0008, 0.0038, 0.0116, 0.0334], [0.0136, 0.0281, 0.1067]),
       ('C_nested', c17, c5): ([0.0026, 0.0130, 0.0412, 0.0985], [0.0308, 0.0629, 0.2177]),
       ('C_same', c15, c3): ([0.0040, 0.0224, 0.0675, 0.1412], [0.0437, 0.0888, 0.2789])}
mx_mcse = 0.0
for (est, cref, cind), (rv, iv) in tab.items():
    comp = [SIM[cref]['views'][v][est]['bias'] for v in ('L16', 'L64', 'L256', 'L1024')] + [SIM[cind]['views'][v][est]['bias'] for v in ('income', 'occupation', 'pair')]
    mx_mcse = max([mx_mcse] + [SIM[cref]['views'][v][est]['mcse'] for v in ('L16', 'L64', 'L256', 'L1024')]
                  + [SIM[cind]['views'][v][est]['mcse'] for v in ('income', 'occupation', 'pair')])
    add(f'A2.table {est} cells {cref}/{cind}', 64, f'{est} bias row (REF L16..L1024 / IND income, occupation, pair)', comp, rv + iv, 4)
add('A2.mcse', 59, 'MCSE at most 0.0005 for the table entries', mx_mcse, 0.0005, 4)
for n, rng_ in ((20000, (0.033, 0.056)), (8000, (0.063, 0.117)), (2000, (0.137, 0.244))):
    v = [SIM[c]['comparisons']['L1024-L16']['C_same']['mean'] for c in cells('REF', n=n)]
    add(f'A2.L1024-L16 C_same n={n}', 69, f'C_same L1024 - L16 {rng_[0]} to {rng_[1]}', [min(v), max(v)], list(rng_), 3)
v = [SIM[c]['comparisons']['L1024-L16']['C_nested']['mean'] for c in cells('REF', n=20000)]
add('A2.L1024-L16 C_nested n=20000', 70, 'C_nested +0.096 to +0.182', [min(v), max(v)], [0.096, 0.182], 3)
v1 = [SIM[c]['views']['pair']['C_same']['mean'] for c in cells('IND', n=2000)]
v2 = [SIM[c]['views']['pair']['C_nested']['mean'] for c in cells('IND', n=2000)]
add('A2.ceiling', 72, 'IND 1,024 outputs at n = 2,000: C_same up to 0.91, C_nested up to 0.99 (cell means)', [max(v1), max(v2)], [0.91, 0.99], 2)
# ---------------- answer 3 table
c29, c41 = cells('RED', 0.5, 'skewed', 20000)[0], cells('COMP', 0.5, 'skewed', 20000)[0]
for lab, c, v, claim in (('IND pair', c5, 'pair', (0.218, 0.218, 0.000, -0.000)), ('REF L1024', c17, 'L1024', (0.122, 0.099, 0.023, -0.001)),
                         ('REF L256', c17, 'L256', (0.050, 0.041, 0.008, -0.001)), ('RED pair', c29, 'pair', (0.061, 0.050, 0.010, -0.000)),
                         ('COMP pair', c41, 'pair', (0.092, 0.075, 0.016, -0.001))):
    w = SIM[c]['views'][v]
    comp = [w['C_nested_minus_D_nested']['mean'], w['C_nested']['bias'], w['D_nested_pop_pooled__estimation_loss_A_minus_pop']['mean'],
            w['D_nested_pop_pooled__test_variation_value_minus_pop']['mean']]
    add(f'A3.table {lab}', 87, f'{lab}: gap / optimism / LOOKUP estimation loss / test variation + pooling', comp, list(claim), 3)
tv = [SIM[c]['views'][v]['D_nested_pop_pooled__test_variation_value_minus_pop']['mean'] for c in SIM for v in SIM[c]['views']]
add('A3.neg160', 99, 'D_nested third term negative in all 160 cell-views (-0.00004 to -0.011)', [len(tv), sum(x < 0 for x in tv), max(tv), min(tv)],
    [160, 160, -0.00004, -0.011], 5)
tb = [SIM[c]['views'][v]['D_bank_pop__test_variation_value_minus_pop']['mean'] for c in SIM for v in SIM[c]['views']]
add('A3.dbank_all', 96, 'D_bank test variation mean -0.0029 to +0.0027 over all cells', [min(tb), max(tb)], [-0.0029, 0.0027], 4)
cf = P['cross_fit_term_figure2_cells']
below = sum(1 for x in cf.values() if not x['within_2_mcse_of_0'] and x['mean'] < 0)
others = [x['mean'] for k, x in cf.items() if k != 'IND|pair']
add('A3.crossfit', 100, 'post hoc: >2 MCSE below zero in 12 of 13 Figure-2 views (-0.0006 to -0.0027); IND pair -0.0002 +- 0.0003',
    [len(cf), below, max(others), min(others), cf['IND|pair']['mean'], cf['IND|pair']['mcse']], [13, 12, -0.0006, -0.0027, -0.0002, 0.0003], 4)
pf = max(abs(SIM[c]['views'][v]['D_nested_pop_pooled']['mean'] - SIM[c]['views'][v]['D_nested_pop_fold_mean']['mean']) for c in SIM for v in SIM[c]['views'])
add('A3.pooled_vs_fold', 103, 'pooled and per-fold population AUCs agree within 0.0001', pf, 0.0001, 4)
# ---------------- answer 4 (bank)
B = S['bank']
inf = [(k, v) for k, b in B.items() for v, x in b['views'].items() if x['A'] > 0.5 + 1e-9]
ab = [B[k]['views'][v]['decomposition']['finite_training_plus_class_loss_A_minus_best_pop']['mean'] for k, v in inf]
sl = [B[k]['views'][v]['decomposition']['selection_loss_best_pop_minus_winner_pop']['mean'] for k, v in inf]
add('A4.A_minus_best', 125, 'A - best population AUC in the bank 0.0012-0.0121 (informative views)', [min(ab), max(ab)], [0.0012, 0.0121], 4)
add('A4.selection', 126, 'selection loss 0.0004-0.0034 (informative views)', [min(sl), max(sl)], [0.0004, 0.0034], 4)
wc = [x['decomposition']['winners_curse_val_minus_winner_pop']['mean'] for b in B.values() for x in b['views'].values()]
add('A4.curse', 127, "winner's curse up to +0.029", max(wc), 0.029, 3)
bp = P['bank_paired_differences']['REF|0.5|skewed|8000|L256-L16']
add('A4.paired', 129, 'REF L256 - L16: winner pop -0.0083 +- 0.0013; winner SCORE -0.0074 +- 0.0023',
    [bp['winner_pop_diff']['mean'], bp['winner_pop_diff']['mcse'], bp['winner_score_diff']['mean'], bp['winner_score_diff']['mcse']],
    [-0.0083, 0.0013, -0.0074, 0.0023], 4)
cl = S['bank_class_limits']['COMP|0.5|uniform|8000|pair']
lr = [cl['pop_auc_large_fit'][r] for r in ('OUT_LR', 'LOG_LR', 'DEC_LR')]
add('A4.class', 130, 'COMP pair OUT_LR / LOG_LR / DEC_LR at 30,000: 0.4999-0.5049, class loss 0.078-0.083',
    [min(lr), max(lr), cl['A'] - max(lr), cl['A'] - min(lr)], [0.4999, 0.5049, 0.078, 0.083], 4)
add('A4.saturated', 131, 'COMP pair LOOKUP / CATI01 at 30,000: 0.573 / 0.577', [cl['pop_auc_large_fit']['LOOKUP'], cl['pop_auc_large_fit']['CATI01']],
    [0.573, 0.577], 3)
add('A4.lookup140', 139, 'bank LOOKUP reproduced grid D_bank in all 140 shared views',
    [P['bank_lookup_equals_grid_dbank']['checked_views'], P['bank_lookup_equals_grid_dbank']['equal_count']], [140, 140], 0)
red_w = {k: sum(c for w, c in B[k]['views']['pair']['winners'].items() if w.startswith('IGNORE_occupation__')) for k in B if k.startswith('RED')}
add('A4.ignore_one', 141, 'RED pair winner an income-only ignore-one candidate in 10 of 10 repetitions', list(red_w.values()), [10] * len(red_w), 0)
# ---------------- answer 5
m = P['m20_direction_null']
add('A5.m20', 165, 'C_same reached +0.020 in 20% / 61% / 100%; C_nested 60% / 100% / 100% (REF, post hoc)',
    [100 * m[f'C_same|REF|{c}']['P_d_ge_+0.020'] for c in REFCMP] + [100 * m[f'C_nested|REF|{c}']['P_d_ge_+0.020'] for c in REFCMP],
    [20, 61, 100, 60, 100, 100], 0)
nested_part = [k for k in m if k.startswith('C_nested|') and not k.endswith('occupation-income')]
add('A5.never_neg', 167, 'C_nested never at or below -0.020 on a nested partition', max(m[k]['P_d_le_-0.020'] for k in nested_part), 0.0, 6)
r20 = cells('REF', n=20000)
d1 = [SIM[c]['views']['L256']['C_nested']['bias'] - SIM[c]['views']['L64']['C_nested']['bias'] for c in r20]
d2 = [SIM[c]['views']['L1024']['C_nested']['bias'] - SIM[c]['views']['L256']['C_nested']['bias'] for c in r20]
add('A5.support_optimism', 172, 'C_nested optimism differs by 0.03-0.05 (64 vs 256) and 0.06-0.12 (256 vs 1,024), REF n = 20,000',
    [min(d1), max(d1), min(d2), max(d2)], [0.03, 0.05, 0.06, 0.12], 2)
c16 = cells('REF', 0.5, 'skewed', 8000)[0]
add('A5.dbank_L256', 183, "D_bank's mean difference L256 - L16 at n = 8,000 was -0.021", SIM[c16]['comparisons']['L256-L16']['D_bank']['mean'], -0.021, 3)
ag = [x['difference'] for x in P['drf1_type_agreement'].values()]
add('A5.agreement', 192, 'D_nested vs bank winner SCORE differed by -0.007 to +0.005 across the 14 subset views', [len(ag), min(ag), max(ag)],
    [14, -0.007, 0.005], 3)
# ---------------- further results
add('F.ind_income', 249, 'IND income C_same +0.044 / +0.022 / +0.014 at n = 2,000 / 8,000 / 20,000 (pi 0.5, skewed)',
    [SIM[cells('IND', 0.5, 'skewed', n)[0]]['views']['income']['C_same']['bias'] for n in (2000, 8000, 20000)], [0.044, 0.022, 0.014], 3)
add('F.pi02', 251, 'REF L1024, n = 20,000, skewed: +0.049 (pi 0.2) against +0.033 (pi 0.5)',
    [SIM[cells('REF', 0.2, 'skewed', 20000)[0]]['views']['L1024']['C_same']['bias'], SIM[c17]['views']['L1024']['C_same']['bias']], [0.049, 0.033], 3)
add('F.unseen', 253, 'up to 37.7% of SCORE people unseen in FIT (IND pair, n = 2,000, uniform)',
    [100 * max(SIM[c]['views'][v]['D_bank_unseen_score_share']['mean'] for c in SIM for v in SIM[c]['views']),
     100 * SIM[cells('IND', 0.5, 'uniform', 2000)[0]]['views']['pair']['D_bank_unseen_score_share']['mean']], [37.7, 37.7], 1)
wr = {e: cmp_vals(RED, 'income-occupation', e, ('R2SE', 'wrong_direction', 'rate')) for e in ('C_nested', 'C_same', 'D_nested', 'D_bank', 'B_indep')}
pw = cmp_vals(RED, 'income-occupation', 'B_indep', ('R2SE', 'power', 'rate'))
add('F.red_wrong', 260, 'RED income - occupation: C_nested wrong 96.6-100%; C_same up to 78.2%; D_* at most 0.2%; B_indep never wrong, power 0.32-1.00',
    [100 * min(wr['C_nested']), 100 * max(wr['C_nested']), 100 * max(wr['C_same']), 100 * max(wr['D_nested'] + wr['D_bank']),
     100 * max(wr['B_indep']), min(pw), max(pw)], [96.6, 100.0, 78.2, 0.2, 0.0, 0.32, 1.00], 1)
wn = cmp_vals(RED, 'pair-occupation', 'D_nested', ('R2SE', 'wrong_direction', 'rate'))
wb = cmp_vals(RED, 'pair-occupation', 'D_bank', ('R2SE', 'wrong_direction', 'rate'))
add('F.red_pair_occ', 265, 'RED pair - occupation: D_nested wrong 2.8-96.4%, D_bank 6.4-88.8%', [100 * min(wn), 100 * max(wn), 100 * min(wb), 100 * max(wb)],
    [2.8, 96.4, 6.4, 88.8], 1)
p2 = [r for cm in ('pair-income', 'pair-occupation') for r in cmp_vals(cells('COMP', n=2000), cm, 'D_bank', ('R2SE', 'power', 'rate'))]
p8 = [r for cm in ('pair-income', 'pair-occupation') for r in cmp_vals(cells('COMP', n=8000), cm, 'D_bank', ('R2SE', 'power', 'rate'))]
add('F.comp_power', 266, "COMP D_bank power 5-9% at n = 2,000 and 26-71% at n = 8,000", [100 * min(p2), 100 * max(p2), 100 * min(p8), 100 * max(p8)],
    [5, 9, 26, 71], 0)
x = SIM['48']['comparisons']['L64-L16']['B_indep']
add('F.clustered', 272, 'clustered: B_indep L64 - L16 false positives 0.040 (household SE) against 0.036 (person SE)',
    [x['R2SE']['false_positive']['rate'], x['R2SE_person_se']['false_positive']['rate']], [0.040, 0.036], 3)
add('F.undefined', 241, '0 undefined values over all 24,500 repetitions', [sum(v['undefined_values'] for v in SIM.values()), sum(v['reps'] for v in SIM.values())],
    [0, 24500], 0)
occ = {f'{c}|{v}': SIM[c]['views'][v]['occupied_mean'] for c in cells(n=2000) for v in SIM[c]['views'] if SIM[c]['views'][v]['support'] == 1024}
best = min(occ, key=lambda k: abs(occ[k] - 495))
add('F.occupied', 324, '495 of 1,024 wires occupied at n = 2,000 (closest cell-view)', occ[best], 495, 0)
items[-1]['which'] = best
mism = [i for i in items if not i['match']]
rec = {'check': 10, 'name': 'report_numbers_vs_SUMMARY_POST_HOC', 'n_items': len(items), 'n_scalar_values': sum(len(i['claimed']) if isinstance(i['claimed'], list) else 1 for i in items),
       'n_mismatch_items': len(mism), 'mismatches': [{k: i[k] for k in ('id', 'report_line', 'claim', 'claimed', 'computed')} for i in mism],
       'items': items, 'status': 'PASS' if not mism else 'FAIL'}
print(rec['status'], rec['n_items'], rec['n_scalar_values'], len(mism))
for i in mism:
    print('MISMATCH', i['id'], 'line', i['report_line'], i['claimed'], [round(x, 5) if isinstance(x, float) else x for x in (i['computed'] if isinstance(i['computed'], list) else [i['computed']])])
V.write_out('CHECK10_report_numbers.json', rec, started_utc=T0)
