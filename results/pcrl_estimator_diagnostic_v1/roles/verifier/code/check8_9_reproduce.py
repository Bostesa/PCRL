"""ROUND 2, CHECK8 + CHECK9.

CHECK8: reproduce reported summaries from the saved synthetic outputs with the verifier's OWN code (not edg.analyze / posthoc):
  (a) (cell, view, estimator) entries of SUMMARY.json['sim']: R, mean, SD (ddof 1), MCSE = SD / sqrt(R), bias against the
      verifier's EXACT A (vlib brute force), RMSE;
  (b) comparisons from the saved [d, se_unit, se_person] triples: mean d and the R2SE rates (resolved iff d - 2 se > 0 or
      d + 2 se < 0; null iff the exact dA is 0);
  (c) bank entries from bank_all.jsonl.gz (winner SCORE / pop means, selection loss) and POST_HOC dbank_target REF|L1024-L16
      coverage (own re-implementation: |d - dpop| <= 2 se, dpop = D_bank_pop(x) - D_bank_pop(y)).
CHECK9: re-run 3 repetitions with the study's code (edg.simulate.one_rep, laws.build, scenarios(); repetition 0 with the source
  check on, as in the study) and compare the JSON line with the saved line byte for byte; for one of them recompute C_same and
  B_indep with the verifier's own exact code from the sample drawn by edg.simulate.sample_rep.
Reads D/out/*.jsonl.gz (never writes there)."""
import gzip
import json
import math
from fractions import Fraction as Fr

import numpy as np

import vlib as V

V.study_imports()
from edg import laws as L             # noqa: E402
from edg import simulate as SIM       # noqa: E402

T0 = V.utc_now()
OUTD = V.D / 'out'
S = json.load(open(V.D / 'SUMMARY.json'))
P = json.load(open(V.D / 'POST_HOC.json'))
SCS = {s['cell']: s for s in SIM.scenarios()}
NEED_A = [(15, 'L1024', 'C_same'), (15, 'L1024', 'C_nested'), (15, 'L1024', 'D_nested'), (17, 'L1024', 'C_same'),
          (17, 'L1024', 'D_bank'), (17, 'L1024', 'B_indep'), (5, 'pair', 'C_same'), (5, 'pair', 'B_indep'), (29, 'pair', 'C_same'),
          (29, 'pair', 'D_nested'), (41, 'pair', 'C_nested'), (41, 'pair', 'D_bank'), (48, 'L1024', 'C_same'), (48, 'L1024', 'B_indep')]
NEED_B = [(17, 'L1024-L16', 'C_same'), (17, 'L1024-L16', 'C_nested'), (17, 'L1024-L16', 'D_bank'), (29, 'pair-income', 'C_same'),
          (29, 'pair-income', 'D_nested'), (29, 'pair-income', 'B_indep'), (28, 'income-occupation', 'C_nested'),
          (28, 'income-occupation', 'C_same'), (28, 'income-occupation', 'B_indep'), (48, 'L64-L16', 'B_indep')]
REF_CELLS = [s['cell'] for s in SIM.scenarios() if s['family'] == 'REF']
cells = sorted({c for c, _, _ in NEED_A} | {c for c, _, _ in NEED_B} | set(REF_CELLS))
RERUN = [(17, 0), (48, 7), (41, 499)]
rows, lines = {c: [] for c in cells}, {}
pref = {c: f'{{"cell": {c}, "rep": ' for c in cells}
with gzip.open(OUTD / 'sim_all.jsonl.gz', 'rt') as fh:
    for line in fh:
        c = int(line[9:line.index(',')])
        if c in rows:
            r = json.loads(line)
            rows[c].append(r)
            if (c, r['rep']) in RERUN:
                lines[(c, r['rep'])] = line.rstrip('\n')
rec8 = {'check': 8, 'name': 'reproduce_reported_summaries_own_code', 'a_views': [], 'b_comparisons': [], 'c_bank': [], 'c_post_hoc': []}
fails = []
EXA = {}


def exact_A(cell, view):
    sc = SCS[cell]
    k = (sc['family'], sc['pi'], sc['occ'], view)
    if k not in EXA:
        law = L.build(sc['family'], sc['pi'], sc['occ'])
        _, _, m1, m0 = V.study_law_exact_view(law, view)
        EXA[k] = V.pop_optimal_auc_exact(m1, m0)
    return EXA[k]


def stats(x, truth=None):
    x = np.array([v for v in x if v is not None and np.isfinite(v)], dtype=np.float64)
    R = len(x)
    sd = math.sqrt(sum((v - x.mean()) ** 2 for v in x) / (R - 1))
    o = {'R': R, 'mean': float(x.mean()), 'sd': sd, 'mcse': sd / math.sqrt(R)}
    if truth is not None:
        o['bias'] = float(x.mean()) - float(truth)
        o['rmse'] = math.sqrt(float(np.mean((x - float(truth)) ** 2)))
    return o


def cmpd(own, rep, keys, tol=1e-12):
    bad = [k for k in keys if (abs(own[k] - rep[k]) > tol if isinstance(own[k], float) else own[k] != rep[k])]
    return bad


for cell, view, est in NEED_A:
    A = exact_A(cell, view)
    own = stats([r['views'][view][est] for r in rows[cell]], A)
    rep = S['sim'][str(cell)]['views'][view][est]
    bad = cmpd(own, rep, ('R', 'mean', 'sd', 'mcse', 'bias', 'rmse'))
    rec8['a_views'].append({'cell': cell, 'scenario': SCS[cell], 'view': view, 'estimator': est, 'A_exact': A, 'own': own,
                            'summary': {k: rep[k] for k in ('R', 'mean', 'sd', 'mcse', 'bias', 'rmse')}, 'mismatch': bad})
    if bad:
        fails.append(f'a {cell} {view} {est} {bad}')
for cell, comp, est in NEED_B:
    x, y = comp.split('-')
    dA = exact_A(cell, x) - exact_A(cell, y)
    trip = np.array([r['cmp'][comp][est] for r in rows[cell]], dtype=np.float64)
    ok = np.isfinite(trip).all(1)
    d, se, sep = trip[ok, 0], trip[ok, 1], trip[ok, 2]
    up, dn = d - 2 * se > 0, d + 2 * se < 0
    own = {'R': int(ok.sum()), 'mean_d': float(d.mean()), 'P_d_pos': float(np.mean(d > 0))}
    rep = S['sim'][str(cell)]['comparisons'][comp][est]
    repv = {'R': rep['R'], 'mean_d': rep['mean'], 'P_d_pos': rep['P_d_pos']['rate']}
    if dA == 0:
        own['R2SE_false_positive'] = float(np.mean(up | dn))
        repv['R2SE_false_positive'] = rep['R2SE']['false_positive']['rate']
        if SCS[cell]['clustered']:
            own['R2SE_person_se_false_positive'] = float(np.mean((d - 2 * sep > 0) | (d + 2 * sep < 0)))
            repv['R2SE_person_se_false_positive'] = rep['R2SE_person_se']['false_positive']['rate']
    else:
        right, wrong = (up, dn) if dA > 0 else (dn, up)
        own['R2SE_power'], own['R2SE_wrong_direction'] = float(np.mean(right)), float(np.mean(wrong))
        repv['R2SE_power'], repv['R2SE_wrong_direction'] = rep['R2SE']['power']['rate'], rep['R2SE']['wrong_direction']['rate']
    bad = cmpd(own, repv, list(own))
    rec8['b_comparisons'].append({'cell': cell, 'comparison': comp, 'estimator': est, 'dA_exact': float(dA), 'own': own,
                                  'summary': repv, 'mismatch': bad})
    if bad:
        fails.append(f'b {cell} {comp} {est} {bad}')
# (c) bank
brows = []
with gzip.open(OUTD / 'bank_all.jsonl.gz', 'rt') as fh:
    brows = [json.loads(x) for x in fh]
for key, view in (('REF|0.5|skewed|8000', 'L16'), ('REF|0.5|skewed|8000', 'L256'), ('RED|0.5|skewed|8000', 'pair'),
                  ('COMP|0.5|uniform|8000', 'pair')):
    e = [r['views'][view] for r in brows if r['bank_cell'] == key]
    own = {'reps': len(e), 'winner_score_auc_mean': float(np.mean([x['winner_score_auc'] for x in e])),
           'winner_pop_auc_mean': float(np.mean([x['winner_pop_auc'] for x in e])),
           'selection_loss_mean': float(np.mean([x['best_pop_auc_in_bank'] - x['winner_pop_auc'] for x in e])),
           'A_minus_best_pop_mean': float(np.mean([e[0]['A'] - x['best_pop_auc_in_bank'] for x in e]))}
    sb = S['bank'][key]['views'][view]
    repv = {'reps': sb['reps'], 'winner_score_auc_mean': sb['winner_score_auc']['mean'], 'winner_pop_auc_mean': sb['winner_pop_auc']['mean'],
            'selection_loss_mean': sb['decomposition']['selection_loss_best_pop_minus_winner_pop']['mean'],
            'A_minus_best_pop_mean': sb['decomposition']['finite_training_plus_class_loss_A_minus_best_pop']['mean']}
    bad = cmpd(own, repv, list(own))
    A_ex = exact_A(next(r['cell'] for r in brows if r['bank_cell'] == key), view)
    rec8['c_bank'].append({'bank_cell': key, 'view': view, 'own': own, 'summary': repv, 'mismatch': bad,
                           'A_recorded_minus_exact': e[0]['A'] - float(A_ex)})
    if bad or abs(e[0]['A'] - float(A_ex)) > 1e-12:
        fails.append(f'c bank {key} {view} {bad}')
# (c) POST_HOC dbank_target REF|L1024-L16 and REF|L64-L16 (own re-implementation over the 13 REF cells)
for comp in ('L1024-L16', 'L64-L16'):
    x, y = comp.split('-')
    cov, res, dp = [], [], []
    for c in REF_CELLS:
        for r in rows[c]:
            d, se, _ = r['cmp'][comp]['D_bank']
            dpop = r['views'][x]['D_bank_pop'] - r['views'][y]['D_bank_pop']
            cov.append(abs(d - dpop) <= 2 * se)
            res.append((d - 2 * se > 0) or (d + 2 * se < 0))
            dp.append(dpop)
    own = {'R': len(cov), 'coverage_of_dpop': float(np.mean(cov)), 'resolved_rate': float(np.mean(res)), 'mean_dpop': float(np.mean(dp))}
    ph = P['dbank_target'][f'REF|{comp}']
    bad = cmpd(own, ph, list(own))
    rec8['c_post_hoc'].append({'entry': f'dbank_target REF|{comp}', 'own': own, 'post_hoc': ph, 'mismatch': bad})
    if bad:
        fails.append(f'c post hoc {comp} {bad}')
rec8['failures'] = fails
rec8['status'] = 'PASS' if not fails else 'FAIL'
rec8['n_compared'] = {'a': len(rec8['a_views']), 'b': len(rec8['b_comparisons']), 'c_bank': len(rec8['c_bank']), 'c_post_hoc': len(rec8['c_post_hoc'])}
print('CHECK8', rec8['status'], fails)

# ------------------------------------------------------------------ CHECK9: re-run repetitions from their seeds
rec9 = {'check': 9, 'name': 'rerun_repetitions_from_seeds', 'reruns': []}
f9 = []
for cell, rep in RERUN:
    sc = SCS[cell]
    law = L.build(sc['family'], sc['pi'], sc['occ'])
    keys = {k: SIM._wire_keys(v) for k, v in law['views'].items()} if rep == 0 else None
    row = SIM.one_rep(sc, law, rep, keys=keys)
    s_new = json.dumps(row, allow_nan=True)
    same = s_new == lines[(cell, rep)]
    rec9['reruns'].append({'cell': cell, 'rep': rep, 'source_check_on': rep == 0, 'json_line_byte_identical': same,
                           'len': len(s_new)})
    if not same:
        f9.append(f'{cell} {rep}')
# own recomputation of C_same and B_indep for (cell 17, rep 0) from the drawn sample
cell, rep = 17, 0
sc = SCS[cell]
law = L.build(sc['family'], sc['pi'], sc['occ'])
c, s, unit, ct, st, unit_t, folds, parts = SIM.sample_rep(sc, law, rep)
saved = json.loads(lines[(cell, rep)])
own9 = {}
for view in ('L16', 'L1024'):
    vec = law['vectors']['levels'][view]
    wid, K = V.merge_own(vec)                       # own merging of the 1,024 internal cells' emitted vectors
    w_s, w_t = wid[c], wid[ct]
    n1 = np.bincount(w_s, weights=s, minlength=K).astype(np.int64)
    n0 = np.bincount(w_s, minlength=K).astype(np.int64) - n1
    o = np.flatnonzero(n1 + n0 > 0)
    C_ex = V.sample_auc_by_wire([Fr(int(n1[w]), int(n1[w] + n0[w])) for w in o], n1[o], n0[o])
    _, _, m1, m0 = V.study_law_exact_view(law, view)
    pe = V.posterior_exact(m1, m0)
    own2st = np.full(K, -1)
    own2st[wid] = law['views'][view]['inv']
    t1 = np.bincount(w_t, weights=st, minlength=K).astype(np.int64)
    t0 = np.bincount(w_t, minlength=K).astype(np.int64) - t1
    ot = np.flatnonzero(t1 + t0 > 0)
    B_float = V.sample_auc_by_wire([float(law['views'][view]['post'][own2st[w]]) for w in ot], t1[ot], t0[ot])
    B_exact = V.sample_auc_by_wire([pe[w] for w in ot], t1[ot], t0[ot])
    own9[view] = {'C_same_own_exact': C_ex, 'C_same_saved': saved['views'][view]['C_same'],
                  'C_same_diff': float(C_ex) - saved['views'][view]['C_same'],
                  'B_indep_own_float_posterior_keys': B_float, 'B_indep_saved': saved['views'][view]['B_indep'],
                  'B_indep_diff': float(B_float) - saved['views'][view]['B_indep'],
                  'B_indep_own_exact_ties_(D1)': B_exact, 'B_indep_exact_ties_minus_saved': float(B_exact) - saved['views'][view]['B_indep'],
                  'n_study_sample': int(len(c)), 'n_test_sample': int(len(ct))}
    if abs(own9[view]['C_same_diff']) > 1e-12 or abs(own9[view]['B_indep_diff']) > 1e-12:
        f9.append(f'own recompute {view}')
rec9['own_recompute_cell17_rep0'] = own9
rec9['failures'] = f9
rec9['status'] = 'PASS' if not f9 else 'FAIL'
print('CHECK9', rec9['status'], f9, rec9['reruns'], {v: (x['C_same_diff'], x['B_indep_diff'], x['B_indep_exact_ties_minus_saved']) for v, x in own9.items()})
V.write_out('CHECK8_reproduce_summaries.json', rec8, started_utc=T0)
V.write_out('CHECK9_rerun_repetitions.json', rec9, started_utc=T0)
