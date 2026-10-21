"""Verifier checks 2 and 3 (MODEL_SELECT labels, read once through ddr.common.load, operation 'verify', inside a queue unit).

Part U (check 2): the 12 clauses + 2 guards per version (24 + 4) of every saved DA / DB configuration from MODEL_SELECT labels with
own metric code (accuracy loss = error(release) - error(reference), first argmax; NLL excess with the 1e-12 floor; summed Brier
excess; references U0_S<v> / Ucal_S<v> fetched with tsz.evaluate.refs_of; men's and women's occupation summed-Brier excess vs U0;
household-clustered SEs, linearized subgroup SEs) and the paired costs recorded in EVAL_MS.json (every saved comparator except W),
compared with runs/<F>/<C>/EVAL_MS.json (tolerance 1e-9 on point values and SEs).

Part G (check 3, exact replay; only when GRID/DDR1_MAIN/S0 and S1 exist): for the decisive members (both candidates, the twin,
every saved control point, ZTILT, SHUF, NEURAL_RAW, H_LJ), every registered cell and view: the recorded winner's SCORE AUC
recomputed from the saved reader scores (cache entries located by the registered key) and own AUC code; the winner's VALIDATION AUC
equals the maximum over its candidate set; the household bootstrap (own weights, registered seed) reproduces BOOT.npz; paired cell
differences and cell SEs recomputed.

Part R (check 3, independent approximation): own readers (logistic regression on standardized log-probability ratios; a 15-leaf
gradient-boosting reader), fitted on the FIT parts of R1 / R2 (own implementation of the registered household dealing), oriented and
chosen on VALIDATION, scored on SCORE, both versions: 4 cells per comparison.

    python3 -m ddr.jobqueue run --unit DDR_VER_LABEL_V1 --stage verify --rss-gib 2 -- python3 <this file>
"""
from __future__ import annotations
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))

import numpy as np  # noqa: E402

import vlib as V  # noqa: E402

CONFIGS = {'DA': ('DA_TWIN', 'DA_STD', 'DA_SBC_LO', 'DA_SBC_HI', 'DA_SPD_LO', 'DA_SPD_HI', 'DA_COMP', 'DA_STD_OCC'),
           'DB': ('DB_PTC', 'DB_SBC_LO', 'DB_SBC_HI', 'DB_SPD_LO', 'DB_SPD_HI', 'DB_ZTILT', 'DB_SHUF', 'DB_PTC_OCC')}
FAMILY_OF = {c: f for f, cs in CONFIGS.items() for c in cs}
DECISIVE = ('DA_TWIN', 'DA_STD', 'DA_SBC_LO', 'DA_SBC_HI', 'DA_SPD_LO', 'DA_SPD_HI', 'DB_PTC', 'DB_SBC_LO', 'DB_SBC_HI',
            'DB_SPD_LO', 'DB_SPD_HI', 'DB_ZTILT', 'DB_SHUF', 'NEURAL_RAW', 'H_LJ')
COMPARISONS = {'DA_STD': ('DA_TWIN', 'DA_SBC_LO', 'DA_SBC_HI', 'DA_SPD_LO', 'DA_SPD_HI', 'NEURAL_RAW', 'H_LJ'),
               'DB_PTC': ('DA_TWIN', 'DB_SBC_LO', 'DB_SBC_HI', 'DB_SPD_LO', 'DB_SPD_HI', 'DB_ZTILT', 'DB_SHUF', 'NEURAL_RAW', 'H_LJ')}
TOL = 1e-9
GRID_DIR = V.RUNS / 'GRID' / 'DDR1_MAIN'
CACHE_DIR = V.RUNS / 'devgrid_cache'


def per_person(y, rel):
    return {t: V.losses(y[t], rel[t], V.K[t]) for t in V.TASKS}


def part_u(y, sex, unit, refs, out):
    res, worst = [], {'value': 0.0, 'se': 0.0, 'paired_value': 0.0, 'paired_se': 0.0}
    mine = {}
    for fam, cfgs in CONFIGS.items():
        for cfg in cfgs:
            ev_path = V.RUNS / fam / cfg / 'EVAL_MS.json'
            ev = json.loads(ev_path.read_text()) if ev_path.is_file() else None
            for v in (0, 1):
                arr, rec = V.load_release_ms(fam, cfg, v)
                rel = arr['MODEL_SELECT']
                own = per_person(y, rel)
                mine[(cfg, v)] = own
                ref_l = {r: per_person(y, refs[v][r]) for r in V.REFS}
                clauses = []
                for task in V.TASKS:
                    for r in V.REFS:
                        for m in V.METRICS:
                            key = {'accuracy_loss': 'err', 'nll_excess': 'nll', 'brier_excess': 'brier'}[m]
                            d = own[task][key] - ref_l[r][task][key]
                            clauses.append({'task': task, 'reference': r, 'metric': m, 'value': float(d.mean()),
                                            'se': V.clustered_se(d, unit), 'cap': V.CAPS[m]})
                d = own['occupation']['brier'] - ref_l['U0']['occupation']['brier']
                guards = []
                for g, name in ((0, 'sex0'), (1, 'sex1')):
                    m_, se_ = V.subgroup_mean_se(d, sex == g, unit)
                    guards.append({'group': name, 'value': m_, 'se': se_, 'cap': V.GUARD_CAP})
                for c in clauses + guards:
                    c['point_pass'] = bool(c['value'] <= c['cap'])
                entry = {'family': fam, 'config': cfg, 'version': v, 'clauses': clauses, 'guards': guards,
                         'g1_version_pass': all(c['point_pass'] for c in clauses + guards),
                         'max_value_over_cap': max(c['value'] / c['cap'] for c in clauses + guards),
                         'release_sha256': rec['release_sha256']}
                if ev is None:
                    entry['coordinator_record'] = 'MISSING (no EVAL_MS.json)'
                else:
                    ve = ev['versions'][str(v)]
                    entry['release_sha_matches_eval'] = ve.get('release_sha256') == rec['release_sha256']
                    cmp_ = []
                    for c in clauses:
                        hit = [x for x in ve['utility']['clauses'] if x['task'] == c['task'] and x['reference'] == c['reference']
                               and x['metric'] == c['metric']]
                        if len(hit) != 1:
                            cmp_.append({'clause': f"{c['task']}|{c['reference']}|{c['metric']}", 'error': f'{len(hit)} matches'})
                            continue
                        dv, ds = abs(hit[0]['value'] - c['value']), abs(hit[0]['se'] - c['se'])
                        worst['value'] = max(worst['value'], dv); worst['se'] = max(worst['se'], ds)
                        cmp_.append({'clause': f"{c['task']}|{c['reference']}|{c['metric']}", 'dv': dv, 'dse': ds,
                                     'pass_agrees': bool(hit[0]['point_pass']) == c['point_pass']})
                    for gd in guards:
                        hit = [x for x in ve['utility']['guards'] if x['group'] == gd['group']]
                        dv, ds = abs(hit[0]['value'] - gd['value']), abs(hit[0]['se'] - gd['se'])
                        worst['value'] = max(worst['value'], dv); worst['se'] = max(worst['se'], ds)
                        cmp_.append({'clause': f"guard|{gd['group']}", 'dv': dv, 'dse': ds,
                                     'pass_agrees': bool(hit[0]['point_pass']) == gd['point_pass']})
                    entry['comparison'] = cmp_
                    entry['all_within_tol'] = all(x.get('dv', 1) <= TOL and x.get('dse', 1) <= TOL and x.get('pass_agrees')
                                                  for x in cmp_)
                    entry['point_pass_agrees'] = bool(ve['point']['point_pass_12_plus_2']) == entry['g1_version_pass']
                res.append(entry)
    # paired costs recorded in EVAL_MS.json (comparators of this campaign; W skipped)
    paired = []
    for fam, cfgs in CONFIGS.items():
        for cfg in cfgs:
            ev_path = V.RUNS / fam / cfg / 'EVAL_MS.json'
            if not ev_path.is_file():
                continue
            ev = json.loads(ev_path.read_text())
            for v in (0, 1):
                a = mine[(cfg, v)]
                for comp, ent in ev['versions'][str(v)]['paired_costs'].items():
                    if comp == 'W':
                        continue
                    cfam, ccfg = comp.split('/')
                    b = mine.get((ccfg, v))
                    if b is None:
                        arr, _ = V.load_release_ms(cfam, ccfg, v)
                        b = per_person(y, arr['MODEL_SELECT'])
                    costs = {}
                    for task in V.TASKS:
                        for m, key in (('acc_loss', 'err'), ('nll', 'nll'), ('brier', 'brier')):
                            d = a[task][key] - b[task][key]
                            costs[f'{task}|{m}'] = (float(d.mean()), V.clustered_se(d, unit))
                            if m == 'brier':
                                for g, nm in ((0, 'men_brier'), (1, 'women_brier')):
                                    costs[f'{task}|{nm}'] = V.subgroup_mean_se(d, sex == g, unit)
                    tot = (a['income']['nll'] + a['occupation']['nll']) - (b['income']['nll'] + b['occupation']['nll'])
                    costs['total|nll'] = (float(tot.mean()), V.clustered_se(tot, unit))
                    dv = max(abs(ent['costs'][k]['value'] - costs[k][0]) for k in costs)
                    ds = max(abs(ent['costs'][k]['se'] - costs[k][1]) for k in costs)
                    worst['paired_value'] = max(worst['paired_value'], dv); worst['paired_se'] = max(worst['paired_se'], ds)
                    paired.append({'config': cfg, 'version': v, 'comparator': comp, 'max_dv': dv, 'max_dse': ds,
                                   'total_nll': costs['total|nll'][0], 'total_nll_se': costs['total|nll'][1],
                                   'recorded_total_nll': ent['costs']['total|nll']['value'],
                                   'keys_equal': sorted(ent['costs']) == sorted(costs)})
    # G1 over both versions
    g1 = {}
    for cfg in FAMILY_OF:
        ents = [e for e in res if e['config'] == cfg]
        g1[cfg] = {'G1_24_plus_4': all(e['g1_version_pass'] for e in ents),
                   'failed': [f"V{e['version']}|{c.get('task', 'guard')}|{c.get('reference', c.get('group'))}|{c.get('metric', '')}"
                              for e in ents for c in e['clauses'] + e['guards'] if not c['point_pass']],
                   'max_value_over_cap': max(e['max_value_over_cap'] for e in ents)}
    out['U'] = {'per_config_version': res, 'paired': paired, 'worst_abs_diff': worst, 'G1': g1,
                'all_within_tol': all(e.get('all_within_tol', False) for e in res if 'comparison' in e)
                and all(p['max_dv'] <= TOL and p['max_dse'] <= TOL and p['keys_equal'] for p in paired),
                'missing_eval_records': sorted({e['config'] for e in res if 'coordinator_record' in e})}
    return mine


def member_arrays(name, v, bank):
    if name in FAMILY_OF:
        arr, rec = V.load_release_ms(FAMILY_OF[name], name, v)
        return arr['MODEL_SELECT']
    return {t: np.asarray(bank[name]['MODEL_SELECT'][t], dtype=np.float64) for t in V.TASKS}


def read_entry(path):
    with np.load(path, allow_pickle=False) as z:
        info = json.loads(str(z['info']))
        scores = {k[len('score__'):]: z[k].copy() for k in z.files if k.startswith('score__')}
    return scores, info


def part_g(sex, unit, refs, out):
    if not all((GRID_DIR / f'S{v}' / 'GRID.json').is_file() for v in (0, 1)):
        out['G'] = {'status': 'NOT_RUN', 'reason': 'GRID.json of DDR1_MAIN missing for a version at run time'}
        return
    sx = V.arr_sha(np.asarray(sex))
    res = {'cells': [], 'paired': [], 'wire_checks': [], 'worst': {'score_auc': 0.0, 'boot': 0.0, 'cell_diff': 0.0, 'cell_se': 0.0},
           'winner_not_max': [], 'missing_cache': []}
    for v in (0, 1):
        G = json.loads((GRID_DIR / f'S{v}' / 'GRID.json').read_text())
        boot = np.load(GRID_DIR / f'S{v}' / 'BOOT.npz')
        rel = {n: member_arrays(n, v, refs[v][2]) for n in DECISIVE}
        for n in DECISIVE:
            for t in V.TASKS:
                rec_w = G['members'][n]['wire_sha256'][t]
                res['wire_checks'].append({'v': v, 'member': n, 'task': t, 'match': V.wire_sha(rel[n][t]) == rec_w})
        mine_auc, mine_boot = {}, {}
        for rep in ('R1', 'R2'):
            sc = V.split_masks(unit, rep)['SCORE']
            ysc = sex[sc]
            W = V.household_bootstrap_weights(unit[sc], V.BOOT_BASE + 100 * v + V.REP_INDEX[rep])
            for k in (0, 1):
                rs = V.reader_seed(v, k)
                cell = f'S{v}{rep}k{k}'
                for n in DECISIVE:
                    ent = {}
                    for view in V.VIEWS:
                        p = CACHE_DIR / f'{V.cache_key(rep, rs, V.view_hash(view, rel[n]["income"], rel[n]["occupation"]), sx)}.npz'
                        if not p.is_file():
                            res['missing_cache'].append(f'{cell}|{n}|{view}')
                            ent = None
                            break
                        ent[view] = read_entry(p)
                    if ent is None:
                        continue
                    for view in V.VIEWS:
                        g = G['cells'][cell][n][view]
                        w = g['winner']
                        if view == 'pair':
                            if w.startswith('IGNORE_occupation__'):
                                src_view, recipe = 'income', w[len('IGNORE_occupation__'):]
                            elif w.startswith('IGNORE_income__'):
                                src_view, recipe = 'occupation', w[len('IGNORE_income__'):]
                            else:
                                src_view, recipe = 'pair', w
                            cand_vals = ([i['val_auc'] for i in ent['pair'][1].values()] + [i['val_auc'] for i in ent['income'][1].values()]
                                         + [i['val_auc'] for i in ent['occupation'][1].values()])
                        else:
                            src_view, recipe = view, w
                            cand_vals = [i['val_auc'] for i in ent[view][1].values()]
                        s = ent[src_view][0][recipe]
                        wval = ent[src_view][1][recipe]['val_auc']
                        if abs(wval - max(cand_vals)) > 1e-15 or abs(wval - g['val_auc']) > 1e-15:
                            res['winner_not_max'].append({'cell': cell, 'member': n, 'view': view, 'winner': w, 'val': wval,
                                                          'max': max(cand_vals), 'recorded_val': g['val_auc']})
                        a = V.auc(ysc, s)
                        bw = V.weighted_auc(ysc, s, W)
                        bg = boot[f'{cell}__{n}__{view}']
                        da, db = abs(a - g['score_auc']), float(np.max(np.abs(bw - bg)))
                        res['worst']['score_auc'] = max(res['worst']['score_auc'], da)
                        res['worst']['boot'] = max(res['worst']['boot'], db)
                        mine_auc[(cell, n, view)] = a
                        mine_boot[(cell, n, view)] = bw
                        res['cells'].append({'cell': cell, 'member': n, 'view': view, 'winner': w, 'score_auc': a, 'd_auc': da,
                                             'd_boot': db})
        for cand, comps in COMPARISONS.items():
            for comp in comps:
                if comp not in G['paired'].get(cand, {}):
                    res['paired'].append({'v': v, 'cand': cand, 'comp': comp, 'error': 'no paired entry in GRID.json'})
                    continue
                pe = G['paired'][cand][comp]
                for view in V.VIEWS:
                    for cell in pe[view]['cells']:
                        if (cell, cand, view) not in mine_auc or (cell, comp, view) not in mine_auc:
                            continue
                        if view == 'pair':
                            d = mine_auc[(cell, comp, view)] - mine_auc[(cell, cand, view)]
                            dbs = mine_boot[(cell, comp, view)] - mine_boot[(cell, cand, view)]
                        else:
                            d = mine_auc[(cell, cand, view)] - mine_auc[(cell, comp, view)]
                            dbs = mine_boot[(cell, cand, view)] - mine_boot[(cell, comp, view)]
                        se = float(np.std(dbs, ddof=1))
                        dd, dse = abs(d - pe[view]['cells'][cell]), abs(se - pe[view]['cell_se'][cell])
                        res['worst']['cell_diff'] = max(res['worst']['cell_diff'], dd)
                        res['worst']['cell_se'] = max(res['worst']['cell_se'], dse)
                        res['paired'].append({'v': v, 'cand': cand, 'comp': comp, 'view': view, 'cell': cell, 'diff': d, 'se': se,
                                              'recorded_diff': pe[view]['cells'][cell], 'recorded_se': pe[view]['cell_se'][cell]})
    res['status'] = 'RUN'
    res['n_cells_replayed'] = len(res['cells'])
    res['all_wires_match'] = all(x['match'] for x in res['wire_checks'])
    out['G'] = res


def logratio(p):
    p = np.asarray(p, dtype=np.float64)
    return np.log(p[:, 1:]) - np.log(p[:, :1])


def part_r(sex, unit, refs, out):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    res = {'readers': 'LR: LogisticRegression(C=1.0, lbfgs) on FIT-standardized log-probability ratios log(p_k / p_0); HGB: '
                      'HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.05, max_iter=200, min_samples_leaf=40, '
                      'early_stopping=False, random_state=9100 + 10 v + rep index); both fitted on FIT, oriented on VALIDATION; '
                      'winner = higher VALIDATION AUC; SCORE AUC reported', 'cells': [], 'diffs': []}
    aucs = {}
    for v in (0, 1):
        rel = {n: member_arrays(n, v, refs[v][2]) for n in DECISIVE}
        for rep in ('R1', 'R2'):
            m = V.split_masks(unit, rep)
            fit, val, sc = m['FIT'], m['VALIDATION'], m['SCORE']
            cell = f'S{v}{rep}'
            for n in DECISIVE:
                li, lo = logratio(rel[n]['income']), logratio(rel[n]['occupation'])
                for view, X in (('income', li), ('occupation', lo), ('pair', np.column_stack([li, lo]))):
                    mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-12
                    Z = (X - mu) / sd
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Z[fit], sex[fit])
                    s_lr = lr.decision_function(Z)
                    h = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
                                                       early_stopping=False,
                                                       random_state=9100 + 10 * v + V.REP_INDEX[rep]).fit(X[fit], sex[fit])
                    s_h = h.decision_function(X)
                    rec = {}
                    for nm, s in (('LR', s_lr), ('HGB', s_h)):
                        va = V.auc(sex[val], s[val])
                        if va < 0.5:
                            s = -s; va = 1.0 - va
                        rec[nm] = {'val': va, 'score': V.auc(sex[sc], s[sc])}
                    win = 'LR' if rec['LR']['val'] >= rec['HGB']['val'] else 'HGB'
                    rec['WIN'] = {'reader': win, 'score': rec[win]['score']}
                    aucs[(cell, n, view)] = rec
                    res['cells'].append({'cell': cell, 'member': n, 'view': view, 'LR': rec['LR'], 'HGB': rec['HGB'],
                                         'WIN': rec['WIN']})
    for cand, comps in COMPARISONS.items():
        for comp in comps:
            for view in V.VIEWS:
                for rd in ('LR', 'HGB', 'WIN'):
                    per = {}
                    for v in (0, 1):
                        for rep in ('R1', 'R2'):
                            cell = f'S{v}{rep}'
                            a, b = aucs[(cell, cand, view)][rd]['score'], aucs[(cell, comp, view)][rd]['score']
                            per[cell] = (b - a) if view == 'pair' else (a - b)
                    vm = {v: float(np.mean([per[f'S{v}R1'], per[f'S{v}R2']])) for v in (0, 1)}
                    res['diffs'].append({'cand': cand, 'comp': comp, 'view': view, 'reader': rd, 'cells': per,
                                         'version_means': vm, 'mean': float(np.mean(list(per.values())))})
    out['R'] = res


def main():
    t0 = time.process_time()
    parts = ('U', 'G', 'R')
    if '--parts' in sys.argv:
        parts = tuple(sys.argv[sys.argv.index('--parts') + 1].split(','))
    from ddr import common as C
    S = C.load('MODEL_SELECT', ('income', 'occupation', 'sex'), 'verify',
               'independent verifier: MODEL_SELECT utility clauses, guards and paired costs of every saved DA / DB configuration, '
               'exact replay of the DDR1_MAIN reader-grid SCORE AUCs and bootstrap, and independent reader re-fits (own code)')
    y = {'income': np.asarray(S['income'], dtype=np.int64), 'occupation': np.asarray(S['occupation'], dtype=np.int64)}
    sex = np.asarray(S['sex'], dtype=np.int64)
    unit = np.asarray(S['unit'])
    from tsz.evaluate import refs_of                       # frozen U0 / Ucal / bank vectors (no label inside)
    refs = {v: refs_of(v, S) for v in (0, 1)}
    refsUC = {v: refs[v][0] for v in (0, 1)}
    out = {'unit_id': V.unit_id(), 'utc_start': V.utc(), 'people': int(len(sex)), 'households': int(len(np.unique(unit))),
           'errors': [], 'parts': list(parts)}
    for name, fn in (('U', lambda: part_u(y, sex, unit, refsUC, out)), ('G', lambda: part_g(sex, unit, refs, out)),
                     ('R', lambda: part_r(sex, unit, refs, out))):
        if name not in parts:
            continue
        t1 = time.process_time()
        try:
            fn()
        except Exception as e:
            out['errors'].append({'part': name, 'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()[-3000:]})
        out[f'cpu_s_{name}'] = round(time.process_time() - t1, 2)
    out['cpu_s'] = round(time.process_time() - t0, 2)
    out['utc_end'] = V.utc()
    p = V.write_record(f'V2_LABEL__{V.unit_id() or "NOQUEUE"}.json', out)
    summ = {'record': str(p), 'errors': [e['part'] + ': ' + e['error'] for e in out['errors']], 'cpu_s': out['cpu_s']}
    if 'U' in out:
        summ['U_all_within_tol'] = out['U']['all_within_tol']
        summ['U_worst'] = out['U']['worst_abs_diff']
        summ['U_missing'] = out['U']['missing_eval_records']
        summ['G1'] = {k: v['G1_24_plus_4'] for k, v in out['U']['G1'].items()}
    if 'G' in out:
        summ['G'] = {k: out['G'].get(k) for k in ('status', 'reason', 'worst', 'n_cells_replayed', 'all_wires_match')}
        summ['G_winner_not_max'] = len(out['G'].get('winner_not_max', []))
        summ['G_missing_cache'] = len(out['G'].get('missing_cache', []))
    print(json.dumps(summ, default=V._default))
    return 0 if not out['errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
