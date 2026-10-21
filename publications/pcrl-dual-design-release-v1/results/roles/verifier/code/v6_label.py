"""Verifier addendum, label part (MODEL_SELECT labels read once through ddr.common.load, operation 'verify', inside a queue unit).

Part U2 (addendum A2): the 24 clauses + 4 guards, the paired costs recorded in EVAL_MS.json (all comparators except W) and the
two-SE reserve (own implementation of the registered OHN1 rule: per version and condition ratio = (cap - value) / max(own SE, floor
SE x sqrt(5306 / households)), floor = the H_LJ SEs in the OHN1 RULES.json, met iff ratio >= 2; RESERVE_MET if both versions meet all
14 conditions, RESERVE_ONE_SEED_ONLY if one, else RESERVE_NOT_MET) for the post-registration configurations, compared with
runs/<F>/<C>/EVAL_MS.json (tolerance 1e-9).

Part G2 (inputs of A3, A4, A6; exact replay): for every post-registration grid bank with GRID.json in both versions, every this-
campaign member and NEURAL_RAW / H_LJ, every recorded cell (registered and robustness repetitions R1-R4, reader seeds k0-k3) and
view: the recorded winner's SCORE AUC from the saved reader scores (cache key of the registered definition; robustness repetitions
keyed '<R>:DDR1_devgrid_<R>'), the winner = maximum of its candidate set, the household bootstrap with the registered seeds
(2026102200 + 100 v + repetition index), and every recorded paired cell difference and cell SE among the replayed members.

    python3 -m ddr.jobqueue run --unit DDR_VER_ADD_LABEL_V<n> --stage verify --rss-gib 2 -- python3 <this file>
"""
from __future__ import annotations
import json
import os
import sys
import time
import traceback
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))

import numpy as np  # noqa: E402

import vlib as V  # noqa: E402

CONFIGS = {'DA': ('DA_IPW', 'DA_IPW_SBC_LO', 'DA_IPW_SBC_HI'),
           'DB': ('DB_PTC_R1', 'DB_SBC_R1_LO', 'DB_SBC_R1_HI', 'DB_PTC_R1_OCC', 'DB_ZTILT_R1')}
ALL_DDR = {'DA': ('DA_TWIN', 'DA_STD', 'DA_SBC_LO', 'DA_SBC_HI', 'DA_SPD_LO', 'DA_SPD_HI', 'DA_COMP', 'DA_STD_OCC', 'DA_IPW',
                  'DA_IPW_SBC_LO', 'DA_IPW_SBC_HI'),
           'DB': ('DB_PTC', 'DB_SBC_LO', 'DB_SBC_HI', 'DB_SPD_LO', 'DB_SPD_HI', 'DB_ZTILT', 'DB_SHUF', 'DB_PTC_OCC', 'DB_PTC_R1',
                  'DB_SBC_R1_LO', 'DB_SBC_R1_HI', 'DB_PTC_R1_OCC', 'DB_ZTILT_R1')}
FAMILY_OF = {c: f for f, cs in ALL_DDR.items() for c in cs}
REFS_REPLAYED = ('NEURAL_RAW', 'H_LJ')
BANKS = ('DDR1_R1', 'DDR1_DA_R1', 'DDR1_ROB', 'DDR1_ROB_DB_R1', 'DDR1_R1_ZTILT', 'DDR1_R1_BESIDE', 'DDR1_DA_R1_BESIDE')
TOL = 1e-9
OHN_RULES = ROOT / 'results' / 'pcrl_occupation_headroom_overnight_v1' / 'roles' / 'statistical_reviewer' / 'RULES.json'
CACHE_DIR = V.RUNS / 'devgrid_cache'


def per_person(y, rel):
    return {t: V.losses(y[t], rel[t], V.K[t]) for t in V.TASKS}


def reserve(entries, floor, G0):
    """entries: {v: {'clauses': [...], 'guards': [...], 'households': G}} (own values and SEs)."""
    per = {}
    for v, e in entries.items():
        scale = (G0 / e['households']) ** 0.5
        ratios = {}
        for c in e['clauses'] + e['guards']:
            name = f"{c['task']}|{c['reference']}|{c['metric']}" if 'task' in c else f"guard|{c['group']}"
            fl = float(floor[str(v)][name]) * scale
            se_used = max(c['se'], fl)
            ratios[name] = (c['cap'] - c['value']) / se_used
        failed = sorted(n for n, r in ratios.items() if r < 2.0)
        per[v] = {'min_ratio': min(ratios.values()), 'binding': min(ratios, key=ratios.get), 'failed': failed, 'met': not failed}
    met = [v for v in per if per[v]['met']]
    status = 'RESERVE_MET' if len(met) == 2 else ('RESERVE_ONE_SEED_ONLY' if met else 'RESERVE_NOT_MET')
    return {'status': status, 'per_version': per}


def part_u2(y, sex, unit, refs, out):
    rules = json.loads(OHN_RULES.read_text())['reserve']
    floor, G0 = rules['floor']['se'], int(rules['floor']['households'])
    res, paired, worst = [], [], {'value': 0.0, 'se': 0.0, 'paired_value': 0.0, 'paired_se': 0.0}
    cache = {}

    def losses_of(cfg, v):
        if (cfg, v) not in cache:
            arr, rec = V.load_release_ms(FAMILY_OF[cfg], cfg, v)
            cache[(cfg, v)] = (per_person(y, arr['MODEL_SELECT']), rec['release_sha256'])
        return cache[(cfg, v)]
    reserves = {}
    for fam, cfgs in CONFIGS.items():
        for cfg in cfgs:
            ev = json.loads((V.RUNS / fam / cfg / 'EVAL_MS.json').read_text())
            ents = {}
            for v in (0, 1):
                own, rsha = losses_of(cfg, v)
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
                ve = ev['versions'][str(v)]
                cmp_ = []
                for c in clauses:
                    h = [x for x in ve['utility']['clauses'] if x['task'] == c['task'] and x['reference'] == c['reference']
                         and x['metric'] == c['metric']][0]
                    cmp_.append((abs(h['value'] - c['value']), abs(h['se'] - c['se']), bool(h['point_pass']) == c['point_pass']))
                for gd in guards:
                    h = [x for x in ve['utility']['guards'] if x['group'] == gd['group']][0]
                    cmp_.append((abs(h['value'] - gd['value']), abs(h['se'] - gd['se']), bool(h['point_pass']) == gd['point_pass']))
                dv, ds = max(x[0] for x in cmp_), max(x[1] for x in cmp_)
                worst['value'] = max(worst['value'], dv); worst['se'] = max(worst['se'], ds)
                g1v = all(c['point_pass'] for c in clauses + guards)
                ents[v] = {'clauses': clauses, 'guards': guards, 'households': int(len(np.unique(unit)))}
                res.append({'config': cfg, 'version': v, 'g1_version_pass': g1v, 'max_dv': dv, 'max_dse': ds,
                            'flags_agree': all(x[2] for x in cmp_) and bool(ve['point']['point_pass_12_plus_2']) == g1v,
                            'release_sha_matches_eval': ve.get('release_sha256') == rsha,
                            'max_value_over_cap': max(c['value'] / c['cap'] for c in clauses + guards),
                            'clauses': clauses, 'guards': guards})
                for comp, ent in ve['paired_costs'].items():
                    if comp == 'W':
                        continue
                    b, _ = losses_of(comp.split('/')[1], v)
                    costs = {}
                    for task in V.TASKS:
                        for m, key in (('acc_loss', 'err'), ('nll', 'nll'), ('brier', 'brier')):
                            dd = own[task][key] - b[task][key]
                            costs[f'{task}|{m}'] = (float(dd.mean()), V.clustered_se(dd, unit))
                            if m == 'brier':
                                for g, nm in ((0, 'men_brier'), (1, 'women_brier')):
                                    costs[f'{task}|{nm}'] = V.subgroup_mean_se(dd, sex == g, unit)
                    tot = (own['income']['nll'] + own['occupation']['nll']) - (b['income']['nll'] + b['occupation']['nll'])
                    costs['total|nll'] = (float(tot.mean()), V.clustered_se(tot, unit))
                    pdv = max(abs(ent['costs'][k]['value'] - costs[k][0]) for k in costs)
                    pds = max(abs(ent['costs'][k]['se'] - costs[k][1]) for k in costs)
                    worst['paired_value'] = max(worst['paired_value'], pdv); worst['paired_se'] = max(worst['paired_se'], pds)
                    paired.append({'config': cfg, 'version': v, 'comparator': comp, 'max_dv': pdv, 'max_dse': pds,
                                   'keys_equal': sorted(ent['costs']) == sorted(costs), 'total_nll': costs['total|nll'][0]})
            rv = reserve(ents, floor, G0)
            rec_r = ev.get('reserve_rules') or {}
            reserves[cfg] = {'mine': rv, 'recorded_status': rec_r.get('status'),
                             'recorded_min_ratio_by_version': rec_r.get('min_ratio_by_version'),
                             'status_agrees': rv['status'] == rec_r.get('status'),
                             'max_ratio_diff': max(abs(rv['per_version'][v]['min_ratio'] - float(rec_r['min_ratio_by_version'][str(v)]))
                                                   for v in (0, 1)) if rec_r.get('min_ratio_by_version') else None}
    g1 = {cfg: all(e['g1_version_pass'] for e in res if e['config'] == cfg) for f in CONFIGS for cfg in CONFIGS[f]}
    out['U2'] = {'per_config_version': res, 'paired': paired, 'worst': worst, 'G1': g1, 'reserve': reserves,
                 'all_within_tol': all(e['max_dv'] <= TOL and e['max_dse'] <= TOL and e['flags_agree'] and e['release_sha_matches_eval']
                                       for e in res) and all(p['max_dv'] <= TOL and p['max_dse'] <= TOL and p['keys_equal'] for p in paired)}


def member_arrays(name, v, bank_refs):
    if name in FAMILY_OF:
        arr, _ = V.load_release_ms(FAMILY_OF[name], name, v)
        return arr['MODEL_SELECT']
    return {t: np.asarray(bank_refs[name]['MODEL_SELECT'][t], dtype=np.float64) for t in V.TASKS}


def read_entry(path):
    with np.load(path, allow_pickle=False) as z:
        info = json.loads(str(z['info']))
        scores = {k[len('score__'):]: z[k].copy() for k in z.files if k.startswith('score__')}
    return scores, info


def cache_key(rep, rs, vw, sx):
    tag = rep if rep in ('R1', 'R2') else f'{rep}:DDR1_devgrid_{rep}'
    import hashlib
    return hashlib.sha256(f'{V.GRID_VERSION}|{tag}|{rs}|1.0|{vw}|{sx}'.encode()).hexdigest()


def part_g2(sex, unit, refs, out):
    sx = V.arr_sha(np.asarray(sex))
    res = {'banks': {}, 'worst': {'score_auc': 0.0, 'boot': 0.0, 'cell_diff': 0.0, 'cell_se': 0.0}, 'winner_not_max': [],
           'missing_cache': [], 'wire_mismatch': [], 'n_replayed': 0, 'n_paired_cells': 0}
    masks, weights = {}, {}
    banks = BANKS
    if '--banks' in sys.argv:
        banks = tuple(sys.argv[sys.argv.index('--banks') + 1].split(','))
    for bank in banks:
        gd = V.RUNS / 'GRID' / bank
        if not all((gd / f'S{v}' / 'GRID.json').is_file() for v in (0, 1)):
            res['banks'][bank] = 'NOT_RUN: GRID.json missing for a version'
            continue
        bres = {'cells': 0, 'paired_cells': 0, 'members': None}
        for v in (0, 1):
            G = json.loads((gd / f'S{v}' / 'GRID.json').read_text())
            boot = np.load(gd / f'S{v}' / 'BOOT.npz')
            members = [n for n in G['members'] if n in FAMILY_OF or n in REFS_REPLAYED]
            bres['members'] = members
            rel = {n: member_arrays(n, v, refs[v][2]) for n in members}
            for n in members:
                for t in V.TASKS:
                    if V.wire_sha(rel[n][t]) != G['members'][n]['wire_sha256'][t]:
                        res['wire_mismatch'].append(f'{bank}|V{v}|{n}|{t}')
            mine_auc, mine_boot = {}, {}
            for cell in sorted(G['cells']):
                if '@' in cell:
                    continue
                cv, rep, k = int(cell[1]), cell[2:4], int(cell[5:])
                if rep not in masks:
                    masks[rep] = V.split_masks(unit, rep)
                sc = masks[rep]['SCORE']
                ysc = sex[sc]
                wk = (v, rep)
                if wk not in weights:
                    weights[wk] = V.household_bootstrap_weights(unit[sc], V.BOOT_BASE + 100 * v + V.REP_INDEX[rep])
                W = weights[wk]
                rs = V.reader_seed(v, k)
                for n in members:
                    ent = {}
                    for view in V.VIEWS:
                        p = CACHE_DIR / f'{cache_key(rep, rs, V.view_hash(view, rel[n]["income"], rel[n]["occupation"]), sx)}.npz'
                        if not p.is_file():
                            res['missing_cache'].append(f'{bank}|{cell}|{n}|{view}')
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
                                sv, recipe = 'income', w[len('IGNORE_occupation__'):]
                            elif w.startswith('IGNORE_income__'):
                                sv, recipe = 'occupation', w[len('IGNORE_income__'):]
                            else:
                                sv, recipe = 'pair', w
                            cvals = [i['val_auc'] for vv in ('pair', 'income', 'occupation') for i in ent[vv][1].values()]
                        else:
                            sv, recipe = view, w
                            cvals = [i['val_auc'] for i in ent[view][1].values()]
                        s = ent[sv][0][recipe]
                        wval = ent[sv][1][recipe]['val_auc']
                        if abs(wval - max(cvals)) > 1e-15 or abs(wval - g['val_auc']) > 1e-15:
                            res['winner_not_max'].append(f'{bank}|{cell}|{n}|{view}')
                        a = V.auc(ysc, s)
                        bw = V.weighted_auc(ysc, s, W)
                        bg = boot[f'{cell}__{n}__{view}']
                        res['worst']['score_auc'] = max(res['worst']['score_auc'], abs(a - g['score_auc']))
                        res['worst']['boot'] = max(res['worst']['boot'], float(np.max(np.abs(bw - bg))))
                        mine_auc[(cell, n, view)] = a
                        mine_boot[(cell, n, view)] = bw
                        bres['cells'] += 1
            for cand, comps in G['paired'].items():
                for comp, views in comps.items():
                    for view, e in views.items():
                        for cell, dval in e['cells'].items():
                            if (cell, cand, view) not in mine_auc or (cell, comp, view) not in mine_auc:
                                continue
                            if view == 'pair':
                                d = mine_auc[(cell, comp, view)] - mine_auc[(cell, cand, view)]
                                dbs = mine_boot[(cell, comp, view)] - mine_boot[(cell, cand, view)]
                            else:
                                d = mine_auc[(cell, cand, view)] - mine_auc[(cell, comp, view)]
                                dbs = mine_boot[(cell, cand, view)] - mine_boot[(cell, comp, view)]
                            res['worst']['cell_diff'] = max(res['worst']['cell_diff'], abs(d - dval))
                            res['worst']['cell_se'] = max(res['worst']['cell_se'], abs(float(np.std(dbs, ddof=1)) - e['cell_se'][cell]))
                            bres['paired_cells'] += 1
        res['banks'][bank] = bres
        res['n_replayed'] += bres['cells']; res['n_paired_cells'] += bres['paired_cells']
    out['G2'] = res


def main():
    t0 = time.process_time()
    parts = ('U2', 'G2')
    if '--parts' in sys.argv:
        parts = tuple(sys.argv[sys.argv.index('--parts') + 1].split(','))
    from ddr import common as C
    S = C.load('MODEL_SELECT', ('income', 'occupation', 'sex'), 'verify',
               'independent verifier addendum: MODEL_SELECT utility, paired costs and two-SE reserve of the post-registration '
               'configurations, and exact replay of the post-registration reader-grid banks (own code)')
    y = {'income': np.asarray(S['income'], dtype=np.int64), 'occupation': np.asarray(S['occupation'], dtype=np.int64)}
    sex = np.asarray(S['sex'], dtype=np.int64)
    unit = np.asarray(S['unit'])
    from tsz.evaluate import refs_of
    refs = {v: refs_of(v, S) for v in (0, 1)}
    out = {'unit_id': V.unit_id(), 'utc_start': V.utc(), 'parts': list(parts), 'errors': []}
    for name, fn in (('U2', lambda: part_u2(y, sex, unit, {v: refs[v][0] for v in (0, 1)}, out)),
                     ('G2', lambda: part_g2(sex, unit, refs, out))):
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
    p = V.write_record(f'V6_ADD_LABEL__{V.unit_id() or "NOQUEUE"}.json', out)
    summ = {'record': str(p), 'errors': [e['part'] + ': ' + e['error'] for e in out['errors']], 'cpu_s': out['cpu_s']}
    if 'U2' in out:
        summ['U2'] = {'all_within_tol': out['U2']['all_within_tol'], 'worst': out['U2']['worst'], 'G1': out['U2']['G1'],
                      'reserve': {k: (x['mine']['status'], x['recorded_status'], x['max_ratio_diff'],
                                      {v: round(x['mine']['per_version'][v]['min_ratio'], 4) for v in (0, 1)})
                                  for k, x in out['U2']['reserve'].items()}}
    if 'G2' in out:
        g = out['G2']
        summ['G2'] = {'banks': {b: (x if isinstance(x, str) else {'cells': x['cells'], 'paired_cells': x['paired_cells']})
                                for b, x in g['banks'].items()}, 'worst': g['worst'], 'winner_not_max': len(g['winner_not_max']),
                      'missing_cache': len(g['missing_cache']), 'wire_mismatch': g['wire_mismatch']}
    print(json.dumps(summ, default=V._default))
    return 0 if not out['errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
