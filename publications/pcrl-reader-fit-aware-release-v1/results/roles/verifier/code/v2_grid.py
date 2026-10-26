#!/usr/bin/env python3
"""RFA1 verifier, wave 2, task 4: recompute the main reader grid RFA1_MAIN (S0, S1) from its saved outputs with the verifier's own
code, then the decisive comparisons of RF_FULL, its aggregates, Q* frontiers, reader targeting and the D1-D4 conditions.

Per version: members' MODEL_SELECT arrays (this campaign's releases by the verifier's own hash-checked reader; references through the
campaign loader rfa.devreaders.load_member, checked against the grid's recorded wire hashes); per cell (S<v><R>k<k>): the verifier's
reader roles, reader seed, cache keys, cache entries; per member and view: the VALIDATION winner (vlib.pick order), its SCORE AUC
(vlib.auc), its household-bootstrap draws (vlib.boot_weights / weighted_auc), all 16 per-recipe SCORE AUCs, the non-table winner
(LOOKUP excluded) and its SCORE AUC; each compared with GRID.json / BOOT.npz. Paired per-cell differences and SEs of RF_FULL vs every
comparator recomputed from the verifier's own AUCs and draws; aggregates, recipient gates, the 0.020 target and the Q* frontier of
STOPFIT / ALTCRIT / RAWJ (paths START + saved points; costs from runs/FAM/RF_FULL/EVAL_MS.json, reproduced exactly by unit
RFA_VER_USEFULNESS_v1) with vagg; reader targeting with the verifier's non-table winners; D1 (from V2_USEFULNESS.json), D2, D3 and D4
assembled by the verifier from RULES.md 2, 4.1, 6. SEX through rfa.common.load(..., 'verify') from this __main__, in a queue unit.

    python3 -m rfa.jobqueue run --unit RFA_VER_GRID_v1 --stage verify --rss-gib 3 -- \\
        <repository>/.venv/bin/python results/pcrl_reader_fit_aware_release_v1/roles/verifier/code/v2_grid.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))

import numpy as np  # noqa: E402

import vagg as A  # noqa: E402
import vlib as V  # noqa: E402

OUT = HERE.parent / 'out' / 'V2_GRID.json'
CAMPAIGN = 'pcrl_reader_fit_aware_release_v1'
RUNS = ROOT / '.private' / CAMPAIGN / 'runs'
BANK = 'RFA1_MAIN'
CAND = 'RF_FULL'
START, CONT = 'NEURAL_RAW', 'RF_TASK'
PATHS = {'STOPFIT': ['RF_STOP_LO', 'RF_STOP_HI'], 'ALTCRIT': ['RF_ALTC_HI'], 'RAWJ': ['RF_RAWJ_HI']}
ROLES6 = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
NONTABLE = tuple(r for r in V.RECIPES if r != 'LOOKUP')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_sex():
    from rfa import common as C
    S = C.load('MODEL_SELECT', ('features', 'sex'), 'verify',
               f'verify: independent recomputation of reader grid {BANK} S0/S1 (winners, SCORE AUCs, household bootstrap, paired '
               'differences, aggregates, frontiers and decision conditions; verifier wave 2)')
    S.pop('X', None)
    return S


def own_release(config, version):
    d = RUNS / 'FAM' / config / f'V{int(version)}'
    rec = json.loads((d / 'RELEASE.json').read_text())
    if sha256(d / 'RELEASE.npz') != rec['release_sha256']:
        raise PermissionError(f'{config}/V{version}: RELEASE.npz differs from its record')
    with np.load(d / 'RELEASE.npz', allow_pickle=False) as z:
        out = {}
        for t in V.TASKS:
            a = z[f'MODEL_SELECT__{t}']
            if V.arr_sha(a) != rec['wire_sha256'][f'MODEL_SELECT__{t}']:
                raise PermissionError(f'{config}/V{version} {t}: array hash differs')
            out[t] = np.asarray(a, dtype=np.float64).copy()
    return out


def members_of(grid, version, row_id):
    from rfa import devreaders as DR
    out, prov_ok = {}, {}
    for name, spec in grid['spec']['members'].items():
        if spec.get('type') == 'rfa':
            rel = own_release(spec['config'], version)
        else:
            rel, _ = DR.load_member(spec, version, n=len(row_id), row_id=row_id)
            if any(np.asarray(rel[t]).ndim != 2 for t in V.TASKS):
                raise NotImplementedError(f'{name}: randomized member not handled by this verifier code')
        got = {t: V.wire_sha(rel[t]) for t in V.TASKS}
        prov_ok[name] = got == grid['members'][name]['wire_sha256']
        out[name] = {t: np.asarray(rel[t], dtype=np.float64) for t in V.TASKS}
    return out, prov_ok


def nontable_winner(info):
    out = {}
    for view in ('income', 'occupation'):
        out[view] = V.pick([(n, info[view][n]['val_auc'], f'{view}__{n}', 'source' if n in V.SRC8 else 'new') for n in NONTABLE])
    c = []
    for group, org in ((V.SRC8, 'source'), (V.NEW8, 'new')):
        g = [n for n in group if n != 'LOOKUP']
        c += [(n, info['pair'][n]['val_auc'], f'pair__{n}', org) for n in g]
        c += [(f'IGNORE_occupation__{n}', info['income'][n]['val_auc'], f'income__{n}', org) for n in g]
        c += [(f'IGNORE_income__{n}', info['occupation'][n]['val_auc'], f'occupation__{n}', org) for n in g]
    out['pair'] = V.pick(c)
    return out


def recompute_version(version, sex, unit, row_id):
    gdir = RUNS / 'GRID' / BANK / f'S{version}'
    grid = json.loads((gdir / 'GRID.json').read_text())
    with np.load(gdir / 'BOOT.npz') as z:
        boot = {k: z[k].copy() for k in z.files}
    members, prov_ok = members_of(grid, version, row_id)
    sx = V.arr_sha(np.asarray(sex))
    worst = {'val_auc': 0.0, 'score_auc': 0.0, 'boot': 0.0, 'per_recipe_score_auc': 0.0, 'nontable_score_auc': 0.0,
             'paired_d': 0.0, 'paired_se': 0.0}
    mism, mine = [], {}
    seed_eq = {}
    for cell, cres in grid['cells'].items():
        vv, rep, k, frac = __import__('vcore').parse_cell(cell)
        if vv != version or frac != 1.0:
            mism.append(f'{cell}: unexpected cell in S{version}')
            continue
        roles = V.reader_roles(unit, rep)
        sc = roles['SCORE']
        y = np.asarray(sex)[sc]
        W = V.boot_weights(np.asarray(unit)[sc], version, rep)
        rs = V.reader_seed(version, k)
        for name, ent in cres.items():
            rel = members[name]
            E = {}
            for view in V.VIEWS:
                p = RUNS / 'devgrid_cache' / f"{V.cache_key(rep, rs, 1.0, V.view_hash(view, rel['income'], rel['occupation']), sx)}.npz"
                if not p.is_file():
                    mism.append(f'{cell}/{name}/{view}: cache entry missing')
                    continue
                E[view] = V.read_cache_entry(p)
            if len(E) != 3:
                continue
            info = {vw: {n: E[vw][n][1] for n in V.RECIPES} for vw in V.VIEWS}
            w, wn = V.winners(info), nontable_winner(info)
            for view in V.VIEWS:
                g = ent[view]
                vw, rname = w[view]['reader_key'].split('__', 1)
                s = E[vw][rname][0]
                a = V.auc(y, s)
                bd = V.weighted_auc(y, s, W)
                if w[view]['winner'] != g['winner'] or w[view]['near_tie'] != g['near_tie'] or w[view]['gap_to_runner_up'] != g['gap_to_runner_up']:
                    mism.append(f'{cell}/{name}/{view}: winner {g["winner"]} vs verifier {w[view]["winner"]}')
                worst['val_auc'] = max(worst['val_auc'], abs(w[view]['val_auc'] - g['val_auc']))
                worst['score_auc'] = max(worst['score_auc'], abs(a - g['score_auc']))
                key = f'{cell}__{name}__{view}'
                worst['boot'] = max(worst['boot'], float(np.max(np.abs(bd - boot[key]))) if key in boot else float('inf'))
                pr = {n: A.auc_fast(y, E[view][n][0]) for n in V.RECIPES}
                worst['per_recipe_score_auc'] = max(worst['per_recipe_score_auc'],
                                                    max(abs(pr[n] - g['per_recipe_score_auc'][n]) for n in V.RECIPES))
                nvw, nname = wn[view]['reader_key'].split('__', 1)
                na = V.auc(y, E[nvw][nname][0])
                gx = ((g.get('extras') or {}).get('nolookup') or {})
                if gx.get('winner') != wn[view]['winner']:
                    mism.append(f'{cell}/{name}/{view}: non-table winner {gx.get("winner")} vs verifier {wn[view]["winner"]}')
                worst['nontable_score_auc'] = max(worst['nontable_score_auc'], abs(na - gx.get('score_auc', float('nan'))) if gx else float('inf'))
                mine[(cell, name, view)] = {'auc': a, 'boot': bd, 'winner': w[view]['winner'], 'nontable_auc': na,
                                            'nontable_winner': wn[view]['winner'], 'near_tie': w[view]['near_tie']}
                if k == 1:                                   # reader-seed effect: k0 vs k1 cached scores of seed-dependent recipes
                    p0 = RUNS / 'devgrid_cache' / f"{V.cache_key(rep, V.reader_seed(version, 0), 1.0, V.view_hash(view, rel['income'], rel['occupation']), sx)}.npz"
                    if p0.is_file():
                        E0 = V.read_cache_entry(p0)
                        for n in ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP'):
                            seed_eq.setdefault(n, [0, 0])
                            seed_eq[n][0] += int(np.array_equal(E0[n][0], E[view][n][0]))
                            seed_eq[n][1] += 1
    paired = {}
    for comp in (grid.get('paired') or {}).get(CAND, {}):
        for view in V.VIEWS:
            rec = {'cells': {}, 'cell_se': {}}
            for cell in grid['cells']:
                if (cell, CAND, view) not in mine or (cell, comp, view) not in mine:
                    continue
                a, b = mine[(cell, CAND, view)], mine[(cell, comp, view)]
                d, se = V.paired_cell(view, a['auc'], b['auc'], a['boot'], b['boot'])
                rec['cells'][cell], rec['cell_se'][cell] = d, se
                gp = grid['paired'][CAND][comp][view]
                worst['paired_d'] = max(worst['paired_d'], abs(d - gp['cells'][cell]))
                worst['paired_se'] = max(worst['paired_se'], abs(se - gp['cell_se'][cell]))
            paired.setdefault(comp, {})[view] = rec
    nontable = {}
    for comp in paired:
        nontable[comp] = {'cells': {cell: mine[(cell, comp, 'pair')]['nontable_auc'] - mine[(cell, CAND, 'pair')]['nontable_auc']
                                    for cell in grid['cells'] if (cell, comp, 'pair') in mine and (cell, CAND, 'pair') in mine}}
    winners = {f'{c}|{n}|{vw}': {'winner': m['winner'], 'auc': m['auc'], 'nontable_winner': m['nontable_winner'], 'near_tie': m['near_tie']}
               for (c, n, vw), m in mine.items()}
    return {'grid_sha256': sha256(gdir / 'GRID.json'), 'boot_sha256': sha256(gdir / 'BOOT.npz'), 'n_cells': len(grid['cells']),
            'members': sorted(members), 'provenance_wire_hashes_match': prov_ok, 'worst_abs_diff': worst, 'mismatches': mism,
            'seed_dependent_equal_k0_k1': seed_eq, 'paired': paired, 'nontable': nontable, 'winners': winners}


def merge(per_version, comp, view):
    e = {'cells': {}, 'cell_se': {}}
    for v in (0, 1):
        x = per_version[v]['paired'].get(comp, {}).get(view)
        if x:
            e['cells'].update(x['cells'])
            e['cell_se'].update(x['cell_se'])
    return e


def summarize_comparison(c):
    a = c['benefit']['aggregate']
    out = {'benefit_label': c['benefit']['label'], 'recipient_gate_pass': c['recipient_gate_pass'], 'passes': c['passes'],
           'recipient_labels': {v: c['recipient'][v]['label'] for v in ('income', 'occupation')}}
    if a.get('complete'):
        out.update({'mean': a['mean'], 'se_agg': a['se_agg'], 'lower': a['lower'], 'upper': a['upper'],
                    'version_means': {v: a['per_version'][v]['mean'] for v in (0, 1)}, 'se_hh': a['se_hh'], 'se_rs': a['se_rs']})
    out['recipient_version_means'] = {view: {v: c['recipient'][view]['aggregate']['per_version'][v]['mean'] for v in (0, 1)}
                                      for view in ('income', 'occupation') if c['recipient'][view]['aggregate'].get('complete')}
    return out


def main():
    if not os.environ.get('RFA_UNIT_ID'):
        sys.exit('refused: labels are read only inside an rfa.jobqueue verify unit')
    t0 = time.process_time()
    S = load_sex()
    sex, unit, row_id = S['sex'], S['unit'], S['row_id']
    per_version = {v: recompute_version(v, sex, unit, row_id) for v in (0, 1)}
    res = {'schema': 'RFA1-verifier-grid-v2', 'unit_id': os.environ.get('RFA_UNIT_ID'), 'bank': BANK, 'checks': {},
           'per_version': {str(v): {k: x for k, x in pv.items() if k not in ('paired', 'nontable', 'winners')} for v, pv in per_version.items()}}
    chk = res['checks']
    for v, pv in per_version.items():
        chk[f'S{v}|member_wires_match_grid_provenance'] = all(pv['provenance_wire_hashes_match'].values())
        chk[f'S{v}|no_winner_or_cache_mismatch'] = not pv['mismatches']
        for key, x in pv['worst_abs_diff'].items():
            chk[f'S{v}|{key}_reproduced (abs <= 1e-12)'] = x <= 1e-12
        chk[f'S{v}|four_registered_cells'] = pv['n_cells'] == 4
    # merged 8-cell entries from the verifier's own per-cell values
    comps = sorted(set(per_version[0]['paired']) & set(per_version[1]['paired']))
    entries = {c: {view: merge(per_version, c, view) for view in V.VIEWS} for c in comps}
    nontable = {c: {'cells': {**per_version[0]['nontable'].get(c, {}).get('cells', {}), **per_version[1]['nontable'].get(c, {}).get('cells', {})}}
                for c in comps}
    res['inputs'] = {'entries': entries, 'nontable': nontable}
    res['winners'] = {str(v): pv['winners'] for v, pv in per_version.items()}
    cmp_all = {c: A.comparison(entries[c]) for c in comps}
    res['comparisons'] = {c: summarize_comparison(x) for c, x in cmp_all.items()}
    # Q* frontiers of the PATH controls; costs from RF_FULL's EVAL_MS (reproduced exactly by RFA_VER_USEFULNESS_v1)
    ev = json.loads((RUNS / 'FAM' / CAND / 'EVAL_MS.json').read_text())
    def costs(spec):
        return {v: ev['versions'][str(v)]['paired_costs'][spec]['costs'] for v in (0, 1)}
    costs_vs = {START: costs(f'REF:{START}'), **{m: costs(f'FAM/{m}') for ms in PATHS.values() for m in ms}, CONT: costs(f'FAM/{CONT}')}
    fr = {key: A.frontier(costs_vs, [START] + ids, entries) for key, ids in PATHS.items()}
    res['frontiers'] = fr
    # reader targeting: the non-table winner's point pair benefit vs START, CONT and every gating Q*
    targets = [START, CONT] + [f['bracket'] for f in fr.values() if f.get('bracket')]
    tg = {t: A.point_positive(nontable.get(t)) for t in dict.fromkeys(targets)}
    res['targeting'] = tg
    # D1 from the verifier's usefulness unit
    uf = HERE.parent / 'out' / 'V2_USEFULNESS.json'
    d1 = None
    if uf.is_file():
        u = json.loads(uf.read_text())
        d1 = 'TASK_FEASIBLE' if u['configs'][CAND]['G1']['G1'] else 'CAND_INFEASIBLE'
    gating = {'START': cmp_all[START]['passes'], 'CONT': cmp_all[CONT]['passes'],
              **{k: bool(f.get('passes')) for k, f in fr.items()}}
    targeting_pass = all(x is not None and x['pass'] for x in tg.values())
    order = ('START', 'CONT', 'STOPFIT', 'ALTCRIT', 'RAWJ')
    if d1 != 'TASK_FEASIBLE':
        d2 = 'NOT_ASSESSED_TASK_INFEASIBLE' if d1 else 'INCOMPLETE'
    elif all(gating.values()):
        d2 = 'COMPONENT_GAIN_BEYOND_START_AND_MECHANISM_CONTROLS' if targeting_pass else 'READER_TARGETED'
    elif not any(gating.values()):
        d2 = 'NOT_SEPARATED'
    else:
        d2 = 'PARTIAL_' + '+'.join(k for k in order if gating[k]) + '_ONLY'
    refs = [r for r in A.BENCHMARK if r in cmp_all]
    d3_fail = [r for r in refs if not cmp_all[r]['passes']]
    d3 = 'INCOMPLETE' if len(refs) != 9 else ('NOT_SUPERIOR' if d3_fail else
                                              ('SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES' if d1 == 'TASK_FEASIBLE' else 'MET_WITHOUT_TASK_FEASIBILITY'))
    tsc = {r: A.target_screen(entries[r]['pair']) for r in A.TARGET_REFS}
    labs = [tsc[r]['label'] for r in A.TARGET_REFS]
    rg = all(cmp_all[r]['recipient_gate_pass'] for r in A.TARGET_REFS)
    if all(x == 'TARGET_AGG_RESOLVED' for x in labs):
        lvl = 'MATERIAL_0020_RESOLVED' if rg else 'TARGET_RESOLVED_RECIPIENT_GATE_FAILED'
    elif all(x in ('TARGET_AGG_RESOLVED', 'TARGET_AGG_POINT_ONLY') for x in labs):
        lvl = 'MATERIAL_0020_POINT_ONLY'
    elif all(cmp_all[r]['benefit_positive'] for r in A.TARGET_REFS):
        lvl = 'POSITIVE_BELOW_MATERIAL_TARGET'
    else:
        lvl = 'NOT_MET'
    d4 = lvl if (d1 == 'TASK_FEASIBLE' or lvl == 'NOT_MET') else 'MET_WITHOUT_TASK_FEASIBILITY'
    res['decisions_verifier'] = {
        'D1': d1, 'D2': d2, 'D2_gating': gating, 'D2_targeting_pass': targeting_pass, 'D3': d3, 'D3_failing': d3_fail,
        'D3_passed': [r for r in refs if r not in d3_fail], 'D4': d4,
        'D4_target_screens': {r: {'label': tsc[r]['label'], 'mean': tsc[r]['aggregate'].get('mean'), 'se_agg': tsc[r]['aggregate'].get('se_agg'),
                                  'lower': tsc[r]['aggregate'].get('lower')} for r in A.TARGET_REFS},
        'note': 'reader-budget credit rule (READER_BUDGET_ASYMMETRIC references) not applied here: it can only remove credit, and is '
                'read from DECISIONS.json in the comparison step'}
    res['grid_validity_light'] = {'seed_dependent_equal_k0_k1_counts': {str(v): pv['seed_dependent_equal_k0_k1'] for v, pv in per_version.items()},
                                  'reader_seeds': {f'S{v}k{k}': V.reader_seed(v, k) for v in (0, 1) for k in (0, 1)},
                                  'boot_seeds': {f'S{v}{r}': V.BOOT_SEED + 100 * v + V.REP_INDEX[r] for v in (0, 1) for r in ('R1', 'R2')}}
    res['all_pass'] = all(bool(x) for x in chk.values())
    res['cpu_s'] = time.process_time() - t0
    res['code_sha256'] = {f: sha256(HERE / f) for f in ('v2_grid.py', 'vagg.py', 'vlib.py', 'vcore.py')}
    if OUT.exists():
        raise PermissionError(f'{OUT.name} exists (write-once)')
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({'all_pass': res['all_pass'], 'failed': [k for k, x in chk.items() if not x], 'cpu_s': res['cpu_s'],
                      'decisions_verifier': {k: res['decisions_verifier'][k] for k in ('D1', 'D2', 'D3', 'D4')}}, indent=1))
    return 0 if res['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
