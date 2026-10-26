"""RFA1 verifier recomputation cores (file-reading, data passed in). No label access here: the caller supplies SEX / labels.

grid_core(grid_dir, cache_dir, version, sex, unit, members, pairs)  recompute a saved grid cell by cell from GRID.json, BOOT.npz and the
                                                                    cache entries (winners, SCORE AUCs, bootstrap draws, per-recipe
                                                                    SCORE AUCs, paired differences / SEs) with vlib only
refit_core(grid_dir, cache_dir, version, sex, unit, members, picks)  bounded sklearn refit of chosen (cell, member, view) winners
usefulness_core(labels, sex, unit, releases, refs, eval_records)    the 28 conditions per configuration, compared with EVAL_MS records
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

import vlib as V

CELL_RE = re.compile(r'^S(?P<v>\d)(?P<rep>R\d)k(?P<k>\d)(?:@(?P<frac>[0-9.]+))?$')


def parse_cell(cell):
    m = CELL_RE.match(cell)
    if not m:
        raise ValueError(f'unparsed cell name {cell}')
    return int(m['v']), m['rep'], int(m['k']), float(m['frac']) if m['frac'] else 1.0


def _cell_ctx(cell, sex, unit):
    v, rep, k, frac = parse_cell(cell)
    roles = V.reader_roles(unit, rep, frac)
    sc = roles['SCORE']
    return {'version': v, 'rep': rep, 'k': k, 'frac': frac, 'roles': roles, 'score': sc, 'y': np.asarray(sex)[sc],
            'W': V.boot_weights(np.asarray(unit)[sc], v, rep), 'rs': V.reader_seed(v, k)}


def _entries(cache_dir, ctx, rel, sx):
    out = {}
    for view in V.VIEWS:
        p = Path(cache_dir) / f"{V.cache_key(ctx['rep'], ctx['rs'], ctx['frac'], V.view_hash(view, rel['income'], rel['occupation']), sx)}.npz"
        if not p.is_file():
            raise FileNotFoundError(f'cache entry missing for {view} ({p.name})')
        out[view] = V.read_cache_entry(p)
    return out


def _score(entries, key):
    view, name = key.split('__', 1)
    return entries[view][name][0]


def grid_core(grid_dir, cache_dir, version, sex, unit, members, pairs=None):
    """members: {name: {'income': (n, 2), 'occupation': (n, 5)}} on the grid's MODEL_SELECT rows (deterministic releases).
    pairs: [(cand, comp)] (default: every pair recorded in GRID.json 'paired'). Returns a record with per-check worst differences."""
    t0 = time.process_time()
    grid_dir = Path(grid_dir)
    grid = json.loads((grid_dir / 'GRID.json').read_text())
    with np.load(grid_dir / 'BOOT.npz') as z:
        boot = {k: z[k].copy() for k in z.files}
    sx = V.arr_sha(np.asarray(sex))
    worst = {'score_auc': 0.0, 'boot': 0.0, 'per_recipe_score_auc': 0.0, 'val_auc': 0.0, 'paired_d': 0.0, 'paired_se': 0.0}
    mism, cells_out, mine_boot, mine_auc = [], {}, {}, {}
    for cell, cres in grid['cells'].items():
        ctx = _cell_ctx(cell, sex, unit)
        if ctx['version'] != int(version):
            raise ValueError(f'cell {cell} is not version {version}')
        cells_out[cell] = {}
        for name, ent in cres.items():
            if name not in members:
                mism.append(f'{cell}/{name}: member arrays not supplied')
                continue
            E = _entries(cache_dir, ctx, members[name], sx)
            w = V.winners({vw: {n: E[vw][n][1] for n in V.RECIPES} for vw in V.VIEWS})
            cells_out[cell][name] = {}
            for view in V.VIEWS:
                g = ent[view]
                s = _score(E, w[view]['reader_key'])
                a = V.auc(ctx['y'], s)
                bd = V.weighted_auc(ctx['y'], s, ctx['W'])
                mine_boot[(cell, name, view)], mine_auc[(cell, name, view)] = bd, a
                if w[view]['winner'] != g['winner'] or w[view]['near_tie'] != g['near_tie']:
                    mism.append(f'{cell}/{name}/{view}: winner {g["winner"]} vs verifier {w[view]["winner"]}')
                worst['val_auc'] = max(worst['val_auc'], abs(w[view]['val_auc'] - g['val_auc']))
                worst['score_auc'] = max(worst['score_auc'], abs(a - g['score_auc']))
                key = f'{cell}__{name}__{view}'
                worst['boot'] = max(worst['boot'], float(np.max(np.abs(bd - boot[key]))) if key in boot else float('inf'))
                if 'per_recipe_score_auc' in g:
                    worst['per_recipe_score_auc'] = max(worst['per_recipe_score_auc'],
                                                        max(abs(V.auc(ctx['y'], E[view][n][0]) - g['per_recipe_score_auc'][n]) for n in V.RECIPES))
                cells_out[cell][name][view] = {'winner': w[view]['winner'], 'score_auc': a, 'grid_score_auc': g['score_auc'],
                                               'near_tie': w[view]['near_tie'], 'gap': w[view]['gap_to_runner_up']}
    pairs = pairs or [(c, x) for c, comps in grid.get('paired', {}).items() for x in comps]
    paired = {}
    for cand, comp in pairs:
        for view in V.VIEWS:
            rec = {'cells': {}, 'cell_se': {}}
            for cell in grid['cells']:
                if (cell, cand, view) not in mine_auc or (cell, comp, view) not in mine_auc:
                    continue
                d, se = V.paired_cell(view, mine_auc[(cell, cand, view)], mine_auc[(cell, comp, view)], mine_boot[(cell, cand, view)],
                                      mine_boot[(cell, comp, view)])
                rec['cells'][cell], rec['cell_se'][cell] = d, se
                gp = grid.get('paired', {}).get(cand, {}).get(comp, {}).get(view)
                if gp is not None:
                    worst['paired_d'] = max(worst['paired_d'], abs(d - gp['cells'][cell]))
                    worst['paired_se'] = max(worst['paired_se'], abs(se - gp['cell_se'][cell]))
            vals = list(rec['cells'].values())
            rec['mean_over_cells'] = float(np.mean(vals)) if vals else None
            paired.setdefault(cand, {}).setdefault(comp, {})[view] = rec
    tol = 1e-12
    ok = (not mism) and all(worst[k] <= tol for k in worst)
    return {'grid_dir': str(grid_dir), 'version': int(version), 'n_cells': len(grid['cells']), 'worst_abs_diff': worst,
            'mismatches': mism, 'agree': bool(ok), 'tolerance': tol, 'cells': cells_out, 'paired': paired,
            'cpu_s': time.process_time() - t0}


def refit_core(grid_dir, cache_dir, version, sex, unit, members, picks, max_fits=8):
    """picks: [(cell, member, view)]; refit the grid winner's recipe (the ignore-one pair winners refit their single view) with vlib
    and compare the SCORE AUC with the grid and the oriented SCORE scores with the cache entry."""
    t0 = time.process_time()
    grid = json.loads((Path(grid_dir) / 'GRID.json').read_text())
    sx = V.arr_sha(np.asarray(sex))
    out, n = {}, 0
    for cell, name, view in picks[:max_fits]:
        ctx = _cell_ctx(cell, sex, unit)
        E = _entries(cache_dir, ctx, members[name], sx)
        w = V.winners({vw: {r: E[vw][r][1] for r in V.RECIPES} for vw in V.VIEWS})[view]
        vw, recipe = w['reader_key'].split('__', 1)
        rel = members[name]
        s, va, _ = V.refit(recipe, vw, rel['income'], rel['occupation'], sex, ctx['roles']['FIT'], ctx['roles']['VALIDATION'], ctx['rs'])
        stored, info = E[vw][recipe]
        a = V.auc(ctx['y'], s[ctx['score']])
        g = grid['cells'][cell][name][view]
        out[f'{cell}/{name}/{view}'] = {'winner': g['winner'], 'recipe_refit': recipe, 'refit_view': vw, 'grid_score_auc': g['score_auc'],
                                        'refit_score_auc': a, 'score_auc_diff': a - g['score_auc'],
                                        'refit_val_auc': va, 'cached_val_auc': info['val_auc'],
                                        'max_abs_score_diff': float(np.max(np.abs(s[ctx['score']] - stored))),
                                        'bitwise_scores': bool(np.array_equal(s[ctx['score']], stored))}
        n += 1
    return {'n_refits': n, 'refits': out, 'max_abs_score_auc_diff': max((abs(r['score_auc_diff']) for r in out.values()), default=None),
            'all_bitwise': all(r['bitwise_scores'] for r in out.values()), 'cpu_s': time.process_time() - t0}


def usefulness_core(labels, sex, unit, releases, refs, eval_records=None):
    """releases: {config: {version: {'income', 'occupation'}}}; refs: {version: {'U0': {...}, 'Ucal': {...}}};
    eval_records: {config: EVAL_MS json} (optional). Returns per configuration the 28 conditions, G1 and the comparison."""
    out = {}
    for cfg, by_v in releases.items():
        per_v = {v: V.usefulness_version(labels, rel, refs[v], sex, unit) for v, rel in by_v.items()}
        rec = {'per_version': {str(v): pv for v, pv in per_v.items()}, 'G1': V.g1(per_v)}
        if eval_records and cfg in eval_records:
            ev = eval_records[cfg]
            rec['vs_campaign_eval'] = {str(v): V.compare_with_campaign_utility(per_v[v], ev['versions'][str(v)]['utility'])
                                       for v in per_v if not ev['versions'].get(str(v), {}).get('missing')}
            rec['campaign_point_pass'] = {str(v): ev['versions'][str(v)].get('point', {}).get('point_pass_12_plus_2') for v in per_v}
        out[cfg] = rec
    return out
