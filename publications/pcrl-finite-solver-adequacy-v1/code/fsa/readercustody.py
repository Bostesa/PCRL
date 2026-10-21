"""Representative reader custody of the PCRL finite-output solver-adequacy campaign (coordinator; derived from nrm.readercustody). Refits a few readers of a registered grid
exactly as fsa.devreaders fits them (same repetition splits, reader seed, recipe, wire and FIT rows; the randomized member's instance
from the declared instance map), requires their oriented SCORE scores to equal the grid's cache entry bitwise, and saves each fitted
reader with its SCORE design rows and raw scores to runs/READER_CUSTODY/<bank>__<cell>__<member>__<view>__<recipe>.pkl.
frc.restore_check re-loads every pickle from a backup alone and requires predict(model, X_score) to reproduce score_raw bitwise.

Reads MODEL_SELECT features and SEX through the access layer (operation 'readers'; this entry point is authorized in fsa.common
ENTRY_POINTS from the registration on): the same exposure as the registered grid. It writes no grid output and changes no decision.

    python3 -m fsa.jobqueue run --unit FSA_READER_CUSTODY_V1 --stage readers --rss-gib 3 -- <torch venv python> -m fsa.readercustody run
        --bank FSA1_MAIN --seed 0 --rep R1 --k 0 --item PRC_S2:pair:WINNER --item PRC_S2:pair:LOOKUP ...
(recipe WINNER = the grid's registered winner for that member, view and cell)
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
import warnings

import numpy as np

from . import common as C
from . import devreaders as DR


def _fit_keep(name, view, inc, occ, sex, fit, rs):
    """fsa.devreaders._fit_one's fit, keeping the model, its prediction function and the design (rows or keys)."""
    if name == 'LOOKUP':
        keys = DR.lookup_keys(inc) if view == 'income' else DR.lookup_keys(occ) if view == 'occupation' else DR.lookup_keys(inc, occ)
        m = DR.fit_lookup([keys[i] for i in np.flatnonzero(fit)], sex[fit])
        return m, DR.predict_lookup, keys, DR.predict_lookup(m, keys)
    if name in dict(DR.SLATE):
        X = DR._src_tab(view, inc, occ, dict(DR.SLATE)[name])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = DR.fit_reader(X[fit], sex[fit], name, rs)
        return m, DR.predict_reader, X, DR.predict_reader(m, X)
    tf, _, _ = DR.design_for(name, view, inc[fit], occ[fit])
    X = tf(inc, occ)
    m, _ = DR.fit_new(name, X[fit], sex[fit], rs)
    return m, DR.predict_new, X, DR.predict_new(m, X)


def run(bank, seed, rep, k, items):
    spec = json.loads((C.RESULTS / 'grid_specs' / f'{bank}.json').read_text())
    grid = json.loads((C.RUNS / 'GRID' / bank / f'S{seed}' / 'GRID.json').read_text())
    cell = DR.cell_name(seed, rep, k)
    S = C.load('MODEL_SELECT', ('features', 'sex'), operation='readers',
               purpose=f'representative reader custody refits, bank {bank} cell {cell} (same exposure as the registered grid)')
    sex, unit, row_id = S['sex'], S['unit'], S['row_id']
    del S                                                  # readers never see Census inputs
    r = DR.roles_of(unit, rep)
    fit, val, sc = DR.lc_fit(unit, r['FIT'], 1.0), r['VALIDATION'], r['SCORE']
    rs = DR.reader_seed(seed, k)
    sx = C.arr_sha(np.asarray(sex))
    out = C.RUNS / 'READER_CUSTODY'
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    recs = []
    for item in items:
        member, view, recipe = item.split(':')
        winner = grid['cells'][cell][member][view]['winner']
        recipe = winner if recipe == 'WINNER' else recipe
        rel, _ = DR.load_member(spec['members'][member], seed, n=len(sex), row_id=row_id)
        inc, occ = rel['income'], rel['occupation']
        if inc.ndim == 3 or occ.ndim == 3:                 # randomized: this cell's one instance (declared map)
            d = DR.draw_for(rel.get('_draw_rule', {}), rep)
            inc = inc[d] if inc.ndim == 3 else inc
            occ = occ[d] if occ.ndim == 3 else occ
        m, pred, X, raw = _fit_keep(recipe, view, inc, occ, sex, fit, rs)
        sign = DR.orientation(sex[val], raw[val])
        s = np.ascontiguousarray(DR.orient(raw, sign)[sc], dtype=np.float64)
        key = DR.cache_key(rep, rs, 1.0, DR.wire_hash(view, inc, occ), sx)
        cached = DR._read_entry(DR.CACHE / f'{key}.npz', int(sc.sum()))[recipe][0]
        if not np.array_equal(s, cached):
            raise RuntimeError(f'{item}: refit scores differ from the grid cache entry {key}')
        idx = np.flatnonzero(sc)
        xs = [X[i] for i in idx] if isinstance(X, list) else np.ascontiguousarray(X[sc])
        name = f'{bank}__{cell}__{member}__{view}__{recipe}.pkl'
        obj = {'schema': 'FSA1-reader-custody-v1', 'bank': bank, 'cell': cell, 'member': member, 'view': view, 'recipe': recipe,
               'grid_winner': winner, 'model': m, 'predict': pred, 'X_score': xs, 'score_raw': np.ascontiguousarray(raw[sc]),
               'orientation': int(sign), 'grid_cache_key': key, 'oriented_scores_equal_grid_cache': True,
               'python': sys.version.split()[0]}
        p = out / name
        if p.exists():
            raise PermissionError(f'{name} exists (custody pickles are write-once)')
        with open(p, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(p, 'rb') as f:
            back = pickle.load(f)
        reload_ok = bool(np.array_equal(back['predict'](back['model'], back['X_score']), back['score_raw']))
        recs.append({'item': item, 'recipe': recipe, 'grid_winner': winner, 'file': name, 'sha256': C.sha(p),
                     'oriented_scores_equal_grid_cache': True, 'reload_rescore_bitwise': reload_ok, 'n_score': int(sc.sum())})
    rec = {'schema': 'FSA1-reader-custody-v1', 'utc': C.now(), 'bank': bank, 'cell': cell, 'reader_seed': rs, 'readers': recs,
           'python': sys.version.split()[0], 'all_ok': all(x['reload_rescore_bitwise'] for x in recs),
           'note': 'fitted readers and their SCORE rows stay private (runs/READER_CUSTODY); this record holds hashes only'}
    C.write_json_x(C.RESULTS / 'custody' / f'READER_CUSTODY_{bank}_{cell}.json', rec)
    print(json.dumps({'all_ok': rec['all_ok'], 'readers': [(x['item'], x['recipe']) for x in recs]}, indent=1))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--bank', required=True); r.add_argument('--seed', type=int, required=True)
    r.add_argument('--rep', required=True, choices=DR.REPS); r.add_argument('--k', type=int, required=True)
    r.add_argument('--item', action='append', required=True)
    a = ap.parse_args(argv)
    run(a.bank, a.seed, a.rep, a.k, a.item)
    return 0


if __name__ == '__main__':
    sys.exit(main())
