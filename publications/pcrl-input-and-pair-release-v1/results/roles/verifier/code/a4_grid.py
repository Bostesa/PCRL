"""Addendum check A4 (MODEL_SELECT SEX and households, read once through ipr.common.load, operation 'verify', inside a queue unit).

Exact replay, as for IPR1_MAIN in the first report, of the banks IPR1_R1 (registered cells of the IA revision reading), IPR1_ROB_R1
(its robustness bank) and IPR1_ROB (the robustness bank of the registered readings): for every decisive member and every recorded
cell and view, the member's MODEL_SELECT wire hashes equal the grid's; the cache entry is located by the registered key (own split,
seed and hash code; repetitions R3 / R4 carry their split tag); the recorded winner is the first maximum of VALIDATION AUC in its
registered candidate set; its SCORE AUC is recomputed from the cached scores with own AUC code; the household bootstrap (own weights,
seed 2026102300 + 100 v + repetition index) reproduces BOOT.npz; the non-table (NOLOOKUP) winner's SCORE AUC is recomputed; paired cell
differences and bootstrap cell SEs are recomputed and compared with GRID.json 'paired'. The replayed per-cell entries are written for
checks A5 and A6.

    python3 -m ipr.jobqueue run --unit IPR_VER_A4_GRID_V1 --stage verify --rss-gib 3 -- python3 <this file>
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
import a_lib as A  # noqa: E402

R1_MEMBERS = ('CTL_TWIN', 'IA_OSG', 'IA_OSG_SBC_LO', 'IA_OSG_SBC_HI', 'IA_OSG_SPD_LO', 'IA_OSG_SPD_HI', 'IA_OSG_RO_LO', 'IA_OSG_RO_HI',
              'IA_OSG_BLIND', 'NEURAL_RAW', 'H_LJ')
R1_COMPS = {'IA_OSG': ('CTL_TWIN', 'IA_OSG_SBC_LO', 'IA_OSG_SBC_HI', 'IA_OSG_SPD_LO', 'IA_OSG_SPD_HI', 'IA_OSG_RO_LO', 'IA_OSG_RO_HI',
                       'IA_OSG_BLIND', 'NEURAL_RAW', 'H_LJ')}
BANKS = {
    'IPR1_R1': {'members': R1_MEMBERS, 'comparisons': R1_COMPS},
    'IPR1_ROB_R1': {'members': R1_MEMBERS, 'comparisons': R1_COMPS},
    'IPR1_ROB': {'members': ('CTL_TWIN', 'IA_SFG', 'IA_SBC_HI', 'IA_SPD_HI', 'IA_RO_HI', 'IA_BLIND', 'IB_CSF', 'IB_SPD_LO', 'IB_SPD_HI',
                             'IB_CSFT', 'IB_IND', 'NEURAL_RAW', 'H_LJ'),
                 'comparisons': {'IA_SFG': ('CTL_TWIN', 'IA_SBC_HI', 'IA_SPD_HI', 'IA_RO_HI', 'NEURAL_RAW', 'H_LJ', 'IA_BLIND'),
                                 'IB_CSF': ('CTL_TWIN', 'IB_SPD_LO', 'IB_SPD_HI', 'NEURAL_RAW', 'H_LJ', 'IB_CSFT', 'IB_IND')}},
}


def replay(bank_id, spec, sex, unit, fam_of, roles, frozen):
    gdir = V.RUNS / 'GRID' / bank_id
    sx = V.arr_sha(np.asarray(sex))
    res = {'cells': [], 'paired': [], 'wire_checks': [], 'missing_cache': [], 'boot_keys_missing': [], 'winner_not_max': [],
           'worst': {'score_auc': 0.0, 'boot': 0.0, 'cell_diff': 0.0, 'cell_se': 0.0, 'nolookup_auc': 0.0}, 'meta': {}}
    for v in (0, 1):
        G = json.loads((gdir / f'S{v}' / 'GRID.json').read_text())
        res['meta'][str(v)] = {k: G.get(k) for k in ('unit_id', 'utc', 'cpu_s', 'stats', 'reps', 'seed')}
        boot = np.load(gdir / f'S{v}' / 'BOOT.npz')
        bkeys = set(boot.files)
        rel = {n: A.member_ms(n, v, fam_of, roles, frozen) for n in spec['members']}
        for n in spec['members']:
            for t in V.TASKS:
                res['wire_checks'].append({'v': v, 'member': n, 'task': t,
                                           'match': V.wire_sha(rel[n][t]) == G['members'][n]['wire_sha256'][t]})
        mine_auc, mine_boot = {}, {}
        cells = sorted(G['cells'])
        for cell in cells:
            vv, rep, k = A.cell_parts(cell)
            sc = V.split_masks(unit, rep)['SCORE']
            ysc = sex[sc]
            W = V.household_bootstrap_weights(unit[sc], V.BOOT_BASE + 100 * v + V.REP_INDEX[rep])
            rs = V.reader_seed(v, k)
            for n in spec['members']:
                ent = {}
                for view in V.VIEWS:
                    p = V.CACHE_DIR / f'{V.cache_key(rep, rs, V.view_hash(view, rel[n]["income"], rel[n]["occupation"]), sx)}.npz'
                    if not p.is_file():
                        res['missing_cache'].append(f'{cell}|{n}|{view}')
                        ent = None
                        break
                    ent[view] = A.read_entry(p)
                if ent is None:
                    continue
                for view in V.VIEWS:
                    g = G['cells'][cell][n][view]
                    w = g['winner']
                    if view == 'pair':
                        src, recipe = A.resolve(w)
                        cands = A.pair_candidates(ent)
                    else:
                        src, recipe = view, w
                        cands = [(r, ent[view][1][r]['val_auc']) for r in A.SRC8 + A.NEW8]
                    s = ent[src][0][recipe]
                    wval = ent[src][1][recipe]['val_auc']
                    mx = max(c[1] for c in cands)
                    first_max = [c[0] for c in cands if c[1] == mx][0]
                    if abs(wval - mx) > 1e-15 or abs(wval - g['val_auc']) > 1e-15 or w != first_max:
                        res['winner_not_max'].append({'cell': cell, 'member': n, 'view': view, 'winner': w, 'first_max': first_max})
                    a = V.auc(ysc, s)
                    bw = V.weighted_auc(ysc, s, W)
                    bk = f'{cell}__{n}__{view}'
                    if bk in bkeys:
                        db = float(np.max(np.abs(bw - boot[bk])))
                        res['worst']['boot'] = max(res['worst']['boot'], db)
                    else:
                        res['boot_keys_missing'].append(bk)
                        db = None
                    da = abs(a - g['score_auc'])
                    res['worst']['score_auc'] = max(res['worst']['score_auc'], da)
                    mine_auc[(cell, n, view)] = a
                    mine_boot[(cell, n, view)] = bw
                    row = {'cell': cell, 'member': n, 'view': view, 'winner': w, 'score_auc': a, 'd_auc': da, 'd_boot': db}
                    nl = (g.get('extras') or {}).get('nolookup')
                    if nl:
                        nsv, nrec = A.resolve(nl['winner']) if view == 'pair' else (view, nl['winner'])
                        if nrec in ent[nsv][0]:
                            an = V.auc(ysc, ent[nsv][0][nrec])
                            row['nolookup'] = {'winner': nl['winner'], 'score_auc': an, 'd_auc': abs(an - nl['score_auc'])}
                            res['worst']['nolookup_auc'] = max(res['worst']['nolookup_auc'], abs(an - nl['score_auc']))
                    res['cells'].append(row)
        for cand, comps in spec['comparisons'].items():
            for comp in comps:
                pe = (G['paired'].get(cand) or {}).get(comp)
                if pe is None:
                    res['paired'].append({'v': v, 'cand': cand, 'comp': comp, 'error': 'no paired entry in GRID.json'})
                    continue
                for view in V.VIEWS:
                    for cell in pe[view]['cells']:
                        if (cell, cand, view) not in mine_auc or (cell, comp, view) not in mine_auc:
                            res['paired'].append({'v': v, 'cand': cand, 'comp': comp, 'view': view, 'cell': cell, 'error': 'not replayed'})
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
    res['n_cells_replayed'] = len(res['cells'])
    res['all_wires_match'] = all(x['match'] for x in res['wire_checks'])
    res['n_paired_errors'] = sum(1 for x in res['paired'] if 'error' in x)
    return res


def main():
    t0 = time.process_time()
    from ipr import common as C
    S = C.load('MODEL_SELECT', ('features', 'sex'), 'verify',
               'independent verifier addendum: exact replay of the IPR1_R1, IPR1_ROB_R1 and IPR1_ROB reader-grid SCORE AUCs, bootstrap '
               'and paired cell differences of the decisive members (own code)')
    sex = np.asarray(S['sex'])
    unit = np.asarray(S['unit'])
    from ohn import common as OC
    frozen = {}
    for v in (0, 1):
        b, ids = OC.frozen_bank(v, roles=('MODEL_SELECT',))
        if not np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id']):
            raise AssertionError('frozen bank MODEL_SELECT row order differs')
        frozen[v] = b
    fam_of, _ = A.ledger_family()
    roles = A.roles_map()
    out = {'unit_id': V.unit_id(), 'utc_start': V.utc(), 'people': int(len(sex)), 'households': int(len(np.unique(unit))),
           'banks': {}, 'errors': []}
    for bank_id, spec in BANKS.items():
        t1 = time.process_time()
        try:
            out['banks'][bank_id] = replay(bank_id, spec, sex, unit, fam_of, roles, frozen)
        except Exception as e:
            out['errors'].append({'bank': bank_id, 'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()[-3000:]})
        out[f'cpu_s_{bank_id}'] = round(time.process_time() - t1, 2)
    out['cpu_s'] = round(time.process_time() - t0, 2)
    out['utc_end'] = V.utc()
    p = V.write_record(f'A4_GRID__{V.unit_id() or "NOQUEUE"}.json', out)
    summ = {'record': str(p), 'errors': [e['bank'] + ': ' + e['error'] for e in out['errors']], 'cpu_s': out['cpu_s'],
            'banks': {b: {'cells': r['n_cells_replayed'], 'wires': r['all_wires_match'], 'worst': r['worst'],
                          'missing_cache': len(r['missing_cache']), 'boot_missing': len(r['boot_keys_missing']),
                          'winner_not_first_max': len(r['winner_not_max']), 'paired': len(r['paired']),
                          'paired_errors': r['n_paired_errors']} for b, r in out['banks'].items()}}
    print(json.dumps(summ, default=V._default))
    return 0 if not out['errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
