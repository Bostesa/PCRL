"""Verifier check 3 (MODEL_SELECT SEX and households, read once through ipr.common.load, operation 'verify', inside a queue unit).

Part G (exact replay; only when GRID/IPR1_MAIN/S0 and S1 exist): for the decisive members (both candidates, the twin, every gating
path point, the ablations, NEURAL_RAW, H_LJ) and every registered cell and view: the member's MODEL_SELECT wire hashes equal the
grid's records; the recorded winner's SCORE AUC is recomputed from the saved reader scores (cache entries located by the registered
key, computed here from own split / seed / hash code) with own AUC code; the winner's VALIDATION AUC equals the maximum over its
candidate set; the household bootstrap (own weights, registered seed) reproduces BOOT.npz; the non-table (NOLOOKUP) winner's SCORE AUC
is recomputed; paired cell differences and cell SEs are recomputed and compared with GRID.json 'paired'.

Part R (independent approximation): own readers (logistic regression on FIT-standardized log-probability ratios; a 15-leaf gradient-
boosting reader), fitted on the FIT parts of R1 / R2 (own household dealing), oriented and chosen on VALIDATION, scored on SCORE, both
versions: 4 cells per comparison.

    python3 -m ipr.jobqueue run --unit IPR_VER_GRID_V1 --stage verify --rss-gib 2 -- python3 <this file> [--parts G,R]
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

FAMILY_OF = {'CTL_TWIN': 'CTL', 'IA_SFG': 'IA', 'IA_SBC_HI': 'IA', 'IA_BLIND': 'IA', 'IB_CSF': 'IB', 'IB_CSFT': 'IB', 'IB_IND': 'IB'}
ROLE_MEMBERS = ('IA_SPD_HI', 'IA_RO_HI', 'IB_SPD_LO', 'IB_SPD_HI')
REFS = ('NEURAL_RAW', 'H_LJ')
DECISIVE = tuple(FAMILY_OF) + ROLE_MEMBERS + REFS
COMPARISONS = {'IA_SFG': ('CTL_TWIN', 'IA_SBC_HI', 'IA_SPD_HI', 'IA_RO_HI', 'NEURAL_RAW', 'H_LJ', 'IA_BLIND'),
               'IB_CSF': ('CTL_TWIN', 'IB_SPD_LO', 'IB_SPD_HI', 'NEURAL_RAW', 'H_LJ', 'IB_CSFT', 'IB_IND')}
SRC8 = ('OUT_LR', 'OUT_MLP', 'OUT_HGB', 'LOG_LR', 'LOG_MLP', 'DEC_LR', 'CONF_HGB', 'LOOKUP')
NEW8 = ('ET5', 'ET25', 'HGB15', 'HGB31', 'MLP64', 'MLP6432', 'CATI01', 'CATI1')


def member_arrays(name, v, roles, bank):
    if name in FAMILY_OF:
        d = V.RUNS / FAMILY_OF[name] / name / f'V{v}'
        rec = json.loads((d / 'RELEASE.json').read_text())
        if V.file_sha(d / 'RELEASE.npz') != rec['release_sha256']:
            raise RuntimeError(f'{name} V{v}: release hash differs')
        with np.load(d / 'RELEASE.npz', allow_pickle=False) as z:
            return {t: z[f'MODEL_SELECT__{t}'].copy() for t in V.TASKS}
    if name in ROLE_MEMBERS:
        fam, cfg = roles[name][str(v)].split('/')
        d = V.RUNS / fam / cfg / f'V{v}'
        rec = json.loads((d / 'RELEASE.json').read_text())
        if V.file_sha(d / 'RELEASE.npz') != rec['release_sha256']:
            raise RuntimeError(f'{name} V{v}: release hash differs')
        with np.load(d / 'RELEASE.npz', allow_pickle=False) as z:
            return {t: z[f'MODEL_SELECT__{t}'].copy() for t in V.TASKS}
    return {t: np.asarray(bank[v][name]['MODEL_SELECT'][t], dtype=np.float64) for t in V.TASKS}


def read_entry(path):
    with np.load(path, allow_pickle=False) as z:
        info = json.loads(str(z['info']))
        scores = {k[len('score__'):]: z[k].copy() for k in z.files if k.startswith('score__')}
    return scores, info


def resolve(w):
    if w.startswith('IGNORE_occupation__'):
        return 'income', w[len('IGNORE_occupation__'):]
    if w.startswith('IGNORE_income__'):
        return 'occupation', w[len('IGNORE_income__'):]
    return 'pair', w


def pair_candidates(ent, allowed=None):
    """(name, val) in the registered order: per group (source 8, new 8): pair readers, ignore-occupation, ignore-income."""
    out = []
    for group in (SRC8, NEW8):
        g = [n for n in group if allowed is None or n in allowed]
        out += [(n, ent['pair'][1][n]['val_auc']) for n in g]
        out += [(f'IGNORE_occupation__{n}', ent['income'][1][n]['val_auc']) for n in g]
        out += [(f'IGNORE_income__{n}', ent['occupation'][1][n]['val_auc']) for n in g]
    return out


def part_g(sex, unit, roles, bank, out):
    if not all((V.GRID_DIR / f'S{v}' / 'GRID.json').is_file() for v in (0, 1)):
        out['G'] = {'status': 'NOT_RUN', 'reason': 'GRID.json of IPR1_MAIN missing for a version at run time'}
        return
    sx = V.arr_sha(np.asarray(sex))
    res = {'cells': [], 'paired': [], 'wire_checks': [], 'worst': {'score_auc': 0.0, 'boot': 0.0, 'cell_diff': 0.0, 'cell_se': 0.0,
                                                                     'nolookup_auc': 0.0},
           'winner_not_max': [], 'missing_cache': [], 'boot_keys_missing': [], 'grid_meta': {}}
    for v in (0, 1):
        G = json.loads((V.GRID_DIR / f'S{v}' / 'GRID.json').read_text())
        res['grid_meta'][str(v)] = {k: G.get(k) for k in ('unit_id', 'utc', 'cpu_s', 'stats', 'exact_aliases', 'reps', 'seed')}
        boot = np.load(V.GRID_DIR / f'S{v}' / 'BOOT.npz')
        bkeys = set(boot.files)
        rel = {n: member_arrays(n, v, roles, bank) for n in DECISIVE}
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
                        p = V.CACHE_DIR / f'{V.cache_key(rep, rs, V.view_hash(view, rel[n]["income"], rel[n]["occupation"]), sx)}.npz'
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
                            src_view, recipe = resolve(w)
                            cands = pair_candidates(ent)
                        else:
                            src_view, recipe = view, w
                            cands = [(r, ent[view][1][r]['val_auc']) for r in SRC8 + NEW8]
                        s = ent[src_view][0][recipe]
                        wval = ent[src_view][1][recipe]['val_auc']
                        mx = max(c[1] for c in cands)
                        first_max = [c[0] for c in cands if c[1] == mx][0]
                        if abs(wval - mx) > 1e-15 or abs(wval - g['val_auc']) > 1e-15:
                            res['winner_not_max'].append({'cell': cell, 'member': n, 'view': view, 'winner': w, 'val': wval, 'max': mx,
                                                          'first_max': first_max, 'recorded_val': g['val_auc']})
                        a = V.auc(ysc, s)
                        bw = V.weighted_auc(ysc, s, W)
                        bk = f'{cell}__{n}__{view}'
                        if bk in bkeys:
                            db = float(np.max(np.abs(bw - boot[bk])))
                        else:
                            res['boot_keys_missing'].append(bk)
                            db = float('nan')
                        da = abs(a - g['score_auc'])
                        res['worst']['score_auc'] = max(res['worst']['score_auc'], da)
                        if np.isfinite(db):
                            res['worst']['boot'] = max(res['worst']['boot'], db)
                        mine_auc[(cell, n, view)] = a
                        mine_boot[(cell, n, view)] = bw
                        row = {'cell': cell, 'member': n, 'view': view, 'winner': w, 'score_auc': a, 'd_auc': da, 'd_boot': db,
                               'winner_is_first_max': w == first_max}
                        nl = (g.get('extras') or {}).get('nolookup')
                        if nl:
                            nsv, nrec = resolve(nl['winner']) if view == 'pair' else (view, nl['winner'])
                            if nrec in ent[nsv][0]:
                                an = V.auc(ysc, ent[nsv][0][nrec])
                                row['nolookup'] = {'winner': nl['winner'], 'score_auc': an, 'd_auc': abs(an - nl['score_auc'])}
                                res['worst']['nolookup_auc'] = max(res['worst']['nolookup_auc'], abs(an - nl['score_auc']))
                                if view == 'pair':
                                    allowed = [r for r in SRC8 + NEW8 if r != 'LOOKUP']
                                    nc = pair_candidates(ent, allowed)
                                    nmx = max(c[1] for c in nc)
                                    row['nolookup']['winner_val_is_max'] = abs(ent[nsv][1][nrec]['val_auc'] - nmx) <= 1e-15
                        res['cells'].append(row)
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
    res['n_winner_not_first_max'] = sum(1 for c in res['cells'] if not c['winner_is_first_max'])
    out['G'] = res


def logratio(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-300, None)
    return np.log(p[:, 1:]) - np.log(p[:, :1])


def part_r(sex, unit, roles, bank, out):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    res = {'readers': 'LR: LogisticRegression(C=1.0, lbfgs) on FIT-standardized log-probability ratios log(p_k / p_0); HGB: '
                      'HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.05, max_iter=200, min_samples_leaf=40, '
                      'early_stopping=False, random_state=9900 + 10 v + rep index); both fitted on FIT, oriented on VALIDATION; '
                      'winner = higher VALIDATION AUC; SCORE AUC reported', 'cells': [], 'diffs': []}
    aucs = {}
    for v in (0, 1):
        rel = {n: member_arrays(n, v, roles, bank) for n in DECISIVE}
        if v == 0:                      # IA_SBC_LO exists in version 0 only (saved, not in the bank): descriptive own readers
            d = V.RUNS / 'IA' / 'IA_SBC_LO' / 'V0'
            if json.loads((d / 'RELEASE.json').read_text())['release_sha256'] == V.file_sha(d / 'RELEASE.npz'):
                with np.load(d / 'RELEASE.npz', allow_pickle=False) as z:
                    rel['IA_SBC_LO'] = {t: z[f'MODEL_SELECT__{t}'].copy() for t in V.TASKS}
        for rep in ('R1', 'R2'):
            m = V.split_masks(unit, rep)
            fit, val, sc = m['FIT'], m['VALIDATION'], m['SCORE']
            cell = f'S{v}{rep}'
            for n in rel:
                li, lo = logratio(rel[n]['income']), logratio(rel[n]['occupation'])
                for view, X in (('income', li), ('occupation', lo), ('pair', np.column_stack([li, lo]))):
                    mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-12
                    Zs = (X - mu) / sd
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Zs[fit], sex[fit])
                    s_lr = lr.decision_function(Zs)
                    h = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
                                                       early_stopping=False,
                                                       random_state=9900 + 10 * v + V.REP_INDEX[rep]).fit(X[fit], sex[fit])
                    s_h = h.decision_function(X)
                    rec = {}
                    for nm, s in (('LR', s_lr), ('HGB', s_h)):
                        va = V.auc(sex[val], s[val])
                        if va < 0.5:
                            s = -s
                            va = 1.0 - va
                        rec[nm] = {'val': va, 'score': V.auc(sex[sc], s[sc])}
                    win = 'LR' if rec['LR']['val'] >= rec['HGB']['val'] else 'HGB'
                    rec['WIN'] = {'reader': win, 'score': rec[win]['score']}
                    aucs[(cell, n, view)] = rec
                    res['cells'].append({'cell': cell, 'member': n, 'view': view, 'LR': rec['LR'], 'HGB': rec['HGB'], 'WIN': rec['WIN']})
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
                    res['diffs'].append({'cand': cand, 'comp': comp, 'view': view, 'reader': rd, 'cells': per, 'version_means': vm,
                                         'mean': float(np.mean(list(per.values())))})
    if ('S0R1', 'IA_SBC_LO', 'pair') in aucs:
        for comp in ('IA_SBC_LO',):
            for view in V.VIEWS:
                for rd in ('LR', 'HGB', 'WIN'):
                    per = {}
                    for rep in ('R1', 'R2'):
                        cell = f'S0{rep}'
                        a, b = aucs[(cell, 'IA_SFG', view)][rd]['score'], aucs[(cell, comp, view)][rd]['score']
                        per[cell] = (b - a) if view == 'pair' else (a - b)
                    res['diffs'].append({'cand': 'IA_SFG', 'comp': comp, 'view': view, 'reader': rd, 'cells': per,
                                         'version_means': {0: float(np.mean(list(per.values())))}, 'mean': float(np.mean(list(per.values()))),
                                         'note': 'version 0 only (IA_SBC_LO exists in version 0 only; not in the registered bank)'})
    out['R'] = res


def main():
    t0 = time.process_time()
    parts = ('G', 'R')
    if '--parts' in sys.argv:
        parts = tuple(sys.argv[sys.argv.index('--parts') + 1].split(','))
    from ipr import common as C
    S = C.load('MODEL_SELECT', ('features', 'sex'), 'verify',
               'independent verifier: exact replay of the IPR1_MAIN reader-grid SCORE AUCs, bootstrap and paired cell differences of '
               'the decisive members, and independent reader re-fits (own code)')
    sex = np.asarray(S['sex'])
    unit = np.asarray(S['unit'])
    from ohn import common as OC                            # frozen NEURAL_RAW / H_LJ vectors (label-free, MODEL_SELECT role only)
    bank = {}
    for v in (0, 1):
        b, ids = OC.frozen_bank(v, roles=('MODEL_SELECT',))
        if not np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id']):
            raise AssertionError('frozen bank MODEL_SELECT row order differs')
        bank[v] = b
    roles = json.loads((V.RESULTS / 'paths' / 'SHARED_ROLES.json').read_text())['roles']
    out = {'unit_id': V.unit_id(), 'utc_start': V.utc(), 'people': int(len(sex)), 'households': int(len(np.unique(unit))),
           'sex_dtype': str(sex.dtype), 'errors': [], 'parts': list(parts)}
    for name, fn in (('G', lambda: part_g(sex, unit, roles, bank, out)), ('R', lambda: part_r(sex, unit, roles, bank, out))):
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
    p = V.write_record(f'V3_GRID__{V.unit_id() or "NOQUEUE"}.json', out)
    summ = {'record': str(p), 'errors': [e['part'] + ': ' + e['error'] for e in out['errors']], 'cpu_s': out['cpu_s']}
    if 'G' in out:
        summ['G'] = {k: out['G'].get(k) for k in ('status', 'reason', 'worst', 'n_cells_replayed', 'all_wires_match',
                                                  'n_winner_not_first_max')}
        summ['G_winner_not_max'] = len(out['G'].get('winner_not_max', []))
        summ['G_missing_cache'] = len(out['G'].get('missing_cache', []))
        summ['G_boot_keys_missing'] = len(out['G'].get('boot_keys_missing', []))
    print(json.dumps(summ, default=V._default))
    return 0 if not out['errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
