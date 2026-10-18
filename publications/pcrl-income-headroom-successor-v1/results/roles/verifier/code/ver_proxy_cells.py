"""Independent verifier, check 5: recompute cells of the development reader grid (PROTOCOL.md section 6; NOMINATION_RULE.md (c))
with separate code. Public-use ACS 2019 Florida person records (MODEL_SELECT role, 8,001 people / 5,306 households); readers
estimate the recorded SEX column from released probability vectors (income view, occupation view, linked pair).

Own code: household halves (A: the source campaign's permutation of household ids, seed = first 8 hex digits of
SHA-256('SCM1proxysplit'), first half; B: SHA-256 order of '<tag>|<household id>', alternate assignment, fold 0), reader
seeds (MODEL_SEED_BASE + 100 + k for LOG_MLP and ET25, + 100 for the seed-invariant readers), both directions, cell value =
mean AUC of the two directions, own rank-based AUC (Mann-Whitney with mid-ranks), own exact-vector LOOKUP reader (pseudo-count
2.0 toward the training-half SEX rate; unseen vector -> the rate), own view maxima (pair maximum includes the single views).
Allowed constructors only for the fitted readers: task_conditioned_representation.attacks.fit_reader / predict_reader on
scm_bench.readers.kinds features (OUT_LR, OUT_HGB on 'out', LOG_MLP on 'log'); ffv.readers.fit_new / predict_new on
ffv.features.rich_view (ET25, HGB31) and ffv.features.CatiDesign fitted label-free on the training half (CATI1).
ihs.evaluate is NOT imported.

usage: python3 ver_proxy_cells.py <release seed> <cells comma list, e.g. A0,A1,B0,B1>
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import rankdata

from ihs import common as C
from ihs.devrun import load_release
from scm_bench.readers import kinds
from task_conditioned_representation.attacks import fit_reader, predict_reader, MODEL_SEED_BASE
from task_conditioned_representation.inference import auc as auc_lib
from ffv import readers as FR
from ffv import features as FF

READERS = ('OUT_LR', 'OUT_HGB', 'LOG_MLP', 'ET25', 'HGB31', 'CATI1', 'LOOKUP')
SEED_DEP = ('LOG_MLP', 'ET25')
VIEWS = ('income', 'occupation', 'pair')
CONFIGS = ('H3_STRONG_LJDIR_PB', 'HC_STRONG', 'HC_TTREE_LJDIR_PB')
REFS = ('NEURAL_RAW', 'H_LJ')
OUT = ROOT / '.private' / 'pcrl_income_headroom_successor_v1' / 'verifier'


def my_auc(y, s):
    y = np.asarray(y).astype(np.int64); s = np.asarray(s, dtype=np.float64)
    r = rankdata(s, method='average'); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def seed_for(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def my_halves(unit, rep):
    groups = np.unique(unit)
    if rep == 'A':
        perm = np.random.default_rng(seed_for('SCM1proxysplit')).permutation(len(groups))
        a = np.isin(unit, groups[perm[:len(groups) // 2]])
    elif rep == 'B':
        keys = np.array([hashlib.sha256(f'IHS1_dev_proxy_rep_B|{g}'.encode()).hexdigest() for g in groups])
        rank = np.empty(len(groups), dtype=np.int64); rank[np.argsort(keys)] = np.arange(len(groups))
        a = (rank % 2 == 0)[np.searchsorted(groups, unit)]
    else:
        raise ValueError(rep)
    return a, ~a


def view_arrays(view, inc, occ):
    return {'income': (inc,), 'occupation': (occ,), 'pair': (inc, occ)}[view]


def my_view_hash(view, inc, occ):
    h = hashlib.sha256(view.encode())
    for a in view_arrays(view, inc, occ):
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


def lookup_scores(view, inc, occ, sex, tr, te):
    z = np.ascontiguousarray(np.column_stack([np.asarray(a, dtype=np.float64) for a in view_arrays(view, inc, occ)]))
    keys = [row.tobytes() for row in z]
    pi = float(sex[tr].mean()); table = {}
    for i in np.flatnonzero(tr):
        n1, n = table.get(keys[i], (0, 0)); table[keys[i]] = (n1 + int(sex[i]), n + 1)
    return np.array([(table[keys[i]][0] + 2.0 * pi) / (table[keys[i]][1] + 2.0) if keys[i] in table else pi
                     for i in np.flatnonzero(te)], dtype=np.float64)


def src_feats(view, inc, occ, kind):
    a, b = kinds(inc, 2)[kind], kinds(occ, 5)[kind]
    return a if view == 'income' else b if view == 'occupation' else np.column_stack((a, b))


def scores(name, view, inc, occ, sex, tr, te, k):
    rs = MODEL_SEED_BASE + 100 + (k if name in SEED_DEP else 0)
    if name == 'LOOKUP':
        return lookup_scores(view, inc, occ, sex, tr, te)
    if name in ('OUT_LR', 'OUT_HGB', 'LOG_MLP'):
        X = src_feats(view, inc, occ, 'log' if name == 'LOG_MLP' else 'out')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = fit_reader(X[tr], sex[tr], name, rs)
        return predict_reader(m, X[te])
    if name == 'CATI1':
        X = FF.CatiDesign(view).fit(inc[tr], occ[tr]).transform(inc, occ)
    else:
        X = FF.rich_view(view, inc, occ)
    m, _ = FR.fit_new(name, X[tr], sex[tr], rs)
    return FR.predict_new(m, X[te])


def main():
    seed = int(sys.argv[1]); cells = sys.argv[2].split(',')
    S = C.load_role('MODEL_SELECT', ('features', 'sex'),
                    purpose=f'independent verification: development proxy grid cells (IHS verifier), release seed {seed}')
    sex, unit = S['sex'], S['unit']
    bank, ids = C.frozen_bank(seed, roles=('MODEL_SELECT',))
    assert np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id'])
    rel = {}
    for cfg in CONFIGS:
        o, rec = load_release('H', cfg, seed, roles=('MODEL_SELECT',))
        rel[cfg] = {t: o['MODEL_SELECT'][t] for t in C.TASKS}
    for r in REFS:
        rel[r] = {t: bank[r]['MODEL_SELECT'][t] for t in C.TASKS}
    stored = {}
    for cfg in CONFIGS:
        stored[cfg] = json.loads((C.RUNS / 'H' / cfg / f'S{seed}' / 'EVAL_MS.json').read_text())
    refev = json.loads((C.RUNS / 'REF' / f'S{seed}' / 'EVAL_MS_REFERENCES.json').read_text())
    for r in REFS:
        stored[r] = refev[r]
    sex_sha, unit_sha = C.arr_sha(np.asarray(sex)), C.arr_sha(np.asarray(unit))
    halves = {rep: my_halves(unit, rep) for rep in ('A', 'B')}
    memo = {}
    out = {'seed': seed, 'cells': cells, 'releases': {}, 'halves': {rep: {'n_first': int(halves[rep][0].sum()),
           'households_first': int(len(np.unique(unit[halves[rep][0]]))), 'households_second': int(len(np.unique(unit[halves[rep][1]])))}
           for rep in halves}, 'max_abs_diff': 0.0, 'n_compared': 0, 'n_exact': 0, 'dir_compared': 0, 'dir_max_abs_diff': 0.0}
    t_start = time.process_time()
    for name_rel, outs in rel.items():
        inc, occ = outs['income'], outs['occupation']
        wire_ok = {t: C.wire_sha(outs[t]) == stored[name_rel]['wire_sha256'][t] for t in C.TASKS}
        rrec = {'wire_matches_stored_eval': wire_ok, 'cells': {}}
        for cell in cells:
            rep, k = cell[0], int(cell[1])
            tr0, te0 = halves[rep]
            crec = {}
            for view in VIEWS:
                vh = my_view_hash(view, inc, occ); vrec = {}
                for name in READERS:
                    eff = k if name in SEED_DEP else 0
                    key = (vh, name, rep, eff)
                    if key not in memo:
                        dirs_own, dirs_lib = [], []
                        for tr, te in ((tr0, te0), (te0, tr0)):
                            s = scores(name, view, inc, occ, sex, tr, te, k)
                            dirs_own.append(my_auc(sex[te], s)); dirs_lib.append(auc_lib(sex[te], s))
                        memo[key] = (dirs_own, dirs_lib)
                    dirs_own, dirs_lib = memo[key]
                    val = float(np.mean(dirs_own)); st = stored[name_rel]['proxy']['cells'][cell][view][name]
                    # per-direction comparison with the evaluator's cache record, when it exists
                    ck = hashlib.sha256(f'IHS1-dev-proxy-v1|{vh}|{name}|{rep}|{eff}|{sex_sha}|{unit_sha}'.encode()).hexdigest()
                    cp = C.RUNS / 'proxy_cache' / f'{ck}.json'
                    dir_diff = None
                    if cp.is_file():
                        cached = json.loads(cp.read_text())
                        dir_diff = max(abs(a - b) for a, b in zip(cached['auc_dirs'], dirs_own))
                        out['dir_compared'] += 1; out['dir_max_abs_diff'] = max(out['dir_max_abs_diff'], dir_diff)
                    d = abs(val - st)
                    vrec[name] = {'stored': st, 'recomputed': val, 'abs_diff': d, 'dirs_own_auc': dirs_own,
                                  'own_vs_lib_auc_max_diff': max(abs(a - b) for a, b in zip(dirs_own, dirs_lib)),
                                  'cache_dir_abs_diff': dir_diff}
                    out['max_abs_diff'] = max(out['max_abs_diff'], d); out['n_compared'] += 1; out['n_exact'] += int(d == 0.0)
                vmax = max(vrec[n]['recomputed'] for n in READERS)
                vrec['view_max'] = {'stored': stored[name_rel]['proxy']['cells'][cell][view]['view_max'], 'recomputed': vmax}
                crec[view] = vrec
            mis = max(crec[v]['view_max']['recomputed'] for v in VIEWS)
            crec['pair']['max_incl_single'] = {'stored': stored[name_rel]['proxy']['cells'][cell]['pair']['max_incl_single'], 'recomputed': mis}
            for v in VIEWS:
                d = abs(crec[v]['view_max']['stored'] - crec[v]['view_max']['recomputed']); out['max_abs_diff'] = max(out['max_abs_diff'], d)
            out['max_abs_diff'] = max(out['max_abs_diff'], abs(mis - crec['pair']['max_incl_single']['stored']))
            rrec['cells'][cell] = crec
        out['releases'][name_rel] = rrec
    # benefits of each configuration against each eligible reference, recomputed from own cells vs the stored comparisons
    ben = {}
    for cfg in CONFIGS:
        for r in REFS:
            for cell in cells:
                b_pair = out['releases'][r]['cells'][cell]['pair']['max_incl_single']['recomputed'] - out['releases'][cfg]['cells'][cell]['pair']['max_incl_single']['recomputed']
                st = stored[cfg]['comparisons'][r]['pair']['cells'][cell]
                rec_v = {'pair_benefit': {'stored': st, 'recomputed': b_pair, 'abs_diff': abs(st - b_pair)}}
                for v in ('income', 'occupation'):
                    dv = out['releases'][cfg]['cells'][cell][v]['view_max']['recomputed'] - out['releases'][r]['cells'][cell][v]['view_max']['recomputed']
                    sv = stored[cfg]['comparisons'][r][v]['cells'][cell]
                    rec_v[f'{v}_recipient_diff'] = {'stored': sv, 'recomputed': dv, 'abs_diff': abs(sv - dv)}
                ben[f'{cfg}|{r}|{cell}'] = rec_v
    out['benefits'] = ben
    out['benefit_max_abs_diff'] = max(x['abs_diff'] for b in ben.values() for x in b.values())
    out['cpu_s'] = time.process_time() - t_start
    OUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    p = OUT / f'PROXY_CELLS_S{seed}.json'
    p.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: out[k] for k in ('seed', 'cells', 'max_abs_diff', 'n_compared', 'n_exact', 'dir_compared', 'dir_max_abs_diff',
                                          'benefit_max_abs_diff', 'cpu_s', 'halves')}, indent=1))
    for nr, rr in out['releases'].items():
        print(nr, 'wire_ok', rr['wire_matches_stored_eval'])
        for cell, cr in rr['cells'].items():
            print('  ', cell, {v: (round(cr[v]['view_max']['stored'], 6), cr[v]['view_max']['stored'] - cr[v]['view_max']['recomputed']) for v in VIEWS},
                  'pair_max_incl_single', cr['pair']['max_incl_single'])


if __name__ == '__main__':
    main()
