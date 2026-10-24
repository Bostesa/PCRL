"""Verifier unit (label read; torch venv = the grid's locked interpreter): refit a random sample of winning readers from scratch.

Sampling frame: every registered-cell winner (cells S<v>R<1|2>k<0|1>, views income / occupation / pair) of the banks SAR1_FB_R1 and
SAR1_FA_R1, members other than the U0 / Ucal diagnostics; 6 drawn without replacement by numpy.random.default_rng(SEED).
Each sampled winner is refitted on the frozen FIT part of its repetition (the verifier's own implementation of the frozen split rule:
households ordered by SHA-256('SAR1_devgrid_<R>|<household>'), dealt into 10 parts), with the reader seed MODEL_SEED_BASE + 1000 +
10 v + k, using the locked recipe code only (scm_bench.readers, task_conditioned_representation.attacks, ffv.readers: the modules
the grid's DEVGRID_CODE_LOCK.json pins); oriented on VALIDATION with the recipe module's orientation rule; scored on SCORE with the
verifier's own AUC. Compared with the grid: VALIDATION AUC, the SCORE score vector (bitwise) and the SCORE AUC.
Also: the MODEL_FIT share of men used by FA_TFRO_R1's guard estimator (p = 0.5103), and the seed-effectiveness of the seed-dependent
recipes on the minimum-set wires (k0 vs k1 stored scores differ).

    <repository>/.venv/bin/python results/.../roles/verifier/code/v_refit.py --out results/.../roles/verifier/work/refit.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

sys.path.insert(0, str(V.ROOT))
import numpy as np  # noqa: E402

from sar import common as C  # noqa: E402
from sar import release as RL  # noqa: E402
from scm_bench.readers import SLATE, kinds, lookup_keys, fit_lookup, predict_lookup  # noqa: E402  (locked recipe code)
from task_conditioned_representation.attacks import fit_reader, predict_reader, orientation, orient  # noqa: E402
from ffv.readers import design_for, fit_new, predict_new, versions  # noqa: E402

SEED = 20261024
N_SAMPLE = 6
BANKS = ('SAR1_FB_R1', 'SAR1_FA_R1')
SEED_DEP = ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP')
MIN_SET = {'SAR1_FB_R1': ('FB_PFO', 'NEURAL_RAW', 'FB_PFO_TASK', 'FB_R1_CTRL_HI', 'FB_R1_ALT_HI'),
           'SAR1_FA_R1': ('FA_TFRO_R1', 'SUCC', 'FA_TFRO_R1_CONT', 'FA_R1_CTRL_LO')}
INPUTS = {}


def note(p):
    INPUTS[str(Path(p).relative_to(V.ROOT))] = V.fsha(p)


def member_arrays(m, v):
    t = m['type']
    if t == 'sar':
        arr, _ = RL.load(m['family'], m['config'], v, roles=('MODEL_SELECT',))
        note(RL.rdir(m['family'], m['config'], v) / 'RELEASE.npz')
        rel = arr['MODEL_SELECT']
    elif t == 'arrays':
        rel = {}
        for task in V.TASKS:
            src = m['seeds'][str(v)][task]
            p = V.ROOT / src['path']
            if V.fsha(p) != src['sha256']:
                raise PermissionError(f'{p}: hash differs from the bank spec')
            note(p)
            with np.load(p, allow_pickle=False) as z:
                rel[task] = z[src['key']].copy()
    elif t == 'reference':
        from ohn.devrun import reference_release
        rel = reference_release(m['name'], v, 'MODEL_SELECT')
    elif t == 'witness':
        from tsz.evaluate import source_a1, h3_income
        rel = {'income': h3_income(v), 'occupation': source_a1(v)}
    else:
        raise ValueError(t)
    return {k: np.ascontiguousarray(np.asarray(rel[k], dtype=np.float64)) for k in V.TASKS}


def refit_one(recipe, view, inc, occ, sex, fit, val, rs):
    """The locked recipe dispatch (definition: PROTOCOL 7 / the grid's recipe set), written out by the verifier."""
    src_kind = dict(SLATE)
    if recipe == 'LOOKUP':
        keys = lookup_keys(inc) if view == 'income' else lookup_keys(occ) if view == 'occupation' else lookup_keys(inc, occ)
        model = fit_lookup([keys[i] for i in np.flatnonzero(fit)], sex[fit])
        raw = predict_lookup(model, keys)
    elif recipe in src_kind:
        a, b = kinds(inc, 2)[src_kind[recipe]], kinds(occ, 5)[src_kind[recipe]]
        X = a if view == 'income' else b if view == 'occupation' else np.column_stack((a, b))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = fit_reader(X[fit], sex[fit], recipe, rs)
        raw = predict_reader(model, X)
    else:
        tf, _, _ = design_for(recipe, view, inc[fit], occ[fit])
        X = tf(inc, occ)
        model, _ = fit_new(recipe, X[fit], sex[fit], rs)
        raw = predict_new(model, X)
    sign = orientation(sex[val], raw[val])
    return orient(raw, sign), int(sign)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.process_time()
    S = C.load('MODEL_SELECT', ('features', 'sex'), 'verify',
               'independent verifier: refit a random sample (seed 20261024) of 6 winning readers of SAR1_FB_R1 / SAR1_FA_R1 on '
               'the frozen splits and seeds; seed-effectiveness spot check (verifier wave 1)')
    sex, unit = np.asarray(S['sex']), np.asarray(S['unit'])
    del S
    MF = C.load('MODEL_FIT', ('features', 'sex'), 'verify',
                'independent verifier: the MODEL_FIT share of men used by FA_TFRO_R1\'s guard estimator (p = 0.5103)')
    p_men = float(np.mean(np.asarray(MF['sex']) == 0))
    del MF
    out = {'schema': 'SAR1-verifier-refit-v1', 'utc_start': C.now(), 'seed': SEED, 'versions': versions(),
           'interpreter': sys.executable, 'p_men_model_fit': p_men, 'refits': [], 'seed_effectiveness': []}
    sex_hash = V.ahash(sex)
    cache = V.RUNS / 'devgrid_cache'
    frame, grids, specs = [], {}, {}
    for bank in BANKS:
        sp = V.RES / 'grid_specs' / f'{bank}.json'
        note(sp)
        specs[bank] = json.loads(sp.read_text())
        for v in (0, 1):
            gp = V.RUNS / 'GRID' / bank / f'S{v}' / 'GRID.json'
            note(gp)
            grids[(bank, v)] = json.loads(gp.read_text())
            for cell in sorted(grids[(bank, v)]['cells']):
                for name in sorted(grids[(bank, v)]['cells'][cell]):
                    if name in ('U0', 'Ucal'):
                        continue
                    for view in ('income', 'occupation', 'pair'):
                        frame.append((bank, v, cell, name, view))
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(len(frame), size=N_SAMPLE, replace=False).tolist())
    out['frame_size'] = len(frame)
    out['sample_indices'] = pick
    for idx in pick:
        bank, v, cell, name, view = frame[idx]
        rep, k = cell[2:4], int(cell[5:])
        e = grids[(bank, v)]['cells'][cell][name][view]
        winner = e['winner']
        if winner.startswith('IGNORE_occupation__'):
            wview, recipe = 'income', winner.split('__', 1)[1]
        elif winner.startswith('IGNORE_income__'):
            wview, recipe = 'occupation', winner.split('__', 1)[1]
        else:
            wview, recipe = view, winner
        rel = member_arrays(specs[bank]['members'][name], v)
        roles = V.split_roles(unit, rep)
        rs = V.reader_seed(v, k)
        t1 = time.process_time()
        s, sign = refit_one(recipe, wview, rel['income'], rel['occupation'], sex, roles['FIT'], roles['VALIDATION'], rs)
        cpu = time.process_time() - t1
        val_auc = V.auc_weighted(sex[roles['VALIDATION']], s[roles['VALIDATION']])
        sc_auc = V.auc_weighted(sex[roles['SCORE']], s[roles['SCORE']])
        key = V.cache_key(rep, rs, 1.0, V.view_hash(wview, rel['income'], rel['occupation']), sex_hash)
        with np.load(cache / f'{key}.npz', allow_pickle=False) as z:
            stored = z[f'score__{recipe}'].copy()
            info = json.loads(str(z['info']))[recipe]
        bitwise = bool(np.array_equal(stored, s[roles['SCORE']]))
        out['refits'].append({'bank': bank, 'version': v, 'cell': cell, 'member': name, 'view': view, 'winner': winner,
                              'refit_view': wview, 'recipe': recipe, 'reader_seed': rs, 'orientation': sign,
                              'orientation_record': info.get('orientation'),
                              'val_auc_refit': val_auc, 'val_auc_record': e['val_auc'], 'val_auc_cache': info['val_auc'],
                              'score_auc_refit': sc_auc, 'score_auc_record': e['score_auc'],
                              'score_vector_bitwise_equal_to_stored': bitwise,
                              'max_abs_score_diff': float(np.abs(stored - s[roles['SCORE']]).max()), 'cpu_s': cpu,
                              'n_fit': int(roles['FIT'].sum()), 'n_validation': int(roles['VALIDATION'].sum()),
                              'n_score': int(roles['SCORE'].sum())})
        print(json.dumps(out['refits'][-1]), flush=True)
    # seed effectiveness of the seed-dependent recipes on the minimum-set wires (stored SCORE scores, k0 vs k1)
    for bank in BANKS:
        for v in (0, 1):
            for name in MIN_SET[bank]:
                rel = member_arrays(specs[bank]['members'][name], v)
                for rep in ('R1', 'R2'):
                    for view in ('income', 'occupation', 'pair'):
                        vh = V.view_hash(view, rel['income'], rel['occupation'])
                        ent = []
                        for k in (0, 1):
                            with np.load(cache / f'{V.cache_key(rep, V.reader_seed(v, k), 1.0, vh, sex_hash)}.npz') as z:
                                ent.append({r: z[f'score__{r}'].copy() for r in SEED_DEP})
                        eq = [r for r in SEED_DEP if np.array_equal(ent[0][r], ent[1][r])]
                        out['seed_effectiveness'].append({'bank': bank, 'v': v, 'member': name, 'rep': rep, 'view': view,
                                                          'equal_k0_k1': eq, 'distinct_inputs':
                                                          int(len(np.unique(np.column_stack([rel['income']] if view == 'income' else [rel['occupation']] if view == 'occupation' else [rel['income'], rel['occupation']]), axis=0)))})
    out['inputs_sha256'] = INPUTS
    out['cpu_s'] = time.process_time() - t0
    out['utc_end'] = C.now()
    V.jdump(Path(a.out), out)
    print(json.dumps({'refits': len(out['refits']), 'bitwise': [r['score_vector_bitwise_equal_to_stored'] for r in out['refits']],
                      'p_men': p_men, 'seed_equalities_on_minimum_set': sum(bool(x['equal_k0_k1']) for x in out['seed_effectiveness']),
                      'cpu_s': out['cpu_s']}))


if __name__ == '__main__':
    main()
