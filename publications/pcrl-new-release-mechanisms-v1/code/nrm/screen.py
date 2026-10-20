"""Development SCREENING readers of the PCRL new-release-mechanisms campaign (operation 'screen', logged; adaptive diagnosis on
MODEL_SELECT only; never a claim and never a substitute for the registered grid nrm.devreaders).

Readers estimate the recorded SEX column (sex 0 = men, sex 1 = women) from released outputs only: the income view, the occupation
view and the linked pair (each recipient's float64 probability vector and argmax decision, as in the grid). The screening split is
registered label-free in READER_SPLITS.json: MODEL_SELECT households ordered by SHA-256('NRM1_screen|<household>') and dealt into
two halves A / B. Each recipe is fitted on one half (oriented on the same half) and scored on the other, both ways; the screen AUC
of a view is the mean of the two directions, reported per recipe and as the maximum over the screening recipes (an optimistic
summary, identical for every release). Recipes (locked source code, as in the grid): LOG_LR, OUT_HGB, LOOKUP, HGB15. Randomized
releases: draws 0 and 1 (one draw per person each) are screened separately and averaged; the target and auxiliary people never
contribute more than one draw to any fit or score.

    python3 -m nrm.screen release --family F --config C [--tag T]      -> runs/<F>/<C>/SCREEN[__T].json, dev/SCREEN_LEDGER.csv
    python3 -m nrm.screen reference --name REF:<name> | W              -> runs/SCREEN_REFS/<name>.json (frozen inherited releases)
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

from task_conditioned_representation.inference import auc
from . import common as C
from . import release as RL

SPLIT_TAG = 'NRM1_screen'
RECIPES = ('LOG_LR', 'OUT_HGB', 'LOOKUP', 'HGB15')
VIEWS = ('income', 'occupation', 'pair')
SEED = 2026102014
MAX_DRAWS = 2
LEDGER = C.RUNS / 'SCREEN_LEDGER.jsonl'
OUT_PUBLIC = C.RESULTS / 'dev'


def halves(unit):
    part = C.household_folds(unit, 2, SPLIT_TAG)
    return part == 0, part == 1


def _screen_wire(inc, occ, sex, unit):
    from tsz.devreaders import _fit_one
    A, B = halves(unit)
    res = {}
    for view in VIEWS:
        per = {}
        for name in RECIPES:
            aucs = []
            for fit, sc in ((A, B), (B, A)):
                s, _ = _fit_one(name, view, inc, occ, sex, fit, fit, SEED)
                aucs.append(float(auc(sex[sc], s[sc])))
            per[name] = float(np.mean(aucs))
        res[view] = {'per_recipe': per, 'max': max(per.values()), 'argmax': max(per, key=per.get)}
    return res


def screen_release(rel, sex, unit):
    inc, occ = rel['income'], rel['occupation']
    if inc.ndim == 2 and occ.ndim == 2:
        return _screen_wire(inc, occ, sex, unit), 1
    R = min(MAX_DRAWS, inc.shape[0] if inc.ndim == 3 else occ.shape[0])
    outs = [_screen_wire(inc[r] if inc.ndim == 3 else inc, occ[r] if occ.ndim == 3 else occ, sex, unit) for r in range(R)]
    res = {}
    for view in VIEWS:
        per = {n: float(np.mean([o[view]['per_recipe'][n] for o in outs])) for n in RECIPES}
        res[view] = {'per_recipe': per, 'max': max(per.values()), 'argmax': max(per, key=per.get),
                     'per_draw_max': [o[view]['max'] for o in outs]}
    return res, R


def _load_sex(purpose):
    S = C.load('MODEL_SELECT', ('features', 'sex'), 'screen', purpose)
    return S['sex'], S['unit']


def release_cmd(family, config, tag=None):
    t0 = time.process_time()
    d = RL.rdir(family, config, 0).parent
    out = d / (f'SCREEN__{tag}.json' if tag else 'SCREEN.json')
    if out.exists():
        raise PermissionError(f'{out.name} exists (write-once)')
    sex, unit = _load_sex(f'screen: development screening readers on {family}/{config} (registered screening halves; adaptive)')
    res = {'schema': 'NRM1-screen-v1', 'family': family, 'config_id': config, 'utc': C.now(), 'recipes': list(RECIPES), 'versions': {}}
    rows = []
    for v in C.VERSIONS:
        if not RL.exists(family, config, v):
            continue
        arr, rec = RL.load(family, config, v, roles=('MODEL_SELECT',))
        r, R = screen_release(arr['MODEL_SELECT'], sex, unit)
        res['versions'][str(v)] = {'screen': r, 'draws_screened': R, 'release_sha256': rec['release_sha256']}
        rows.append({'family': family, 'config_id': config, 'version': v, 'tag': tag or '', 'draws_screened': R,
                     **{f'{view}_max': round(r[view]['max'], 4) for view in VIEWS},
                     **{f'{view}_{n}': round(r[view]['per_recipe'][n], 4) for view in VIEWS for n in RECIPES}})
    res['cpu_s'] = time.process_time() - t0
    res['unit_id'] = os.environ.get(C.UNIT_ENV)
    C.write_json_x(out, res)
    _append(rows, out)
    print(json.dumps([{k: r[k] for k in ('family', 'config_id', 'version', 'income_max', 'occupation_max', 'pair_max')} for r in rows]))
    return res


def reference_cmd(name):
    from .evaluate import comparator_release
    t0 = time.process_time()
    out = C.RUNS / 'SCREEN_REFS' / f'{name.replace(":", "_")}.json'
    if out.exists():
        raise PermissionError(f'{out.name} exists (write-once)')
    sex, unit = _load_sex(f'screen: development screening readers on the frozen inherited release {name} (adaptive comparison)')
    res = {'schema': 'NRM1-screen-v1', 'reference': name, 'utc': C.now(), 'recipes': list(RECIPES), 'versions': {}}
    rows = []
    for v in C.VERSIONS:
        rel, prov = comparator_release(name, v)
        r, R = screen_release(rel, sex, unit)
        res['versions'][str(v)] = {'screen': r, 'draws_screened': R, 'provenance': prov}
        rows.append({'family': 'REF', 'config_id': name, 'version': v, 'tag': '', 'draws_screened': R,
                     **{f'{view}_max': round(r[view]['max'], 4) for view in VIEWS},
                     **{f'{view}_{n}': round(r[view]['per_recipe'][n], 4) for view in VIEWS for n in RECIPES}})
    res['cpu_s'] = time.process_time() - t0
    C.write_json_x(out, res)
    _append(rows, out)
    print(json.dumps([{k: r[k] for k in ('config_id', 'version', 'income_max', 'occupation_max', 'pair_max')} for r in rows]))
    return res


def _append(rows, out):
    LEDGER.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(LEDGER, 'a') as f:
        for r in rows:
            f.write(json.dumps({**r, 'utc': C.now(), 'screen_file': str(out.relative_to(C.ROOT))}, sort_keys=True) + '\n')
    allr = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    keys = sorted({k for r in allr for k in r if k != 'screen_file'})
    lead = ['utc', 'family', 'config_id', 'version', 'tag', 'draws_screened', 'pair_max', 'income_max', 'occupation_max']
    cols = [k for k in lead if k in keys] + [k for k in keys if k not in lead]
    OUT_PUBLIC.mkdir(parents=True, exist_ok=True)
    with open(OUT_PUBLIC / 'SCREEN_LEDGER.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader(); w.writerows(allr)


def selftest():
    from tsz.devreaders import _invented
    unit, sex, inc, occ_leaky, occ_safe = _invented(1200)
    a, _ = screen_release({'income': inc, 'occupation': occ_leaky}, sex, unit)
    b, _ = screen_release({'income': inc, 'occupation': occ_safe}, sex, unit)
    A, B = halves(unit)
    ok = {'halves_partition': bool(np.array_equal(A | B, np.ones(len(unit), bool)) and not (A & B).any()),
          'halves_household_disjoint': not np.intersect1d(unit[A], unit[B]).size,
          'leaky_reads_higher': a['occupation']['max'] > b['occupation']['max'],
          'income_identical': a['income']['max'] == b['income']['max']}
    st = {'income': np.stack([inc, inc]), 'occupation': np.stack([occ_leaky, occ_leaky])}
    c, R = screen_release(st, sex, unit)
    ok['constant_draws_equal'] = R == 2 and abs(c['pair']['max'] - a['pair']['max']) < 1e-12
    ok['all_ok'] = all(ok.values())
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('release'); r.add_argument('--family', required=True); r.add_argument('--config', required=True)
    r.add_argument('--tag', default=None)
    f = sub.add_parser('reference'); f.add_argument('--name', required=True)
    sub.add_parser('selftest')
    a = ap.parse_args(argv)
    if a.cmd == 'release':
        release_cmd(a.family, a.config, a.tag); return 0
    if a.cmd == 'reference':
        reference_cmd(a.name); return 0
    res = selftest(); print(json.dumps(res)); return 0 if res['all_ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
