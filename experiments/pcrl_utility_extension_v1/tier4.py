"""Tier 4: test-split comparisons for the nominated configurations (PROTOCOL sections 6-7).

Paired household-cluster bootstrap (predecessor machinery), seed-averaged per-person test-loss
differences, both weightings. ONE declared level for every primary contrast: studentized
Bonferroni over the realised primary family. The full grid vs J is a separate exploratory family.
Sign convention (inherited): utility rows are loss differences (negative = candidate better);
recovery rows are recovery differences (negative = candidate leaks less).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

from experiments.pcrl_direct_adversarial_v1 import inputs as dax
from experiments.pcrl_direct_adversarial_v1.run_report import build_bootstrap, contrast_intervals
from experiments.pcrl_nonlinear_rank_v1.report import FAMILY_ENDPOINTS
from experiments.pcrl_nonlinear_rank_v1.run_report import _losses

from . import program as pg

ALPHA = 0.05
RES = 'utility/same_residence'
SENS = tuple('recovery/' + e for e in pg.SENSITIVE4)


def vectors_for(names, frames):
    vec = {}
    for seed in pg.SEEDS:
        labels = frames[seed][2]
        for name in names:
            path = pg.OUT / f'seed_{seed}' / pg.canonical(seed, name) / 'predictions.npz'
            with np.load(path) as store:
                for w in pg.WEIGHTS:
                    p = pg.point(seed, name, 'test', w)
                    for e in FAMILY_ENDPOINTS:
                        kind, rest = e.split('/', 1)
                        if kind == 'utility':
                            row = p['selected']['utility/' + rest]
                            target, key = rest, f"utility/{row['view']}/{rest}/None/{row['candidate_id']}/test"
                        else:
                            row = p['selected']['audit/' + rest]
                            target = rest.split('/')[1]
                            key = f"audit/{row['view']}/{target}/{row['audit_budget']}/{row['candidate_id']}/test"
                        vec[seed, name, w, e] = (target, _losses(store[key], labels['test'][target])[0])
    return vec


def adjust(rows):
    m = len(rows)
    z2, z1 = NormalDist().inv_cdf(1 - ALPHA / (2 * m)), NormalDist().inv_cdf(1 - ALPHA / m)
    for r in rows:
        r.update(family_size=m, z_two_sided=z2, z_one_sided=z1,
                 low=r['estimate'] - z2 * r['bootstrap_se'], high=r['estimate'] + z2 * r['bootstrap_se'],
                 upper_one_sided=r['estimate'] + z1 * r['bootstrap_se'])
    return rows


def decide(rows, cand, comp):
    out = {}
    for w in pg.WEIGHTS:
        get = {r['endpoint']: r for r in rows if r['left'] == cand and r['right'] == comp and r['weight'] == w}
        if RES not in get or any(s not in get for s in SENS):
            out[w] = {'assessable': False}
            continue
        res = get[RES]
        out[w] = {'assessable': True,
                  'residence_improved': res['estimate'] <= -0.003 and res['high'] < 0,
                  'sensitive_within_001': all(get[s]['upper_one_sided'] <= 0.001 for s in SENS),
                  'sensitive_within_0': all(get[s]['upper_one_sided'] <= 0 for s in SENS),
                  'residence_estimate': res['estimate'], 'residence_interval': [res['low'], res['high']],
                  'sensitive_upper': {s: get[s]['upper_one_sided'] for s in SENS}}
    ok = all(v.get('assessable') and v['residence_improved'] and v['sensitive_within_001'] for v in out.values())
    return {'pass': bool(ok), 'by_weight': out}


def matched(cfg):
    spec = pg.unit_spec(cfg)
    r, b = spec['r'], int(round(spec['beta']))
    return {'L1': f'X_r{r}_L1_b{b:03d}', 'L2': f'X_r{r}_L2_b{b:03d}', 'U': f'U_r{r}', 'P': f'P_r{r}', 'LEACE': f'LEACE_r{r}'}


def run(out: Path, gate):
    sel = json.loads((pg.OUT / 'SELECTION.json').read_text())['selected']
    boot, frames = build_bootstrap(pg.SEEDS, dax.Registry.new())
    comps = {c: matched(c) for c in sel}
    names = sorted(set(sel) | {'ref_J', 'ref_leace_A0'} | {v for m in comps.values() for v in m.values()})
    vec = vectors_for(names, frames)
    specs = [{'left': c, 'right': r, 'family': 'primary'}
             for c in sel for r in ['ref_J', 'ref_leace_A0', *comps[c].values()]]
    rows, _ = contrast_intervals(boot, vec, pg.SEEDS, specs)
    rows = adjust(rows)
    decision = {}
    for c in sel:
        vs_j, vs_l = decide(rows, c, 'ref_J'), decide(rows, c, 'ref_leace_A0')
        local = {k: decide(rows, c, comps[c][k]) for k in ('L1', 'L2')}
        decision[c] = {'development_candidate_vs_J': vs_j, 'competitive_vs_leace_A0': vs_l,
                       'coalition_specific_vs_L1_L2': local,
                       'label': ('prospective development candidate' if vs_j['pass'] else 'no pass vs J')}
    grid = [n for n in pg.units(pg.FULL_R, leace=True)]
    gvec = vectors_for(sorted(set(grid) | {'ref_J'}), frames)
    grows, _ = contrast_intervals(boot, gvec, pg.SEEDS, [{'left': g, 'right': 'ref_J', 'family': 'exploratory_grid'} for g in grid])
    grows = adjust(grows)
    for name, data in (('INTERVALS_PRIMARY.csv', rows), ('INTERVALS_GRID_EXPLORATORY.csv', grows)):
        with open(pg.OUT / name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in data for k in r}))
            w.writeheader(); w.writerows(data)
    (pg.OUT / 'TIER4_DECISION.json').write_text(json.dumps(decision, indent=1))
    any_pass = any(d['development_candidate_vs_J']['pass'] for d in decision.values())
    return gate('T4', 'PASS' if any_pass else 'FAIL', decision={c: d['label'] for c, d in decision.items()},
                primary_family_size=len(rows), grid_family_size=len(grows),
                next_action=('2017 exploratory transport (pending implementation; resume from this gate)'
                             if any_pass else 'closeout'))
