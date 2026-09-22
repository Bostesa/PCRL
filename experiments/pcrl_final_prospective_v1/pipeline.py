"""Stage driver used on the single Linux x86 execution host.

Stages: parity (2018 encoding vs archived host), emulated (timing/integration
dry run on resampled 2018 rows), prepare2016, fit2016, lock, score2016, infer.
Every stage is resumable from its markers; completed units are never refit.
"""
from __future__ import annotations
import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np

from experiments.pcrl_task_directed_release_v1.uncertainty import paired_household_bounds
from . import inference_panel as ip
from .audit_panel import all_units, run_all, unit_dir
from .common import ANCHORS, OUT, PANEL, PRIMARY_ROLES, PRIVATE, RESTORED, ROOT, TASK_ROLE, atomic_json, now, sha


def root_for(dataset):
    return PRIVATE/'runs'/dataset


def parity():
    """Cross-host encoding parity against the archived Linux encodings (2018 fitting pools)."""
    from .releases import FrozenReleases
    out = {'host': platform.platform(), 'machine': platform.machine(), 'created_utc': now(), 'anchors': {}}
    for a in ANCHORS:
        prep = joblib.load(RESTORED/f'anchor_{a}'/'prepared.joblib')
        fr = FrozenReleases(a)
        res = {}
        for pool, d in prep['ctx']['pools'].items():
            mine, old = fr.encode(d['x'], d['ha']), prep['encoded'][pool]
            res[pool] = {'n': int(len(d['x'])), 'T0_code_flips': int((mine['codes']['T0'] != old['codes']['T0']).sum()),
                         'p_bitwise': bool(np.array_equal(mine['p'], old['p'])),
                         'b_bitwise': bool(np.array_equal(mine['b'], old['b'])),
                         'actions17_bitwise': bool(np.array_equal(mine['actions'][17], old['actions'][17])),
                         'p_max_abs': float(np.abs(mine['p']-old['p']).max()),
                         'b_max_abs': float(np.abs(mine['b']-old['b']).max())}
        out['anchors'][a] = res
    atomic_json(OUT/'private'/f'ENCODING_PARITY_{platform.machine()}.json', out)
    return out


def load_losses(root, pool='final'):
    losses = {}
    for short in PANEL:
        for a in ANCHORS:
            for role in PRIMARY_ROLES+(TASK_ROLE,):
                d = unit_dir(root, short, a, role)
                with np.load(d/f'score_{pool}.npz') as z:
                    losses[(short, a, role)] = {k: z[k] for k in ('ids', 'households', 'weights', 'loss')}
    return losses


def infer(root, dataset):
    root = Path(root)
    losses = load_losses(root)
    prim, sec, desc = ip.primary_endpoints(), ip.secondary_endpoints(), ip.descriptive_endpoints()
    endpoints = prim+sec+desc
    tick = time.perf_counter()
    result = paired_household_bounds(ip.contrasts_from_losses(endpoints, losses), n_boot=ip.BOOT['n_boot'],
                                     seed=ip.BOOT['seed'], alpha=.05)
    bounds = result['bounds']
    record = {'created_utc': now(), 'dataset': dataset, 'bootstrap': result['bootstrap'],
              'households': result['households_union'], 'counts': ip.family_counts(),
              'primary': ip.decide(prim, bounds), 'secondary': ip.secondary_intervals(sec, bounds),
              'descriptive': ip.descriptive_intervals(desc, bounds), 'seconds': time.perf_counter()-tick,
              'se_source': 'paired_household_bounds (predecessor, unchanged); its own Bonferroni fields are ignored'}
    record['secondary_interpretation'] = ip.interpret_secondary(record['secondary'])
    atomic_json(root/'INFERENCE.json', record)
    return record


def absolute_scores(root):
    out = {}
    for short in PANEL:
        for a in ANCHORS:
            for role in PRIMARY_ROLES+(TASK_ROLE,):
                m = json.loads((unit_dir(root, short, a, role)/'SCORED_final.json').read_text())
                out[f'{short}|{a}|{role}'] = {k: m[k] for k in ('selection', 'rows', 'ce', 'independent_ce', 'fixed_decoder_ce')}
    atomic_json(Path(root)/'ABSOLUTE_SCORES.json', out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('smoke', 'parity', 'emulated', 'prepare2016', 'fit2016', 'score2016', 'infer2016'))
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    if a.stage == 'smoke':
        from .audit_panel import roles_for
        root = root_for('smoke')
        only = [(s, 0, r) for s in ('H', 'Q', 'D17', 'D33', 'E', 'W75') for r in roles_for(s)]
        print(run_all(root, 'smoke', a.workers, phase='fit', only=only))
        print(run_all(root, 'smoke', a.workers, phase='score', only=[u for u in only if '/B/' not in u[2] and ':B/' not in u[2]]))
    elif a.stage == 'parity':
        print(json.dumps(parity(), indent=1))
    elif a.stage == 'emulated':
        from .prepare import build_emulated, data_dir
        if not (data_dir('emulated')/'PREPARED.json').exists():
            t = time.perf_counter(); build_emulated(); prep_s = time.perf_counter()-t
        else:
            prep_s = None
        root = root_for('emulated')
        t = time.perf_counter(); fit = run_all(root, 'emulated', a.workers, phase='fit'); fit_s = time.perf_counter()-t
        t = time.perf_counter(); sc = run_all(root, 'emulated', a.workers, phase='score'); score_s = time.perf_counter()-t
        t = time.perf_counter(); infer(root, 'emulated'); inf_s = time.perf_counter()-t
        atomic_json(root/'TIMING.json', {'created_utc': now(), 'workers': a.workers, 'prepare_seconds': prep_s,
                                         'fit_seconds': fit_s, 'score_seconds': score_s, 'infer_seconds': inf_s,
                                         'fit': fit, 'score': sc, 'host': platform.platform()})
    elif a.stage == 'prepare2016':
        from .prepare import build_2016, labels_2016
        print(build_2016()); print(labels_2016(['fit', 'attack_val', 'task_val']))
    elif a.stage == 'fit2016':
        print(run_all(root_for('acs2016'), 'acs2016', a.workers, phase='fit'))
    elif a.stage == 'score2016':
        from .prepare import labels_2016, data_dir
        if not (data_dir('acs2016')/'labels_final.npz').exists():
            labels_2016(['final'])
        print(run_all(root_for('acs2016'), 'acs2016', a.workers, phase='score'))
    elif a.stage == 'infer2016':
        root = root_for('acs2016')
        absolute_scores(root)
        r = infer(root, 'acs2016')
        print(json.dumps({m: r['primary'][m]['decision'] for m in r['primary']}))


if __name__ == '__main__':
    main()
