"""Frozen MODEL_SELECT rule: one common configuration across both seeds for every tunable control.

Rule (SCIENCE_LOCK 'selection'): (1) task feasibility first: a configuration is FEASIBLE iff all 24 point utility
clauses (2 tasks x {U0, Ucal} x {accuracy loss <= 0.010, NLL excess <= 0.010, summed Brier excess <= 0.005} x 2 seeds)
pass on MODEL_SELECT; (2) among feasible configurations, minimise the mean over seeds of the validation recovery proxy:
coalition AUC on MODEL_SELECT of a fixed OUT_LR reader (LogisticRegression C=1 on the released pair wire, finite
scaler) trained on the MODEL_FIT pair wire with MODEL_FIT SEX; ties by the fixed grid order; (3) if no configuration
is feasible, freeze one diagnostic fallback minimising the maximum normalised clause value (value / cap) over all 24
clauses, ties by grid order; the fallback is INELIGIBLE (descriptive only). LOCAL_TREE (rho1 E20), TASK_REFIT (E20),
T_TREE and the references are fixed and reported, never selected. No DEV_REPORT, ATTACK or SEALED label is read.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import warnings
from pathlib import Path

import numpy as np

from . import data as D
from . import registry

FIXED = ('T_TREE', 'LOCAL_TREE', 'TASK_REFIT', 'JOINT_TREE')


def family_grid(services):
    nr = sorted([s for s in services if s.startswith('NEURAL_RAW_rho')],
                key=lambda s: (float(s.split('rho')[1].split('_')[0]), int(s.split('_E')[1])))
    fg = sorted([s for s in services if s.startswith('FAGTB_lam')], key=lambda s: float(s.split('lam')[1]))
    fa = sorted([s for s in services if s.startswith('FARE_F')], key=lambda s: int(s.split('FARE_F')[1]))
    return {'NEURAL_RAW': nr, 'FAGTB': fg, 'FARE': fa}


def pair_out(svc_role):
    cols = []
    for t in D.TASKS:
        p = np.asarray(svc_role[t], dtype=np.float64)
        cols += [p, p.argmax(1)[:, None].astype(np.float64)]
    return np.column_stack(cols)


def proxy_auc(services_fit, services_sel, sex_fit, sex_sel, seed):
    from task_conditioned_representation.attacks import fit_reader, predict_reader, MODEL_SEED_BASE
    from task_conditioned_representation.inference import auc
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rec = fit_reader(pair_out(services_fit), sex_fit, 'OUT_LR', MODEL_SEED_BASE + seed)
    return auc(sex_sel, predict_reader(rec, pair_out(services_sel)))


def run(out_path, detail_path):
    if not D.lock_committed('SCIENCE_LOCK.json') or not D.lock_committed('TRAINING_COMPLETE_LOCK.json'):
        raise PermissionError('selection requires the committed SCIENCE_LOCK and TRAINING_COMPLETE_LOCK')
    S = D.load_role('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'))
    Fsex = D.load_role('MODEL_FIT', ('features', 'sex'))['sex']
    labels = {t: S[t] for t in D.TASKS}
    per_seed, hashes = {}, {}
    for seed in (0, 1):
        outs, h, missing = registry.load(seed, roles=('MODEL_FIT', 'MODEL_SELECT'))
        hashes.update(h)
        refs = {r: [outs[r]['MODEL_SELECT'][t] for t in D.TASKS] for r in ('U0', 'Ucal')}
        rows = {}
        for svc, roles in outs.items():
            if svc in ('U0', 'Ucal') or 'MODEL_SELECT' not in roles:
                continue
            probs = [roles['MODEL_SELECT'][t] for t in D.TASKS]
            cl = D.clauses(labels, probs, refs, seed)
            rows[svc] = {'clauses': cl, 'proxy_auc': proxy_auc(roles['MODEL_FIT'], roles['MODEL_SELECT'], Fsex, S['sex'], seed),
                         'summary': {t: D.summary(labels[t], roles['MODEL_SELECT'][t], D.K[t]) for t in D.TASKS}}
        per_seed[seed] = {'services': rows, 'missing_families': missing}
    common = set(per_seed[0]['services']) & set(per_seed[1]['services'])
    grid = family_grid(common)
    frozen = {}
    for fam, configs in grid.items():
        if not configs:
            frozen[fam] = {'status': 'NOT_RUN_OR_INVALID', 'configuration': None}; continue
        table = []
        for c in configs:
            cl = per_seed[0]['services'][c]['clauses'] + per_seed[1]['services'][c]['clauses']
            table.append({'configuration': c, 'feasible': all(x['point_pass'] for x in cl),
                          'max_normalized': max(x['normalized'] for x in cl),
                          'proxy_auc_mean': float(np.mean([per_seed[s]['services'][c]['proxy_auc'] for s in (0, 1)])),
                          'proxy_auc_per_seed': [per_seed[s]['services'][c]['proxy_auc'] for s in (0, 1)],
                          'failed_clauses': [f"s{x['seed']}|{x['task']}|{x['reference']}|{x['metric']}" for x in cl if not x['point_pass']]})
        feas = [r for r in table if r['feasible']]
        if feas:
            win = min(feas, key=lambda r: (r['proxy_auc_mean'], configs.index(r['configuration'])))
            frozen[fam] = {'status': 'ELIGIBLE', 'configuration': win['configuration'], 'rule': 'feasible; minimum mean MODEL_SELECT proxy AUC', 'table': table}
        else:
            win = min(table, key=lambda r: (r['max_normalized'], configs.index(r['configuration'])))
            frozen[fam] = {'status': 'INELIGIBLE_FALLBACK', 'configuration': win['configuration'],
                           'rule': 'no feasible configuration; minimum maximum normalised clause value', 'table': table}
    if frozen['FARE'].get('configuration'):
        frozen['FARE']['matched_compression_twin'] = frozen['FARE']['configuration'].replace('FARE_F', 'FARE_Z')
    # FAGTB counts as a two-task baseline only if both the faithful binary income model and the documented
    # FAGTB_MC_ADAPTATION occupation model are valid on both seeds (FAGTB_SEED.json flags written by the worker).
    flags = []
    for seed in (0, 1):
        p = registry.RUNS / f'FAGTB_{seed}' / 'FAGTB_SEED.json'
        if p.is_file():
            r = json.loads(p.read_text()); flags.append(bool(r.get('binary_valid')) and bool(r.get('mc_valid')))
        else:
            flags.append(False)
    frozen['FAGTB']['two_task_valid'] = bool(all(flags))
    fixed = {}
    for svc in FIXED:
        if svc in common:
            cl = per_seed[0]['services'][svc]['clauses'] + per_seed[1]['services'][svc]['clauses']
            fixed[svc] = {'point_feasible_24': all(x['point_pass'] for x in cl), 'failed_clauses': [f"s{x['seed']}|{x['task']}|{x['reference']}|{x['metric']}" for x in cl if not x['point_pass']],
                          'proxy_auc_per_seed': [per_seed[s]['services'][svc]['proxy_auc'] for s in (0, 1)]}
    roster = ['T_TREE', 'LOCAL_TREE', 'TASK_REFIT', 'U0'] + [s for s in ('JOINT_TREE',) if s in common]
    for fam in ('NEURAL_RAW', 'FAGTB', 'FARE'):
        if frozen[fam].get('configuration'):
            roster.append(frozen[fam]['configuration'])
    if frozen['FARE'].get('matched_compression_twin') in common:
        roster.append(frozen['FARE']['matched_compression_twin'])
    lock = {'schema': 'ACSR1-selection-lock-v1', 'status': 'SELECTION_LOCKED', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
            'rule': __doc__.strip(), 'frozen': {k: {kk: vv for kk, vv in v.items() if kk != 'table'} for k, v in frozen.items()},
            'fixed_services': fixed, 'attack_roster': roster, 'output_file_sha256': hashes,
            'DEV_REPORT_labels_opened': False, 'ATTACK_labels_used': False, 'sealed_assessment_touched': False}
    detail = {'schema': 'ACSR1-selection-detail-v1', 'per_seed': {str(s): {'services': {k: {'clauses': v['clauses'], 'proxy_auc': v['proxy_auc'], 'summary': v['summary']}
                                                                                  for k, v in per_seed[s]['services'].items()},
                                                                     'missing_families': per_seed[s]['missing_families']} for s in (0, 1)},
              'family_tables': {k: v.get('table') for k, v in frozen.items()}}
    with open(out_path, 'x') as f:
        json.dump(lock, f, indent=2, sort_keys=True)
    with open(detail_path, 'x') as f:
        json.dump(detail, f, indent=2, sort_keys=True)
    return lock


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--detail', required=True)
    a = ap.parse_args(); r = run(a.out, a.detail)
    print(json.dumps({'frozen': r['frozen'], 'roster': r['attack_roster'], 'fixed': {k: v['point_feasible_24'] for k, v in r['fixed_services'].items()}}, indent=1))
