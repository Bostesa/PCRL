"""Enumerate the registered matrix and hash every dependency before any new outcome.

`MATRIX.json` lists every nominal slot of both tracks with its audit-queue priority.
`DEPENDENCY_MANIFEST.json` hashes this study's code, its fixtures and the predecessor
modules it imports, so a later reader can tell whether anything moved after the lock.
`--verify` recomputes the hashes and fails on any drift.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import projection as pj
from .common import OUT, PREDECESSORS, ROOT, read_json, sha_file, utcnow, write_json_atomic
from .run_fit_n import BETAS, GAMMAS, INITS, POLICIES, plan as n_plan

SEEDS = (0, 1, 2)
CHANNELS = ('A0', 'J')

# Audit-queue priority, fixed now (PROTOCOL §9). Whatever §13 would cut first is queued
# last, so a time cut can only ever remove exactly the registered reduction slices, in
# the registered order, uniformly across matched methods.
PRIORITY_MAIN = 1
PRIORITY_K1_K6 = 2          # second registered reduction
PRIORITY_GAMMA_001 = 3      # first registered reduction, therefore audited last


def track_e_slots() -> list:
    slots = []
    for channel in CHANNELS:
        for policy in ('L', 'C', 'LX'):
            for k in pj.REMOVAL_RANKS:
                slots.append(dict(unit=f'E_{channel}_{policy}_k{k}', block='core', channel=channel,
                                  policy=policy, k=k))
        for control, block in (('marginal', 'control_marginal'), ('pca', 'control_pca'),
                               ('rand', 'control_random')):
            for k in pj.REMOVAL_RANKS:
                slots.append(dict(unit=f'E_{channel}_{control}_k{k}', block=block,
                                  channel=channel, policy=control, k=k))
        for policy in ('L', 'C', 'LX'):
            slots.append(dict(unit=f'E_{channel}_{policy}_full', block='full_span_diagnostic',
                              channel=channel, policy=policy, k='full'))
    for name in ('leace_J', 'splince_J'):
        slots.append(dict(unit=name, block='ordinary_erasure_on_J', channel='J',
                          policy=name.split('_')[0], k=None))
    for slot in slots:
        slot['track'] = 'E'
        slot['priority'] = PRIORITY_K1_K6 if slot['k'] in (1, 6) else PRIORITY_MAIN
    return slots


def track_n_slots() -> list:
    slots = []
    for unit in n_plan():
        slot = {**unit, 'track': 'N',
                'priority': PRIORITY_GAMMA_001 if unit['gamma'] == 0.01 else PRIORITY_MAIN}
        slots.append(slot)
    return slots


def matrix() -> dict:
    e, n = track_e_slots(), track_n_slots()
    per_seed = len(e) + len(n)
    return {
        'generated_utc': utcnow(),
        'seeds': list(SEEDS),
        'track_N': {'gammas': list(GAMMAS), 'betas': list(BETAS), 'policies': list(POLICIES),
                    'inits': list(INITS), 'slots_per_seed': len(n),
                    'nominal_total': len(n) * len(SEEDS),
                    'main_nominal': sum(1 for s in n if s['block'] == 'main') * len(SEEDS),
                    'utility_only_continuations_nominal':
                        sum(1 for s in n if s['block'] != 'main') * len(SEEDS),
                    'slots': n},
        'track_E': {'removal_ranks': list(pj.REMOVAL_RANKS), 'channels': list(CHANNELS),
                    'slots_per_seed': len(e), 'nominal_total': len(e) * len(SEEDS),
                    'core_nominal': sum(1 for s in e if s['block'] == 'core') * len(SEEDS),
                    'controls_nominal': sum(1 for s in e if s['block'].startswith('control'))
                                        * len(SEEDS),
                    'diagnostic_nominal': sum(1 for s in e if s['block'] in
                                              ('full_span_diagnostic', 'ordinary_erasure_on_J'))
                                          * len(SEEDS),
                    'slots': e},
        'references': {'ref_A0': 'untouched A0 wire, must reproduce the historical matched '
                                 'primary audit exactly',
                       'ref_J': 'untouched J wire, must reproduce the historical matched '
                                'primary audit exactly'},
        'nominal_registry_total': per_seed * len(SEEDS),
        'audit_priority': {'1': 'main', '2': 'Track E k in {1,6} (second registered reduction)',
                           '3': 'Track N gamma = .01 (first registered reduction)'},
        'note': ('Nominal slots are not distinct learned systems. Exact reuse, duplicate '
                 'releases (for example step-0 selections identical to the untouched starting '
                 'channel, or constant full-span maps) and infeasible units are counted '
                 'separately after fitting.')}


MANIFEST_FILES = (
    'experiments/pcrl_competitive_method_v1/common.py',
    'experiments/pcrl_competitive_method_v1/diagnose.py',
    'experiments/pcrl_competitive_method_v1/projection.py',
    'experiments/pcrl_competitive_method_v1/run_fit_e.py',
    'experiments/pcrl_competitive_method_v1/run_fit_n.py',
    'experiments/pcrl_competitive_method_v1/run_dev.py',
    'experiments/pcrl_competitive_method_v1/freeze.py',
    'tests/pcrl_competitive_method_v1/test_projection_math.py',
    'experiments/pcrl_direct_adversarial_v1/train.py',
    'experiments/pcrl_direct_adversarial_v1/attackers.py',
    'experiments/pcrl_direct_adversarial_v1/channel.py',
    'experiments/pcrl_direct_adversarial_v1/inputs.py',
    'experiments/pcrl_direct_adversarial_v1/run_fit.py',
    'experiments/pcrl_direct_adversarial_v1/run_dev_2018.py',
    'experiments/pcrl_direct_adversarial_v1/report.py',
    'experiments/pcrl_invariant_baselines_v1/erasure_baselines.py',
    'pcrl/baselines/splince.py',
    'experiments/acs_spectral_audits.py',
    'experiments/acs_fixed_predictions_audits.py',
    'experiments/run_acs_residual_spectral.py',
    'scripts/report_acs_residual_spectral.py',
)
PROTOCOL_DOCS = ('PROTOCOL.md', 'METHOD.md', 'MATRIX.json', 'CORRECTIONS.md',
                 'SELECTION_DIAGNOSIS.md', 'CHECKPOINT_LEDGER.json')


def manifest() -> dict:
    import numpy, scipy, sklearn, torch
    files = {}
    for rel in MANIFEST_FILES:
        path = ROOT / rel
        files[rel] = sha_file(path) if path.exists() else None
    docs = {name: sha_file(OUT / name) for name in PROTOCOL_DOCS if (OUT / name).exists()}
    return {'generated_utc': utcnow(), 'predecessors': PREDECESSORS,
            'code': files, 'protocol_documents': docs,
            'versions': {'python': sys.version.split()[0], 'numpy': numpy.__version__,
                         'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
                         'torch': torch.__version__}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        stored = read_json(OUT / 'DEPENDENCY_MANIFEST.json')
        now = manifest()
        drift = {k: (stored['code'].get(k), v) for k, v in now['code'].items()
                 if stored['code'].get(k) != v}
        drift.update({k: (stored['protocol_documents'].get(k), v)
                      for k, v in now['protocol_documents'].items()
                      if stored['protocol_documents'].get(k) != v})
        print('DRIFT' if drift else 'NO DRIFT', drift)
        raise SystemExit(1 if drift else 0)
    write_json_atomic(OUT / 'MATRIX.json', matrix())
    write_json_atomic(OUT / 'DEPENDENCY_MANIFEST.json', manifest())
    m = read_json(OUT / 'MATRIX.json')
    print('N', m['track_N']['nominal_total'], 'E', m['track_E']['nominal_total'],
          'total', m['nominal_registry_total'])


if __name__ == '__main__':
    main()
