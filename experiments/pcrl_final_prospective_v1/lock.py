"""Build EVALUATION_LOCK.json after every fitting unit completed (before any final label).

The lock hashes each unit's registry, completion receipt and artifact manifest (which
itself hashes every fitted model and validation record), the scoring and inference
sources, the registration documents and the label-free 2016 feature caches.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from .audit_panel import all_units, unit_dir
from .common import OUT, ROOT, atomic_json, now, sha
from .pipeline import root_for

REGISTRATION = ('PROTOCOL.md', 'RELEASE_CONTRACT.md', 'PANEL.json', 'MODEL_MANIFEST.json', 'DATA_ADMISSION_PLAN.md',
                'ATTACK_RECIPES.json', 'UTILITY_RECIPES.json', 'PRIMARY_CLAIMS.json', 'SECONDARY_CONTRASTS.json',
                'RANDOMNESS_AND_LOSS.md', 'BOOTSTRAP_SPEC.json', 'RESOURCE_PLAN.json', 'PREDICTIONS.md')
CODE = ('audit_panel.py', 'common.py', 'inference_panel.py', 'pipeline.py', 'prepare.py', 'releases.py', 'transport.py')
PREDECESSOR = ('audits.py', 'evaluation.py', 'mechanisms.py', 'encoding.py', 'data.py', 'finite.py', 'baselines.py',
               'uncertainty.py', 'config.py')


def build(dataset='acs2016'):
    root = root_for(dataset)
    files, missing = {}, []
    for u in all_units():
        d = unit_dir(root, *u)
        for name in ('registry.joblib', 'COMPLETE.json', 'ARTIFACTS.json'):
            p = d/name
            if not p.exists():
                missing.append(str(p))
            else:
                files[str(p.relative_to(ROOT))] = sha(p)
    if missing:
        raise SystemExit(f'Incomplete fitting: {len(missing)} missing unit files')
    for name in REGISTRATION:
        files[str((OUT/name).relative_to(ROOT))] = sha(OUT/name)
    for name in CODE:
        p = ROOT/'experiments/pcrl_final_prospective_v1'/name
        files[str(p.relative_to(ROOT))] = sha(p)
    for name in PREDECESSOR:
        p = ROOT/'experiments/pcrl_task_directed_release_v1'/name
        files[str(p.relative_to(ROOT))] = sha(p)
    data = OUT/'private/data'/dataset
    for p in [data/'cohort.npz', data/'PREPARED.json', *sorted(data.glob('anchor_*/features.npz')),
              data/'labels_fit.npz', data/'labels_attack_val.npz', data/'labels_task_val.npz']:
        files[str(p.relative_to(ROOT))] = sha(p)
    selections = {'|'.join(map(str, u)): json.loads((unit_dir(root, *u)/'COMPLETE.json').read_text())['selection']
                  for u in all_units()}
    lock = {'created_utc': now(), 'dataset': dataset, 'units': len(all_units()), 'files': files,
            'selections': selections, 'final_labels_read_before_lock': False,
            'rule': 'final-partition labels are readable only while every hash below verifies'}
    if (OUT/'EVALUATION_LOCK.json').exists():
        raise SystemExit('Lock exists; a second lock would reopen selection')
    atomic_json(OUT/'EVALUATION_LOCK.json', lock)
    return lock


if __name__ == '__main__':
    lock = build(sys.argv[1] if len(sys.argv) > 1 else 'acs2016')
    print(len(lock['files']), 'files locked')
