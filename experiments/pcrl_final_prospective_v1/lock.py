"""Build EVALUATION_LOCK.json after every fitting unit completed (before any final label).

The lock hashes each unit's registry, completion receipt and artifact manifest (which
itself hashes every fitted model and validation record), the scoring and inference
sources, the registration documents and the label-free 2016 feature caches.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from experiments.pcrl_task_directed_release_v1.audits import select_losses
from .audit_panel import all_units, sha_of, unit_dir
from .common import INPUTS, OUT, RESTORED, ROOT, atomic_json, now, read_json, sha
from .pipeline import root_for
from .releases import object_hashes

REGISTRATION = ('PROTOCOL.md', 'RELEASE_CONTRACT.md', 'PANEL.json', 'MODEL_MANIFEST.json', 'DATA_ADMISSION_PLAN.md',
                'ATTACK_RECIPES.json', 'UTILITY_RECIPES.json', 'PRIMARY_CLAIMS.json', 'SECONDARY_CONTRASTS.json',
                'RANDOMNESS_AND_LOSS.md', 'BOOTSTRAP_SPEC.json', 'RESOURCE_PLAN.json', 'PREDICTIONS.md')
CODE = ('audit_panel.py', 'common.py', 'inference_panel.py', 'pipeline.py', 'prepare.py', 'releases.py',
        'transport.py', 'lock.py')
PREDECESSOR = ('audits.py', 'evaluation.py', 'mechanisms.py', 'encoding.py', 'data.py', 'finite.py', 'baselines.py',
               'uncertainty.py', 'config.py')
SERVICE_CODE = ('acs_transfer_data.py', 'acs_transfer_heads.py')


def validate_unit(root, unit, dataset):
    """Verify a completed fit, every persisted artifact and its validation choice."""
    d = unit_dir(root, *unit)
    required = (d/'registry.joblib', d/'val_losses.npz', d/'ARTIFACTS.json', d/'COMPLETE.json')
    if any(not p.is_file() for p in required):
        raise ValueError(f'Incomplete fitting unit {unit}: missing {[str(p) for p in required if not p.is_file()]}')
    record = read_json(d/'COMPLETE.json')
    registry = joblib.load(d/'registry.joblib')
    for key, expected in zip(('release', 'anchor', 'role'), unit):
        if record.get(key) != expected or registry.get(key) != expected:
            raise ValueError(f'Fitting unit identity mismatch: {unit}, {key}')
    if record.get('dataset') != dataset or registry.get('dataset') != dataset:
        raise ValueError(f'Fitting dataset mismatch: {unit}')
    artifacts = read_json(d/'ARTIFACTS.json')
    if (record.get('registry_sha256') != sha(d/'registry.joblib')
            or record.get('artifacts_sha256') != sha_of(artifacts)
            or record.get('artifact_count') != len(artifacts)):
        raise ValueError(f'Fitting receipt mismatch: {unit}')
    actual = {str(p.relative_to(root)) for p in d.rglob('*') if p.is_file() and p.name not in ('ARTIFACTS.json', 'COMPLETE.json')}
    if set(artifacts) != actual:
        raise ValueError(f'Fitting artifact inventory mismatch: {unit}')
    files = {}
    for rel, expected in artifacts.items():
        p = Path(root)/rel
        if sha(p) != expected:
            raise ValueError(f'Fitting artifact hash mismatch: {unit}, {rel}')
        files[str(p.relative_to(ROOT))] = expected
    with np.load(d/'val_losses.npz') as z:
        losses = {str(k): z['losses'][:, j] for j, k in enumerate(z['candidate_ids'])}
        selection = select_losses(losses, z['weights'])
        independent_ids = [cid for cid, candidate in registry['candidates'].items()
                           if candidate.get('origin') == 'independent']
        independent = select_losses({cid: losses[cid] for cid in independent_ids}, z['weights'])
    if (record.get('selection') != selection['selection']
            or registry.get('selection') != selection['selection']
            or record.get('independent_selection') != independent['selection']
            or registry.get('independent_selection') != independent['selection']):
        raise ValueError(f'Validation selection mismatch: {unit}')
    for cid, score in selection['scores'].items():
        old = registry['validation_scores'].get(cid, {})
        if any(k not in old or not np.isfinite(old[k]) or abs(score[k]-old[k]) > 1e-12
               for k in ('unweighted', 'weighted', 'balanced')):
            raise ValueError(f'Validation score mismatch: {unit}, {cid}')
    files[str((d/'COMPLETE.json').relative_to(ROOT))] = sha(d/'COMPLETE.json')
    files[str((d/'ARTIFACTS.json').relative_to(ROOT))] = sha(d/'ARTIFACTS.json')
    return files


def build(dataset='acs2016'):
    root = root_for(dataset)
    data = OUT/'private/data'/dataset
    if (data/'labels_final.npz').exists() or list(root.glob('units/**/SCORED_final.json')):
        raise SystemExit('Final labels or scores already exist; cannot claim a pre-scoring lock')
    files = {}
    for u in all_units():
        files.update(validate_unit(root, u, dataset))
    for name in REGISTRATION:
        files[str((OUT/name).relative_to(ROOT))] = sha(OUT/name)
    for name in CODE:
        p = ROOT/'experiments/pcrl_final_prospective_v1'/name
        files[str(p.relative_to(ROOT))] = sha(p)
    for name in PREDECESSOR:
        p = ROOT/'experiments/pcrl_task_directed_release_v1'/name
        files[str(p.relative_to(ROOT))] = sha(p)
    for name in SERVICE_CODE:
        p = ROOT/'experiments'/name
        files[str(p.relative_to(ROOT))] = sha(p)
    staged = INPUTS/'STAGED_INPUTS.json'
    files[str(staged.relative_to(ROOT))] = sha(staged)
    for item in read_json(staged)['files']:
        p = INPUTS/item['path']
        if sha(p) != item['sha256']:
            raise ValueError(f'Staged input changed: {item["path"]}')
        files[str(p.relative_to(ROOT))] = item['sha256']
    for anchor in (0, 1, 2):
        for rel, digest in object_hashes(anchor).items():
            p = RESTORED/rel
            files[str(p.relative_to(ROOT))] = digest
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
