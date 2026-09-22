"""Verify final score receipts before inference; preserve failed/partial evidence."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import joblib
import numpy as np

from experiments.pcrl_task_directed_release_v1.audits import expected_token_loss, loss_scores
from .audit_panel import all_units, unit_dir
from .common import array_hash, atomic_json, now, sha
from .pipeline import root_for


def check_one(root, unit):
    d = unit_dir(root, *unit)
    marker, score, prediction = (d/name for name in ('SCORED_final.json', 'score_final.npz', 'pred_final.npz'))
    if not marker.is_file() or not score.is_file() or not prediction.is_file():
        raise ValueError('Incomplete final score artifact set')
    from .common import read_json
    receipt = read_json(marker)
    registry = joblib.load(d/'registry.joblib')
    if (receipt.get('release'), receipt.get('anchor'), receipt.get('role')) != unit:
        raise ValueError('Score receipt unit identity differs')
    if receipt.get('selection') != registry['selection']:
        raise ValueError('Final score selection differs from locked validation choice')
    if receipt.get('score_sha256') != sha(score) or receipt.get('prediction_sha256') != sha(prediction):
        raise ValueError('Final score or prediction file hash differs')
    with np.load(score) as s, np.load(prediction) as p:
        q, law = p['probabilities'], p['token_probs']
        recomputed = expected_token_loss(q, law, s['y'])
        if not np.allclose(recomputed, s['loss'], atol=1e-12, rtol=0):
            raise ValueError('Expected token loss differs from persisted score')
        if (receipt.get('rows') != len(s['ids']) or receipt.get('loss_hash') != array_hash(s['loss'])
                or receipt.get('prediction_hash') != array_hash(q)
                or receipt.get('token_hash') != array_hash(law)):
            raise ValueError('Final score array identity differs')
        scores = loss_scores(s['loss'], s['weights'])
    if any(abs(scores[k]-receipt['ce'][k]) > 1e-12 for k in scores):
        raise ValueError('Final score aggregate differs')
    return {'unit': list(unit), 'rows': receipt['rows'], 'selection': receipt['selection'],
            'score_sha256': receipt['score_sha256'], 'prediction_sha256': receipt['prediction_sha256']}


def inspect(root, *, quarantine=False):
    root = Path(root)
    valid, missing, invalid, quarantined = [], [], {}, []
    for unit in all_units():
        d = unit_dir(root, *unit)
        files = [d/name for name in ('SCORED_final.json', 'score_final.npz', 'pred_final.npz')]
        files += list(d.glob('score_final.*.tmp*')) + list(d.glob('pred_final.*.tmp*'))
        if not any(p.exists() for p in files):
            missing.append(list(unit))
            continue
        try:
            valid.append(check_one(root, unit))
        except Exception as error:
            key = '|'.join(map(str, unit))
            invalid[key] = repr(error)
            if quarantine:
                target = root/'quarantine'/'score'/f'{unit[0]}-{unit[1]}-{unit[2].replace(":", "_").replace("/", "_")}-{time.time_ns()}'
                target.mkdir(parents=True)
                original = {p.name: sha(p) for p in files if p.is_file()}
                for p in files:
                    if p.is_file():
                        os.replace(p, target/p.name)
                atomic_json(target/'FAILURE.json', {'created_utc': now(), 'unit': list(unit),
                                                     'error': repr(error), 'original_hashes': original})
                quarantined.append(str(target))
    return {'created_utc': now(), 'expected': len(all_units()), 'valid': len(valid),
            'missing': len(missing), 'invalid': invalid, 'quarantined': quarantined,
            'valid_units': valid, 'missing_units': missing}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='acs2016')
    ap.add_argument('--quarantine', action='store_true')
    ap.add_argument('--require-complete', action='store_true')
    args = ap.parse_args()
    record = inspect(root_for(args.dataset), quarantine=args.quarantine)
    from .common import OUT
    atomic_json(OUT/'private'/'SCORE_INTEGRITY.json', record)
    print({k: record[k] for k in ('expected', 'valid', 'missing', 'invalid', 'quarantined')})
    if args.require_complete and (record['valid'] != record['expected'] or record['invalid']):
        raise SystemExit(1)
