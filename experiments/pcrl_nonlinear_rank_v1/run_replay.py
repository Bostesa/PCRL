"""Independent replay: recompute every stored prediction in a separate process.

The completed transport study found rare, load-dependent corrupted blocks in CPU
prediction outputs under extreme machine memory pressure (its Lock amendment 2).
Memory pressure remains a SUSPECTED contributing condition, not a demonstrated
root cause. This machine is again under heavy swap pressure, so every new
prediction array is recomputed here from the fitted candidate and the release,
in a process that did not write it, and must match bitwise.

A surviving mismatch quarantines the unit: both versions and the logs are kept,
the unit is invalidated, and unaffected work continues. Nothing is averaged or
majority-voted into a report.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import joblib
import numpy as np

from experiments import acs_fixed_predictions_audits as oldaudit
from experiments import acs_spectral_audits as audit
from experiments.acs_transfer_data import array_hash
from experiments.run_acs_residual_spectral import load_labels, wires

from .inputs import HIST_ROOT, OUT, read_json, sha_file, write_json
from .maps import HISTORICAL_ALIAS
from .run_fit import limit_threads, machine_state


def replay_unit(out: Path, seed: int, condition: str, labels, weights) -> dict:
    dest = out / f'seed_{seed}' / condition
    stored = dict(np.load(dest / 'predictions.npz'))
    w, d = wires(out / f'seed_{seed}' / 'releases' / condition / 'releases.npz')
    u = joblib.load(dest / 'utility_state.joblib')
    attacks = audit.load_audits(dest / 'audits')

    checked, mismatches, cache = 0, [], {}
    def compare(role, view, target, budget, cid, candidate):
        nonlocal checked
        for split in ('validation', 'test'):
            pool = (('downstream_validation' if role == 'utility' else 'attacker_validation')
                    if split == 'validation' else 'test')
            y = labels[pool][target]
            valid = y >= 0
            key = f'{role}/{view}/{target}/{budget}/{cid}/{split}'
            if key not in stored:
                mismatches.append({'key': key, 'reason': 'missing_from_stored_predictions'})
                continue
            if role == 'audit':
                x = w[view][pool][valid] if candidate.space == 'wire' else d[view][pool][valid]
                if candidate.columns is not None:
                    x = x[:, candidate.columns]
                x = np.ascontiguousarray(x)
                ident = (candidate.metadata['base_candidate_directory'], pool, array_hash(x))
                if ident not in cache:
                    cache[ident] = candidate.base.predict_proba(x)
                prediction = cache[ident]
            else:
                prediction = candidate.predict_proba(w[view][pool][valid])
            checked += 1
            if not np.array_equal(np.asarray(prediction, float), np.asarray(stored[key], float)):
                a = np.asarray(prediction, float)
                b = np.asarray(stored[key], float)
                mismatches.append({
                    'key': key, 'reason': 'value_mismatch',
                    'max_abs_difference': float(np.max(np.abs(a - b))) if a.shape == b.shape else None,
                    'differing_rows': int((np.abs(a - b) > 0).any(axis=1).sum()) if a.shape == b.shape else None,
                    'shape_replay': list(a.shape), 'shape_stored': list(b.shape)})

    for role, candidates in u['candidates'].items():
        view, target = role.split('/')
        for cid, candidate in candidates.items():
            compare('utility', view, target, None, cid, candidate)
    for budget, roles in attacks['candidates'].items():
        for role, candidates in roles.items():
            view, target = role.split('/')
            for cid, candidate in candidates.items():
                compare('audit', view, target, budget, cid, candidate)

    record = {'seed': seed, 'condition': condition, 'predictions_checked': checked,
              'stored_arrays': len(stored), 'mismatches': mismatches,
              'clean': not mismatches,
              'predictions_sha256': sha_file(dest / 'predictions.npz')}
    if mismatches:
        quarantine = dest.with_name(condition + f'__quarantined_{time.time_ns()}')
        shutil.copytree(dest, quarantine)
        write_json(quarantine / 'QUARANTINE.json', {
            'reason': 'independent replay disagreed with stored predictions',
            'machine': machine_state(), 'record': record,
            'disposition': 'unit invalidated; both versions retained; no averaging or voting'})
        (dest / 'complete.json').rename(dest / 'complete.invalidated.json')
        record['quarantine'] = str(quarantine)
    return record


def run(out: Path = OUT, seeds=(0, 1, 2), conditions=None) -> dict:
    limit_threads()
    out = Path(out)
    summary, clean, total = {}, 0, 0
    for seed in seeds:
        _frame, _pools, labels, weights = load_labels(HIST_ROOT, seed)
        names = conditions or read_json(out / f'seed_{seed}' / 'fit_complete.json')['conditions']
        rows = []
        for condition in names:
            if condition in HISTORICAL_ALIAS:
                continue          # frozen historical unit; this study wrote no prediction for it
            if not (out / f'seed_{seed}' / condition / 'complete.json').exists():
                continue
            tick = time.perf_counter()
            record = replay_unit(out, seed, condition, labels, weights)
            record['runtime_seconds'] = time.perf_counter() - tick
            rows.append(record)
            total += record['predictions_checked']
            clean += 0 if record['mismatches'] else record['predictions_checked']
            print('REPLAY', seed, condition, record['predictions_checked'],
                  'clean' if record['clean'] else f'MISMATCH {len(record["mismatches"])}', flush=True)
        summary[str(seed)] = rows
    result = {'seeds': summary, 'predictions_checked': total, 'predictions_clean': clean,
              'all_clean': clean == total, 'machine': machine_state(),
              'purpose': ('detect the load-dependent block corruption previously observed under '
                          'extreme memory pressure; memory pressure is a suspected contributing '
                          'condition, not a demonstrated root cause')}
    write_json(out / 'PREDICTION_REPLAY.json', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--only', nargs='+')
    a = p.parse_args()
    r = run(a.out, tuple(a.seeds), a.only)
    print('REPLAY_TOTAL', r['predictions_checked'], 'all_clean', r['all_clean'])


if __name__ == '__main__':
    main()
