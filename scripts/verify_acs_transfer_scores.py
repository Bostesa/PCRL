"""Independently replay ACS scores from saved predictions and original raw rows.

Requires the original local CSV, test_predictions.npz and split_rows.npz files,
plus committed metrics and selections. No fitting or experiment scorer imports.
Default output is stdout; --report exclusively creates a fresh JSON file.
See PORTABLE_VERIFIERS.md alongside the original ACS result evidence for the
relationship between this portable wrapper and the historically executed script.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support, roc_auc_score

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                    help='Repository root used to resolve the configured raw-data path')
parser.add_argument('--out', type=Path, help='Existing ACS transfer result directory')
parser.add_argument('--report', type=Path, help='Fresh report destination; omitted means stdout')
args = parser.parse_args()
ROOT = args.root.resolve()
OUT = args.out.resolve() if args.out is not None else ROOT/'results/redesign_20260907_acs_transfer_v1'
if args.report is not None and args.report.exists():
    raise FileExistsError('Report already exists; preserve original evidence and choose a fresh path')
TOL = 2e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def finite(value):
    return None if not np.isfinite(value) else float(value)


def labels(frame, target):
    if target == 'same_residence':
        original = frame.MIG.to_numpy(float)
        valid = np.isin(original, [1, 2, 3])
        y = (original == 1).astype(int)
    elif target == 'commute_over20':
        original = frame.JWMNP.to_numpy(float)
        valid = np.isfinite(original) & (original >= 1) & (original <= 200) & (original == np.floor(original))
        y = (original > 20).astype(int)
    else:
        original = frame[target].to_numpy(float)
        valid = np.isin(original, range(1, 3 if target == 'SEX' else 10))
        y = np.full(len(frame), -1, dtype=int)
        y[valid] = original[valid].astype(int)-1
    return y, valid


def independent_scores(y, probability, k, weight=None):
    p = np.asarray(probability, dtype=float)
    assert p.shape == (len(y), k)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1+1e-8).all()
    assert np.allclose(p.sum(1), 1., atol=1e-6, rtol=0)
    predicted = p.argmax(1)
    classes = np.arange(k)
    weights = np.ones(len(y)) if weight is None else weight
    counts = confusion_matrix(y, predicted, labels=classes)
    confusion = confusion_matrix(y, predicted, labels=classes, sample_weight=weights)
    support, guessed = counts.sum(1), counts.sum(0)
    weighted_support, weighted_guessed = confusion.sum(1), confusion.sum(0)
    total = confusion.sum()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, labels=classes, sample_weight=weights, zero_division=np.nan)
    aucs = []
    per_class = []
    for category in classes:
        auc = None
        if weighted_support[category] > 0 and weighted_support[category] < total:
            auc = float(roc_auc_score(y == category, p[:, category], sample_weight=weights))
        aucs.append(auc)
        per_class.append({'class_index': int(category), 'support': int(support[category]),
                          'weighted_support': float(weighted_support[category]),
                          'prevalence': float(weighted_support[category]/total),
                          'predicted_support': int(guessed[category]),
                          'predicted_weight': float(weighted_guessed[category]),
                          'precision': finite(precision[category]), 'recall': finite(recall[category]),
                          'f1': finite(f1[category]), 'auroc': auc,
                          'recall_defined': bool(np.isfinite(recall[category])), 'auroc_defined': auc is not None})
    complete = bool((weighted_support > 0).all())
    usable_auc = [a for a in aucs if a is not None]
    macro_auc = float(np.mean(usable_auc)) if len(usable_auc) == k else None
    clipped = np.maximum(p, 1e-12)
    clipped /= clipped.sum(1, keepdims=True)
    return {'n': len(y), 'n_classes': k, 'class_schema': classes.tolist(),
            'weighted': weight is not None, 'weight_sum': float(total),
            'support': support.tolist(), 'weighted_support': weighted_support.tolist(),
            'prevalence': (weighted_support/total).tolist(), 'coverage_complete': complete,
            'raw_coverage_complete': bool((support > 0).all()), 'valid_class_mask': (weighted_support > 0).tolist(),
            'log_loss': float(log_loss(y, clipped, labels=classes, sample_weight=weights)),
            'log_loss_probability_floor': 1e-12,
            'accuracy': float(np.trace(confusion)/total),
            'balanced_accuracy': float(np.mean(recall)) if complete else None,
            'observed_balanced_accuracy': float(np.nanmean(recall)),
            'auroc': aucs[1] if k == 2 else macro_auc, 'macro_auroc': macro_auc,
            'observed_macro_auroc': float(np.mean(usable_auc)) if usable_auc else None,
            'per_class': per_class}


errors = []
max_error = 0.
max_error_path = None
numeric_comparisons = 0


def compare(expected, actual, path):
    global max_error, max_error_path, numeric_comparisons
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            errors.append({'path': path, 'different_keys': [sorted(expected), sorted(actual)]})
            return
        for key in expected:
            compare(expected[key], actual[key], f'{path}/{key}')
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            errors.append({'path': path, 'different_lengths': [len(expected), len(actual)]})
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f'{path}/{index}')
    elif expected is None or isinstance(expected, (bool, str)):
        if expected != actual:
            errors.append({'path': path, 'expected': expected, 'actual': actual})
    else:
        if actual is None:
            errors.append({'path': path, 'expected': expected, 'actual': actual})
            return
        delta = abs(float(expected)-float(actual))
        numeric_comparisons += 1
        if delta > max_error:
            max_error, max_error_path = delta, path
        if delta > TOL:
            errors.append({'path': path, 'expected': expected, 'actual': actual, 'absolute_error': delta})


tick = time.perf_counter()
config = json.loads((OUT/'config.json').read_text())
raw_path = ROOT/config['raw_path']
raw = pd.read_csv(raw_path, usecols=['MIG', 'JWMNP', 'SEX', 'RAC1P', 'PWGTP', 'SERIALNO', 'SPORDER'],
                  dtype={'SERIALNO': str})
result = {
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'Independent saved-prediction metric replay; no training, model selection changes, or new test data.',
    'method': 'Reload original CSV using saved raw test-row indices. Recode MIG==1, valid integer JWMNP>20, SEX-1 and RAC1P-1 independently. Use sklearn log_loss/confusion_matrix/precision_recall_fscore_support/roc_auc_score; no experiment scorer imports. Repeat with the same predictions and original PWGTP.',
    'absolute_tolerance': TOL, 'raw_sha256': sha(raw_path), 'script_sha256': sha(__file__),
    'candidate_records_replayed': 0, 'test_score_sets_replayed': 0,
    'family_selections_checked': 0, 'mlp_epoch_selections_checked': 0,
    'fit_target_hashes_checked': 0, 'mlp_schedule_groups_checked': 0,
    'seeds': {}, 'warnings': [], 'fallbacks': [], 'errors': errors,
}
for seed in config['seeds']:
    directory = OUT/f'seed_{seed}'
    measured = json.loads((directory/'metrics.json').read_text())
    selection = json.loads((directory/'selection_before_test.json').read_text())
    with np.load(directory/'split_rows.npz') as saved:
        split_rows = {name: saved[name].copy() for name in saved.files}
    test = raw.iloc[split_rows['test']]
    coverage, controls = {}, {}
    per_seed_error_start = len(errors)
    with np.load(directory/'test_predictions.npz') as saved:
        assert len(saved.files) == len(measured['raw_metrics'])
        for row in measured['raw_metrics']:
            role, release, target, family = (row[key] for key in ('role', 'release', 'target', 'family'))
            key = '/'.join((role, release, target))
            probability = saved[f'{key}/{family}']
            full_y, valid = labels(test, target)
            y, weights = full_y[valid], test.PWGTP.to_numpy(float)[valid]
            k = 9 if target == 'RAC1P' else 2
            for metric_name, weight in [('test', None), ('test_person_weighted', weights)]:
                independent = independent_scores(y, probability, k, weight)
                compare(independent, row[metric_name], f'seed_{seed}/{key}/{family}/{metric_name}')
                result['test_score_sets_replayed'] += 1
            assert row['selected'] == (selection['head_selections'][key] == family)
            if target == 'RAC1P':
                coverage = {name: row['test'][name] for name in ('support', 'coverage_complete', 'balanced_accuracy', 'macro_auroc')}
                # Macro values depend on the family; only common support/flags retained below.
                coverage.pop('balanced_accuracy'); coverage.pop('macro_auroc')
            if release in ('prior', 'exposed'):
                controls[f'{role}/{release}/{target}/{family}'] = {
                    'selected': row['selected'], 'log_loss': row['test']['log_loss'],
                    'accuracy': row['test']['accuracy'], 'balanced_accuracy': row['test']['balanced_accuracy'],
                    'observed_balanced_accuracy': row['test']['observed_balanced_accuracy'],
                    'per_class_recall': [v['recall'] for v in row['test']['per_class']],
                    'coverage_complete': row['test']['coverage_complete']}
            result['candidate_records_replayed'] += 1
    group_schedules = {}
    for key, record in selection['fitting_records'].items():
        selected = selection['head_selections'][key]
        candidates = record['candidates']
        minimum = min(candidates, key=lambda f: (candidates[f]['validation_scores']['log_loss'], f))
        assert selected == minimum, (seed, key, selected, minimum)
        result['family_selections_checked'] += 1
        role, release, target = key.split('/')
        pool = 'downstream_fit' if role == 'transfer' else 'attacker_fit'
        full_y, valid = labels(raw.iloc[split_rows[pool]], target)
        target_order = ['same_residence', 'commute_over20'] if role == 'transfer' else ['SEX', 'RAC1P']
        seed_base = 1230000 if role == 'transfer' else 1240000
        limit = config['head_budget'] if role == 'transfer' else config['attacker_budget']
        idx = np.random.default_rng(seed_base+100*seed+target_order.index(target)).permutation(np.flatnonzero(valid))[:limit]
        expected_hash = array_hash(full_y[idx].astype(np.int64))
        for family, candidate in candidates.items():
            actual_hash = candidate.get('fit_hashes', {}).get('y', candidate.get('fit_label_hash'))
            assert expected_hash == actual_hash, (seed, key, family, 'fit target hash mismatch')
            result['fit_target_hashes_checked'] += 1
            if candidate.get('fit_warnings'):
                result['warnings'].append({'seed': seed, 'key': key, 'family': family, 'warnings': candidate['fit_warnings']})
            if 'fallback_reason' in candidate:
                result['fallbacks'].append({'seed': seed, 'key': key, 'family': family, 'reason': candidate['fallback_reason']})
            if family == 'mlp' and 'validation_curve' in candidate:
                best = min(candidate['validation_curve'], key=lambda v: (v['validation_log_loss'], v['epoch']))
                assert candidate['selected_epoch'] == best['epoch']
                assert candidate['selected_optimizer_steps'] == best['optimizer_steps']
                assert candidate['validation_scores']['log_loss'] == best['validation_log_loss']
                assert candidate['validation_curve'][0]['optimizer_steps'] == 0
                epochs = candidate['parameters']['epochs']
                assert candidate['training_row_exposures'] == len(idx)*epochs
                assert candidate['optimizer_steps'] == int(np.ceil(len(idx)/candidate['parameters']['batch_size']))*epochs
                group_schedules.setdefault((role, target), set()).add((candidate['schedule_hash'], candidate['optimizer_steps'], candidate['training_row_exposures']))
                result['mlp_epoch_selections_checked'] += 1
    for group, schedules in group_schedules.items():
        assert len(schedules) == 1, (seed, group, 'unmatched schedules')
        result['mlp_schedule_groups_checked'] += 1
    race_fit, race_valid = labels(raw.iloc[split_rows['attacker_fit']], 'RAC1P')
    result['seeds'][str(seed)] = {
        'test_rows': len(test), 'prediction_archive_sha256': sha(directory/'test_predictions.npz'),
        'split_rows_sha256': sha(directory/'split_rows.npz'),
        'metrics_sha256': sha(directory/'metrics.json'),
        'selection_sha256': sha(directory/'selection_before_test.json'),
        'race_test_coverage': coverage,
        'race_attacker_fit_support': np.bincount(race_fit[race_valid], minlength=9).tolist(),
        'controls': controls, 'new_errors': len(errors)-per_seed_error_start,
    }
result['numeric_values_compared'] = numeric_comparisons
result['max_absolute_metric_error'] = max_error
result['max_error_path'] = max_error_path
result['passed'] = not errors
result['runtime_seconds'] = time.perf_counter()-tick
rendered = json.dumps(result, indent=2, allow_nan=False)+'\n'
if args.report is not None:
    with args.report.open('x') as handle:
        handle.write(rendered)
else:
    print(rendered, end='')
if errors:
    raise SystemExit(1)
