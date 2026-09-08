"""Replay ACS protection scores from frozen saved predictions; no fitting.

Checks five utility labels and SEX/RAC1P against original raw rows, validation
and reused development-evaluation predictions, and PWGTP sensitivity. Also
checks primary/family/AUROC choices, MLP epoch choices and matched fit schedules.
--report creates a fresh file exclusively; default JSON output goes to stdout.
Requires local raw data and cached predictions, plus the published records.
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
parser.add_argument('--out', type=Path, help='Existing ACS protection result directory')
parser.add_argument('--report', type=Path, help='Fresh report destination; omitted means stdout')
parser.add_argument('--seeds', type=int, nargs='+', help='Declared completed seeds; default all configured seeds')
args = parser.parse_args()
ROOT = args.root.resolve()
OUT = args.out.resolve() if args.out is not None else ROOT/'results/redesign_20260908_acs_protection_v1'
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
    elif target in ('income_binary', 'civilian_at_work', 'public_coverage'):
        column = {'income_binary': 'PINCP', 'civilian_at_work': 'ESR', 'public_coverage': 'PUBCOV'}[target]
        original = frame[column].to_numpy(float)
        if target == 'income_binary':
            valid = np.isfinite(original) & (original >= -19998) & (original <= 4209995)
            y = (original > 50000).astype(int)
        else:
            valid = np.isin(original, range(1, 7) if target == 'civilian_at_work' else [1, 2])
            y = (original == 1).astype(int)
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
PARENT = ROOT/config['parent_results']
parent_config = json.loads((PARENT/'config.json').read_text())
selected_seeds = config['seeds'] if args.seeds is None else args.seeds
if (not selected_seeds or len(set(selected_seeds)) != len(selected_seeds)
        or set(selected_seeds)-set(config['seeds'])):
    raise ValueError('Use a nonempty subset of declared seeds without duplicates')
freeze = json.loads((OUT/'protocol_freeze.json').read_text())
for path, expected in freeze['sha256'].items():
    if sha(ROOT/path) != expected:
        raise ValueError('Frozen execution/config/protocol source changed: '+path)
for path, expected in freeze['parent_record_hashes'].items():
    if sha(ROOT/path) != expected:
        raise ValueError('Recorded historical parent changed: '+path)
raw_path = ROOT/parent_config['raw_path']
raw_hash = sha(raw_path)
if raw_hash != json.loads((PARENT/'schema_support.json').read_text())['raw_sha256']:
    raise ValueError('Raw CSV differs from the original source')
raw = pd.read_csv(raw_path, usecols=['MIG', 'JWMNP', 'PINCP', 'ESR', 'PUBCOV',
                                    'SEX', 'RAC1P', 'PWGTP', 'SERIALNO', 'SPORDER'],
                  dtype={'SERIALNO': str})
result = {
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'Independent saved-prediction metric replay. No fitting, source/map updates, selection changes, or new evaluation data.',
    'evaluation_status': config['evaluation_status'],
    'method': 'Independent raw-column masks/labels; sklearn log_loss, confusion_matrix, precision_recall_fscore_support and roc_auc_score. No experiment fitter/scorer imports. Numerical probability floor1e-12 before log-loss renormalization. Identical probabilities reused for PWGTP sensitivity.',
    'absolute_tolerance': TOL, 'raw_sha256': raw_hash, 'script_sha256': sha(__file__),
    'frozen_source_entries_checked': len(freeze['sha256']),
    'historical_parent_records_checked': len(freeze['parent_record_hashes']),
    'candidate_records_replayed': 0, 'score_sets_replayed': 0,
    'primary_selections_checked': 0, 'family_selections_checked': 0,
    'auroc_selections_checked': 0, 'mlp_epoch_selections_checked': 0,
    'fit_target_hashes_checked': 0, 'validation_target_hashes_checked': 0,
    'matched_mlp_schedule_groups_checked': 0, 'seeds': {},
    'warnings': [], 'fallbacks': [], 'errors': errors,
}
for seed in selected_seeds:
    directory = OUT/f'seed_{seed}'
    measured = json.loads((directory/'metrics.json').read_text())
    selection = json.loads((directory/'selection_before_test.json').read_text())
    with np.load(directory/'split_rows.npz') as saved:
        split_rows = {name: saved[name].copy() for name in saved.files}
    with np.load(PARENT/f'seed_{seed}/split_rows.npz') as saved:
        for name, indices in split_rows.items():
            assert np.array_equal(indices, saved[name]), (seed, name, 'historical row alignment')
    assert sha(directory/'release_freeze.json') == selection['release_freeze_sha256']
    assert sha(OUT/'protocol_freeze.json') == selection['protocol_freeze_sha256']
    assert sha(directory/'selection_before_test.json') == measured['integrity']['selection_sha256']
    assert selection['created_utc'] <= measured['integrity']['evaluation_started_utc']
    data = {name: raw.iloc[indices] for name, indices in split_rows.items()}
    label_cache = {}
    def scoring_data(split, role, target):
        key = split, role, target
        if key not in label_cache:
            pool = 'test' if split == 'test' else ('downstream_validation' if role == 'transfer' else 'attacker_validation')
            complete_y, valid = labels(data[pool], target)
            label_cache[key] = complete_y[valid], data[pool].PWGTP.to_numpy(float)[valid]
        return label_cache[key]
    controls, coverage = {}, {}
    with np.load(directory/'predictions.npz') as saved:
        assert len(saved.files) == 2*len(measured['raw_metrics'])
        for row in measured['raw_metrics']:
            role, release, target, candidate_id = (row[key] for key in ('role', 'release', 'target', 'candidate_id'))
            key = '/'.join((role, release, target))
            k = 9 if target == 'RAC1P' else 2
            for split in ('validation', 'test'):
                y, weights = scoring_data(split, role, target)
                probability = saved[f'{split}/{key}/{candidate_id}']
                for suffix, weight in [('', None), ('_person_weighted', weights)]:
                    independent = independent_scores(y, probability, k, weight)
                    compare(independent, row[split+suffix], f'seed_{seed}/{key}/{candidate_id}/{split}{suffix}')
                    result['score_sets_replayed'] += 1
            assert row['selected'] == (selection['head_selections'][key] == candidate_id)
            assert row['selected_within_family'] == (selection['family_selections'][key][row['family']] == candidate_id)
            assert row['auc_selected'] == (selection['auroc_selections'][key] == candidate_id)
            stored_fit_score = selection['fitting_records'][key]['candidates'][candidate_id]['validation_scores']
            compare(stored_fit_score, row['validation'], f'seed_{seed}/{key}/{candidate_id}/fitting_validation_replay')
            if target == 'RAC1P':
                coverage = {split: {name: row[split][name] for name in ('support', 'coverage_complete')}
                            for split in ('validation', 'test')}
            if release in ('prior', 'exposed'):
                controls[f'{role}/{release}/{target}/{candidate_id}'] = {
                    'selected': row['selected'], 'auc_selected': row['auc_selected'],
                    'log_loss': row['test']['log_loss'], 'accuracy': row['test']['accuracy'],
                    'balanced_accuracy': row['test']['balanced_accuracy'],
                    'observed_balanced_accuracy': row['test']['observed_balanced_accuracy'],
                    'per_class_recall': [v['recall'] for v in row['test']['per_class']],
                    'coverage_complete': row['test']['coverage_complete']}
            result['candidate_records_replayed'] += 1
    schedule_groups = {}
    for key, record in selection['fitting_records'].items():
        candidates = record['candidates']
        best = min(candidates, key=lambda c: (candidates[c]['validation_scores']['log_loss'], c))
        assert selection['head_selections'][key] == best, (seed, key, 'primary selection')
        result['primary_selections_checked'] += 1
        for family, actual in selection['family_selections'][key].items():
            matching = [c for c in candidates if candidates[c]['family'] == family]
            best_family = min(matching, key=lambda c: (candidates[c]['validation_scores']['log_loss'], c))
            assert actual == best_family, (seed, key, family, 'within-family selection')
            result['family_selections_checked'] += 1
        role, release, target = key.split('/')
        if role == 'audit' and release != 'prior':
            valid_auc = [c for c in candidates if candidates[c]['validation_scores']['auroc'] is not None
                         and np.isfinite(candidates[c]['validation_scores']['auroc'])]
            best_auc = min(valid_auc, key=lambda c: (-candidates[c]['validation_scores']['auroc'], c)) if valid_auc else None
            assert selection['auroc_selections'][key] == best_auc, (seed, key, 'diagnostic AUROC choice')
            result['auroc_selections_checked'] += 1
        else:
            assert selection['auroc_selections'][key] is None
        pool = 'downstream_fit' if role == 'transfer' else 'attacker_fit'
        validation_pool = 'downstream_validation' if role == 'transfer' else 'attacker_validation'
        fit_y, valid = labels(data[pool], target)
        target_order = config['utility_tasks'] if role == 'transfer' else ['SEX', 'RAC1P']
        seed_base = 1230000 if role == 'transfer' else 1240000
        limit = config['head_budget'] if role == 'transfer' else config['attacker_budget']
        idx = np.random.default_rng(seed_base+100*seed+target_order.index(target)).permutation(np.flatnonzero(valid))[:limit]
        expected_hash = array_hash(fit_y[idx].astype(np.int64))
        val_y, val_valid = labels(data[validation_pool], target)
        expected_val_hash = array_hash(val_y[val_valid].astype(np.int64))
        for candidate_id, candidate in candidates.items():
            assert expected_hash == candidate.get('fit_hashes', {}).get('y', candidate.get('fit_label_hash'))
            result['fit_target_hashes_checked'] += 1
            if candidate['family'] != 'prior':
                assert expected_val_hash == candidate['validation_hashes']['y']
                result['validation_target_hashes_checked'] += 1
            if candidate.get('fit_warnings'):
                result['warnings'].append({'seed': seed, 'key': key, 'candidate_id': candidate_id,
                                           'warnings': candidate['fit_warnings']})
            if 'fallback_reason' in candidate:
                result['fallbacks'].append({'seed': seed, 'key': key, 'candidate_id': candidate_id,
                                            'reason': candidate['fallback_reason']})
            if candidate['family'] == 'mlp' and 'validation_curve' in candidate:
                best_epoch = min(candidate['validation_curve'], key=lambda c: (c['validation_log_loss'], c['epoch']))
                assert candidate['selected_epoch'] == best_epoch['epoch']
                assert candidate['selected_optimizer_steps'] == best_epoch['optimizer_steps']
                assert candidate['validation_scores']['log_loss'] == best_epoch['validation_log_loss']
                assert candidate['validation_curve'][0]['epoch'] == 0
                assert candidate['validation_curve'][0]['optimizer_steps'] == 0
                epochs = config['head_mlp_epochs'] if role == 'transfer' else config['audit_mlp_epochs']
                assert candidate['parameters']['epochs'] == epochs
                assert candidate['training_row_exposures'] == len(idx)*epochs
                assert candidate['optimizer_steps'] == int(np.ceil(len(idx)/candidate['parameters']['batch_size']))*epochs
                schedule_groups.setdefault((role, target, candidate_id), set()).add(
                    (candidate['schedule_hash'], candidate['optimizer_steps'], candidate['training_row_exposures']))
                result['mlp_epoch_selections_checked'] += 1
    for group, schedules in schedule_groups.items():
        assert len(schedules) == 1, (seed, group, 'unmatched MLP schedules')
        result['matched_mlp_schedule_groups_checked'] += 1
    race_fit, valid = labels(data['attacker_fit'], 'RAC1P')
    result['seeds'][str(seed)] = {
        'development_evaluation_rows': len(data['test']),
        'prediction_archive_sha256': sha(directory/'predictions.npz'),
        'metrics_sha256': sha(directory/'metrics.json'),
        'selection_sha256': sha(directory/'selection_before_test.json'),
        'split_rows_sha256': sha(directory/'split_rows.npz'),
        'race_evaluation_coverage': coverage,
        'race_attacker_fit_support': np.bincount(race_fit[valid], minlength=9).tolist(),
        'controls': controls,
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
