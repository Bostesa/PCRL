"""Independently replay ACS bottleneck scores and saved fitted models; no fitting.

All candidate metrics use raw labels, cached probabilities, and independent
sklearn/scalar formulas. New selected model files are replayed using saved
normalizers and plain inference, without importing experiment fitters/scorers.
Historical reference rows are checked exactly and scored from original caches.
--report requires a fresh path; otherwise JSON goes to stdout.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support, roc_auc_score

TOL = 2e-12
errors = []
max_error = 0.
max_error_path = None
numeric_comparisons = 0

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


def read(path):
    return json.loads(Path(path).read_text())


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(array_hash(value.detach().cpu().numpy()).encode())
    return digest.hexdigest()


def network_probability(state, x):
    """Literal saved 64/32 ReLU network; no fitting or mode-dependent layers."""
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            h = torch.as_tensor(np.array(x[start:start+4096], dtype=np.float32, copy=True))
            for layer in ('0', '2', '4'):
                h = torch.nn.functional.linear(h, state[layer+'.weight'], state[layer+'.bias'])
                if layer != '4':
                    h = torch.relu(h)
            outputs.append(torch.softmax(h, 1).double().numpy())
    return np.concatenate(outputs)


def load_inference(path, metadata):
    """Rebuild only inference from original persisted parameters/statistics."""
    pre = np.load(path/'preprocessing.npz')
    mean, scale = pre['mean'].copy(), pre['scale'].copy()
    pre.close()
    if metadata['family'] == 'mlp':
        checkpoint = torch.load(path/'model.pt', map_location='cpu', weights_only=True)
        state = checkpoint['state']
        assert state_hash(state) == metadata['selected_state_hash']
        def predict(x):
            return network_probability(state, (np.asarray(x, dtype=np.float64)-mean)/scale)
    else:
        model = joblib.load(path/'model.joblib')
        state = None
        def predict(x):
            x = (np.asarray(x, dtype=np.float64)-mean)/scale
            probability = np.zeros((len(x), metadata['n_classes']))
            probability[:, model.classes_.astype(int)] = model.predict_proba(x)
            return probability
    return predict, mean, scale, state


def primary(candidates, ids, auc=False):
    metric = 'auroc' if auc else 'log_loss'
    eligible = [cid for cid in ids if candidates[cid]['validation_scores'][metric] is not None
                and np.isfinite(candidates[cid]['validation_scores'][metric])]
    return min(eligible, key=lambda cid: ((-1 if auc else 1)*candidates[cid]['validation_scores'][metric], cid)) if eligible else None


def replay_schedule(candidate, n):
    digest = hashlib.sha256()
    rng = np.random.default_rng(candidate['schedule_seed'])
    for _ in range(candidate['parameters']['epochs']):
        digest.update(rng.permutation(n).tobytes())
    assert digest.hexdigest() == candidate['schedule_hash']


def run(root, out, seeds=None):
    started = time.perf_counter()
    config = read(out/'config.json')
    parent, reference = root/config['parent_results'], root/config['reference_results']
    pcfg = read(parent/'config.json')
    seeds = config['seeds'] if seeds is None else seeds
    assert seeds and len(seeds) == len(set(seeds)) and not set(seeds)-set(config['seeds'])
    freeze = read(out/'protocol_freeze.json')
    for key in ('sha256', 'reference_record_hashes'):
        for path, expected in freeze[key].items():
            assert sha(root/path) == expected, ('frozen file', path)
    raw_path = root/pcfg['raw_path']
    assert sha(raw_path) == read(parent/'schema_support.json')['raw_sha256']
    raw = pd.read_csv(raw_path, usecols=['MIG','JWMNP','PINCP','ESR','PUBCOV','SEX','RAC1P','PWGTP'], low_memory=False)
    report = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Inference-only independent metric and fitted-model replay; no fitting, parameter updates, new evaluation rows, or selection changes.',
        'method': 'Raw-column labels and masks; independent sklearn/scalar metrics, probability floor1e-12; literal saved-network inference and saved sklearn prediction. No experiment fitter/scorer imports.',
        'evaluation_status': config['evaluation_status'], 'absolute_tolerance': TOL,
        'script_sha256': sha(__file__), 'raw_sha256': sha(raw_path),
        'frozen_source_entries_checked': len(freeze['sha256']),
        'reference_records_checked': len(freeze['reference_record_hashes']),
        'candidate_records_replayed': 0, 'score_sets_replayed': 0,
        'unchanged_reference_records_checked': 0, 'new_saved_model_prediction_sets_replayed': 0,
        'new_primary_selections_checked': 0, 'new_independent_selections_checked': 0,
        'new_family_selections_checked': 0, 'new_auroc_selections_checked': 0,
        'mlp_epoch_and_schedule_checks': 0, 'matched_schedule_groups': 0,
        'identity_coordinate_audits': 0, 'saved_adversary_fidelity_checks': 0,
        'inherited_exposure_checks': 0, 'fitting_standardizers_checked': 0,
        'fit_label_hashes_checked': 0, 'validation_label_hashes_checked': 0,
        'new_release_arrays_replayed': 0, 'warnings': [], 'fallbacks': [], 'seeds': {},
    }
    for seed in seeds:
        directory, rd = out/f'seed_{seed}', reference/f'seed_{seed}'
        measured, selection = read(directory/'metrics.json'), read(directory/'selection_before_test.json')
        previous, old_selection = read(rd/'metrics.json'), read(rd/'selection_before_test.json')
        training = read(directory/'training/training.json')
        releases = {name: dict(np.load(directory/f'release_{name}.npz')) for name in ('C_bottleneck','D_protected')}
        checkpoints = {name: torch.load(directory/'training'/name/'final.pt', map_location='cpu', weights_only=True)
                       for name in releases}
        with np.load(directory/'split_rows.npz') as archive:
            rows = {p: archive[p].copy() for p in archive.files}
        with np.load(parent/f'seed_{seed}/split_rows.npz') as original:
            assert set(rows) == set(original.files)
            assert all(np.array_equal(v, original[k]) for k, v in rows.items())
        frames = {pool: raw.iloc[idx] for pool, idx in rows.items()}
        assert sha(directory/'selection_before_test.json') == measured['integrity']['selection_sha256']
        assert sha(directory/'release_freeze.json') == selection['release_freeze_sha256']
        assert sha(out/'protocol_freeze.json') == selection['protocol_freeze_sha256']
        assert selection['created_utc'] <= measured['integrity']['evaluation_started_utc']
        assert all(v for k, v in measured['integrity'].items() if k.endswith('_unchanged'))
        frozen = read(directory/'release_freeze.json')
        manifest = read(directory/'local_artifacts.json')
        for record in manifest:
            assert sha(root/record['path']) == record['sha256']
        pca = dict(np.load(rd/'release_E_pca.npz'))
        fit64 = pca['representation_fit'].astype(np.float64)
        expected_mean, std = fit64.mean(0), fit64.std(0)
        expected_scale = np.where(std > 1e-12, std, 1.)
        assert training['fit_input_sha256'] == array_hash(pca['representation_fit'].astype(np.float32))
        for arm, checkpoint in checkpoints.items():
            state = checkpoint['model_state']
            assert state_hash(state) == frozen['learned_state'][arm]['model']
            assert state_hash(checkpoint['adversary_state']) == frozen['learned_state'][arm]['adversaries']
            np.testing.assert_array_equal(state['input_mean'].numpy(), expected_mean)
            np.testing.assert_array_equal(state['input_scale'].numpy(), expected_scale)
            with torch.no_grad():
                for pool, expected in releases[arm].items():
                    x = torch.from_numpy(((pca[pool].astype(np.float64)-expected_mean)/expected_scale).astype(np.float32))
                    x = torch.relu(torch.nn.functional.linear(x, state['mapper.0.weight'], state['mapper.0.bias']))
                    actual = torch.nn.functional.linear(x, state['mapper.2.weight'], state['mapper.2.bias']).numpy()
                    assert np.array_equal(actual, expected), (seed, arm, pool, 'mapper replay')
                    report['new_release_arrays_replayed'] += 1
        old_rows = {(r['role'],r['release'],r['target'],r['candidate_id']):r for r in previous['raw_metrics']}
        controls, race_coverage = {}, {}
        with np.load(directory/'predictions.npz') as new_probs, np.load(rd/'predictions.npz') as old_probs:
            assert len(new_probs.files) == 96
            assert len(measured['raw_metrics']) == 185
            for row in measured['raw_metrics']:
                role, arm, target, cid = (row[k] for k in ('role','release','target','candidate_id'))
                key = '/'.join((role,arm,target))
                k = 9 if target == 'RAC1P' else 2
                archive = old_probs if row['reused_reference'] else new_probs
                if row['reused_reference']:
                    old = old_rows[role,arm,target,cid]
                    assert {k:v for k,v in row.items() if k not in ('independent_selected','reused_reference')} == old
                    assert row['independent_selected'] == old['selected']
                    report['unchanged_reference_records_checked'] += 1
                for split in ('validation','test'):
                    pool = 'test' if split == 'test' else ('downstream_validation' if role == 'transfer' else 'attacker_validation')
                    y, valid = labels(frames[pool], target)
                    weights = frames[pool].PWGTP.to_numpy(float)[valid]
                    probability = archive[f'{split}/{key}/{cid}']
                    for suffix, weight in (('',None),('_person_weighted',weights)):
                        compare(independent_scores(y[valid],probability,k,weight), row[split+suffix],
                                f'seed_{seed}/{key}/{cid}/{split}{suffix}')
                        report['score_sets_replayed'] += 1
                assert row['selected'] == (selection['head_selections'][key] == cid)
                assert row['independent_selected'] == (selection['independent_selections'][key] == cid)
                assert row['selected_within_family'] == (selection['family_selections'][key][row['family']] == cid)
                assert row['auc_selected'] == (selection['auroc_selections'][key] == cid)
                compare(selection['fitting_records'][key]['candidates'][cid]['validation_scores'],row['validation'],
                        f'seed_{seed}/{key}/{cid}/saved_validation_selection')
                if target == 'RAC1P':
                    race_coverage = {s:{k:row[s][k] for k in ('support','coverage_complete')} for s in ('validation','test')}
                if arm in ('prior','exposed'):
                    controls[key+'/'+cid] = {'selected':row['selected'], 'log_loss':row['test']['log_loss'],
                        'accuracy':row['test']['accuracy'], 'per_class_recall':[v['recall'] for v in row['test']['per_class']]}
                report['candidate_records_replayed'] += 1

            schedule_groups = {}
            for key, record in selection['fitting_records'].items():
                role, arm, target = key.split('/')
                if arm not in releases:
                    assert record == old_selection['fitting_records'][key]
                    assert selection['head_selections'][key] == old_selection['head_selections'][key]
                    assert selection['family_selections'][key] == old_selection['family_selections'][key]
                    assert selection['auroc_selections'][key] == old_selection['auroc_selections'][key]
                    continue
                candidates = record['candidates']
                eligible = [c for c in candidates if c != 'saved_adversary']
                independent = [c for c in eligible if c != 'catchup']
                assert set(record['candidate_ids']) == set(candidates)
                assert set(record['primary_selection_candidates']) == set(eligible)
                assert set(record['independent_candidate_ids']) == set(independent)
                assert selection['head_selections'][key] == primary(candidates,eligible)
                assert selection['independent_selections'][key] == primary(candidates,independent)
                assert selection['auroc_selections'][key] == primary(candidates,eligible,auc=True)
                assert record == read(directory/'fitted'/key/'selection.json')
                assert record['selected_family'] == selection['head_selections'][key]
                report['new_primary_selections_checked'] += 1
                report['new_independent_selections_checked'] += 1
                report['new_auroc_selections_checked'] += 1
                family = lambda cid: cid if cid in ('catchup','saved_adversary') else candidates[cid]['family']
                for f, chosen in selection['family_selections'][key].items():
                    assert chosen == primary(candidates,[c for c in candidates if family(c) == f])
                    report['new_family_selections_checked'] += 1
                fit_pool, val_pool = ('downstream_fit','downstream_validation') if role == 'transfer' else ('attacker_fit','attacker_validation')
                yf, valid = labels(frames[fit_pool],target)
                yv, vv = labels(frames[val_pool],target)
                j = (config['utility_tasks'] if role == 'transfer' else ['SEX','RAC1P']).index(target)
                fit_seed = (1230000 if role == 'transfer' else 1240000)+100*seed+j
                limit = config['head_budget'] if role == 'transfer' else config['attacker_budget']
                idx = np.random.default_rng(fit_seed).permutation(np.flatnonzero(valid))[:limit]
                xf, xv = releases[arm][fit_pool][idx], releases[arm][val_pool][vv]
                assert record['fit_hashes'] == {'x':array_hash(xf.astype(np.float64)),'y':array_hash(yf[idx].astype(np.int64))}
                for cid, candidate in candidates.items():
                    cpath = directory/'fitted'/key/cid
                    assert read(cpath/'metadata.json') == candidate
                    predict, mean, scale, model_state = load_inference(cpath,candidate)
                    assert candidate['validation_hashes'] == {'x':array_hash(xv.astype(np.float64)),'y':array_hash(yv[vv].astype(np.int64))}
                    report['validation_label_hashes_checked'] += 1
                    if cid != 'saved_adversary':
                        assert candidate['fit_hashes'] == record['fit_hashes']
                        assert candidate['fit_support'] == np.bincount(yf[idx],minlength=candidate['n_classes']).tolist()
                        report['fit_label_hashes_checked'] += 1
                    if cid in ('saved_adversary','catchup'):
                        np.testing.assert_array_equal(mean,np.zeros(16))
                        np.testing.assert_array_equal(scale,np.ones(16))
                        assert candidate['preprocessing_fit_rows'] == 0
                        report['identity_coordinate_audits'] += 1
                        inherited = candidate['inherited_exposure']
                        ty, mask = labels(frames['representation_fit'],target)
                        ty = np.where(mask,ty,-1).astype(np.int64)
                        arm_meta = training['arms'][arm]
                        saved_state = {k.removeprefix(target+'.'):v for k,v in checkpoints[arm]['adversary_state'].items()
                                       if k.startswith(target+'.')}
                        assert inherited['target'] == target and inherited['fit_pool'] == 'representation_fit'
                        assert inherited['fit_rows'] == len(ty)
                        assert inherited['fit_raw_row_sha256'] == array_hash(rows['representation_fit'])
                        assert inherited['training_label_sha256'] == array_hash(ty)
                        assert inherited['fit_coverage'] == training['attribute_fit_coverage'][target]
                        assert inherited['fit_coverage']['support'] == np.bincount(ty[ty>=0],minlength=candidate['n_classes']).tolist()
                        assert inherited['total_row_passes_equivalent'] == 260
                        assert inherited['total_row_exposures'] == len(ty)*260
                        assert inherited['total_known_label_exposures'] == int(mask.sum())*260
                        assert inherited['total_optimizer_steps'] == int(np.ceil(len(ty)/256))*260
                        assert inherited['continuation_schedule_hash'] == arm_meta['schedule_hash']
                        assert inherited['final_adversary_state_hash'] == state_hash(saved_state) == candidate['source_state_hash']
                        report['inherited_exposure_checks'] += 1
                        if cid == 'saved_adversary':
                            assert candidate['fit_rows'] == 0 and candidate['fit_support'] is None
                            assert candidate['fit_hashes'] is None and candidate['optimizer_steps'] == 0
                            assert state_hash(model_state) == state_hash(saved_state)
                            assert np.array_equal(predict(xv),network_probability(saved_state,xv))
                            report['saved_adversary_fidelity_checks'] += 1
                        else:
                            assert candidate['initial_state_hash'] == state_hash(saved_state)
                            assert candidate['optimizer']['initial_state_entries'] == 0
                            assert candidate['optimizer']['restored_state'] is False
                            assert candidate['validation_curve'][0]['validation_log_loss'] == candidates['saved_adversary']['validation_scores']['log_loss']
                            assert all(s == candidate['optimizer_steps'] for s in candidate['optimizer']['final_state_steps'])
                    else:
                        fit64 = xf.astype(np.float64)
                        np.testing.assert_array_equal(mean,fit64.mean(0))
                        std = fit64.std(0)
                        np.testing.assert_array_equal(scale,np.where(std>1e-12,std,1.))
                        report['fitting_standardizers_checked'] += 1
                    for split,pool in (('validation',val_pool),('test','test')):
                        y, valid = labels(frames[pool],target)
                        actual = predict(releases[arm][pool][valid])
                        expected = new_probs[f'{split}/{key}/{cid}']
                        assert np.array_equal(actual,expected), (seed,key,cid,split,'saved inference')
                        report['new_saved_model_prediction_sets_replayed'] += 1
                    if candidate.get('fit_warnings'):
                        report['warnings'].append({'seed':seed,'key':key,'candidate':cid,'warnings':candidate['fit_warnings']})
                    if 'fallback_reason' in candidate:
                        report['fallbacks'].append({'seed':seed,'key':key,'candidate':cid,'reason':candidate['fallback_reason']})
                    if 'validation_curve' in candidate:
                        epochs = config['head_mlp_epochs'] if role == 'transfer' else config['catchup_epochs'] if cid == 'catchup' else config['audit_mlp_epochs']
                        assert candidate['parameters']['epochs'] == epochs
                        curve = candidate['validation_curve']
                        assert [p['epoch'] for p in curve] == list(range(0,epochs+1,5))
                        steps = int(np.ceil(len(idx)/256))
                        assert all(p['optimizer_steps'] == steps*p['epoch'] for p in curve)
                        best = min(curve,key=lambda p:(p['validation_log_loss'],p['epoch']))
                        assert candidate['selected_epoch'] == best['epoch']
                        assert candidate['selected_optimizer_steps'] == best['optimizer_steps']
                        assert candidate['validation_scores']['log_loss'] == best['validation_log_loss']
                        assert candidate['optimizer_steps'] == steps*epochs
                        assert candidate['training_row_exposures'] == len(idx)*epochs
                        assert candidate['row_exposure_min'] == candidate['row_exposure_max'] == epochs
                        replay_schedule(candidate,len(idx))
                        schedule_groups.setdefault((role,target,cid),set()).add((candidate['schedule_hash'],candidate['optimizer_steps']))
                        report['mlp_epoch_and_schedule_checks'] += 1
                if role == 'audit':
                    assert len(candidates) == 7
                    for field in ('optimizer_steps','training_row_exposures'):
                        independent_total = sum(candidates[c][field] for c in ('mlp_0','mlp_1'))
                        assert record['independent_total_mlp_'+field] == independent_total
                        assert record['total_new_mlp_'+field] == independent_total+candidates['catchup'][field]
            for key, schedules in schedule_groups.items():
                assert len(schedules) == 1, (seed,key,'C/D fitting schedule mismatch')
                report['matched_schedule_groups'] += 1
        report['seeds'][str(seed)] = {'prediction_sha256':sha(directory/'predictions.npz'),
            'metrics_sha256':sha(directory/'metrics.json'), 'selection_sha256':sha(directory/'selection_before_test.json'),
            'race_evaluation_coverage':race_coverage,'controls':controls,
            'new_candidate_rows':48,'unchanged_reference_rows':137}
    report.update(numeric_values_compared=numeric_comparisons,max_absolute_metric_error=max_error,
                  max_error_path=max_error_path,errors=errors,passed=not errors,
                  runtime_seconds=time.perf_counter()-started)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out',type=Path)
    parser.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--report',type=Path,help='Fresh JSON file; default stdout; never overwrites')
    args = parser.parse_args()
    if args.report is not None and args.report.exists():
        raise FileExistsError('Preserve existing evidence; choose a fresh report destination')
    root = args.root.resolve()
    out = args.out.resolve() if args.out else root/'results/redesign_20260908_acs_bottleneck_v1'
    torch.set_num_threads(1)
    report = run(root,out,args.seeds)
    rendered = json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.report:
        with args.report.open('x') as handle:
            handle.write(rendered)
    else:
        print(rendered,end='')
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
