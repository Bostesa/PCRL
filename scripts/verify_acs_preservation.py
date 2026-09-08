"""Replay new preservation checkpoints, releases, heads, scores, and affine maps.

Only new scientific models are inferred. Historical artifacts supply immutable
state, metadata, and prediction references; no historical inference or fitting
is rerun. Affine coefficients receive an independent SVD algebra check.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_acs_bottleneck_scores as check
from scripts.verify_acs_bottleneck_artifacts import adam_steps
from scripts.verify_acs_pca16_init import load, map_output
from experiments.acs_bottleneck_training import tree_digest

ARMS = ('C_warmup_only', 'D_warmup_only', 'C_persistent', 'D_persistent')
SNAPSHOTS = {'I': 'initialization.pt', 'W': 'warm_base.pt'}


def beta_name(beta):
    assert beta in (.1, 1.)
    return 'beta_0p1' if beta == .1 else 'beta_1'


def effective_execution_hashes(out, frozen):
    """Keep original execution identities; admit only an explicit source chain."""
    expected, amendments = frozen['sha256'].copy(), {}
    original = check.sha(out / 'protocol_freeze.json')
    for path in sorted(out.glob('EXECUTION_AMENDMENT_*.json')):
        amendment = check.read(path)
        assert amendment['original_protocol_freeze_sha256'] == original
        for name, change in amendment['source_changes'].items():
            assert name.startswith('experiments/') and expected[name] == change['before_sha256']
            expected[name] = change['after_sha256']
        amendments[path.name] = check.sha(path)
    return expected, amendments


def state_checks(directory, historical, pca, arrays, meta, freeze, frames, beta):
    train = directory / 'training'
    old = check.read(historical / 'training/training.json')
    initial, base, warm = (load(train / name) for name in ('initialization.pt', 'warm_base.pt', 'warm_adversary.pt'))
    old_initial = load(historical / 'training/initialization.pt')
    assert tree_digest(initial) == tree_digest(old_initial), 'Exact historical whole initialization and empty Adam'
    state = initial['model_state']
    n = len(pca['representation_fit'])
    batches = int(np.ceil(n / 256))
    for field in ('config', 'preprocessing', 'schedules', 'adversary_initialization_hash', 'source_label_hashes',
                  'attribute_label_hashes', 'source_validation_label_hashes', 'fit_input_sha256',
                  'source_validation_input_sha256', 'common_optimizer_counts'):
        assert meta[field] == old[field], ('historical recipe', field)
    fit = pca['representation_fit'].astype(np.float64)
    std = fit.std(0)
    mean, scale = fit.mean(0), np.where(std > 1e-12, std, 1.)
    np.testing.assert_array_equal(state['input_mean'].numpy(), mean)
    np.testing.assert_array_equal(state['input_scale'].numpy(), scale)
    teacher = meta['teacher']
    assert teacher['sha256'] == check.array_hash(pca['representation_fit'][:, :16].astype(np.float32))
    assert teacher['coordinate_scale_sha256'] == check.array_hash(scale[:16])
    np.testing.assert_array_equal(teacher['coordinate_scale'], scale[:16])
    assert teacher['rows'] == n and teacher['fit_pool'] == 'representation_fit'
    assert all(teacher[k] for k in ('detached', 'immutable_verified', 'saved_statistics_supplied', 'saved_statistics_exact_fit_parity'))
    assert not teacher['labels_received'] and teacher['raw_coordinates'] == list(range(16))
    assert meta['preservation_config']['study'] == 'acs_pca16_output_preservation_v1'
    assert meta['preservation_config']['beta'] == beta
    for phase, epochs, offset in (('warm_base', 60, 0), ('warm_adversary', 20, 100), ('continuation', 80, 200)):
        schedule = meta['schedules'][phase]
        assert schedule['seed'] == 1280000 + 100 * meta['seed'] + offset
        rng = np.random.default_rng(schedule['seed'])
        digest = hashlib.sha256()
        for _ in range(epochs):
            digest.update(rng.permutation(n).tobytes())
        assert digest.hexdigest() == schedule['sha256']
    diagnostic_indices = np.random.default_rng(meta['schedules']['warm_base']['seed']).permutation(n)[:256]
    assert meta['gradient_diagnostic_indices_sha256'] == check.array_hash(diagnostic_indices)
    for pool, targets, hash_key in (
            ('representation_fit', ('income_binary', 'civilian_at_work', 'public_coverage'), 'source_label_hashes'),
            ('representation_fit', ('SEX', 'RAC1P'), 'attribute_label_hashes'),
            ('source_validation', ('income_binary', 'civilian_at_work', 'public_coverage'), 'source_validation_label_hashes')):
        for target in targets:
            y, mask = check.labels(frames[pool], target)
            assert meta[hash_key][target] == check.array_hash(np.where(mask, y, -1).astype(np.int64))
    assert not meta['reserved_labels_received'] and not meta['final_evaluation_received']
    assert check.state_hash(base['model_state']) == check.state_hash(warm['model_state']) == meta['snapshots']['W']['model_hash']
    assert tree_digest(base['mapper_optimizer_state']) == tree_digest(warm['mapper_optimizer_state'])
    adam_steps(base['mapper_optimizer_state'], 60 * batches)
    adam_steps(warm['adversary_optimizer_state'], 20 * batches)
    assert meta['common_base_row_exposures'] == n * 60 and meta['common_adversary_row_exposures'] == n * 20
    for phase in ('warm_base', 'warm_adversary'):
        for row in meta['shared_curves'][phase]:
            assert row['preservation_coefficient'] == (beta if phase == 'warm_base' else 0.)
            assert row['source_coefficient'] == 1. and row['reconstruction_coefficient'] == .1
    forks = {arm: load(train / arm / 'fork.pt') for arm in ARMS}
    assert len({tree_digest(fork) for fork in forks.values()}) == 1
    checkpoints = {name: load(train / path) for name, path in SNAPSHOTS.items()}
    diagnostics = list(meta['stage_gradient_diagnostics'].values())
    assert set(meta['stage_gradient_diagnostics']) == {'initialization', 'after_base_warmup', 'shared_fork'}
    assert meta['stage_gradient_diagnostics']['initialization']['preservation_loss'] < 1e-10
    perturbation = meta['initialization']['perturbed_preservation_gradient_check']
    assert all(perturbation[k] for k in ('analytic_gradient_passed', 'actual_model_unchanged', 'torch_rng_unchanged', 'disposable_model'))
    assert perturbation['optimizer_steps'] == 0 and perturbation['readout_bias_gradient_l2'] > 0
    for arm in ARMS:
        checkpoint = load(train / arm / 'final.pt')
        checkpoints[arm] = checkpoint
        record = meta['arms'][arm]
        old_arm = old['arms']['C_bottleneck' if arm.startswith('C_') else 'D_protected']
        for field in ('schedule_hash', 'continuation_mapper_optimizer_steps', 'continuation_adversary_optimizer_steps',
                      'optimizer_counts_including_common', 'mapper_row_exposures', 'adversary_row_exposures'):
            assert record[field] == old_arm[field]
        assert record['fork_hashes'] == meta['shared_fork_hashes']
        for field, part in (('model_state', 'model'), ('adversary_state', 'adversaries'),
                            ('mapper_optimizer_state', 'mapper_optimizer'), ('adversary_optimizer_state', 'adversary_optimizer')):
            hash_function = tree_digest if 'optimizer' in field else check.state_hash
            assert hash_function(forks[arm][field]) == meta['shared_fork_hashes'][part] == hash_function(warm[field])
        assert checkpoint['counters'] == {'mapper_optimizer_steps': 140 * batches,
            'adversary_optimizer_steps': 260 * batches, 'continuation_epoch': 80}
        assert record['selected_epoch'] == 80
        adam_steps(checkpoint['mapper_optimizer_state'], 140 * batches)
        adam_steps(checkpoint['adversary_optimizer_state'], 260 * batches)
        active = beta if arm.endswith('_persistent') else 0.
        protected = arm.startswith('D_')
        assert record['preservation_beta'] == beta and record['warmup_preservation_coefficient'] == beta
        assert record['continuation_preservation_coefficient'] == active
        assert record['mapper_loss_uses_protection_gradient'] == protected
        for row in record['curve']:
            assert row['mapper_optimizer_steps'] == (60 + row['epoch']) * batches
            assert row['adversary_optimizer_steps'] == (20 + 3 * row['epoch']) * batches
            assert row['preservation_coefficient'] == active and row['protection_coefficient'] == (-.1 if protected else 0.)
            assert row['source_coefficient'] == 1. and row['reconstruction_coefficient'] == .1
            assert row['fit_objective'] == row['fit_base_loss'] + row['fit_applied_preservation'] + row['fit_applied_protection']
        for field, key in (('model_state', 'final_model_hash'), ('adversary_state', 'final_adversary_hash')):
            assert check.state_hash(checkpoint[field]) == record[key]
        assert tree_digest(checkpoint['mapper_optimizer_state']) == record['final_mapper_optimizer_hash']
        assert tree_digest(checkpoint['adversary_optimizer_state']) == record['final_adversary_optimizer_hash']
        diagnostics.extend(record[key] for key in ('fixed_batch_gradient_diagnostics_at_shared_fork', 'fixed_batch_gradient_diagnostics_at_final'))
    for diag in diagnostics:
        assert all(diag[k] for k in ('training_state_unchanged', 'torch_rng_unchanged', 'preservation_nonmapper_gradients_all_absent'))
        assert diag['optimizer_steps'] == 0
        assert np.isclose(diag['applied_preservation_mapper_l2'], diag['preservation_coefficient'] * diag['preservation_mapper_l2'], rtol=2e-6, atol=1e-12)
    count, parity = 0, {}
    for arm, checkpoint in checkpoints.items():
        assert check.state_hash(checkpoint['model_state']) == freeze['learned_state'][arm]['model']
        if arm in ARMS:
            assert check.state_hash(checkpoint['adversary_state']) == freeze['learned_state'][arm]['adversaries']
        for pool, values in arrays[arm].items():
            np.testing.assert_array_equal(map_output(checkpoint['model_state'], pca[pool]), values)
            if pool != 'test':
                assert freeze['output_hashes'][arm][pool] == check.array_hash(values)
            if arm == 'I':
                np.testing.assert_allclose(values, pca[pool][:, :16], atol=1e-5, rtol=1e-5)
                parity[pool] = float(np.abs(values.astype(np.float64) - pca[pool][:, :16]).max())
            count += 1
    with np.load(historical / 'release_I.npz') as old_arrays:
        for pool in arrays['I']:
            np.testing.assert_array_equal(arrays['I'][pool], old_arrays[pool])
    return checkpoints, {'initial_full_state_and_empty_Adam_exact': True, 'I_parity_max_errors': parity,
        'raw_teacher_and_saved_fitting_scales_exact': True, 'four_way_full_state_and_Adam_fork_exact': True,
        'W_model_and_mapper_Adam_unchanged_through_adversary_warmup': True,
        'all_saved_Adam_steps_verified': True, 'historical_recipe_and_schedules_match': True,
        'frozen_release_arrays_replayed': count, 'gradient_diagnostic_records_checked': len(diagnostics)}


def affine_checks(directory, arrays, pca, meta, selected, release_freeze):
    frozen = check.read(directory / 'affine_freeze.json')
    report = check.read(directory / 'preservation_diagnostics.json')
    assert release_freeze['created_utc'] <= frozen['created_utc'] < selected['created_utc']
    for path, digest in frozen['sha256'].items():
        assert check.sha(directory / path) == digest
    mean, scale = np.asarray(meta['preprocessing']['mean'])[:16], np.asarray(meta['preprocessing']['scale'])[:16]
    errors = []
    for arm, values in arrays.items():
        record = report['snapshots'][arm]
        assert {k: v for k, v in record.items() if k != 'development_evaluation'} == frozen['fits']['snapshots'][arm]
        assert record['fit_pool'] == 'representation_fit' and record['intercept_fitted'] and record['rank_tolerance'] == 1e-12
        with np.load(directory / f'affine_{arm}.npz') as saved:
            coefficient, prior = saved['coefficient'], saved['prior']
        x = values['representation_fit'].astype(np.float64)
        y = (pca['representation_fit'][:, :16].astype(np.float64) - mean) / scale
        design = np.column_stack((x, np.ones(len(x))))
        assert record['fit_release_sha256'] == check.array_hash(x)
        assert record['fit_teacher_sha256'] == check.array_hash(y)
        u, singular, vt = np.linalg.svd(design, full_matrices=False)
        valid = singular > 1e-12 * singular[0]
        reconstructed = (vt[valid].T / singular[valid]) @ (u[:, valid].T @ y)
        np.testing.assert_allclose(coefficient, reconstructed, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(record['singular_values'], singular, rtol=1e-12, atol=1e-12)
        assert record['rank'] == int(valid.sum())
        np.testing.assert_array_equal(prior, y.mean(0))
        errors.append(float(np.abs(coefficient - reconstructed).max()))
        for pool, name in (('representation_fit', 'fit'), ('source_validation', 'source_validation'), ('test', 'development_evaluation')):
            x, raw = values[pool].astype(np.float64), pca[pool][:, :16].astype(np.float64)
            y = (raw - mean) / scale
            prediction = np.column_stack((x, np.ones(len(x)))) @ coefficient
            square_errors = {'direct': ((x - raw) / scale) ** 2,
                             'affine': (prediction - y) ** 2, 'prior': (y - prior) ** 2}
            expected = {'rows': len(x), **{f'{k}_mean_mse': float(v.mean()) for k, v in square_errors.items()},
                        **{f'{k}_per_coordinate_mse': v.mean(0).tolist() for k, v in square_errors.items()}}
            check.compare(expected, record[name], f'{directory.name}/{arm}/affine/{name}')
    return {'fitting_pool_and_freeze_verified': True, 'decoders_checked': len(arrays),
            'all_coordinate_and_prior_scores_replayed': True, 'independent_SVD_coefficient_max_error': max(errors)}


def verify(out):
    started = time.perf_counter()
    check.errors.clear()
    check.max_error = 0.
    check.max_error_path = None
    check.numeric_comparisons = 0
    cfg, frozen = (check.read(out / name) for name in ('config.json', 'protocol_freeze.json'))
    execution, amendments = effective_execution_hashes(out, frozen)
    for hashes in (execution, frozen['reference_record_hashes']):
        for path, digest in hashes.items():
            assert check.sha(ROOT / path) == digest, path
    parent, reference, historical = (ROOT / cfg[k] for k in ('parent_results', 'reference_results', 'init_reference_results'))
    raw_path = ROOT / check.read(parent / 'config.json')['raw_path']
    assert check.sha(raw_path) == check.read(parent / 'schema_support.json')['raw_sha256']
    raw = pd.read_csv(raw_path, usecols=['MIG', 'JWMNP', 'PINCP', 'ESR', 'PUBCOV', 'SEX', 'RAC1P', 'PWGTP'])
    order = [(b, s) for b in cfg['preservation_betas'] for s in cfg['seeds']]
    completed = [(b, s) for b, s in order if (out / beta_name(b) / f'seed_{s}/metrics.json').exists()]
    assert completed, 'No complete units to verify'
    assert completed == order[:len(completed)], 'Completed units must follow the predeclared order'
    progress = check.read(out / 'progress.json')
    assert progress['completed_units'] == [list(v) for v in completed]
    report = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Only new preservation checkpoints, releases, candidate predictions/scores; old initialization, cached I arrays, metadata/manifests as references. No fitting or old scientific inference rerun. Independent SVD validates saved affine algebra.',
        'evaluation_status': cfg['evaluation_status'], 'script_sha256': check.sha(__file__),
        'original_protocol_freeze_sha256': check.sha(out / 'protocol_freeze.json'),
        'execution_amendments_sha256': amendments,
        'helper_sha256': {p: check.sha(ROOT / p) for p in ('scripts/verify_acs_bottleneck_scores.py',
            'scripts/verify_acs_bottleneck_artifacts.py', 'scripts/verify_acs_pca16_init.py')},
        'completed_units_checked': [list(v) for v in completed],
        'missing_units': [list(v) for v in order if v not in completed],
        'new_candidate_records': 0, 'new_score_dictionaries': 0, 'new_prediction_sets_replayed': 0,
        'unchanged_historical_records_checked': 0, 'primary_choices_checked': 0, 'independent_choices_checked': 0,
        'family_choices_checked': 0, 'MLP_curves_schedules_checked': 0, 'fitting_standardizers_checked': 0,
        'catchup_identity_coordinate_checks': 0, 'saved_adversary_fidelity_checks': 0, 'units': {}}
    schedule_groups = {}
    for beta, seed in completed:
        directory = out / beta_name(beta) / f'seed_{seed}'
        hd = historical / f'seed_{seed}'
        measured, selected, release_freeze, meta = (check.read(directory / path) for path in
            ('metrics.json', 'selection_before_test.json', 'release_freeze.json', 'training/training.json'))
        assert measured['beta'] == beta and measured['seed'] == seed
        assert release_freeze['scientific_training_completed_utc'] <= release_freeze['created_utc'] < selected['created_utc'] <= measured['integrity']['evaluation_started_utc']
        assert check.sha(directory / 'selection_before_test.json') == measured['integrity']['selection_sha256']
        assert check.sha(directory / 'release_freeze.json') == selected['release_freeze_sha256']
        assert check.sha(out / 'protocol_freeze.json') == selected['protocol_freeze_sha256']
        assert all(v for k, v in measured['integrity'].items() if k.endswith('_unchanged'))
        for item in check.read(directory / 'local_artifacts.json'):
            assert check.sha(ROOT / item['path']) == item['sha256']
        provenance = check.read(directory / 'parent_provenance.json')
        for group in ('used_reference_files_sha256', 'used_original_files_sha256', 'additional_history_files_sha256'):
            for path, digest in provenance[group].items():
                assert check.sha(ROOT / path) == digest
        rows, old_rows = (dict(np.load(path / 'split_rows.npz')) for path in (directory, hd))
        assert set(rows) == set(old_rows) and all(np.array_equal(v, old_rows[p]) for p, v in rows.items())
        assert len(set(np.concatenate(list(rows.values())).tolist())) == sum(map(len, rows.values()))
        frames = {p: raw.iloc[i] for p, i in rows.items()}
        pca = dict(np.load(reference / f'seed_{seed}/release_E_pca.npz'))
        arrays = {a: dict(np.load(directory / f'release_{a}.npz')) for a in (*SNAPSHOTS, *ARMS)}
        checkpoints, state_report = state_checks(directory, hd, pca, arrays, meta, release_freeze, frames, beta)
        affine_report = affine_checks(directory, arrays, pca, meta, selected, release_freeze)
        assert check.read(directory / 'evaluation_parity.json')['selected_before_access'] == measured['integrity']['selection_sha256']
        old_selection = check.read(hd / 'selection_before_test.json')
        prior_records = {}
        for path in (reference / f'seed_{seed}', hd):
            for row in check.read(path / 'metrics.json')['raw_metrics']:
                prior_records[row['role'], row['release'], row['target'], row['candidate_id']] = row
        probabilities = dict(np.load(directory / 'predictions.npz'))
        assert len(probabilities) == 212
        new_rows = [r for r in measured['raw_metrics'] if not r['reused_reference']]
        assert len(new_rows) == 106
        for row in measured['raw_metrics']:
            role,arm,target,cid=(row[k] for k in ('role','release','target','candidate_id'))
            if row['reused_reference']:
                old=prior_records[role,'W' if arm=='W_historical' else arm,target,cid]
                for field in old:
                    if field not in ('reused_reference','independent_selected','release'):assert row[field]==old[field]
                assert row['independent_selected']==old.get('independent_selected',old['selected'])
                report['unchanged_historical_records_checked']+=1;continue
            assert arm in arrays and arm!='I' and (arm not in SNAPSHOTS or role=='transfer')
            key=f'{role}/{arm}/{target}';k=9 if target=='RAC1P' else 2
            for split in ('validation','test'):
                pool='test' if split=='test' else 'downstream_validation' if role=='transfer' else 'attacker_validation'
                y,mask=check.labels(frames[pool],target);p=probabilities[f'{split}/{key}/{cid}']
                for suffix,weight in (('',None),('_person_weighted',frames[pool].PWGTP.to_numpy(float)[mask])):
                    check.compare(check.independent_scores(y[mask],p,k,weight),row[split+suffix],f'{seed}/{key}/{cid}/{split}{suffix}')
                    report['new_score_dictionaries']+=1
            check.compare(selected['fitting_records'][key]['candidates'][cid]['validation_scores'],row['validation'],f'{seed}/{key}/{cid}/recorded_validation')
            assert row['selected']==(selected['head_selections'][key]==cid)
            assert row['independent_selected']==(selected['independent_selections'][key]==cid)
            assert row['auc_selected']==(selected['auroc_selections'][key]==cid)
            assert row['selected_within_family']==(selected['family_selections'][key][row['family']]==cid)
            report['new_candidate_records']+=1
        for key,record in selected['fitting_records'].items():
            role,arm,target=key.split('/')
            if arm not in arrays or arm=='I':continue
            assert record==check.read(directory/'fitted'/key/'selection.json')
            candidates=record['candidates'];eligible=[c for c in candidates if c!='saved_adversary'];independent=[c for c in eligible if c!='catchup']
            assert set(candidates)==({'logistic','mlp'} if role=='transfer' else
                {'logistic','mlp_0','mlp_1','hist_gb_20','hist_gb_5','catchup','saved_adversary'})
            assert set(record['candidate_ids'])==set(candidates)
            assert set(record['primary_selection_candidates'])==set(eligible)
            assert set(record['independent_candidate_ids'])==set(independent)
            assert selected['head_selections'][key]==record['selected_family']==check.primary(candidates,eligible)
            assert selected['independent_selections'][key]==check.primary(candidates,independent)
            assert selected['auroc_selections'][key]==check.primary(candidates,eligible,auc=True)
            report['primary_choices_checked']+=1;report['independent_choices_checked']+=1
            family=lambda c:c if c in ('catchup','saved_adversary') else candidates[c]['family']
            for f,cid in selected['family_selections'][key].items():
                assert cid==check.primary(candidates,[c for c in candidates if family(c)==f]);report['family_choices_checked']+=1
            pool,vpool=('downstream_fit','downstream_validation') if role=='transfer' else ('attacker_fit','attacker_validation')
            yf,valid=check.labels(frames[pool],target);yv,vv=check.labels(frames[vpool],target)
            targets=cfg['utility_tasks'] if role=='transfer' else ['SEX','RAC1P']
            base=1230000 if role=='transfer' else 1240000;cap=2048 if role=='transfer' else 4096
            idx=np.random.default_rng(base+100*seed+targets.index(target)).permutation(np.flatnonzero(valid))[:cap]
            field='task_fit_indices' if role=='transfer' else 'audit_fit_indices'
            assert selected[field][target]==old_selection[field][target]
            assert selected[field][target]['pool_indices_sha256']==check.array_hash(idx)
            assert selected[field][target]['raw_rows_sha256']==check.array_hash(rows[pool][idx])
            xf,xv=arrays[arm][pool][idx].astype(np.float64),arrays[arm][vpool][vv].astype(np.float64)
            for cid,candidate in candidates.items():
                path=directory/'fitted'/key/cid;assert check.read(path/'metadata.json')==candidate
                predict,mean,scale,network_state=check.load_inference(path,candidate)
                assert candidate['validation_hashes']=={'x':check.array_hash(xv),'y':check.array_hash(yv[vv].astype(np.int64))}
                if cid!='saved_adversary':
                    assert candidate['fit_hashes']=={'x':check.array_hash(xf),'y':check.array_hash(yf[idx].astype(np.int64))}
                    assert candidate['fit_support']==np.bincount(yf[idx],minlength=candidate['n_classes']).tolist()
                if cid in ('catchup','saved_adversary'):
                    np.testing.assert_array_equal(mean,np.zeros(16));np.testing.assert_array_equal(scale,np.ones(16))
                    assert candidate['preprocessing_fit_rows']==0 and candidate['initial_fidelity_exact']
                    saved_state={k.removeprefix(target+'.'):v for k,v in checkpoints[arm]['adversary_state'].items() if k.startswith(target+'.')}
                    assert candidate['source_state_hash']==check.state_hash(saved_state)
                    assert candidate['original_fit_probability_hash']==check.array_hash(check.network_probability(saved_state,xf))
                    assert candidate['original_validation_probability_hash']==check.array_hash(check.network_probability(saved_state,xv))
                    inherited=candidate['inherited_exposure'];ty,valid=check.labels(frames['representation_fit'],target)
                    assert inherited['training_label_sha256']==check.array_hash(np.where(valid,ty,-1).astype(np.int64))
                    assert inherited['fit_pool']=='representation_fit' and inherited['fit_raw_row_sha256']==check.array_hash(rows['representation_fit'])
                    assert inherited['fit_coverage']==meta['attribute_fit_coverage'][target]
                    assert inherited['total_known_label_exposures']==int(valid.sum())*260
                    assert inherited['total_row_passes_equivalent']==260 and inherited['total_row_exposures']==len(ty)*260
                    assert inherited['total_optimizer_steps']==int(np.ceil(len(ty)/256))*260
                    assert inherited['final_adversary_state_hash']==candidate['source_state_hash']
                    report['catchup_identity_coordinate_checks']+=1
                    if cid=='saved_adversary':
                        assert check.state_hash(network_state)==check.state_hash(saved_state)
                        assert candidate['optimizer_steps']==0 and candidate['fit_support'] is None
                        assert np.array_equal(predict(xv),check.network_probability(saved_state,xv))
                        report['saved_adversary_fidelity_checks']+=1
                    else:
                        assert candidate['initial_state_hash']==check.state_hash(saved_state)
                        assert candidate['optimizer']['initial_state_entries']==0 and not candidate['optimizer']['restored_state']
                        assert all(v==candidate['optimizer_steps'] for v in candidate['optimizer']['final_state_steps'])
                        assert candidate['validation_curve'][0]['validation_log_loss']==candidates['saved_adversary']['validation_scores']['log_loss']
                else:
                    np.testing.assert_array_equal(mean,xf.mean(0));std=xf.std(0)
                    np.testing.assert_array_equal(scale,np.where(std>1e-12,std,1.));report['fitting_standardizers_checked']+=1
                for split,p in (('validation',vpool),('test','test')):
                    _,mask=check.labels(frames[p],target)
                    assert np.array_equal(predict(arrays[arm][p][mask]),probabilities[f'{split}/{key}/{cid}'])
                    report['new_prediction_sets_replayed']+=1
                if 'validation_curve' in candidate:
                    epochs=40 if role=='transfer' else 120;curve=candidate['validation_curve'];steps=int(np.ceil(len(idx)/256))
                    assert candidate['parameters']['epochs']==epochs and [r['epoch'] for r in curve]==list(range(0,epochs+1,5))
                    assert all(r['optimizer_steps']==steps*r['epoch'] for r in curve)
                    best=min(curve,key=lambda r:(r['validation_log_loss'],r['epoch']))
                    assert candidate['selected_epoch']==best['epoch'] and candidate['selected_optimizer_steps']==best['optimizer_steps']
                    assert candidate['validation_scores']['log_loss']==best['validation_log_loss']
                    assert candidate['optimizer_steps']==steps*epochs and candidate['training_row_exposures']==len(idx)*epochs
                    assert candidate['row_exposure_min']==candidate['row_exposure_max']==epochs
                    check.replay_schedule(candidate,len(idx))
                    old=old_selection['fitting_records'][f'{role}/C_init/{target}']['candidates'][cid]
                    assert all(candidate[f]==old[f] for f in ('schedule_hash','optimizer_steps','training_row_exposures'))
                    if cid!='catchup':assert candidate['initial_state_hash']==old['initial_state_hash']
                    report['MLP_curves_schedules_checked']+=1
                    schedule_groups.setdefault((seed,role,target,cid),set()).add((candidate['schedule_hash'],candidate['optimizer_steps']))
            if role=='audit':
                for field in ('optimizer_steps','training_row_exposures'):
                    independent_total=sum(candidates[c][field] for c in ('mlp_0','mlp_1'))
                    assert record['independent_total_mlp_'+field]==independent_total
                    assert record['total_new_mlp_'+field]==independent_total+candidates['catchup'][field]
        report['units'][f'{beta_name(beta)}/seed_{seed}'] = {'states': state_report, 'affine': affine_report,
            'metrics_sha256': check.sha(directory / 'metrics.json'),
            'selection_sha256': check.sha(directory / 'selection_before_test.json'),
            'new_candidate_rows': 106}
    for key, schedules in schedule_groups.items():
        assert len(schedules) == 1, (key, 'Same head/audit batch schedules across all beta/schedule/arm units')
    report.update(passed=not check.errors, errors=check.errors, max_absolute_metric_error=check.max_error,
        max_error_path=check.max_error_path, numeric_values_compared=check.numeric_comparisons,
        matched_MLP_schedule_groups=len(schedule_groups), runtime_seconds=time.perf_counter() - started)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/redesign_20260908_acs_preservation_v1')
    parser.add_argument('--report', type=Path, help='Fresh JSON file; default stdout; never overwritten')
    args = parser.parse_args()
    if args.report and args.report.exists():
        raise FileExistsError('Preserve completed evidence; choose a fresh report path')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        report = verify(args.out.resolve())
    rendered = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.report:
        with args.report.open('x') as handle:
            handle.write(rendered)
    else:
        print(rendered, end='')
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
