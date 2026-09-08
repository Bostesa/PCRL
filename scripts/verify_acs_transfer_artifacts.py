#!/usr/bin/env python3
"""Replay ACS artifact checks without fitting models or rewriting prior evidence.

Requires the original raw ACS CSV and local checkpoint/array artifacts. The
published compact JSON alone is insufficient. This is a portable adaptation of
the exact verification source preserved in INDEPENDENT_VERIFICATION.json.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import datetime
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.acs_transfer_data import FEATURES, NUMERIC, CATEGORICAL, CATEGORY_CODES, CovariatePreprocessor
from experiments.acs_transfer_models import SourceEncoder, source_scores, state_digest
ORIGINAL_RESULT_PREFIX = Path('results/redesign_20260907_acs_transfer_v1')
ORIGINAL_VERIFIER_SHA256 = '739d1f2e9623990dfb12e9a4ef6d0e36ee506c56c555f68cdab39f15094936cf'

def recorded_path(value, out):
    """Resolve original repository-relative manifest paths in a relocated run."""
    path = Path(value)
    try:
        return out / path.relative_to(ORIGINAL_RESULT_PREFIX)
    except ValueError:
        return ROOT / path

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ah(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()

def expect(condition, message):
    if not condition:
        raise AssertionError(message)

def known_codes(v, k):
    v = np.asarray(v, float)
    return np.where(np.isin(v, np.arange(1, k + 1)), v - 1, -1).astype(np.int64)

def targets(f, edges):
    income = f.PINCP.to_numpy(float)
    valid = np.isfinite(income) & (income >= -19998) & (income <= 4209995)
    binary = np.where(valid, income > 50000, -1).astype(np.int64)
    bins = np.where(valid, np.searchsorted(edges, income, side='right'), -1).astype(np.int64)
    esr, pub = (known_codes(f.ESR, 6), known_codes(f.PUBCOV, 2))
    complete = valid & (esr >= 0) & (pub >= 0)
    joint = np.where(complete, 4 * binary + 2 * (esr == 0) + (pub == 0), -1).astype(np.int64)
    mig = known_codes(f.MIG, 3)
    commute = f.JWMNP.to_numpy(float)
    cv = np.isfinite(commute) & (commute >= 1) & (commute <= 200) & (commute == np.floor(commute))
    return {'income_binary': binary, 'income_bins': bins, 'esr': esr, 'pubcov': pub, 'joint': joint, 'same_residence': np.where(mig < 0, -1, mig == 0).astype(np.int64), 'commute_over20': np.where(cv, commute > 20, -1).astype(np.int64), 'SEX': known_codes(f.SEX, 2), 'RAC1P': known_codes(f.RAC1P, 9)}

def verify(OUT):
    started = time.perf_counter()
    cfg = read(OUT / 'config.json')
    freeze = read(OUT / 'protocol_freeze.json')
    support = read(OUT / 'schema_support.json')
    for p, h in freeze['sha256'].items():
        expect(sha(recorded_path(p, OUT)) == h, 'frozen file ' + p)
    expect(sha(OUT / 'schema_support.json') == freeze['schema_support_sha256'], 'support freeze')
    expect(sha(ROOT / cfg['raw_path']) == support['raw_sha256'], 'raw data hash')
    columns = list(FEATURES) + ['SERIALNO', 'SPORDER', 'PWGTP', 'PINCP', 'ESR', 'PUBCOV', 'MIG', 'JWMNP', 'SEX', 'RAC1P']
    raw = pd.read_csv(ROOT / cfg['raw_path'], usecols=columns, dtype={'SERIALNO': str})
    raw['_raw_row'] = np.arange(len(raw), dtype=np.int64)
    eligible = raw[raw.AGEP.between(19, 34) & raw.PWGTP.gt(0)].drop_duplicates(['SERIALNO', 'SPORDER'])
    source_keys = ('income_binary', 'income_bins', 'esr', 'pubcov', 'joint')
    report = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'verification': 'independent artifact/data consistency and inference-only replay; no fitting or model selection', 'verification_code_sha256': sha(Path(__file__)), 'command': [sys.executable, *sys.argv], 'original_executed_verifier_sha256': ORIGINAL_VERIFIER_SHA256, 'publication_adaptation': True, 'frozen_source_protocol_config_hashes_verified': True, 'raw_data_sha256': support['raw_sha256'], 'raw_rows': len(raw), 'seeds': {}, 'limitations': ['Optimizer actions are checked against saved counters, epoch logs, checkpoints, and independently regenerated schedules; no optimizer updates were rerun.', 'Before/after integrity claims use recorded prior-state hashes plus independent current-file hashes and source-validation inference replay.', 'Only source-validation outputs are replayed here; no final-test outcomes were used to fit or select anything.', 'joblib object hashes are not serialization-invariant for these HistGB artifacts. Reloaded object hashes are reported but are not treated as file-integrity failures; saved file SHA256 checks, inference replay, and recorded in-process before/after hashes provide the stated evidence.', 'Recorded before/after hashes establish an in-process boundary check, not cryptographic proof of all historical execution actions.']}
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        cohorts = []
        for seed in cfg['seeds']:
            base = OUT / f'seed_{seed}'
            m = read(base / 'metrics.json')
            sel = read(base / 'source_selection.json')
            choice = read(base / 'selection_before_test.json')
            schema = {**sel['class_counts'], 'same_residence': 2, 'commute_over20': 2, 'SEX': 2, 'RAC1P': 9}
            with np.load(base / 'split_rows.npz') as z:
                rows = {k: z[k].copy() for k in z.files}
            cohorts.append(np.sort(np.concatenate(list(rows.values()))))
            expect(len(np.unique(cohorts[-1])) == len(cohorts[-1]), 'person overlap')
            households = {k: set(raw.iloc[v].SERIALNO) for k, v in rows.items()}
            for i, a in enumerate(rows):
                for b in list(rows)[i + 1:]:
                    expect(households[a].isdisjoint(households[b]), 'household overlap')
            selected_groups = set().union(*households.values())
            expected_whole = eligible[eligible.SERIALNO.isin(selected_groups)]._raw_row.to_numpy()
            expect(np.array_equal(np.sort(expected_whole), cohorts[-1]), 'whole cohort households retained')
            pool_y = {k: targets(raw.iloc[v], np.array(sel['income_edges'])) for k, v in rows.items()}
            for pool, indices in rows.items():
                rec = support['seeds'][str(seed)]['pools'][pool]
                expect(ah(indices) == rec['raw_row_hash'], 'raw row hash')
                expect(ah(np.asarray(sorted(households[pool]), dtype='U')) == rec['household_hash'], 'group hash')
                for name, y in pool_y[pool].items():
                    expected = {'valid': int((y >= 0).sum()), 'missing_or_inapplicable': int((y < 0).sum()), 'class_counts': np.bincount(y[y >= 0], minlength=schema[name]).tolist()}
                    expect(expected == rec['support'][name], f'support {pool}/{name}')
            prep = read(base / 'preprocessing.json')
            pre = CovariatePreprocessor()
            pre.numeric, pre.categories, pre.feature_names = (prep['numeric'], prep['categories'], prep['feature_names'])
            expect(prep['fitted_columns'] == list(FEATURES), 'feature allowlist')
            fitting_frame = raw.iloc[rows['representation_fit']]
            for name in NUMERIC:
                v = fitting_frame[name].to_numpy(float).copy()
                lo, hi = (0, 99) if name == 'AGEP' else (1, 99)
                v[~np.isfinite(v) | (v < lo) | (v > hi)] = np.nan
                median = float(np.nanmedian(v)) if np.isfinite(v).any() else 0.0
                filled = np.nan_to_num(v, nan=median)
                expect(np.array_equal(prep['numeric'][name], [median, float(filled.mean()), max(float(filled.std()), 1e-08)]), 'fitting-only numeric statistics')
            for name in CATEGORICAL:
                levels = sorted((int(v) for v in fitting_frame[name].dropna().unique() if v in CATEGORY_CODES[name]))
                expect(levels == prep['categories'][name], 'fitting-only categorical schema')
            with np.load(base / 'release_F_covariates.npz') as z:
                x = {k: z[k].copy() for k in z.files}
            for pool in x:
                expect(ah(pre.transform(raw.iloc[rows[pool]])) == ah(x[pool]), 'stored preprocessing replay')
            train_income = raw.iloc[rows['representation_fit']].PINCP.to_numpy(float)
            valid = np.isfinite(train_income) & (train_income >= -19998) & (train_income <= 4209995)
            expect(np.array_equal(np.unique(np.quantile(train_income[valid], np.arange(1, 8) / 8)), sel['income_edges']), 'fitting-only quantiles')
            nn_records = []
            initial_states = []
            neural = None
            for lr in cfg['source_lr_grid']:
                directory = base / 'source' / f'encoder_{lr}'
                meta = read(directory / 'metadata.json')
                expect(meta == sel['encoder_candidates'][str(lr)], 'source embedded metadata')
                expect(meta['source_target_order'] == list(source_keys), 'source whitelist')
                expect(meta['class_counts'] == sel['class_counts'], 'source schema')
                expect(not meta['downstream_labels_received'] and (not meta['final_test_received']), 'source access flags')
                for split, pool in [('source_fit', 'representation_fit'), ('source_validation', 'source_validation')]:
                    expect(meta['accessed_array_hashes'][split + '_features'] == ah(x[pool]), 'source feature input hash')
                    for name in source_keys:
                        expect(meta['accessed_array_hashes'][split + '_labels'][name] == ah(pool_y[pool][name]), 'source label hash')
                n = len(rows['representation_fit'])
                per_epoch = math.ceil(n / cfg['source_batch_size'])
                steps = per_epoch * cfg['source_epochs']
                expect(meta['optimizer_steps'] == meta['planned_batches'] == steps and meta['skipped_all_missing_batches'] == 0, 'neural budget')
                generator = torch.Generator().manual_seed(meta['schedule_seed'])
                digest = hashlib.sha256()
                for epoch in range(cfg['source_epochs']):
                    for batch in torch.randperm(n, generator=generator).split(cfg['source_batch_size']):
                        digest.update(len(batch).to_bytes(8, 'little'))
                        digest.update(batch.numpy().tobytes())
                expect(digest.hexdigest() == meta['schedule_hash'], 'source schedule reconstruction')
                history = meta['validation_history']
                expect(len(history) == cfg['source_epochs'] + 1, 'all-epoch source history')
                best = min(history, key=lambda v: (v['validation']['mean_source_loss'], v['epoch']))
                expect(best['epoch'] == meta['selected_epoch'] and best['optimizer_steps'] == meta['selected_optimizer_steps'], 'source selected epoch')
                expect(best['validation']['mean_source_loss'] == meta['validation_loss'], 'source selected loss')
                for row in history:
                    expect(row['optimizer_steps'] == row['epoch'] * per_epoch, 'source epoch step count')
                for stage, hashkey in [('initialization', 'initial_state_hash'), ('selected', 'selected_state_hash'), ('final', 'final_state_hash')]:
                    checkpoint = torch.load(directory / (stage + '.pt'), weights_only=True, map_location='cpu')
                    expect(state_digest(checkpoint['state_dict']) == meta[hashkey], 'checkpoint tensor hash ' + stage)
                    if stage == 'initialization':
                        expect(checkpoint['metadata']['optimizer_steps'] == checkpoint['metadata']['epoch'] == 0, 'true initialization counters')
                        initial_states.append(checkpoint['state_dict'])
                    else:
                        expected_steps = meta['selected_optimizer_steps'] if stage == 'selected' else steps
                        expect(checkpoint['metadata']['checkpoint_optimizer_steps'] == expected_steps, 'saved checkpoint selected/final counters')
                loaded = SourceEncoder.load(directory / 'selected.pt')
                expect(not loaded.training and (not any((p.requires_grad for p in loaded.parameters()))), 'source frozen')
                val = source_scores(loaded.probabilities(x['source_validation']), {k: pool_y['source_validation'][k] for k in source_keys}, sel['class_counts'])
                expect(abs(val['mean_source_loss'] - meta['validation_loss']) < 1e-08, 'saved source validation inference')
                if lr == sel['selected_lr']:
                    neural = loaded
                nn_records.append({'lr': lr, 'optimizer_steps': steps, 'selected_epoch': meta['selected_epoch'], 'selected_optimizer_steps': meta['selected_optimizer_steps'], 'validation_loss': meta['validation_loss'], 'initial_state_hash': meta['initial_state_hash'], 'schedule_hash': meta['schedule_hash']})
            expect(all((torch.equal(initial_states[0][k], initial_states[1][k]) for k in initial_states[0])), 'identical actual source initialization tensors')
            expect(len({r['schedule_hash'] for r in nn_records}) == 1, 'same source schedule')
            expect(min(nn_records, key=lambda z: (z['validation_loss'], z['lr']))['lr'] == sel['selected_lr'], 'source LR selection')
            tree_records = []
            tree = None
            for leaves in cfg['source_tree_leaf_grid']:
                directory = base / 'source' / f'tree_{leaves}'
                meta = read(directory / 'metadata.json')
                expect(meta == sel['tree_candidates'][str(leaves)], 'tree embedded metadata')
                expect(meta['accessed_array_hashes'] == sel['encoder_candidates'][str(cfg['source_lr_grid'][0])]['accessed_array_hashes'], 'tree/neural identical source examples/labels')
                expect(meta['config']['l2_regularization'] == 1 and meta['config']['early_stopping'] is False, 'tree objective/protocol')
                bank = joblib.load(directory / 'source_tree_bank.joblib')
                reloaded_tree_hash = joblib.hash((bank.models, bank.constants, bank.class_counts), hash_name='sha1')
                for name, record in meta['training'].items():
                    expected = 0 if record['constant_predictor'] else cfg['source_tree_iterations']
                    expect(record['boosting_iterations'] == expected, 'tree fixed iterations')
                p = bank.probabilities(x['source_validation'])
                for name, record in meta['training'].items():
                    expect(p[name].shape[1] == schema[name], 'tree fixed probability schema')
                    for absent in record['absent_schema_classes']:
                        expect(np.all(p[name][:, absent] == 0), 'tree absent probability column')
                val = source_scores(p, {k: pool_y['source_validation'][k] for k in source_keys}, sel['class_counts'])
                expect(abs(val['mean_source_loss'] - meta['validation_loss']) < 1e-08, 'tree source validation replay')
                if leaves == sel['selected_tree_leaves']:
                    tree = bank
                tree_records.append({'leaves': leaves, 'validation_loss': meta['validation_loss'], 'heads': meta['training'], 'recorded_preserialization_object_hash': meta['model_content_hash'], 'reloaded_object_hash': reloaded_tree_hash, 'object_hash_matches_after_reload': reloaded_tree_hash == meta['model_content_hash'], 'saved_source_validation_loss_replay': val['mean_source_loss']})
            expect(min(tree_records, key=lambda r: (r['validation_loss'], r['leaves']))['leaves'] == sel['selected_tree_leaves'], 'tree source validation selection')
            maps = joblib.load(base / 'release_maps.joblib')
            maps_reloaded_hash = joblib.hash({'tree': tree, 'pca': maps['pca'], 'compression': maps['compression'], 'preprocessing': pre})
            probabilities = neural.probabilities(x['source_validation'])
            latent = neural.release(x['source_validation'])
            tree_probs = tree.probabilities(x['source_validation'])
            replay = {'A_binary_bank': np.column_stack([probabilities['income_binary'][:, 1], probabilities['esr'][:, 0], probabilities['pubcov'][:, 0]]), 'B_rich_bank': np.concatenate([probabilities[k] for k in sel['bank_keys']], 1), 'C_tree_bank': np.concatenate([tree_probs[k] for k in sel['bank_keys']], 1), 'D_features': latent, 'D_compressed': maps['compression'].transform(latent), 'E_pca': maps['pca'].transform(x['source_validation']), 'F_covariates': x['source_validation']}
            release_arrays = {}
            replay_details = {}
            for name, rec in m['release_metadata'].items():
                with np.load(base / f'release_{name}.npz') as z:
                    arrays = {k: z[k].copy() for k in z.files}
                release_arrays[name] = arrays
                for pool, a in arrays.items():
                    expect(ah(a) == rec['development_hashes'][pool], 'frozen development release file ' + name)
                prediction = np.asarray(replay[name], dtype=np.float32)
                maxdelta = float(np.max(np.abs(prediction - arrays['source_validation'])))
                expect(np.allclose(prediction, arrays['source_validation'], atol=2e-06, rtol=2e-06), 'source-validation release replay ' + name)
                replay_details[name] = {'dimension': arrays['source_validation'].shape[1], 'bitwise_equal': ah(prediction) == ah(arrays['source_validation']), 'max_absolute_difference': maxdelta}
            index_summary = {}
            schedule_summary = {}
            for role, items, budget, base_seed, fitpool, valpool in [('transfer', ('same_residence', 'commute_over20'), cfg['head_budget'], 1230000, 'downstream_fit', 'downstream_validation'), ('audit', ('SEX', 'RAC1P'), cfg['attacker_budget'], 1240000, 'attacker_fit', 'attacker_validation')]:
                for j, target in enumerate(items):
                    y = pool_y[fitpool][target]
                    yv = pool_y[valpool][target]
                    idx = np.random.default_rng(base_seed + 100 * seed + j).permutation(np.flatnonzero(y >= 0))[:budget]
                    vi = np.flatnonzero(yv >= 0)
                    saved = choice['task_fit_indices' if role == 'transfer' else 'audit_fit_indices'][target]
                    expect(saved['n'] == len(idx) and saved['pool_indices_sha256'] == ah(idx) and (saved['raw_rows_sha256'] == ah(rows[fitpool][idx])), 'matched target rows')
                    group = [(key, record) for key, record in choice['fitting_records'].items() if key.startswith(role + '/') and key.endswith('/' + target) and ('/prior/' not in key)]
                    schedules = set()
                    counts = set()
                    targethashes = set()
                    initialization_seeds = set()
                    parameters = set()
                    for key, record in group:
                        _, release, _ = key.split('/')
                        expect(record['fit_hashes']['y'] == ah(y[idx]) and record['validation_hashes']['y'] == ah(yv[vi]), 'matched fitting/validation targets')
                        if release != 'exposed':
                            expect(record['fit_hashes']['x'] == ah(np.asarray(release_arrays[release][fitpool][idx], np.float64)), 'actual head input hash')
                            expect(record['validation_hashes']['x'] == ah(np.asarray(release_arrays[release][valpool][vi], np.float64)), 'actual head validation input hash')
                        chosen = min(record['validation_scores'], key=lambda f: (record['validation_scores'][f]['log_loss'], f))
                        expect(chosen == choice['head_selections'][key] == record['selected_family'], 'validation-only family selection')
                        mlp = record['candidates']['mlp']
                        expect('fallback_reason' not in mlp, 'no unreported MLP fallback')
                        steps = cfg['head_mlp_epochs'] * math.ceil(len(idx) / cfg['head_mlp_batch_size'])
                        expect(mlp['optimizer_steps'] == steps and mlp['training_row_exposures'] == len(idx) * cfg['head_mlp_epochs'], 'head fixed budgets')
                        g = np.random.default_rng(mlp['schedule_seed'])
                        h = hashlib.sha256()
                        for epoch in range(cfg['head_mlp_epochs']):
                            h.update(g.permutation(len(idx)).tobytes())
                        expect(h.hexdigest() == mlp['schedule_hash'], 'head schedule reconstructed')
                        expect(mlp['selected_optimizer_steps'] == mlp['selected_epoch'] * math.ceil(len(idx) / cfg['head_mlp_batch_size']), 'selected head step count')
                        best = min(mlp['validation_curve'], key=lambda r: (r['validation_log_loss'], r['epoch']))
                        expect(best['epoch'] == mlp['selected_epoch'], 'head validation-only checkpoint')
                        schedules.add(mlp['schedule_hash'])
                        counts.add(steps)
                        targethashes.add(record['fit_hashes']['y'])
                        initialization_seeds.add(mlp['initialization_seed'])
                        parameters.add(json.dumps(mlp['parameters'], sort_keys=True))
                        expect(mlp['row_exposure_min'] == mlp['row_exposure_max'] == cfg['head_mlp_epochs'] and mlp['restarts'] == 1, 'matched exposure/restart budget')
                        for family, candidate in record['candidates'].items():
                            meta = read(base / 'fitted' / key / family / 'metadata.json')
                            expect(meta == candidate, 'saved candidate metadata')
                            if family == 'histgb':
                                expect(meta['boosting_iterations'] == cfg['attacker_tree_iterations'] and meta['parameters']['l2_regularization'] == 1, 'audit tree fixed budget')
                            if family == 'logistic':
                                expect(meta['parameters']['C'] == 1 and meta['parameters']['max_iter'] == 500, 'fixed logistic configuration')
                    expect(len(schedules) == len(counts) == len(targethashes) == len(initialization_seeds) == len(parameters) == 1, 'matched heads across releases')
                    index_summary[role + '/' + target] = {'fit_rows': len(idx), 'validation_rows': len(vi), 'fit_valid_pool_rows': int((y >= 0).sum()), 'fit_missing_pool_rows': int((y < 0).sum())}
                    schedule_summary[role + '/' + target] = {'compared_interfaces': len(group), 'schedule_hash': next(iter(schedules)), 'optimizer_steps': next(iter(counts)), 'training_row_exposures': len(idx) * cfg['head_mlp_epochs'], 'initialization_seed': next(iter(initialization_seeds)), 'parameters': json.loads(next(iter(parameters)))}
            integrity = m['integrity']
            expect(sha(base / 'selection_before_test.json') == integrity['selection_sha256'], 'selection hash')
            expect(sha(base / 'source_selection.json') == choice['source_selection_sha256'], 'source selection hash')
            expect(sha(OUT / 'protocol_freeze.json') == choice['protocol_freeze_sha256'], 'protocol identity')
            expect(choice['created_utc'] < integrity['test_evaluation_started_utc'], 'test boundary timestamp')
            expect(integrity['source_state_before'] == integrity['source_state_after'] == state_digest(neural.state_dict()), 'source current/before/after tensors')
            expect(integrity['fitted_maps_before'] == integrity['fitted_maps_after'], 'recorded frozen maps')
            expect(all((integrity[k] for k in ['fitted_maps_unchanged', 'source_unchanged', 'development_releases_unchanged', 'selection_still_unchanged'])), 'integrity flags')
            manifest = read(base / 'local_artifacts.json')
            for file in manifest:
                expect(sha(recorded_path(file['path'], OUT)) == file['sha256'], 'saved artifact hash ' + file['path'])
            prior_metadata_complete = all(('validation_scores' in read(base / 'fitted' / key / 'prior/metadata.json') for key in choice['head_selections'] if '/prior/' in key))
            expect(prior_metadata_complete, 'standalone prior validation metadata')
            report['seeds'][str(seed)] = {'passed': True, 'pool_rows': {k: len(v) for k, v in rows.items()}, 'pool_households': {k: len(v) for k, v in households.items()}, 'row_and_household_isolation': True, 'whole_household_cohort': True, 'all_pool_label_masks_and_supports_verified': True, 'source_candidates': nn_records, 'tree_candidates': tree_records, 'selected_lr': sel['selected_lr'], 'selected_tree_leaves': sel['selected_tree_leaves'], 'class_counts': sel['class_counts'], 'joint_supported': sel['joint_supported'], 'source_fit_ESR_support': np.bincount(pool_y['representation_fit']['esr'], minlength=6).tolist(), 'source_input_hashes_and_whitelist_verified': True, 'matching_source_initial_tensors_verified': True, 'release_source_validation_replay': replay_details, 'head_rows': index_summary, 'head_schedules': schedule_summary, 'all_head_and_source_validation_selections_verified': True, 'saved_artifacts_checked': len(manifest), 'frozen_release_file_hashes_verified': True, 'frozen_model_before_after_verified': True, 'reloaded_maps_joblib_hash': maps_reloaded_hash, 'reloaded_maps_hash_matches_recorded': maps_reloaded_hash == integrity['fitted_maps_after'], 'prior_metadata_contains_validation_scores': prior_metadata_complete, 'selection_before_test_verified': True, 'selection_created_utc': choice['created_utc'], 'test_started_utc': integrity['test_evaluation_started_utc']}
        expect(all((np.array_equal(cohorts[0], c) for c in cohorts[1:])), 'common sampled population across seeds')
    report['all_seeds_passed'] = True
    report['same_sampled_cohort_across_seeds'] = True
    report['runtime_seconds'] = time.perf_counter() - started
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / ORIGINAL_RESULT_PREFIX, help='Existing ACS result directory; default is the canonical repository directory.')
    parser.add_argument('--report', type=Path, help='Write JSON to this fresh file; default prints JSON to stdout. Existing files are refused.')
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if not out.is_dir():
        parser.error(f'Result directory does not exist: {out}')
    if args.report is not None and args.report.exists():
        parser.error(f'Report already exists; choose a fresh destination: {args.report}')
    report = verify(out)
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.report is None:
        print(payload, end='')
    else:
        with args.report.open('x', encoding='utf-8') as handle:
            handle.write(payload)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
