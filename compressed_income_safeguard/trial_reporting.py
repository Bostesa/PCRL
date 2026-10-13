"""New trial's frozen selection, reporting, and combined simultaneous family.

Import is metadata only. Numerical imports, model reads and person arrays are
confined to admitted queue workers. Arithmetic primitives belong to the runner
reporting module; independent verification never supplies runner arithmetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from . import reporting as math
from .reader_specs import BASE_READERS, EXTENDED_READERS

FAMILY = 'compressed_income_safeguard'
SERVICES = ('CONSTRAINED', 'ORDINARY', 'LOCAL_TWIN', 'TASK_ONLY', 'RAW_FULL',
            'RAW_HYBRID', 'RAW_PROJ', 'U0_FULL', 'U0_PROJ')
CONTROLS = ('RAW_PROJ', 'RAW_HYBRID', 'RAW_FULL', 'ORDINARY', 'LOCAL_TWIN',
            'TASK_ONLY', 'U0_PROJ', 'U0_FULL')
ARMS = ('CONSTRAINED', 'ORDINARY', 'LOCAL_TWIN', 'TASK_ONLY')
EPOCHS = (0, 5, 10, 20)
TASKS = ('income', 'occupation')
REFERENCES = ('U0', 'Ucal')
CAPS = {'accuracy_loss': .010, 'nll_excess': .010, 'brier_excess': .005}
READERS = BASE_READERS + EXTENDED_READERS
BOOTSTRAP_SEED = 2026101401
REPLICATES = 2048
K = 240


def table_name(prefix, service, seed):
    return f'{prefix}__{service}__S{seed}'


def score_keys(view):
    if view not in ('r1', 'r2', 'pair'):
        raise ValueError('unregistered reader view')
    keys = [f'{view}__{reader}' for reader in READERS]
    if view == 'pair':
        keys += [f'pair__IGNORE_{other}__{reader}' for other in ('r1', 'r2') for reader in READERS]
    return sorted(keys)


def _packages(packages):
    entries = packages['entries'] if isinstance(packages, dict) else packages
    entries = list(entries)
    identities = [(p.get('service', p.get('arm_id')), p.get('seed')) for p in entries]
    required = {(service, seed) for service in SERVICES for seed in (0, 1)}
    if len(identities) != 18 or set(identities) != required or len(set(identities)) != 18:
        raise ValueError('exactly nine final services and both distinct source seeds are mandatory')
    return entries


def build_registries(packages):
    """Derive new membership independently of every predecessor family."""
    _packages(packages)
    endpoints = []
    for service in SERVICES:
        for seed in (0, 1):
            name = table_name('metrics', service, seed)
            for task in TASKS:
                for ref in REFERENCES:
                    for metric, cap in CAPS.items():
                        base = metric.split('_')[0]
                        own, reference = f'own__{task}__{base}', f'REF_{ref}__{task}__{base}'
                        left, right = (reference, own) if base == 'accuracy' else (own, reference)
                        endpoints.append({'endpoint_id': f'utility__{service}__S{seed}__{task}__{ref}__{metric}',
                            'category': 'utility', 'service': service, 'arm_id': service, 'seed': seed,
                            'task': task, 'reference': ref, 'metric': metric,
                            'kind': 'mean_difference', 'direction': 'at_most', 'threshold': cap,
                            'paired_seeds': [{'source_seed': seed, 'left_table': name, 'left_key': left,
                                'right_table': name, 'right_key': right, 'sign': 1}]})
    for control in CONTROLS:
        for view, metric in (('pair', 'pair_benefit'), ('r1', 'income_worsening'), ('r2', 'occupation_worsening')):
            benefit = view == 'pair'
            parts = []
            for seed in (0, 1):
                candidate, comparison = table_name('scores', 'CONSTRAINED', seed), table_name('scores', control, seed)
                left, right = (comparison, candidate) if benefit else (candidate, comparison)
                parts.append({'source_seed': seed, 'truth_table': 'truth', 'truth_key': 'sex',
                    'left_table': left, 'left_keys': score_keys(view),
                    'right_table': right, 'right_keys': score_keys(view), 'sign': 1})
            endpoints.append({'endpoint_id': f'recovery__CONSTRAINED__{control}__{metric}',
                'category': 'recovery', 'service': 'CONSTRAINED', 'arm_id': 'CONSTRAINED',
                'candidate': 'CONSTRAINED', 'control': control, 'view': view, 'metric': metric,
                'kind': 'auc_envelope_difference', 'direction': 'at_least' if benefit else 'at_most',
                'threshold': .020 if benefit else .010, 'paired_seeds': parts,
                'failure_dependencies': [{'table': 'control_gates', 'key': f'{anchor}__S{seed}'}
                    for anchor in ('U0_FULL', 'RAW_FULL') for seed in (0, 1)]})
    assert len(endpoints) == K and len({e['endpoint_id'] for e in endpoints}) == K
    registry = {'schema': 'CIS1-frozen-endpoint-family-v1', 'family': FAMILY,
        'campaign': 'pcrl_compressed_income_safeguard_v1', 'K': K,
        'utility_count': 216, 'recovery_count': 24, 'service_count': 9,
        'services': list(SERVICES), 'control_tie_order': list(CONTROLS),
        'replicates': REPLICATES, 'bootstrap_seed': BOOTSTRAP_SEED, 'alpha': .05,
        'expected_rows': 2235, 'expected_groups': 2234, 'endpoints': endpoints,
        'group_order': 'numpy.unique(unit), sorted ascending',
        'bootstrap_draws': 'default_rng(seed).integers(0,G,size=(B,G)); counts reused for every service and fitted seed',
        'interval': 'point +/- Student-t(1-alpha/(2K),G-1) times paired-group bootstrap SD(ddof=1)',
        'reader_envelope': '18 applicable local readers; 18 direct pair plus 36 literal ignore-other aliases; maximum reselected in every draw',
        'exploratory_fixed_models_readers': True, 'post_INNER_endpoint_reduction_allowed': False}
    return {FAMILY: registry}


def validate_registries(registries):
    expected = build_registries([{'service': service, 'seed': seed} for service in SERVICES for seed in (0, 1)])
    if registries != expected:
        raise PermissionError('combined pre-INNER endpoint registry identity/procedure differs')


def nominate(records, starts):
    """Metadata arithmetic only; never consult INNER or U0/Ucal FIT gates."""
    starts = {r['seed']: r for r in starts}
    if set(starts) != {0, 1}:
        raise ValueError('both exact unchanged RAW_HYBRID starts required')
    identities = [(r['arm_id'], r['seed'], r['epoch']) for r in records]
    required = {(a, s, e) for a in ARMS for s in (0, 1) for e in EPOCHS}
    if len(identities) != 32 or set(identities) != required or len(set(identities)) != 32:
        raise ValueError('nomination requires all four epochs, four arms and both seeds')
    index = {identity: record for identity, record in zip(identities, records)}
    decisions = []
    for arm in ARMS:
        choices = []
        for epoch in EPOCHS:
            pair = [index[arm, seed, epoch] for seed in (0, 1)]
            normalized = []; clauses = []
            for seed, record in enumerate(pair):
                if record.get('INNER_labels_opened_or_hashed', False):
                    raise PermissionError('selection must precede INNER truth access')
                for task in TASKS:
                    own, baseline = record['metrics'][task]['metrics'], starts[seed]['metrics'][task]['metrics']
                    delta = {'accuracy_loss': baseline['accuracy']-own['accuracy'],
                        'nll_excess': own['nll']-baseline['nll'], 'brier_excess': own['brier']-baseline['brier']}
                    for metric, cap in CAPS.items():
                        value = delta[metric]
                        if not isinstance(value, (float, int)) or not __import__('math').isfinite(value):
                            raise ValueError('selection task metric must be finite')
                        normalized.append(value/cap-1)
                        clauses.append({'seed': seed, 'task': task, 'metric': metric,
                            'value': value, 'cap': cap, 'pass': value <= cap})
                if epoch == 0 and any(clause['value'] != 0 for clause in clauses if clause['seed'] == seed):
                    raise PermissionError('epoch zero must exactly witness RAW_HYBRID start task metrics')
            att = sum(r['complete_view_pair_auc'] for r in pair)/2
            nll = sum(r['metrics']['occupation']['metrics']['nll'] for r in pair)/2
            if not __import__('math').isfinite(att) or not 0 <= att <= 1:
                raise ValueError('ATT complete-view pair AUC must be finite')
            choices.append({'epoch': epoch, 'proxy_eligible': all(c['pass'] for c in clauses),
                'maximum_normalized_task_excess': max(normalized), 'paired_ATT_pair_auc': att,
                'paired_occupation_nll': nll, 'incremental_clauses': clauses})
        eligible = [c for c in choices if c['proxy_eligible']]
        selected = min(eligible, key=lambda c: (c['paired_ATT_pair_auc'], c['paired_occupation_nll'], c['epoch'])) if eligible else min(choices, key=lambda c: (c['maximum_normalized_task_excess'], c['paired_occupation_nll'], c['paired_ATT_pair_auc'], c['epoch']))
        decisions.append({'arm_id': arm, 'selected_epoch': selected['epoch'],
            'proxy_eligible': selected['proxy_eligible'], 'selected': selected, 'epochs': choices})
    return {'schema': 'CIS1-common-epoch-nomination-v1', 'status': 'NOMINEES_FROZEN',
        'nominees': decisions, 'selection_reference': 'exact unchanged RAW_HYBRID per seed',
        'common_epoch_both_source_seeds': True, 'both_tasks_incremental_caps': CAPS,
        'original_U0_Ucal_FIT_gate_applied': False, 'INNER_labels_opened_or_hashed': False,
        'source_training_saw_all_FIT': True, 'selection_role': 'FIT fold2 plus ATT validation',
        'post_assessment_selection_allowed': False}


def strongest_control(configurations):
    records = {r.get('service', r.get('arm_id')): r for r in configurations}
    if set(records) != set(SERVICES):
        raise ValueError('all nine service decisions required')
    eligible = [name for name in CONTROLS if records[name]['point_eligible_both_seeds_all24']]
    return min(eligible, key=lambda name: (records[name]['recovery_paired_seed_mean']['pair'], CONTROLS.index(name))) if eligible else None


def _registered(pin, registered):
    if not isinstance(pin, dict) or registered.get(str(Path(pin['path']).resolve())) != pin['sha256']:
        raise PermissionError('required source absent from active queue manifest')


def _queue_sources(active):
    state = json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    return {str(Path(p['path']).resolve()): p['sha256'] for p in state['units'][active['unit_id']]['spec']['sources']}


def _check_complete_scores(scores, truth):
    import numpy as np
    for view in ('r1', 'r2', 'pair'):
        if any(key not in scores for key in score_keys(view)):
            raise PermissionError('mandatory 18-local/54-pair complete-view reader is missing')
    for key, values in scores.items():
        if values.shape != (len(truth['row_id']),) or not np.isfinite(values).all():
            raise PermissionError('frozen score shape/range differs')
    for local, ignored in (('r1', 'r2'), ('r2', 'r1')):
        for reader in READERS:
            if not np.array_equal(scores[f'{local}__{reader}'], scores[f'pair__IGNORE_{ignored}__{reader}']):
                raise PermissionError('pair ignore-other alias is not the literal local curve')


def _envelopes(scores, sex):
    result = {}
    for view in ('r1', 'r2', 'pair'):
        aucs = {key: math._point_auc(sex, scores[key]) for key in score_keys(view)}
        strongest = max(sorted(aucs), key=aucs.get)
        result[view] = {'auc': aucs[strongest], 'strongest_reader': strongest,
            'reader_count': len(aucs), 'reader_auc': aucs}
    return result


def _metrics(truth, release):
    import numpy as np
    vectors = {key: truth[key] for key in ('row_id', 'unit')}; tasks = {}; clauses = []
    for i, task in enumerate(TASKS):
        classes = (2, 6)[i]; order = np.arange(classes); wire = release[f'r{i+1}']
        if wire.shape != (len(truth['row_id']), 16+2*classes+1) or not np.isfinite(wire).all():
            raise PermissionError('actual released wire schema/range differs')
        p = np.asarray(wire[:, 16+classes:16+2*classes], dtype=np.float64)
        if not np.array_equal(wire[:, -1], order[p.argmax(1)]):
            raise PermissionError('actual released decision differs from own probability')
        own = math.task_vectors(truth[task], p, order)
        vectors.update({f'own__{task}__{metric}': values for metric, values in own.items()})
        summary = {metric: float(value.mean()) for metric, value in own.items()}
        summary['confidence'] = float(p.max(1).mean())
        summary['class_order'] = list(map(int, order))
        summary['class_counts'] = [int((truth[task] == label).sum()) for label in order]
        summary['per_class_recall'] = [float((order[p.argmax(1)][truth[task] == label] == label).mean()) if (truth[task] == label).any() else None for label in order]
        refs = {}
        for ref in REFERENCES:
            rv = math.task_vectors(truth[task], truth['references'][f'{ref}_{i+1}'], order)
            vectors.update({f'REF_{ref}__{task}__{metric}': values for metric, values in rv.items()})
            baseline = {metric: float(value.mean()) for metric, value in rv.items()}
            delta = {'accuracy_loss': baseline['accuracy']-summary['accuracy'],
                'nll_excess': summary['nll']-baseline['nll'], 'brier_excess': summary['brier']-baseline['brier']}
            refs[ref] = {'metrics': baseline, 'difference': delta}
            clauses.extend({'task': task, 'reference': ref, 'metric': metric, 'value': delta[metric],
                'cap': cap, 'point_pass': delta[metric] <= cap} for metric, cap in CAPS.items())
        tasks[task] = {'metrics': summary, 'references': refs}
    return vectors, tasks, clauses


def trajectory_worker(spec_path):
    """Canonical batch1024 full-FIT forward, then slice inherited fold2."""
    from .evaluation_worker import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'admission')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from .source import load_manifest, load_role
    from .wires import prepare_geometry, forward_role
    from task_conditioned_representation.models import load_checkpoint
    manifest = load_manifest(verify(cfg['manifest']['path'], cfg['manifest']['sha256']), cfg['manifest']['sha256'])
    registered = _queue_sources(active)
    for field in ('features', 'income', 'occupation'):
        _registered(manifest['roles']['OSF_DEFENSE_FIT'][field], registered)
    model, payload = load_checkpoint(verify(cfg['checkpoint']['path'], cfg['checkpoint']['sha256']), cfg['checkpoint']['sha256'])
    if payload.get('seed') != cfg['seed']:
        raise PermissionError('trajectory source seed differs')
    binding = json.loads(verify(cfg['geometry']['path'], cfg['geometry']['sha256']).read_text())
    geometry = prepare_geometry(binding, source_seed=cfg['seed'])
    fit = load_role(manifest, cfg['seed'], 'OSF_DEFENSE_FIT', {'features', 'income', 'occupation'})
    guard()
    views = forward_role(model, fit['X'], fit['feature_names'], geometry, variant=cfg['service'])
    selected = np.flatnonzero(fit['fold'] == 2)
    if len(selected) != 5211 or len(np.flatnonzero(fit['fold'] != 2)) != 10223:
        raise PermissionError('inherited FIT fold role counts differ')
    arrays = {key: fit[key][selected] for key in ('row_id', 'unit', 'income', 'occupation')}; metrics = {}
    for i, task in enumerate(TASKS):
        p = views[i]['probability'][selected]; order = np.arange((2, 6)[i])
        prior = np.array([(fit[task] == label).mean() for label in order])
        metrics[task] = {'metrics': math.task_summary(fit[task][selected], p, order, prior)}
        arrays[f'p{i+1}'] = p
    root = math._private_directory(cfg['private_output'])
    pin = math._save_arrays(root/'FIT2_TASK_PROBABILITIES.npz', arrays)
    result = {'schema': 'CIS1-FIT2-incremental-selection-input-v1', 'unit_id': active['unit_id'],
        'arm_id': cfg['arm_id'], 'service': cfg['service'], 'seed': cfg['seed'], 'epoch': cfg['epoch'],
        'metrics': metrics, 'task_probability_sha256': pin['sha256'], 'checkpoint_sha256': cfg['checkpoint']['sha256'],
        'geometry_sha256': cfg['geometry']['sha256'], 'forward_full_FIT_before_fold2_slice': True,
        'role_forward_batch_size': 1024, 'original_U0_Ucal_FIT_gate_applied': False,
        'INNER_labels_opened_or_hashed': False, 'source_training_saw_all_FIT': True}
    math._write_json(root/'TRAJECTORY_COMPLETE.json', result, private=True)
    math._write_json(cfg['public_output'], result)
    return {'status': 'TRAJECTORY_COMPLETE', 'task_probability': pin, 'arm_id': cfg['arm_id'], 'seed': cfg['seed']}


def nomination_worker(spec_path):
    from .evaluation_worker import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'attacks')
    records = []
    for item in cfg['records']:
        record = json.loads(verify(item['trajectory']['path'], item['trajectory']['sha256']).read_text())
        freeze = json.loads(verify(item['freeze']['path'], item['freeze']['sha256']).read_text())
        if freeze.get('status') != 'FROZEN_COMPLETE' or freeze.get('source_seed') != record['seed']:
            raise PermissionError('nomination requires corresponding frozen reader bank')
        names = freeze.get('readers', [])
        if set(names) != set(READERS):
            raise PermissionError('nomination requires all18 applicable readers')
        recovery = freeze.get('ATT_validation', {}).get('pair', {}).get('contracts', {}).get('complete_view')
        if recovery is None or recovery.get('reader_count') != 54:
            raise PermissionError('nomination requires ATT complete 54-curve pair envelope')
        records.append({**record, 'complete_view_pair_auc': recovery['auc']})
    starts = [json.loads(verify(pin['path'], pin['sha256']).read_text()) for pin in cfg['starts']]
    if any(r.get('service') != 'RAW_HYBRID' for r in starts):
        raise PermissionError('exact RAW_HYBRID selection starts required')
    result = nominate(records, starts)
    result['unit_id'] = active['unit_id']
    math._write_json(cfg['public_output'], result)
    return result


def report_worker(spec_path):
    """One sealed INNER report derives all nine services' utility/recovery."""
    from .evaluation_worker import queue_config, require_reporting
    active, cfg, verify, guard = queue_config(spec_path, 'attacks')
    seal, lock = require_reporting(cfg, verify)
    math._registry_lock_binding(lock, cfg['endpoint_registries'])
    registries = {family: json.loads(verify(pin['path'], pin['sha256']).read_text()) for family, pin in cfg['endpoint_registries'].items()}
    validate_registries(registries)
    packages = _packages(cfg['packages'])
    receipts = {}
    for bank in packages:
        service = bank.get('service', bank.get('arm_id')); seed = bank['seed']
        receipt = json.loads(verify(bank['freeze']['path'], bank['freeze']['sha256']).read_text())
        if receipt.get('status') != 'FROZEN_COMPLETE' or receipt.get('source_seed') != seed or set(receipt.get('readers', [])) != set(READERS):
            raise PermissionError('complete frozen new-trial bank identity differs')
        for field, receipt_field in (('release', 'inner_release_sha256'), ('scores', 'score_sha256')):
            verify(bank[field]['path'], bank[field]['sha256'])
            if receipt.get(receipt_field) != bank[field]['sha256']:
                raise PermissionError('frozen report input hash differs')
        if receipt.get('bindings', {}).get('service_id') != service:
            raise PermissionError('frozen wire service identity differs')
        receipts[service, seed] = receipt
    from .source import load_manifest, load_role
    manifest = load_manifest(verify(cfg['manifest']['path'], cfg['manifest']['sha256']), cfg['manifest']['sha256'])
    registered = _queue_sources(active)
    for field in ('features', 'references_s0', 'references_s1'):
        _registered(manifest['roles']['INNER_SELECTION'][field], registered)
    for field in ('income', 'occupation', 'sex'):
        _registered(seal['roles']['INNER_SELECTION'][field], registered)
    import numpy as np
    truth = {seed: load_role(manifest, seed, 'INNER_SELECTION', {'features', 'income', 'occupation', 'sex', 'references'}, reporting_seal=seal) for seed in (0, 1)}
    math._aligned(truth[1], truth[0])
    if any(not np.array_equal(truth[0][key], truth[1][key]) for key in ('income', 'occupation', 'sex')):
        raise PermissionError('paired fitted seeds must share exact people/labels')
    root = math._private_directory(cfg['private_output']); public = Path(cfg['public_output']); public.mkdir(parents=True, exist_ok=True)
    identities = {key: truth[0][key] for key in ('row_id', 'unit')}
    unit_pin = math._save_arrays(root/'INNER_UNITS.npz', identities)
    arrays = {'truth': math._save_arrays(root/'INNER_TRUTH.npz', {key: truth[0][key] for key in ('row_id', 'unit', 'income', 'occupation', 'sex')})}
    records = []; csv_rows = []; gates = dict(identities)
    for bank in sorted(packages, key=lambda p: (p.get('service', p.get('arm_id')), p['seed'])):
        guard(); service = bank.get('service', bank.get('arm_id')); seed = bank['seed']
        release = math._arrays(bank['release'], verify); math._aligned(release, truth[seed])
        if set(release) != {'row_id', 'unit', 'r1', 'r2'}:
            raise PermissionError('recipient release contains forbidden extra fields')
        scores = math._arrays(bank['scores'], verify)
        if 'row_id' in scores or 'unit' in scores:
            math._aligned(scores, truth[seed]); scores = {k: v for k, v in scores.items() if k not in ('row_id', 'unit')}
        _check_complete_scores(scores, truth[seed])
        vectors, tasks, clauses = _metrics(truth[seed], release)
        recovery = _envelopes(scores, truth[seed]['sex'])
        record = {'service': service, 'arm_id': service, 'seed': seed, 'tasks': tasks, 'clauses': clauses,
            'recovery': recovery, 'point_eligible_seed': all(c['point_pass'] for c in clauses),
            'freeze_sha256': bank['freeze']['sha256'], 'complete_local_reader_counts': [18, 18], 'complete_pair_reader_count': 54}
        bankroot = root/f'{service}_S{seed}'; bankroot.mkdir(mode=0o700)
        arrays[table_name('metrics', service, seed)] = math._save_arrays(bankroot/'METRIC_VECTORS.npz', vectors)
        arrays[table_name('scores', service, seed)] = math._save_arrays(bankroot/'READER_SCORES.npz', identities | scores)
        if service in ('U0_FULL', 'RAW_FULL'):
            controls = math._report_control_scores(scores, truth[seed], receipts[service, seed])
            development = receipts[service, seed].get('ATT_validation_controls')
            valid = controls is not None and development is not None and all(v['passed'] for v in controls.values()) and all(v['passed'] for v in development.values())
            gates[f'{service}__S{seed}'] = np.full(len(identities['row_id']), valid, dtype=bool)
            record['controls'] = controls
        records.append(record)
        for task in TASKS:
            csv_rows.append({'service': service, 'seed': seed, 'task': task,
                **{k: tasks[task]['metrics'][k] for k in ('accuracy', 'nll', 'brier', 'confidence')},
                'income_auc': recovery['r1']['auc'], 'occupation_auc': recovery['r2']['auc'], 'pair_auc': recovery['pair']['auc'],
                'point_eligible_seed': record['point_eligible_seed'], 'exploratory': True})
    arrays['control_gates'] = math._save_arrays(root/'CONTROL_GATES.npz', gates)
    configurations = []
    for service in SERVICES:
        pair = [next(r for r in records if r['service'] == service and r['seed'] == seed) for seed in (0, 1)]
        configurations.append({'service': service, 'arm_id': service,
            'point_eligible_both_seeds_all24': all(r['point_eligible_seed'] for r in pair),
            'recovery_paired_seed_mean': {view: sum(r['recovery'][view]['auc'] for r in pair)/2 for view in ('r1', 'r2', 'pair')}})
    strongest = strongest_control(configurations)
    index = {'schema': 'CIS1-report-input-index-v1', 'status': 'REPORTED_FIXED_ROSTER', 'unit_id': active['unit_id'],
        'reporting_lock': seal, 'unit_input': unit_pin, 'arrays': arrays,
        'endpoint_registries': cfg['endpoint_registries'], 'configurations': configurations,
        'packages': records, 'strongest_eligible_control': strongest, 'control_tie_order': list(CONTROLS),
        'post_assessment_selection_allowed': False, 'exploratory_fixed_models_readers': True}
    index_pin = math._write_json(root/'REPORT_INPUT_INDEX.json', index, private=True)
    math._csv(public/'UTILITY_AND_RECOVERY.csv', csv_rows, ['service', 'seed', 'task', 'accuracy', 'nll', 'brier', 'confidence', 'income_auc', 'occupation_auc', 'pair_auc', 'point_eligible_seed', 'exploratory'])
    math._write_json(public/'REPORT_PACKAGE_METRICS.json', {'schema': 'CIS1-package-metrics-v1', 'packages': records,
        'reporting_lock_sha256': seal['sha256'], 'package_count': 18, 'service_count': 9, 'K': K,
        'configurations': configurations, 'strongest_eligible_control': strongest,
        'control_gate_validity': {key: bool(value.all()) for key, value in gates.items() if key not in ('row_id', 'unit')},
        'exploratory_assessment': True})
    return {'status': 'REPORTED_FIXED_ROSTER', 'input_index': index_pin, 'unit_input': unit_pin,
        'arrays': arrays, 'endpoint_registries': cfg['endpoint_registries']}


def evaluate_registries(registries, tables, units, *, progress=None):
    """K240 t-Bonferroni using inherited runner primitives and one draw bank."""
    from .scheduler import assert_active_from_env
    assert_active_from_env()
    validate_registries(registries)
    import numpy as np
    from scipy.stats import t
    rows = len(units['unit']); groups = len(np.unique(units['unit']))
    if rows != 2235 or groups != 2234 or len(np.unique(units['row_id'])) != rows:
        raise ValueError('canonical assessment people/groups differ')
    for table in tables.values():
        math._aligned(table, units)
    weights = np.vstack((np.ones((1, rows)), math.bootstrap_weights(units['unit'], B=REPLICATES, seed=BOOTSTRAP_SEED)))
    total = weights.sum(1); cache = {}; curves = {}; result = []; draws = {}
    critical = float(t.ppf(1-.05/(2*K), df=groups-1))
    def envelope(part, side):
        name = part[f'{side}_table']; keys = part[f'{side}_keys']; truth = tables[part['truth_table']][part['truth_key']]
        cache_key = (part['truth_table'], part['truth_key'], name, tuple(keys))
        if cache_key not in cache:
            scores = []
            for key in keys:
                score = tables[name][key]
                digest = hashlib.sha256(np.ascontiguousarray(truth).tobytes()+np.ascontiguousarray(score, dtype=np.float64).tobytes()).hexdigest()
                if digest not in curves:
                    curves[digest] = math.weighted_auc_draws(truth, score, weights)
                scores.append(curves[digest])
            cache[cache_key] = np.maximum.reduce(scores)
            if progress:
                progress(f'cached_envelopes={len(cache)} oriented_reader_curves={len(curves)}')
        return cache[cache_key]
    for endpoint in registries[FAMILY]['endpoints']:
        fields = {key: endpoint[key] for key in ('endpoint_id', 'category', 'service', 'arm_id', 'metric', 'direction', 'threshold')}
        fields.update({key: endpoint[key] for key in ('seed', 'task', 'reference', 'candidate', 'control', 'view') if key in endpoint})
        try:
            for dependency in endpoint.get('failure_dependencies', []):
                gate = tables[dependency['table']][dependency['key']]
                if gate.shape != (rows,) or gate.dtype.kind != 'b' or not np.all(gate == gate[0]):
                    raise ValueError('required recovery control gate must be constant aligned boolean')
                if not gate[0]:
                    raise KeyError('failed recovery control '+dependency['key'])
            parts = []
            for part in endpoint['paired_seeds']:
                if endpoint['kind'] == 'mean_difference':
                    delta = tables[part['left_table']][part['left_key']]-tables[part['right_table']][part['right_key']]
                    if delta.shape != (rows,) or not np.isfinite(delta).all():
                        raise ValueError('registered person metric shape/range differs')
                    parts.append((weights@delta)/total)
                else:
                    parts.append(envelope(part, 'left')-envelope(part, 'right'))
            expected = 1 if endpoint['category'] == 'utility' else 2
            if len(parts) != expected:
                raise ValueError('seed-specific utility/paired-seed recovery definition differs')
            values = np.mean(parts, axis=0); point = float(values[0]); boot = values[1:]
            sd = float(boot.std(ddof=1)); low, high = point-critical*sd, point+critical*sd
            status, point_status, gap = math.endpoint_classification(point, low, high, endpoint['threshold'], endpoint['direction'])
            row = fields | {'point': point, 'bootstrap_se': sd, 'lower': low, 'upper': high, 'status': status,
                'point_status': point_status, 'gap_to_target': gap, 'K': K, 'df': groups-1, 'critical': critical, 'replicates': REPLICATES}
            draws[endpoint['endpoint_id']] = boot
        except KeyError as error:
            row = fields | {'point': None, 'bootstrap_se': None, 'lower': None, 'upper': None, 'status': 'MISSING',
                'point_status': 'MISSING', 'gap_to_target': None, 'K': K, 'df': groups-1,
                'replicates': REPLICATES, 'missing_dependency': str(error)}
            draws[endpoint['endpoint_id']] = np.full(REPLICATES, np.nan)
        result.append(row)
    output = {'schema': 'CIS1-runner-combined-simultaneous-assessment-v1', 'family': FAMILY, 'K': K,
        'endpoints': result, 'rows': rows, 'groups': groups, 'replicates': REPLICATES, 'bootstrap_seed': BOOTSTRAP_SEED,
        'status_counts': {status: sum(r['status'] == status for r in result) for status in ('PASS', 'UNRESOLVED', 'VIOLATION', 'SHORTFALL', 'MISSING')},
        'complete_family': all(r['status'] != 'MISSING' for r in result),
        'approximate_exploratory_fixed_model_reader_coverage': True}
    return {FAMILY: output}, draws, {'cached_envelopes': len(cache), 'cached_oriented_reader_curves': len(curves)}


def inference_worker(spec_path):
    from .evaluation_worker import queue_config, require_reporting
    active, cfg, verify, guard = queue_config(spec_path, 'attacks')
    seal, lock = require_reporting(cfg, verify)
    math._registry_lock_binding(lock, cfg['endpoint_registries'])
    registries = {family: json.loads(verify(pin['path'], pin['sha256']).read_text()) for family, pin in cfg['endpoint_registries'].items()}
    validate_registries(registries)
    units = math._arrays(cfg['unit_input'], verify)
    tables = {name: math._arrays(pin, verify) for name, pin in cfg['arrays'].items()}
    results, draws, cache = evaluate_registries(registries, tables, units, progress=lambda value: print(value, flush=True))
    root = math._private_directory(cfg['private_output']); public = Path(cfg['public_output']); public.mkdir(parents=True, exist_ok=True)
    draw_pin = math._save_arrays(root/'ENDPOINT_BOOTSTRAP_DRAWS.npz', draws)
    result = results[FAMILY]; result.update(unit_id=active['unit_id'], reporting_lock_sha256=seal['sha256'], **cache)
    math._write_json(public/'COMBINED_INFERENCE.json', result)
    math._csv(public/'PRIMARY_ENDPOINTS.csv', result['endpoints'], ['endpoint_id', 'category', 'service', 'seed', 'task', 'reference', 'candidate', 'control', 'view', 'metric', 'direction', 'threshold', 'point', 'lower', 'upper', 'bootstrap_se', 'status', 'point_status', 'gap_to_target', 'K', 'df', 'replicates'])
    return {'status': 'INFERENCE_COMPLETE', 'bootstrap_draws': draw_pin,
        'families': {FAMILY: {'K': K, 'complete': result['complete_family'], 'status_counts': result['status_counts']}}, **cache}


def fixture_worker(spec_path):
    """Data-free registry boundaries plus inherited queued numerical fixture."""
    from .evaluation_worker import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'attacks')
    roster = [{'service': name, 'seed': seed} for name in SERVICES for seed in (0, 1)]
    registry = build_registries(roster)[FAMILY]
    checks = {'K240_derived': len(registry['endpoints']) == K,
        'utility216_recovery24': sum(e['category'] == 'utility' for e in registry['endpoints']) == 216,
        'new_bootstrap_identity': registry['bootstrap_seed'] == 2026101401,
        'complete18_and54': len(score_keys('r1')) == len(score_keys('r2')) == 18 and len(score_keys('pair')) == 54}
    for label, bad in (('missing_seed_rejected', roster[:-1]), ('duplicate_seed_rejected', roster[:-1]+[roster[0]]),
                       ('unknown_service_rejected', roster[:-1]+[{'service': 'OLD_ARM', 'seed': 1}])):
        try:
            build_registries(bad)
        except ValueError:
            checks[label] = True
        else:
            checks[label] = False
    changed = json.loads(json.dumps({FAMILY: registry})); changed[FAMILY]['K'] = 239
    try:
        validate_registries(changed)
    except PermissionError:
        checks['K_boundary_amendment_rejected'] = True
    else:
        checks['K_boundary_amendment_rejected'] = False
    import numpy as np
    y = np.array([0, 0, 1, 1]); score = np.array([.2, .5, .5, .8])
    checks['inherited_tie_weighted_auc'] = bool(np.allclose(math.weighted_auc_draws(y, score, np.array([[1., 1., 1., 1.], [1., 2., 3., 1.]])), [.875, .75], atol=1e-15, rtol=0))
    units = np.array([11, 11, 22, 33]); weights = math.bootstrap_weights(units, B=16, seed=BOOTSTRAP_SEED)
    checks['duplicate_group_shared_draws'] = bool(np.array_equal(weights[:, 0], weights[:, 1]))
    result = {'schema': 'CIS1-reporting-engineering-fixture-v1', 'unit_id': active['unit_id'],
        'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
        'scientific_arrays_opened': False, 'INNER_truth_opened_or_hashed': False}
    math._write_json(cfg['public_output'], result)
    if not all(checks.values()):
        raise AssertionError('new trial reporting engineering fixture failed')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=('registry', 'trajectory', 'nomination', 'report', 'inference', 'fixture'))
    parser.add_argument('--spec'); parser.add_argument('--packages'); parser.add_argument('--output')
    args = parser.parse_args()
    if args.phase == 'registry':
        registries = build_registries(json.loads(Path(args.packages).read_text()))
        root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
        result = {family: math._write_json(root/'COMBINED_ENDPOINT_REGISTRY.json', registry) for family, registry in registries.items()}
    else:
        result = {'trajectory': trajectory_worker, 'nomination': nomination_worker, 'report': report_worker,
                  'inference': inference_worker, 'fixture': fixture_worker}[args.phase](args.spec)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
