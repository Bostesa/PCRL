"""Frozen occupation assessment and runner-owned simultaneous inference.

Importing this module is metadata-only. Numerical imports and every person-array
read happen after the root queue and committed canonical reporting seal verify.
The runner implements its own metrics, tie-aware AUC and paired-group bootstrap;
it never imports the independent verifier's numerical implementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from .reader_specs import BASE_READERS

TASKS = ('income', 'occupation')
REFERENCES = ('U0', 'Ucal')
CAPS = {'accuracy_loss': .010, 'nll_excess': .010, 'brier_excess': .005}
BOOTSTRAP_SEED = 2026101301
REPLICATES = 2048
RAW_CONTROL = 'RAW_JOINT_R03_E5'


def table_name(prefix, arm, seed):
    return f'{prefix}__{arm}__S{seed}'


def primary_keys(view):
    keys = [f'{view}__{name}' for name in BASE_READERS]
    if view == 'pair':
        keys += [f'pair__IGNORE_{other}__{name}' for other in ('r1', 'r2') for name in BASE_READERS]
    return sorted(keys)


def control_dependencies(arm):
    dependencies = [RAW_CONTROL]
    if arm in ('U0', 'U0_PROJ'):
        dependencies.append('U0')
    elif arm.startswith('JOINT_CONDITIONAL'):
        dependencies.append('JOINT_CONDITIONAL_R03_E5')
    return sorted(set(dependencies))


def build_registries(packages):
    """Freeze all endpoint identities from metadata alone, before INNER truth."""
    entries = packages['entries'] if isinstance(packages, dict) else packages
    identities = [(p['arm_id'], p['seed']) for p in entries]
    arms = sorted({p['arm_id'] for p in entries})
    if len(arms) != 36 or len(identities) != 72 or len(set(identities)) != 72:
        raise ValueError('registry requires exactly36 paired configurations/72 distinct packages')
    if any((arm, seed) not in identities for arm in arms for seed in (0, 1)) or RAW_CONTROL not in arms:
        raise ValueError('registry misses a paired seed or the fixed RAW control')
    common = {'schema': 'OB1-frozen-endpoint-family-v1', 'configuration_count': 36,
              'replicates': REPLICATES, 'bootstrap_seed': BOOTSTRAP_SEED,
              'alpha': .05, 'expected_rows': 2235, 'expected_groups': 2234,
              'exploratory_fixed_models_readers': True, 'post_INNER_endpoint_reduction_allowed': False,
              'group_order': 'numpy.unique(unit), sorted ascending',
              'bootstrap_draws': 'default_rng(seed).integers(0,G,size=(B,G)); counts reused for every arm and seed',
              'interval': 'point +/- Student-t(1-alpha/(2K),G-1) times paired-group bootstrap SD(ddof=1)'}
    utility = []; privacy = []
    for arm in arms:
        for seed in (0, 1):
            name = table_name('metrics', arm, seed)
            for task in TASKS:
                for ref in REFERENCES:
                    for endpoint_metric, cap in CAPS.items():
                        metric = endpoint_metric.split('_')[0]
                        own_key = f'own__{task}__{metric}'
                        ref_key = f'REF_{ref}__{task}__{metric}'
                        left, right = (ref_key, own_key) if metric == 'accuracy' else (own_key, ref_key)
                        utility.append({'endpoint_id': f'utility__{arm}__S{seed}__{task}__{ref}__{endpoint_metric}',
                                        'arm_id': arm, 'seed': seed, 'task': task, 'reference': ref,
                                        'metric': endpoint_metric, 'kind': 'mean_difference',
                                        'direction': 'at_most', 'threshold': cap,
                                        'paired_seeds': [{'source_seed': seed, 'left_table': name, 'left_key': left,
                                                          'right_table': name, 'right_key': right, 'sign': 1}]})
        for view in ('pair', 'r1', 'r2'):
            benefit = view == 'pair'
            parts = []
            for seed in (0, 1):
                candidate = table_name('scores', arm, seed); control = table_name('scores', RAW_CONTROL, seed)
                left, right = (control, candidate) if benefit else (candidate, control)
                parts.append({'source_seed': seed, 'truth_table': 'truth', 'truth_key': 'sex',
                              'left_table': left, 'left_keys': primary_keys(view),
                              'right_table': right, 'right_keys': primary_keys(view), 'sign': 1})
            metric = 'pair_benefit' if benefit else f'{view}_worsening'
            privacy.append({'endpoint_id': f'privacy__{arm}__{metric}', 'arm_id': arm,
                            'view': view, 'metric': metric, 'control': RAW_CONTROL,
                            'kind': 'auc_envelope_difference', 'threshold': .020 if benefit else .010,
                            'direction': 'at_least' if benefit else 'at_most', 'paired_seeds': parts,
                            'failure_dependencies': [{'table': 'control_gates', 'key': f'{dependency}__S{seed}'}
                                                     for dependency in control_dependencies(arm) for seed in (0, 1)]})
    assert len(utility) == 864 and len(privacy) == 108
    return {'utility': {**common, 'family': 'utility', 'K': 864, 'endpoints': utility},
            'privacy': {**common, 'family': 'privacy', 'K': 108, 'endpoints': privacy}}


def _write_json(path, value, *, private=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644), 'w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _save_arrays(path, arrays):
    import numpy as np
    p = Path(path)
    with os.fdopen(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
        np.savez_compressed(stream, **arrays)
    return {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def _csv(path, rows, fields):
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def _private_directory(path):
    p = Path(path)
    if '.private' not in p.resolve().parts or p.exists():
        raise PermissionError('fresh private output directory required')
    p.mkdir(parents=True, mode=0o700)
    return p


def _arrays(pin, verify):
    import numpy as np
    verify(pin['path'], pin['sha256'])
    with np.load(pin['path'], allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def _aligned(arrays, identities):
    import numpy as np
    if any(key not in arrays or not np.array_equal(arrays[key], identities[key]) for key in ('row_id', 'unit')):
        raise PermissionError('person alignment differs from canonical assessment order')


def task_vectors(y, probability, order):
    import numpy as np
    y = np.asarray(y); p = np.asarray(probability, dtype=np.float64); order = np.asarray(order)
    if p.ndim != 2 or p.shape != (len(y), len(order)) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('invalid own probability shape/range')
    if not np.allclose(p.sum(1), 1., atol=1e-10, rtol=0.) or not np.isin(y, order).all() or len(np.unique(order)) != len(order):
        raise ValueError('fixed probability or class convention differs')
    target = y[:, None] == order
    return {'accuracy': (order[p.argmax(1)] == y).astype(np.float64),
            'nll': -(target*np.log(np.maximum(p, 1e-12))).sum(1),
            'brier': ((p-target)**2).sum(1)}


def task_summary(y, p, order, prior):
    import numpy as np
    v = task_vectors(y, p, order); decision = np.asarray(order)[p.argmax(1)]
    c = task_vectors(y, np.tile(prior, (len(y), 1)), order)
    present = [bool(np.any(y == k)) for k in order]
    recalls = [float((decision[y == k] == k).mean()) if has else None for k, has in zip(order, present)]
    return {**{key: float(value.mean()) for key, value in v.items()},
            'balanced_accuracy': sum(0. if r is None else r for r in recalls)/len(order),
            'class_order': list(map(int, order)), 'class_counts': [int((y == k).sum()) for k in order],
            'per_class_recall': recalls, 'per_class_recall_available': present,
            'confidence': float(p.max(1).mean()),
            'confidence_by_true_class': [float(p[y == k].max(1).mean()) if has else None for k, has in zip(order, present)],
            'true_class_probability_by_class': [float(p[y == k, j].mean()) if has else None for j, (k, has) in enumerate(zip(order, present))],
            'constant_prediction': {key: float(value.mean()) for key, value in c.items()},
            'constant_reference': 'inherited class prior over all FIT; source model saw all FIT',
            'gain_over_constant': {'accuracy': float(v['accuracy'].mean()-c['accuracy'].mean()),
                                   'nll': float(c['nll'].mean()-v['nll'].mean()),
                                   'brier': float(c['brier'].mean()-v['brier'].mean())}}


def weighted_auc_draws(y, score, weights, *, chunk=256):
    """Weighted Mann-Whitney AUC with half credit for exact score ties."""
    import numpy as np
    y = np.asarray(y); score = np.asarray(score, dtype=np.float64)
    if y.ndim != 1 or score.shape != y.shape or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2 or not np.isfinite(score).all():
        raise ValueError('binary truth or oriented score differs')
    if weights.ndim != 2 or weights.shape[1] != len(y) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('invalid shared row multiplicities')
    idx = np.argsort(score, kind='stable'); s = score[idx]; pos = y[idx] == 1
    cuts = np.r_[0, np.flatnonzero(s[1:] != s[:-1])+1]
    result = np.empty(len(weights), dtype=np.float64)
    for start in range(0, len(weights), chunk):
        w = weights[start:start+chunk, idx]
        positive = np.add.reduceat(w*pos, cuts, axis=1)
        negative = np.add.reduceat(w*(~pos), cuts, axis=1)
        denominator = positive.sum(1)*negative.sum(1)
        if np.any(denominator <= 0):
            raise ValueError('shared bootstrap draw lacks both protected classes')
        result[start:start+chunk] = (positive*(np.cumsum(negative, axis=1)-.5*negative)).sum(1)/denominator
    return result


def _point_auc(y, score):
    import numpy as np
    return float(weighted_auc_draws(y, score, np.ones((1, len(y))))[0])


def surface_summary(scores, sex):
    contracts = {'primary': lambda n: n in BASE_READERS,
                 'representation': lambda n: n.startswith('REP_'),
                 'output': lambda n: n.startswith('OUTPUT_'),
                 'logits_output': lambda n: n.startswith('LOGITS_OUTPUT_'),
                 'decision': lambda n: n.startswith('DECISION_'),
                 'confidence': lambda n: n.startswith('CONFIDENCE_')}
    result = {}
    for view in ('r1', 'r2', 'pair'):
        aucs = {key: _point_auc(sex, value) for key, value in scores.items()
                if key.startswith(view+'__') and not key.split('__')[1].startswith(('SHUFFLED', 'PLANTED'))}
        result[view] = {'reader_auc': aucs, 'contracts': {}}
        for contract, eligible in contracts.items():
            selected = {key: value for key, value in aucs.items() if eligible(key.split('__')[-1])}
            result[view]['contracts'][contract] = None if not selected else {
                'auc': max(selected.values()), 'strongest_reader': max(selected, key=selected.get), 'reader_count': len(selected)}
    return result


def _check_bank(bank, verify):
    receipt = json.loads(verify(bank['freeze']['path'], bank['freeze']['sha256']).read_text())
    if receipt.get('status') != 'FROZEN_COMPLETE' or receipt.get('source_seed') != bank['seed']:
        raise PermissionError('bank identity/status is not frozen')
    for name, field in (('release', 'inner_release_sha256'), ('scores', 'score_sha256'),
                        ('fit_release', 'FIT_selection_release_sha256'), ('fit_scores', 'FIT_selection_scores_sha256')):
        pin = bank[name]; verify(pin['path'], pin['sha256'])
        if receipt.get(field) != pin['sha256']:
            raise PermissionError('bank inputs differ from its exact frozen receipt')
    if receipt.get('bindings', {}).get('release_variant') != bank.get('release_variant', 'full'):
        raise PermissionError('release variant differs from frozen reader bindings')
    return receipt


def _registry_lock_binding(lock, pins):
    """Public lock stores hashes; operational configs carry private paths."""
    frozen = lock.get('endpoint_registries', {})
    if set(frozen) != set(pins) or any(frozen[family].get('sha256') != pin['sha256'] for family, pin in pins.items()):
        raise PermissionError('endpoint registry hashes differ from committed public lock')


def _load_bank_role(bank, role, truth, verify):
    import numpy as np
    release = _arrays(bank['release' if role == 'INNER' else 'fit_release'], verify)
    _aligned(release, truth)
    if set(release) != {'row_id', 'unit', 'r1', 'r2'}:
        raise PermissionError('recipient release carries forbidden fields')
    scores = _arrays(bank['scores' if role == 'INNER' else 'fit_scores'], verify)
    if 'row_id' in scores or 'unit' in scores:
        _aligned(scores, truth)
        scores = {key: value for key, value in scores.items() if key not in ('row_id', 'unit')}
    for view in ('r1', 'r2', 'pair'):
        if any(key not in scores for key in primary_keys(view)):
            raise PermissionError('frozen complete primary envelope missing reader or ignore alias')
    for key, values in scores.items():
        if values.shape != (len(truth['row_id']),) or not np.isfinite(values).all():
            raise PermissionError('frozen reader scores have invalid person alignment')
    for local, ignored in (('r1', 'r2'), ('r2', 'r1')):
        for key, values in scores.items():
            if key.startswith(local+'__') and key.split('__')[-1] in BASE_READERS:
                alias = 'pair__IGNORE_'+ignored+'__'+key.split('__', 1)[1]
                if alias not in scores or not np.array_equal(values, scores[alias]):
                    raise PermissionError('complete envelope ignore alias differs from its local reader')
    return release, scores


def _label_tables(fit, train):
    import numpy as np
    tables = {}; counts = {}
    for view, size, labels in (('r1', 2, fit['income']), ('r2', 6, fit['occupation']),
                                ('pair', 12, fit['income']*6+fit['occupation'])):
        count = np.zeros((size, 2), dtype=np.int64)
        np.add.at(count, (labels[train], fit['sex'][train]), 1)
        counts[view] = count; tables[view] = (count[:, 1]+1)/(count.sum(1)+2)
    return tables, counts


def _association(tables, truth):
    import numpy as np
    scores = {'r1': tables['r1'][truth['income']], 'r2': tables['r2'][truth['occupation']],
              'pair': tables['pair'][truth['income']*6+truth['occupation']]}
    return {view: {'auc': _point_auc(truth['sex'], values), 'rows': len(values),
                   'groups': len(np.unique(truth['unit'])), 'interpretation_only': True,
                   'orientation': 'positive SEX probability fixed by FIT01 Laplace1 table'} for view, values in scores.items()}, scores


def _strata(scores, truth):
    import numpy as np
    result = {}
    for view, size, labels in (('r1', 2, truth['income']), ('r2', 6, truth['occupation']),
                                ('pair', 12, truth['income']*6+truth['occupation'])):
        records = []
        for value in range(size):
            mask = labels == value
            count = [int(len(np.unique(truth['unit'][mask & (truth['sex'] == y)]))) for y in (0, 1)]
            rec = {'class_or_pair_id': value, 'rows': int(mask.sum()), 'SEX_group_counts': count,
                   'minimum_groups_each_SEX': 32, 'auc': None, 'status': 'MISSING_SPARSE_STRATUM'}
            if min(count) >= 32:
                aucs = {key: _point_auc(truth['sex'][mask], scores[key][mask]) for key in primary_keys(view)}
                rec.update(auc=max(aucs.values()), strongest_reader=max(aucs, key=aucs.get), status='ESTIMATED_DESCRIPTIVE')
            records.append(rec)
        result[view] = records
    return result


def _report_control_scores(scores, truth, receipt):
    """Score controls without changing ATT-frozen reader orientation."""
    import numpy as np
    from task_conditioned_representation.attacks import group_shuffle, INNER_NULL_SEED
    if not receipt.get('real_data_control_fits_requested'):
        return None
    shuffled = group_shuffle(truth['sex'], truth['unit'], INNER_NULL_SEED)
    result = {}
    for view in ('r1', 'r2', 'pair'):
        needed = [f'{view}__{name}' for name in ('SHUFFLED', 'PLANTED_0', 'PLANTED_1')]
        if not all(key in scores for key in needed):
            raise PermissionError('required real-data control score absent')
        sh = _point_auc(shuffled, scores[needed[0]])
        planted = np.where(truth['sex'] == 1, scores[needed[2]], scores[needed[1]])
        values = {'null_auc': _point_auc(truth['sex'], np.full(len(shuffled), .5)),
                  'shuffled_independent_auc': max(sh, 1-sh), 'planted_auc': _point_auc(truth['sex'], planted)}
        gates = {'null_exact': values['null_auc'] == .5, 'shuffle_at_most_0.60': values['shuffled_independent_auc'] <= .60,
                 'planted_at_least_0.99': values['planted_auc'] >= .99}
        result[view] = {'controls': values, 'gates': gates, 'passed': all(gates.values())}
    return result


def report_worker(spec_path):
    """One queue unit scores the complete frozen package roster once."""
    from .evaluation_worker import queue_config, require_reporting
    active, cfg, verify, guard = queue_config(spec_path, 'attacks')
    seal, lock = require_reporting(cfg, verify)
    # Check every frozen package and endpoint definition before any truth read.
    registries = {family: json.loads(verify(pin['path'], pin['sha256']).read_text())
                  for family, pin in cfg['endpoint_registries'].items()}
    _registry_lock_binding(lock, cfg['endpoint_registries'])
    expected = build_registries(cfg['packages'])
    if registries != expected:
        raise PermissionError('actual registry differs from complete pre-INNER metadata definition')
    dependency_map = {p['arm_id']: control_dependencies(p['arm_id']) for p in cfg['packages']}
    if cfg.get('control_dependencies') != dependency_map:
        raise PermissionError('control failure dependencies differ from prelocked shared slate rules')
    receipts = {(p['arm_id'], p['seed']): _check_bank(p, verify) for p in cfg['packages']}
    table_freeze = json.loads(verify(cfg['label_table_freeze']['path'], cfg['label_table_freeze']['sha256']).read_text())
    verify(cfg['label_table']['path'], cfg['label_table']['sha256'])
    if table_freeze.get('status') != 'FROZEN_COMPLETE' or table_freeze.get('training_role') != 'FIT folds0/1' or table_freeze.get('table_sha256') != cfg['label_table']['sha256']:
        raise PermissionError('C3 label table must be the frozen FIT01 development table')
    from .source import load_manifest, load_role
    manifest = load_manifest(verify(cfg['manifest']['path'], cfg['manifest']['sha256']), cfg['manifest']['sha256'])
    for role in ('OSF_DEFENSE_FIT', 'INNER_SELECTION'):
        fields = ('features', 'income', 'occupation', 'sex', 'references_s0', 'references_s1') if role == 'OSF_DEFENSE_FIT' else ('features', 'references_s0', 'references_s1')
        for field in fields:
            pin = manifest['roles'][role][field]; verify(pin['path'], pin['sha256'])
    # No truth hashes are computed by the metadata phase. The sealed lock must
    # now contain exact queue-admitted label pins before load_role can open them.
    state = json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    registered = {str(Path(x['path']).resolve()): x['sha256'] for x in state['units'][active['unit_id']]['spec']['sources']}
    for field in ('income', 'occupation', 'sex'):
        pin = seal['roles']['INNER_SELECTION'][field]
        if registered.get(str(Path(pin['path']).resolve())) != pin['sha256']:
            raise PermissionError('sealed INNER truth absent from active queue source manifest')
    import numpy as np
    fit = {}; inner = {}
    for seed in (0, 1):
        fit[seed] = load_role(manifest, seed, 'OSF_DEFENSE_FIT', {'features', 'income', 'occupation', 'sex', 'references'})
    _aligned(fit[1], fit[0])
    train = np.flatnonzero(fit[0]['fold'] != 2); selected = np.flatnonzero(fit[0]['fold'] == 2)
    if set(fit[0]['unit'][train]) & set(fit[0]['unit'][selected]):
        raise PermissionError('FIT table training and selection groups overlap')
    selected_truth = {seed: {key: fit[seed][key][selected] for key in ('row_id', 'unit', 'income', 'occupation', 'sex')} for seed in (0, 1)}
    tables, counts = _label_tables(fit[0], train)
    frozen = _arrays(cfg['label_table'], verify)
    if any(not np.array_equal(tables[view], frozen[view]) or not np.array_equal(counts[view], frozen[view+'__counts']) for view in tables):
        raise PermissionError('FIT01 label table differs from committed preassessment table')
    tables = {view: frozen[view] for view in tables}
    for seed in (0, 1):
        inner[seed] = load_role(manifest, seed, 'INNER_SELECTION', {'features', 'income', 'occupation', 'sex', 'references'}, reporting_seal=seal)
    _aligned(inner[1], inner[0])
    for field in ('income', 'occupation', 'sex'):
        if not np.array_equal(inner[0][field], inner[1][field]) or not np.array_equal(fit[0][field], fit[1][field]):
            raise PermissionError('same-person seed labels differ')
    root = _private_directory(cfg['private_output']); public = Path(cfg['public_output']); public.mkdir(parents=True, exist_ok=True)
    identities = {key: inner[0][key] for key in ('row_id', 'unit')}
    arrays = {'truth': _save_arrays(root/'INNER_TRUTH.npz', {key: inner[0][key] for key in ('row_id', 'unit', 'income', 'occupation', 'sex')})}
    unit_pin = _save_arrays(root/'INNER_UNITS.npz', identities)
    table_pin = _save_arrays(root/'FIT01_LABEL_TABLES.npz', {**tables, **{view+'__counts': c for view, c in counts.items()}})
    association = {}; association_pins = {}
    for role, truth in (('FIT2', selected_truth[0]), ('INNER', inner[0])):
        association[role], values = _association(tables, truth)
        association_pins[role] = _save_arrays(root/f'{role}_LABEL_TABLE_SCORES.npz', {key: truth[key] for key in ('row_id', 'unit')} | values)
    trajectory = []; surfaces = []; package_records = []; c3_banks = []; decision_aliases = []
    bank_lookup = {(p['arm_id'], p['seed']): p for p in cfg['packages']}
    for bank in sorted(cfg['packages'], key=lambda p: (p['arm_id'], p['seed'])):
        guard(); arm = bank['arm_id']; seed = bank['seed']; receipt = receipts[arm, seed]
        bankdir = root/f'{arm}_S{seed}'; bankdir.mkdir(mode=0o700)
        record = {'arm_id': arm, 'seed': seed, 'release_variant': bank.get('release_variant', 'full'),
                  'historical_point_ineligible': bank.get('historical_point_ineligible', False),
                  'freeze_sha256': bank['freeze']['sha256'], 'roles': {}}
        if bool(record['historical_point_ineligible']) != (arm == 'H_RAW_CSTAR'):
            raise PermissionError('historical RAW eligibility designation differs from frozen roster')
        for role in ('FIT2', 'INNER'):
            truth = selected_truth[seed] if role == 'FIT2' else inner[seed]
            release, scores = _load_bank_role(bank, role, truth, verify)
            if bank.get('release_variant') == 'head_row_space':
                full_arm = bank.get('source_arm', arm.removesuffix('_PROJ'))
                full_bank = bank_lookup[full_arm, seed]
                full_scores = _arrays(full_bank['scores' if role == 'INNER' else 'fit_scores'], verify)
                decision_keys = sorted(key for key in scores if key.startswith(('r1__', 'r2__', 'pair__')) and key.split('__')[-1].startswith('DECISION_'))
                if len(decision_keys) != 15:
                    raise PermissionError('projected anchor decision reader bank is incomplete')
                for key in decision_keys:
                    if key not in full_scores or not np.array_equal(scores[key], full_scores[key]):
                        raise PermissionError('decision-preserving full/projection contract produced distinct decision reader scores')
                decision_aliases.append({'arm_id': arm, 'source_arm': full_arm, 'seed': seed, 'role': role,
                                         'reader_keys': decision_keys, 'exact_score_alias': True,
                                         'extra_independent_scientific_evidence': False})
            aligned_scores = {key: truth[key] for key in ('row_id', 'unit')} | scores
            scores_pin = _save_arrays(bankdir/f'{role}_SCORES_ALIGNED.npz', aligned_scores)
            vectors = {key: truth[key] for key in ('row_id', 'unit')}; tasks = {}; clauses = []
            for index, task in enumerate(TASKS):
                order = np.arange(2 if index == 0 else 6); classes = len(order); wire = release[f'r{index+1}']
                if wire.shape != (len(truth['row_id']), 16+2*classes+1) or not np.isfinite(wire).all():
                    raise PermissionError('actual released head wire shape/range differs')
                p = np.asarray(wire[:, 16+classes:16+2*classes], dtype=np.float64)
                if not np.array_equal(wire[:, -1], order[p.argmax(1)]):
                    raise PermissionError('released actual decision differs from own probability argmax')
                prior = np.array([(fit[seed][task] == k).mean() for k in order])
                summary = task_summary(truth[task], p, order, prior); own = task_vectors(truth[task], p, order)
                vectors.update({f'own__{task}__{metric}': values for metric, values in own.items()})
                refs = {}; refs_source = fit[seed]['references'] if role == 'FIT2' else truth['references']
                for ref in REFERENCES:
                    rp = refs_source[f'{ref}_{index+1}'][selected] if role == 'FIT2' else refs_source[f'{ref}_{index+1}']
                    baseline = task_summary(truth[task], rp, order, prior); rv = task_vectors(truth[task], rp, order)
                    vectors.update({f'REF_{ref}__{task}__{metric}': values for metric, values in rv.items()})
                    difference = {'accuracy_loss': baseline['accuracy']-summary['accuracy'],
                                  'nll_excess': summary['nll']-baseline['nll'], 'brier_excess': summary['brier']-baseline['brier']}
                    refs[ref] = {'metrics': baseline, 'difference': difference}
                    for metric, cap in CAPS.items():
                        clauses.append({'task': task, 'reference': ref, 'metric': metric, 'value': difference[metric],
                                        'cap': cap, 'point_pass': difference[metric] <= cap})
                    trajectory.append({'phase': 'REPORT' if role == 'INNER' else 'FIT_DIAGNOSTIC', 'role': role,
                                       'arm_id': arm, 'seed': seed, 'task': task, 'reference': ref,
                                       'release_variant': bank.get('release_variant', 'full'),
                                       **{key: summary[key] for key in ('accuracy', 'nll', 'brier', 'balanced_accuracy', 'confidence')},
                                       **difference, 'accuracy_cap': .010, 'nll_cap': .010, 'brier_cap': .005,
                                       'point_eligible_seed': all(c['point_pass'] for c in clauses) if role == 'INNER' else '',
                                       'assessment_gate_applied': role == 'INNER', 'exploratory': True})
                tasks[task] = {'metrics': summary, 'references': refs}
            vectors_pin = _save_arrays(bankdir/f'{role}_METRIC_VECTORS.npz', vectors)
            recovery = surface_summary(scores, truth['sex'])
            for view, results in recovery.items():
                for contract, metric in results['contracts'].items():
                    surfaces.append({'phase': 'REPORT' if role == 'INNER' else 'FIT_DIAGNOSTIC', 'role': role,
                                     'arm_id': arm, 'seed': seed, 'release_variant': bank.get('release_variant', 'full'),
                                     'view': view, 'contract': contract, 'auc': None if metric is None else metric['auc'],
                                     'reader_count': 0 if metric is None else metric['reader_count'],
                                     'strongest_reader': None if metric is None else metric['strongest_reader'],
                                     'status': 'MISSING_CONTRACT_NOT_FITTED' if metric is None else 'ESTIMATED_DESCRIPTIVE',
                                     'exploratory': True})
            controls = _report_control_scores(scores, truth, receipt) if role == 'INNER' else None
            record['roles'][role] = {'tasks': tasks, 'clauses': clauses, 'recovery': recovery,
                                      'controls': controls, 'point_eligible_seed': all(c['point_pass'] for c in clauses) if role == 'INNER' else None,
                                      'metric_vectors': vectors_pin, 'scores_aligned': scores_pin,
                                      'assessment_gate_applied': role == 'INNER'}
            if role == 'INNER':
                arrays[table_name('metrics', arm, seed)] = vectors_pin
                arrays[table_name('scores', arm, seed)] = scores_pin
            c3_banks.append({'arm_id': arm, 'seed': seed, 'role': role,
                             'within_true_task_strata': _strata(scores, truth)})
        # Fill seed eligibility only after both tasks/references are evaluated.
        for row in trajectory:
            if row['arm_id'] == arm and row['seed'] == seed and row['role'] == 'INNER':
                row['point_eligible_seed'] = record['roles']['INNER']['point_eligible_seed']
        record['eligible_control'] = (record['roles']['INNER']['point_eligible_seed'] and not record['historical_point_ineligible'])
        package_records.append(record)
    by_identity = {(x['arm_id'], x['seed']): x for x in package_records}
    gates = dict(identities)
    for dependency in sorted({a for names in dependency_map.values() for a in names}):
        for seed in (0, 1):
            protected = by_identity[dependency, seed]['roles']['INNER']['controls']
            valid = protected is not None and all(item['passed'] for item in protected.values())
            development = receipts[dependency, seed].get('ATT_validation_controls')
            valid = valid and development is not None and all(item['passed'] for item in development.values())
            gates[f'{dependency}__S{seed}'] = np.full(len(identities['row_id']), valid, dtype=bool)
    arrays['control_gates'] = _save_arrays(root/'RECOVERY_CONTROL_GATES.npz', gates)
    configurations = []
    for arm in expected['utility']['endpoints']:
        if arm['arm_id'] in {c['arm_id'] for c in configurations}:
            continue
        arm_id = arm['arm_id']; records = [by_identity[arm_id, seed] for seed in (0, 1)]
        configurations.append({'arm_id': arm_id, 'point_eligible_both_seeds_all24': all(x['roles']['INNER']['point_eligible_seed'] for x in records),
                               'historical_point_ineligible': any(x['historical_point_ineligible'] for x in records),
                               'eligible_control': all(x['eligible_control'] for x in records),
                               'recovery_paired_seed_mean': {view: sum(x['roles']['INNER']['recovery'][view]['contracts']['primary']['auc'] for x in records)/2 for view in ('r1', 'r2', 'pair')}})
    index = {'schema': 'OB1-report-input-index-v1', 'unit_id': active['unit_id'], 'status': 'REPORTED_FIXED_ROSTER',
             'reporting_lock': seal, 'unit_input': unit_pin, 'arrays': arrays,
             'endpoint_registries': cfg['endpoint_registries'], 'packages': package_records,
             'configurations': configurations, 'post_assessment_selection_allowed': False,
             'exploratory_fixed_models_readers': True}
    index_pin = _write_json(root/'REPORT_INPUT_INDEX.json', index, private=True)
    _csv(public/'TRAJECTORY_METRICS.csv', trajectory, ['phase','role','arm_id','seed','release_variant','task','reference','accuracy','nll','brier','balanced_accuracy','confidence','accuracy_loss','nll_excess','brier_excess','accuracy_cap','nll_cap','brier_cap','point_eligible_seed','assessment_gate_applied','exploratory'])
    _csv(public/'RELEASE_SURFACE_COMPARISONS.csv', surfaces, ['phase','role','arm_id','seed','release_variant','view','contract','auc','reader_count','strongest_reader','status','exploratory'])
    _write_json(public/'C3_TASK_ASSOCIATION.json', {'schema': 'OB1-C3-frozen-report-v1', 'task_association': association,
                'training_role': 'FIT folds0/1', 'scoring_roles': ['FIT fold2', 'INNER_SELECTION'],
                'reader_banks': c3_banks, 'smoothing': 'Laplace1 per binary SEX per task class/label pair',
                'sparse_support_rule': 'at least32 distinct groups of each SEX per stratum; no pooling',
                'source_training_saw_all_FIT': True, 'universal_privacy_floor_established': False,
                'interpretation_only': True, 'reporting_lock_sha256': seal['sha256']})
    _write_json(public/'REPORT_SUMMARY.json', {'schema': index['schema'], 'status': index['status'],
                'reporting_lock_sha256': seal['sha256'], 'package_count': len(package_records),
                'configuration_count': len(configurations), 'configurations': configurations,
                'utility_K': 864, 'privacy_K': 108, 'exploratory_assessment': True})
    public_packages = [{**{key: record[key] for key in ('arm_id','seed','release_variant','historical_point_ineligible','eligible_control','freeze_sha256')},
                        'roles': {role: {key: value for key, value in payload.items() if key not in ('metric_vectors','scores_aligned')}
                                  for role, payload in record['roles'].items()}} for record in package_records]
    _write_json(public/'REPORT_PACKAGE_METRICS.json', {'schema': 'OB1-frozen-package-metrics-v1',
                'packages': public_packages, 'configuration_count': 36, 'package_count': 72,
                'reporting_lock_sha256': seal['sha256'], 'exploratory_assessment': True,
                'control_dependencies': dependency_map,
                'control_gate_validity': {key: bool(value.all()) for key, value in gates.items() if key not in ('row_id', 'unit')}})
    _write_json(public/'DECISION_READER_ALIASES.json', {'schema': 'OB1-fixed-decision-reader-aliases-v1',
                'aliases': decision_aliases, 'exact_duplicate_scores_counted_as_new_evidence': False})
    return {'status': index['status'], 'input_index': index_pin, 'unit_input': unit_pin, 'arrays': arrays,
            'endpoint_registries': cfg['endpoint_registries'], 'label_table': table_pin, 'association_scores': association_pins}


def bootstrap_weights(units, B=REPLICATES, seed=BOOTSTRAP_SEED):
    import numpy as np
    unique, inverse = np.unique(units, return_inverse=True)
    draws = np.random.default_rng(seed).integers(0, len(unique), size=(B, len(unique)))
    counts = np.stack([np.bincount(draw, minlength=len(unique)) for draw in draws])
    return counts[:, inverse].astype(np.float64)


def endpoint_classification(point, low, high, threshold, direction):
    if direction == 'at_most':
        return ('PASS' if high <= threshold else ('VIOLATION' if low > threshold else 'UNRESOLVED'),
                'PASS' if point <= threshold else 'VIOLATION', point-threshold)
    if direction == 'at_least':
        return ('PASS' if low >= threshold else ('SHORTFALL' if high < threshold else 'UNRESOLVED'),
                'PASS' if point >= threshold else 'SHORTFALL', threshold-point)
    raise ValueError('unregistered endpoint direction')


def evaluate_registries(registries, tables, units, *, progress=None):
    """Cache each oriented reader and each envelope for the shared draw bank."""
    import numpy as np
    from scipy.stats import t
    groups = len(np.unique(units['unit'])); rows = len(units['unit'])
    if rows != 2235 or groups != 2234 or len(np.unique(units['row_id'])) != rows:
        raise ValueError('canonical assessment people/groups differ')
    for name, table in tables.items():
        _aligned(table, units)
    weights = np.vstack((np.ones((1, rows)), bootstrap_weights(units['unit'])))
    total = weights.sum(1)
    auc_cache = {}; envelope_cache = {}; endpoints_out = {}; draws_out = {}
    def reader(y, scores):
        # Literal score aliases cache once, including pair ignores and seed
        # aliases. Truth is part of the key so unrelated binary labels cannot
        # accidentally share a curve.
        key = hashlib.sha256(np.ascontiguousarray(y).tobytes()+np.ascontiguousarray(scores, dtype=np.float64).tobytes()).hexdigest()
        if key not in auc_cache:
            auc_cache[key] = weighted_auc_draws(y, scores, weights)
        return auc_cache[key]
    def envelope(truth_table, truth_key, name, keys):
        key = (truth_table, truth_key, name, tuple(keys))
        if key not in envelope_cache:
            y = tables[truth_table][truth_key]
            envelope_cache[key] = np.maximum.reduce([reader(y, tables[name][k]) for k in keys])
            if progress:
                progress(f'cached_envelopes={len(envelope_cache)} oriented_reader_curves={len(auc_cache)}')
        return envelope_cache[key]
    for family, registry in registries.items():
        K = 864 if family == 'utility' else 108
        if registry['K'] != K or len(registry['endpoints']) != K or len({e['endpoint_id'] for e in registry['endpoints']}) != K:
            raise ValueError('complete family or endpoint identity differs')
        if registry['replicates'] != REPLICATES or registry['bootstrap_seed'] != BOOTSTRAP_SEED or registry['alpha'] != .05:
            raise ValueError('frozen bootstrap procedure differs')
        critical = float(t.ppf(1-.05/(2*K), df=groups-1)); out = []
        for endpoint in registry['endpoints']:
            try:
                for dependency in endpoint.get('failure_dependencies', []):
                    gate = tables[dependency['table']][dependency['key']]
                    if gate.shape != (rows,) or gate.dtype.kind != 'b' or not np.all(gate == gate[0]):
                        raise ValueError('recovery control gate must be a person-aligned constant bool vector')
                    if not bool(gate[0]):
                        raise KeyError('failed required recovery control '+dependency['key'])
                parts = []
                for part in endpoint['paired_seeds']:
                    sign = part.get('sign', 1)
                    if sign not in (-1, 1):
                        raise ValueError('endpoint sign must be +/-1')
                    if endpoint['kind'] == 'mean_difference':
                        delta = tables[part['left_table']][part['left_key']]-tables[part['right_table']][part['right_key']]
                        if delta.shape != (rows,) or not np.isfinite(delta).all():
                            raise ValueError('per-person metric difference convention differs')
                        parts.append(sign*((weights@delta)/total))
                    elif endpoint['kind'] == 'auc_envelope_difference':
                        left = envelope(part['truth_table'], part['truth_key'], part['left_table'], part['left_keys'])
                        right = envelope(part['truth_table'], part['truth_key'], part['right_table'], part['right_keys'])
                        parts.append(sign*(left-right))
                    else:
                        raise ValueError('unregistered endpoint kind')
                expected_parts = 1 if family == 'utility' else 2
                if len(parts) != expected_parts:
                    raise ValueError('seed-specific utility/paired-seed privacy requirement differs')
                values = np.mean(parts, axis=0); point = float(values[0]); boot = values[1:]
                se = float(boot.std(ddof=1)); low = point-critical*se; high = point+critical*se; threshold = endpoint['threshold']
                status, point_status, guard_excess = endpoint_classification(point, low, high, threshold, endpoint['direction'])
                row = {**{key: endpoint[key] for key in ('endpoint_id','arm_id','metric','direction','threshold')},
                       'point': point, 'bootstrap_se': se, 'lower': low, 'upper': high, 'status': status,
                       'point_status': point_status, 'gap_to_target': guard_excess, 'K': K,
                       'df': groups-1, 'critical': critical, 'replicates': REPLICATES}
                draws_out[endpoint['endpoint_id']] = boot
            except KeyError as error:
                row = {**{key: endpoint[key] for key in ('endpoint_id','arm_id','metric','direction','threshold')},
                       'point': None, 'bootstrap_se': None, 'lower': None, 'upper': None,
                       'status': 'MISSING', 'point_status': 'MISSING', 'gap_to_target': None, 'K': K,
                       'missing_dependency': str(error), 'df': groups-1, 'replicates': REPLICATES}
                draws_out[endpoint['endpoint_id']] = np.full(REPLICATES, np.nan)
            out.append(row)
        endpoints_out[family] = {'schema': 'OB1-runner-simultaneous-assessment-v1', 'family': family, 'K': K,
                                'endpoints': out, 'status_counts': {status: sum(x['status'] == status for x in out) for status in ('PASS','UNRESOLVED','VIOLATION','SHORTFALL','MISSING')},
                                'rows': rows, 'groups': groups, 'bootstrap_seed': BOOTSTRAP_SEED, 'replicates': REPLICATES,
                                'approximate_exploratory_fixed_model_reader_coverage': True,
                                'complete_family': all(x['status'] != 'MISSING' for x in out)}
    return endpoints_out, draws_out, {'cached_envelopes': len(envelope_cache), 'cached_oriented_reader_curves': len(auc_cache)}


def inference_worker(spec_path):
    from .evaluation_worker import queue_config, require_reporting
    active, cfg, verify, guard = queue_config(spec_path, 'attacks')
    seal, lock = require_reporting(cfg, verify)
    registries = {family: json.loads(verify(pin['path'], pin['sha256']).read_text()) for family, pin in cfg['endpoint_registries'].items()}
    # The canonical committed lock binds the exact endpoint bytes, not merely K.
    _registry_lock_binding(lock, cfg['endpoint_registries'])
    units = _arrays(cfg['unit_input'], verify)
    tables = {name: _arrays(pin, verify) for name, pin in cfg['arrays'].items()}
    results, draws, cache = evaluate_registries(registries, tables, units, progress=lambda m: print(m, flush=True))
    root = _private_directory(cfg['private_output']); public = Path(cfg['public_output']); public.mkdir(parents=True, exist_ok=True)
    bootstrap_pin = _save_arrays(root/'ENDPOINT_BOOTSTRAP_DRAWS.npz', draws)
    for family, result in results.items():
        result.update(unit_id=active['unit_id'], reporting_lock_sha256=seal['sha256'], **cache)
        _write_json(public/f'{family.upper()}_INFERENCE.json', result)
    gap = [{**row, 'family': family} for family, result in results.items() for row in result['endpoints']]
    _csv(public/'GAP_TO_TARGET.csv', gap, ['family','endpoint_id','arm_id','metric','direction','threshold','point','lower','upper','bootstrap_se','status','point_status','gap_to_target','K','df','replicates'])
    return {'status': 'INFERENCE_COMPLETE', 'bootstrap_draws': bootstrap_pin, 'families': {name: {'K': r['K'], 'complete': r['complete_family'], 'status_counts': r['status_counts']} for name, r in results.items()}, **cache}


def fixture_worker(spec_path):
    """Queued synthetic checks of arithmetic risks; no scientific sources."""
    from .evaluation_worker import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'attacks')
    import numpy as np
    checks = {}
    y = np.array([0, 0, 1, 1]); score = np.array([.2, .5, .5, .8])
    values = weighted_auc_draws(y, score, np.array([[1, 1, 1, 1], [1, 2, 3, 1]], dtype=float))
    checks['exact_tie_and_weighted_AUC'] = bool(np.allclose(values, [.875, .75], atol=1e-15, rtol=0))
    w = np.array([[1, 1, 1, 1], [10, 1, 1, 10], [1, 10, 10, 1]], dtype=float)
    a = weighted_auc_draws(y, np.array([0, 2, 1, 3]), w)
    b = weighted_auc_draws(y, np.array([2, 0, 3, 1]), w)
    checks['envelope_reselects_inside_each_draw'] = bool(a[1] > b[1] and b[2] > a[2] and np.maximum(a, b)[2] > a[2])
    p = np.array([[0., 1.], [.2, .8]]); before = p.copy()
    metrics = task_vectors(np.array([0, 1]), p, np.array([0, 1]))
    checks['NLL_floor_1e_minus12'] = bool(np.allclose(metrics['nll'], [-np.log(1e-12), -np.log(.8)], rtol=0, atol=1e-14))
    checks['Brier_is_sum_and_probability_unmutated'] = bool(np.allclose(metrics['brier'], [2., .08], rtol=0, atol=1e-14) and np.array_equal(p, before))
    summary = task_summary(np.array([0, 0]), np.array([[.9, .1], [.9, .1]]), np.array([0, 1]), np.array([.5, .5]))
    checks['missing_class_unavailable_but_balanced_fixedlist'] = bool(summary['per_class_recall'] == [1., None] and summary['balanced_accuracy'] == .5)
    groups = np.array([11, 11, 22, 33]); multiplicity = bootstrap_weights(groups, B=128)
    checks['duplicate_group_rows_share_multiplicity'] = bool(np.array_equal(multiplicity[:, 0], multiplicity[:, 1]) and len(np.unique(multiplicity.sum(1))) > 1)
    checks['both_seeds_reuse_identical_draws'] = bool(np.array_equal(multiplicity, bootstrap_weights(groups, B=128)))
    checks['caps_and_effect_threshold_classification'] = bool(
        endpoint_classification(.005, 0., .010, .010, 'at_most')[0] == 'PASS' and
        endpoint_classification(.012, .011, .013, .010, 'at_most')[0] == 'VIOLATION' and
        endpoint_classification(.005, -.01, .019, .020, 'at_least')[0] == 'SHORTFALL' and
        endpoint_classification(.020, .019, .021, .020, 'at_least')[0] == 'UNRESOLVED')
    result = {'schema': 'OB1-R8-synthetic-arithmetic-fixture-v1', 'unit_id': active['unit_id'],
              'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
              'scientific_arrays_opened': False, 'INNER_truth_opened_or_hashed': False}
    _write_json(cfg['public_output'], result)
    if not all(checks.values()):
        raise AssertionError('R8 synthetic arithmetic fixture failed')
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', required=True, choices=('report','inference','registry','fixture'))
    parser.add_argument('--spec'); parser.add_argument('--packages'); parser.add_argument('--output')
    args = parser.parse_args()
    if args.phase == 'registry':
        registries = build_registries(json.loads(Path(args.packages).read_text()))
        directory = Path(args.output); directory.mkdir(parents=True, exist_ok=True)
        result = {family: _write_json(directory/f'{family.upper()}_ENDPOINT_REGISTRY.json', registry) for family, registry in registries.items()}
    elif args.phase == 'report':
        result = report_worker(args.spec)
    elif args.phase == 'fixture':
        result = fixture_worker(args.spec)
    else:
        result = inference_worker(args.spec)
    print(json.dumps(result, sort_keys=True), flush=True)

if __name__ == '__main__':
    main()
