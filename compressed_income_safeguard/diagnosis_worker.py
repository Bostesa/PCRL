"""Queue-owned pre-INNER C1/C2 diagnostics; imports have no numerical effects.

The coordinator registers explicit source pins and runs these operations. This
worker never launches another numerical worker or opens INNER truth. Original
source nomination choices are recorded, never replaced with diagnostic scores.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

CORE_FAMILIES = ('RAW_JOINT', 'JOINT_CONDITIONAL', 'LOCAL_TWIN')
ANCHORS = ('U0', 'RAW_JOINT_R03_E5', 'JOINT_CONDITIONAL_R03_E5')
ROLES = ('OSF_DEFENSE_FIT', 'ATTACK_FIT_NEW', 'INNER_SELECTION')
RANK_FIELDS = ('source_sensitive_statistic', 'occupation_nll', 'occupation_brier',
               'occupation_mean_confidence', 'ATT_pair_auc', 'ATT_occupation_auc')


def _fresh_private(path):
    p = Path(path)
    if '.private' not in p.resolve().parts or p.exists():
        raise PermissionError('fresh private diagnostic directory required')
    p.mkdir(parents=True, mode=0o700)
    return p


def geometry_worker(spec_path):
    from .evaluation_worker import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'admission')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from .source import load_manifest, load_role, write_json
    from .geometry import RULE, build_geometry, audit_geometry, projected_release, public_geometry
    from task_conditioned_representation.models import load_checkpoint
    from task_conditioned_representation.attacks import save_npz
    expected = {(arm, seed) for arm in ANCHORS for seed in (0, 1)}
    if {(x['arm_id'], x['seed']) for x in cfg['anchors']} != expected or len(cfg['anchors']) != 6:
        raise ValueError('C1 requires exactly the six prelocked anchor packages')
    manifest = load_manifest(verify(cfg['manifest'], cfg['manifest_sha256']), cfg['manifest_sha256'])
    # Verify feature sources explicitly against the active unit registration.
    for role in ROLES:
        pin = manifest['roles'][role]['features']; verify(pin['path'], pin['sha256'])
    data = {role: load_role(manifest, 0, role, {'features'}) for role in ROLES}
    root = _fresh_private(cfg['private_output'])
    geometries = []; audits = []; artifact_pins = []
    for anchor in cfg['anchors']:
        guard()
        model, payload = load_checkpoint(verify(anchor['checkpoint'], anchor['checkpoint_sha256']), anchor['checkpoint_sha256'])
        if payload['seed'] != anchor['seed']:
            raise PermissionError('anchor seed differs')
        package = root / (anchor['arm_id']+'_S'+str(anchor['seed'])); package.mkdir(mode=0o700)
        pair = [build_geometry(h.weight.detach().numpy(), h.bias.detach().numpy(), model.class_orders[i]) for i, h in enumerate(model.head)]
        for i, g in enumerate(pair):
            geometries.append({'arm_id': anchor['arm_id'], 'seed': anchor['seed'], 'recipient': i+1,
                               'checkpoint_sha256': anchor['checkpoint_sha256'], 'geometry': public_geometry(g)})
        for role, rows in data.items():
            guard(); views = model.route_raw(rows['X'], rows['feature_names'])
            arrays = {k: rows[k] for k in ('row_id', 'unit')}
            role_audits = []
            for i, (view, g) in enumerate(zip(views, pair)):
                audit = audit_geometry(view['representation'], g)
                projected = projected_release(view['representation'], g)
                actual_l = view['centered_logits']; actual_p = view['probability']; actual_d = view['decision']
                audit.update(actual_source_centered_logits_match=bool(np.allclose(actual_l, projected['centered_logits'], atol=RULE['logit_atol'], rtol=RULE['logit_rtol'])),
                             actual_source_probabilities_match=bool(np.allclose(actual_p, projected['probabilities'], atol=RULE['probability_atol'], rtol=RULE['probability_rtol'])),
                             actual_source_decisions_match=bool(np.array_equal(actual_d, projected['decision'])),
                             actual_source_logit_max_absolute_difference=float(np.max(np.abs(actual_l-projected['centered_logits']))),
                             actual_source_probability_max_absolute_difference=float(np.max(np.abs(actual_p-projected['probabilities']))))
                audit['passed'] = bool(audit['passed'] and audit['actual_source_centered_logits_match'] and audit['actual_source_probabilities_match'] and audit['actual_source_decisions_match'])
                audit.update(arm_id=anchor['arm_id'], seed=anchor['seed'], recipient=i+1, role=role,
                             checkpoint_sha256=anchor['checkpoint_sha256'])
                audits.append(audit); role_audits.append(audit)
                for name, value in (('z', view['representation']), ('logits', actual_l), ('probabilities', actual_p), ('decisions', actual_d)):
                    arrays[f'r{i+1}__source_{name}'] = value
                for name, value in projected.items():
                    arrays[f'r{i+1}__projected_{name}'] = value
            p = package / (role+'_PARITY.npz'); sha = save_npz(p, **arrays)
            artifact_pins.append({'arm_id': anchor['arm_id'], 'seed': anchor['seed'], 'role': role, 'path': str(p), 'sha256': sha})
    fit = data['OSF_DEFENSE_FIT']; eligible = np.flatnonzero(np.isin(fit['fold'], [0, 1]))
    if len(eligible) < 512:
        raise PermissionError('fixed FIT0/1 batch roster has fewer than512 rows')
    batches = [eligible[:256].tolist(), eligible[256:512].tolist()]
    batch_sha = write_json(root/'FIXED_FIT_BATCHES.json', {
        'schema': 'OB1-prelocked-fit-gradient-batches-v1', 'batch_indices': batches,
        'selection': 'first256 then next256 original FIT indices with inherited fold0/1, source row order',
        'feature_sha256': manifest['roles']['OSF_DEFENSE_FIT']['features']['sha256'],
        'INNER_labels_opened_or_hashed': False}, private=True)
    targets = cfg.get('gradient_checkpoints', [])
    required = {('RAW_JOINT_R03_E5', s) for s in (0, 1)} | {('RAW_JOINT_R3_E20', s) for s in (0, 1)}
    if len(targets) != 4 or {(x['arm_id'], x['seed']) for x in targets} != required:
        raise ValueError('fixed gradient checkpoint roster differs')
    gradient_specs = []
    for target in targets:
        verify(target['checkpoint'], target['checkpoint_sha256'])
        arm = target['arm_id']; seed = target['seed']; unit_id = 'OB_GRADIENT_'+arm+'_S'+str(seed)
        spec = {'operation': 'regenerate_fixed_fit_gradients', 'unit_id': unit_id,
                'manifest': cfg['manifest'], 'manifest_sha256': cfg['manifest_sha256'],
                'seed': seed, 'checkpoint': target['checkpoint'], 'checkpoint_sha256': target['checkpoint_sha256'],
                'batch_indices': batches, 'private_output': str(root / ('GRADIENT_'+arm+'_S'+str(seed))),
                'output': str(Path(cfg['gradient_public_root']) / (arm+'_S'+str(seed)+'.json'))}
        path = Path(cfg['gradient_spec_root']) / (unit_id+'.json')
        sha = write_json(path, spec, private=True)
        gradient_specs.append({'unit_id': unit_id, 'arm_id': arm, 'seed': seed, 'spec': {'path': str(path), 'sha256': sha},
                               'checkpoint': {'path': target['checkpoint'], 'sha256': target['checkpoint_sha256']},
                               'private_output': spec['private_output'], 'public_output': spec['output']})
    private = {'schema': 'OB1-C1-private-audit-index-v1', 'parity_artifacts': artifact_pins,
               'fixed_batch': {'path': str(root/'FIXED_FIT_BATCHES.json'), 'sha256': batch_sha}, 'gradient_specs': gradient_specs}
    private_sha = write_json(root/'PRIVATE_DIAGNOSIS_INDEX.json', private, private=True)
    out = {'schema': 'OB1-C1-real-head-geometry-v1', 'unit_id': active['unit_id'],
           'status': 'PASS' if all(x['passed'] for x in audits) else 'FAILED_PARITY',
           'geometry': geometries, 'audits': audits, 'audit_count': len(audits), 'package_count': 6,
           'private_audit_index_sha256': private_sha, 'fixed_batch_sha256': batch_sha,
           'gradient_units_prepared': len(gradient_specs), 'gradient_units_launched': 0,
           'INNER_labels_opened_or_hashed': False, 'new_fits': 0, 'public_head_parameters_available': True,
           'independent_numerical_verification': 'PENDING_SEPARATE_IMPLEMENTATION',
           'release_contract': 'factorized_Pz plus recomputed actual own outputs only; raw z/residual private audit only'}
    write_json(cfg['public_output'], out)
    return out


def _read_pin(pin, verify):
    if pin is None:
        return None
    return json.loads(verify(pin['path'], pin['sha256']).read_text())


def _auc(bank, view, contract='primary'):
    if bank is None:
        return None
    record = bank.get('ATT_validation', {}).get(view, {}).get('contracts', {}).get(contract)
    return None if record is None else record.get('auc')


def _sign(value):
    return None if value is None else (1 if value > 0 else -1 if value < 0 else 0)


def _contains_alias(receipt, original_sha, canonical_sha):
    packages = receipt.get('checkpoint_packages', [])
    originals = [p for p in packages if p.get('checkpoint_sha256') == original_sha
                 and p.get('exact_source_tensor_alias') is True and p.get('status') == 'PASS']
    canonicals = [p for p in packages if p.get('checkpoint_sha256') == canonical_sha and p.get('status') == 'PASS']
    return any(a.get('canonical_model_alias') == b.get('canonical_model_alias') and a.get('seed') == b.get('seed')
               for a in originals for b in canonicals)


def _ranks(rows, field):
    """Fixed ascending midranks. Missing values are retained as unavailable."""
    values = sorted(set(row[field] for row in rows if row[field] is not None))
    rank = {}; start = 1
    for value in values:
        count = sum(row[field] == value for row in rows)
        rank[value] = start+(count-1)/2; start += count
    return [None if row[field] is None else rank[row[field]] for row in rows]


def path_worker(spec_path):
    from .evaluation_worker import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'admission')
    from .source import write_json
    rows = []; aligned = []
    identities = set()
    for source in cfg['sources']:
        guard()
        identity = (source['family'], source['strength'], source['seed'], source['epoch'])
        if identity in identities:
            raise ValueError('duplicate C2 path state')
        identities.add(identity)
        selection = _read_pin(source.get('selection'), verify)
        logs = _read_pin(source.get('gradient_log'), verify)
        utility = _read_pin(source.get('utility'), verify)
        bank = _read_pin(source.get('bank'), verify)
        if selection is not None:
            if any(selection.get(k) != source[k] for k in ('family', 'strength', 'seed', 'epoch')) or selection.get('checkpoint_sha256') != source['checkpoint_sha256']:
                raise PermissionError('source training selection differs from frozen path checkpoint')
        canonical_sha = source.get('canonical_checkpoint_sha256', source['checkpoint_sha256'])
        if canonical_sha != source['checkpoint_sha256']:
            alias = _read_pin(source.get('alias_receipt'), verify)
            if alias is None:
                raise PermissionError('canonical epoch-zero alias requires pinned exact admission receipt')
            # Root supplies a precise entry from the independent admission alias
            # table, rather than a scientific-score-dependent alias choice.
            declared = source.get('alias_binding', {})
            if declared.get('checkpoint_sha256') != source['checkpoint_sha256'] or declared.get('canonical_checkpoint_sha256') != canonical_sha:
                raise PermissionError('alias checkpoint binding differs')
            if declared.get('exact_model_alias_verified') is not True:
                raise PermissionError('bitwise source alias evidence required')
            if not _contains_alias(alias, source['checkpoint_sha256'], canonical_sha):
                raise PermissionError('admission receipt lacks both pinned alias checkpoints')
        if utility is not None and (utility.get('seed') != source['seed'] or utility.get('checkpoint_sha256') != canonical_sha):
            raise PermissionError('fresh FIT utility checkpoint binding differs')
        if bank is not None:
            if bank.get('status') != 'FROZEN_COMPLETE' or bank.get('source_seed') != source['seed']:
                raise PermissionError('C2 ATT bank not frozen for exact source seed')
            bindings = bank.get('bindings', {})
            if bindings.get('checkpoint_sha256') != canonical_sha:
                raise PermissionError('C2 ATT bank checkpoint binding differs')
        metrics = None if utility is None else utility['metrics']['occupation']['metrics']
        source_metrics = None if selection is None else selection.get('metrics', [None, None])[1]
        row = {k: source[k] for k in ('arm_id', 'family', 'strength', 'seed', 'epoch', 'checkpoint_sha256')}
        row.update(source_sensitive_statistic=None if selection is None else selection.get('sensitive'),
                   source_sensitive_definition='RAW: saved critic entropy-normalized adversarial risk; other core families: conditional kernel objective, not AUC',
                   occupation_nll=None if metrics is None else metrics['nll'],
                   occupation_brier=None if metrics is None else metrics['brier'],
                   occupation_accuracy=None if metrics is None else metrics['accuracy'],
                   occupation_mean_confidence=None if metrics is None else metrics.get('confidence'),
                   saved_source_occupation_mean_confidence=None if source_metrics is None else source_metrics.get('mean_confidence'),
                   ATT_pair_auc=_auc(bank, 'pair'), ATT_income_auc=_auc(bank, 'r1'), ATT_occupation_auc=_auc(bank, 'r2'),
                   ATT_pair_output_auc=_auc(bank, 'pair', 'output'), ATT_pair_logits_output_auc=_auc(bank, 'pair', 'logits_output'),
                   source_selection_violation=None if selection is None else selection.get('violation'),
                   source_critic_ces=None if selection is None else selection.get('diagnostics', {}).get('critic_ces'),
                   fresh_ATT_reader_type_auc={} if bank is None else {view: bank.get('ATT_validation', {}).get(view, {}).get('reader_auc', {}) for view in ('r1','r2','pair')},
                   missing=[name for name, value in (('source_selection',selection),('saved_gradient_log',logs),('fresh_FIT_utility',utility),('frozen_ATT_bank',bank)) if value is None],
                   scientific_score_selection_performed=False)
        rows.append(row)
        # Logged minibatches are historical norm summaries, never regenerated tensors.
        window = [] if logs is None else [x for x in logs if x.get('epoch', -1) <= source['epoch'] and x.get('epoch', -1) > (0 if source['epoch'] <= 5 else 5 if source['epoch'] <= 10 else 10)]
        aligned.append({'arm_id': source['arm_id'], 'seed': source['seed'], 'epoch': source['epoch'],
                        'source_selection_sha256': None if source.get('selection') is None else source['selection']['sha256'],
                        'gradient_log_sha256': None if source.get('gradient_log') is None else source['gradient_log']['sha256'],
                        'saved_gradient_log_rows': window, 'logged_norms_are_regenerated_tensors': False,
                        'training_diagnostics': None if selection is None else selection.get('diagnostics')})
    paths = defaultdict(list)
    for row in rows:
        if row['family'] in CORE_FAMILIES:
            paths[(row['family'], row['strength'], row['seed'])].append(row)
    summaries = []; transitions = []
    for identity, states in sorted(paths.items()):
        states.sort(key=lambda x:x['epoch'])
        rank = {field: _ranks(states, field) for field in RANK_FIELDS}
        summary = {'family': identity[0], 'strength': identity[1], 'seed': identity[2],
                   'epochs': [x['epoch'] for x in states], 'missing_epochs': [e for e in (0,5,10,20) if e not in {x['epoch'] for x in states}],
                   'within_path_ascending_midranks': rank, 'rank_definition': 'ascending exact-value midranks, no correlation search'}
        summaries.append(summary)
        for previous, current in zip(states, states[1:]):
            change = {field: None if previous[field] is None or current[field] is None else current[field]-previous[field] for field in RANK_FIELDS}
            transitions.append({'family': identity[0], 'strength': identity[1], 'seed': identity[2],
                                'from_epoch': previous['epoch'], 'to_epoch': current['epoch'], 'differences': change,
                                'signs': {field:_sign(value) for field,value in change.items()},
                                'occupation_confidence_falls': None if change['occupation_mean_confidence'] is None else change['occupation_mean_confidence']<0,
                                'fresh_pair_recovery_falls': None if change['ATT_pair_auc'] is None else change['ATT_pair_auc']<0})
    paired = defaultdict(dict)
    for transition in transitions:
        key = (transition['family'],transition['strength'],transition['from_epoch'],transition['to_epoch'])
        paired[key][transition['seed']] = transition
    recurrence = []
    for key, seeds in sorted(paired.items()):
        recurrence.append({'family':key[0], 'strength':key[1], 'from_epoch':key[2], 'to_epoch':key[3],
                           'both_seeds_available':set(seeds)=={0,1},
                           'same_sign_both_seeds': {field: None if set(seeds)!={0,1} or any(seeds[s]['signs'][field] is None for s in (0,1)) else seeds[0]['signs'][field]==seeds[1]['signs'][field] for field in RANK_FIELDS}})
    root = _fresh_private(cfg['private_output'])
    private_sha = write_json(root/'SOURCE_DIAGNOSTIC_ALIGNMENT.json', {'schema':'OB1-C2-source-alignment-v1','records':aligned}, private=True)
    out = {'schema':'OB1-C2-pre-INNER-path-diagnosis-v1','unit_id':active['unit_id'],'status':'DIAGNOSTIC_COMPLETE',
           'checkpoint_records':rows,'paths':summaries,'transitions':transitions,'both_seed_signs':recurrence,
           'core_path_count':len(paths),'nominal_core_path_count':18,'private_source_alignment_sha256':private_sha,
           'source_nomination_rule_replaced':False,'fresh_ATT_orientation_changed':False,
           'source_training_saw_all_FIT':True,'FIT_utility_is_diagnostic_training_proxy':True,
           'stored_gradient_norms_are_regenerated_tensors':False,'INNER_labels_opened_or_hashed':False,
           'causal_bottleneck_established':False,'independent_numerical_verification':'PENDING_SEPARATE_IMPLEMENTATION'}
    write_json(cfg['public_output'], out)
    return out


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=('geometry','path'), required=True)
    parser.add_argument('--spec', required=True); args = parser.parse_args()
    out = {'geometry':geometry_worker,'path':path_worker}[args.phase](args.spec)
    print(json.dumps({'status':out['status'],'unit_id':out['unit_id']}))


if __name__ == '__main__':
    main()
