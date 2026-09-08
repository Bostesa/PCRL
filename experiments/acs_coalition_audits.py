"""Frozen coalition audit pools, exact inherited paths and nested budgets.

Only attacker fitting/validation arrays are accepted. Development scoring is a
separate operation. Historical fitters keep their old contracts; the new saved-
observer adapter adds the declared1/2/3/16/32 input widths in this module only.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.acs_transfer_heads import (FittedCandidate, InputStandardizer, _features,
    _labels, _hash_array, _state_hash, fit_candidates, metrics)
from experiments.acs_protection_audits import AUDIT_BUDGET
from experiments.acs_preservation_audits import fit_extended_auditors, save_extended_audits, _trajectory
from experiments.acs_bottleneck_catchup import CATCHUP_CONFIG, _direct_probabilities

TARGETS = ('SEX', 'RAC1P', 'income_binary', 'civilian_at_work', 'public_coverage', 'same_residence', 'commute_over20')
CLASSES = {name: 9 if name == 'RAC1P' else 2 for name in TARGETS}
VIEWS = ('A', 'B', 'AB')
AUDIT_ROLES = {'A': ('public_coverage', 'commute_over20', 'SEX', 'RAC1P'),
               'B': ('income_binary', 'civilian_at_work', 'same_residence', 'SEX', 'RAC1P'),
               'AB': ('SEX', 'RAC1P')}
OBSERVER_ROLES = tuple(f'{view}__{target}' for view in VIEWS for target in AUDIT_ROLES[view]
                       if target not in ('same_residence', 'commute_over20'))
FRESH = ('logistic', 'mlp_0', 'mlp_1', 'hist_gb_20', 'hist_gb_5')
SCOPES = ('standard_independent', 'expanded_independent', 'expanded_catchup')
FIT_POOLS = ('attacker_fit', 'attacker_validation')


def _write(path, value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def role_seed(seed, view, target, *, catchup=False):
    return (1300000 if catchup else 1260000)+100*int(seed)+10*VIEWS.index(view)+TARGETS.index(target)


def fit_static_auditors(xfit, yfit, xval, yval, n_classes, seed, *, budget=None):
    cfg = copy.deepcopy(AUDIT_BUDGET)
    if budget:
        for family, values in budget.items(): cfg[family].update(values)
    first = fit_candidates(xfit, yfit, xval, yval, n_classes, seed,
                           families=('logistic', 'histgb'), budget=cfg)['candidates']
    second_cfg = copy.deepcopy(cfg); second_cfg['histgb']['min_samples_leaf'] = 5
    second = fit_candidates(xfit, yfit, xval, yval, n_classes, seed,
                            families=('histgb',), budget=second_cfg)['candidates']['histgb']
    return {'logistic': first['logistic'], 'hist_gb_20': first['histgb'], 'hist_gb_5': second}


def fit_role_catchup(saved_model, xfit, yfit, xval, yval, n_classes, seed, *,
                     inherited_exposure, epochs=360, nested_epochs=120):
    """Exact saved-start adapter for declared wire widths; no normalizer fit."""
    started = time.perf_counter()
    xf, xv, yf, yv = _features(xfit), _features(xval), _labels(yfit, n_classes), _labels(yval, n_classes)
    width = xf.shape[1]
    if width not in (1, 2, 3, 16, 32) or n_classes not in (2, 9):
        raise ValueError('Undeclared observer input or class schema')
    if xf.shape != (len(yf), width) or xv.shape != (len(yv), width) or not len(yf) or not len(yv):
        raise ValueError('Aligned nonempty fitting/validation examples required')
    if not np.isfinite(xf).all() or not np.isfinite(xv).all(): raise ValueError('Finite original observer coordinates required')
    if not isinstance(saved_model, torch.nn.Sequential) or len(saved_model) != 5:
        raise ValueError('Observer must have the fixed three-linear/two-ReLU architecture')
    if not all(isinstance(saved_model[k], torch.nn.Linear) for k in (0, 2, 4)) or not all(isinstance(saved_model[k], torch.nn.ReLU) for k in (1, 3)):
        raise ValueError('Observer must have the fixed three-linear/two-ReLU architecture')
    if [(saved_model[k].in_features, saved_model[k].out_features) for k in (0, 2, 4)] != [(width, 64), (64, 32), (32, n_classes)]:
        raise ValueError('Observer dimensions do not match the released wire')
    if any(v.device.type != 'cpu' or v.dtype != torch.float32 or not torch.isfinite(v).all() for v in saved_model.state_dict().values()):
        raise ValueError('Finite float32 CPU observer tensors required')
    if not 0 < nested_epochs < epochs: raise ValueError('Require nested budget before total budget')
    initial_hash = _state_hash(saved_model)
    modes = [module.training for module in saved_model.modules()]
    flags = [parameter.requires_grad for parameter in saved_model.parameters()]
    identity = InputStandardizer(np.zeros(width), np.ones(width), fitted=False)
    saved = FittedCandidate('mlp', copy.deepcopy(saved_model).eval().requires_grad_(False), identity, n_classes, {})
    direct_fit, direct_val = _direct_probabilities(saved_model, xf), _direct_probabilities(saved_model, xv)
    assert np.array_equal(saved.predict_proba(xf), direct_fit) and np.array_equal(saved.predict_proba(xv), direct_val)
    parameters = {**copy.deepcopy(CATCHUP_CONFIG), 'input_dim': width, 'epochs': epochs}
    common = {'family': 'mlp', 'n_classes': n_classes, 'class_schema': list(range(n_classes)),
        'input_dim': width, 'seed': int(seed), 'parameters': parameters,
        'preprocessing': 'identity; exact original observer wire coordinates; no fitted normalizer',
        'preprocessing_fit_rows': 0, 'feature_mean': [0.]*width, 'feature_scale': [1.]*width,
        'validation_rows': len(yv), 'validation_hashes': {'x': _hash_array(xv), 'y': _hash_array(yv)},
        'source_state_hash': initial_hash, 'inherited_exposure': copy.deepcopy(inherited_exposure),
        'inherited_exposure_known': inherited_exposure is not None, 'initial_fidelity_exact': True,
        'original_fit_probability_hash': _hash_array(direct_fit), 'original_validation_probability_hash': _hash_array(direct_val),
        'device': 'cpu', 'fit_weighted': False, 'provided_attacker_fit_rows': len(yf),
        'provided_attacker_fit_support': np.bincount(yf, minlength=n_classes).tolist(),
        'scope': 'own saved observer exposure is separate from new frozen-wire attacker fitting'}
    saved.metadata = {**copy.deepcopy(common), 'audit_kind': 'saved', 'candidate_id': 'saved_adversary',
        'parameters': {**parameters, 'epochs': 0}, 'fit_rows': 0, 'fit_support': None, 'fit_hashes': None,
        'optimizer_steps': 0, 'training_row_exposures': 0, 'selected_epoch': 0, 'selected_optimizer_steps': 0,
        'selected_state_hash': initial_hash, 'validation_scores': metrics(yv, direct_val, n_classes),
        'selection': 'unchanged saved observer; diagnostic only'}
    common.update(fit_rows=len(yf), fit_support=np.bincount(yf, minlength=n_classes).tolist(),
        fit_coverage_complete=bool((np.bincount(yf, minlength=n_classes)>0).all()),
        fit_hashes={'x': _hash_array(xf), 'y': _hash_array(yf)})
    candidates, checkpoints = _trajectory(saved_model, identity, xf, yf, xv, yv, n_classes, seed,
                                           parameters, common, nested_epochs, catchup=True)
    assert _state_hash(saved_model) == initial_hash
    assert modes == [module.training for module in saved_model.modules()]
    assert flags == [parameter.requires_grad for parameter in saved_model.parameters()]
    for candidate in candidates.values():
        assert candidate.metadata['validation_curve'][0]['validation_log_loss'] == saved.metadata['validation_scores']['log_loss']
    return {'saved': saved, 'nested120': candidates[nested_epochs], 'nested360': candidates[epochs],
        'training_checkpoints': {'catchup': checkpoints},
        'metadata': {'config': parameters, 'nested_epochs': nested_epochs, 'source_state_hash': initial_hash,
            'initial_fidelity_exact': True, 'identity_coordinates_preserved': True, 'catchup_optimizer_reset': True,
            'source_unchanged': True, 'actual_new_optimizer_steps': candidates[epochs].metadata['optimizer_steps'],
            'actual_new_training_row_exposures': candidates[epochs].metadata['training_row_exposures'],
            'candidates': {'saved': saved.metadata, 'nested120': candidates[nested_epochs].metadata, 'nested360': candidates[epochs].metadata},
            'fit_runtime_seconds': time.perf_counter()-started}}


@dataclass(frozen=True)
class AuditCandidate:
    base: FittedCandidate
    space: str
    columns: tuple | None
    metadata: dict

    def predict_proba(self, bundle):
        if not isinstance(bundle, dict) or self.space not in bundle:
            raise ValueError('Provide only the recipient wire and available public-derived view')
        value = bundle[self.space]
        if self.columns is not None: value = value[:, self.columns]
        return self.base.predict_proba(np.ascontiguousarray(value))

    @property
    def family(self): return self.base.family


def _wrapped(base, cid, space, view, target, budget, path, *, reused=False, columns=None,
             source_view=None, source_cid=None, origin=None):
    source_view = source_view or view
    if origin is None:
        origin = 'saved_diagnostic' if cid == 'saved_adversary' else ('own_catchup' if cid == 'catchup' else ('public_composition' if space == 'derived' else 'fresh_wire'))
    metadata = {**copy.deepcopy(base.metadata), 'candidate_id': cid, 'candidate_origin': origin,
        'target': target, 'view': view, 'audit_budget': budget, 'space': space,
        'projection_columns': list(columns) if columns is not None else None,
        'source_view': source_view, 'source_candidate_id': source_cid or cid,
        'base_candidate_directory': str(Path(path).resolve()), 'base_metadata_sha256': _file_hash(Path(path)/'metadata.json'),
        'reused_historical_fit': reused, 'diagnostic_only': 'saved_adversary' in cid,
        'actual_auditor_input_dimension': base.metadata['input_dim'],
        'inherited_singleton': source_view != view, 'source_fit_hashes': base.metadata.get('fit_hashes'),
        'coordinate_route': 'public source-head probabilities' if space == 'derived' else 'released wire coordinates'}
    return AuditCandidate(base, space, columns, metadata)


def select_pools(candidates):
    pools = {'standard_independent': [], 'expanded_independent': [], 'expanded_catchup': []}
    for cid, candidate in candidates.items():
        meta = candidate.metadata
        if meta['diagnostic_only']: continue
        is_catchup = 'catchup' in cid
        if not is_catchup:
            pools['expanded_independent'].append(cid)
            if candidate.space == 'wire': pools['standard_independent'].append(cid)
        pools['expanded_catchup'].append(cid)
    choices = {}
    for scope, ids in pools.items():
        ids.sort()
        if not ids: raise ValueError('An authorized selection pool is empty')
        choices[scope] = min(ids, key=lambda cid: (candidates[cid].metadata['validation_scores']['log_loss'], cid))
    return choices, pools


def inherit_singletons(candidates, widths):
    """Add every sensitive singleton candidate, never a selected-only subset."""
    for target in ('SEX', 'RAC1P'):
        coalition = candidates['AB/'+target]
        for view in ('A', 'B'):
            for source_cid, source in candidates[view+'/'+target].items():
                width_a, width_b = widths[source.space]
                columns = tuple(range(width_a)) if view == 'A' else tuple(range(width_a, width_a+width_b))
                if source.columns is not None: raise AssertionError('Singleton source was already projected')
                cid = f'inherited_{view}__{source_cid}'
                origin = 'inherited_'+view+('__catchup' if 'catchup' in source_cid else ('__derived' if source.space == 'derived' else '__wire'))
                wrapped = _wrapped(source.base, cid, source.space, 'AB', target, source.metadata['audit_budget'],
                    source.metadata['base_candidate_directory'], reused=source.metadata['reused_historical_fit'], columns=columns,
                    source_view=view, source_cid=source_cid, origin=origin)
                coalition[cid] = wrapped


def _validate_inputs(wire, derived, labels, indices, interface):
    if interface not in ('F', 'P', 'E') or set(wire) != set(VIEWS): raise ValueError('Fixed F/P/E interface and A/B/AB views required')
    if set(labels) != set(FIT_POOLS) or any(set(values) != set(TARGETS) for values in labels.values()):
        raise ValueError('Fitting APIs accept only the two attacker pools and seven fixed targets')
    if set(indices) != set(TARGETS): raise ValueError('All seven original fitting subsets required')
    spaces = {'wire': wire}
    if interface == 'F':
        if derived is None: raise ValueError('F requires the public native-source probabilities')
        spaces['derived'] = derived
    elif derived is not None:
        if interface == 'E': raise ValueError('E has no native heads')
        for view in VIEWS:
            for pool in FIT_POOLS:
                if not np.array_equal(derived[view][pool], wire[view][pool]): raise ValueError('P derived view must equal its wire')
    expected = {'F': (16, 16, 32), 'P': (2, 1, 3), 'E': (16, 16, 32)}[interface]
    for space, views in spaces.items():
        if set(views) != set(VIEWS): raise ValueError('A/B/AB required')
        dims = expected if space == 'wire' else (2, 1, 3)
        for view, dim in zip(VIEWS, dims):
            if set(views[view]) != set(FIT_POOLS): raise ValueError('Development arrays are forbidden during fitting')
            for pool in FIT_POOLS:
                x = _features(views[view][pool])
                if x.shape != (len(labels[pool]['SEX']), dim) or not np.isfinite(x).all(): raise ValueError('View shape/finite mismatch')
        for pool in FIT_POOLS:
            if not np.array_equal(views['AB'][pool], np.column_stack((views['A'][pool], views['B'][pool]))):
                raise ValueError('Coalition must be the exact same-person A/B concatenation')
    if interface == 'E' and any(not np.array_equal(wire['A'][p], wire['B'][p]) for p in FIT_POOLS):
        raise ValueError('Static E requires duplicate singleton outputs')
    for target, ix in indices.items():
        ix = np.asarray(ix)
        if ix.ndim != 1 or ix.dtype.kind not in 'iu' or not len(ix) or len(np.unique(ix)) != len(ix): raise ValueError('Unique nonempty fitting indices required')
        if (ix < 0).any() or (ix >= len(labels['attacker_fit'][target])).any(): raise ValueError('Fitting index out of bounds')
        if (np.asarray(labels['attacker_fit'][target])[ix] < 0).any(): raise ValueError('Fitting indices include missing labels')
    return spaces


def _validate_reuse(reused, xf, yf, xv, yv, n_classes, seed, epochs, nested_epochs):
    for tag, boundary in (('nested120', nested_epochs), ('nested360', epochs)):
        candidates = reused[tag]['candidates']
        if set(candidates) != set(FRESH): raise ValueError('Reusable fixed five-candidate set required')
        for cid, candidate in candidates.items():
            metadata = candidate.metadata
            expected_seed = seed+(10000 if cid == 'mlp_1' else 0)
            assert metadata['seed'] == expected_seed and metadata['n_classes'] == n_classes
            assert metadata['fit_hashes'] == {'x': _hash_array(_features(xf)), 'y': _hash_array(_labels(yf, n_classes))}
            assert metadata['validation_hashes'] == {'x': _hash_array(_features(xv)), 'y': _hash_array(_labels(yv, n_classes))}
            assert metadata['validation_scores'] == metrics(yv, candidate.predict_proba(xv), n_classes)
            expected_parameters = copy.deepcopy(AUDIT_BUDGET[metadata['family']])
            if cid.startswith('mlp'): expected_parameters['epochs'] = boundary
            if cid == 'hist_gb_5': expected_parameters['min_samples_leaf'] = 5
            assert metadata['parameters'] == expected_parameters
            path = Path(reused['paths'][boundary][cid])
            assert json.loads((path/'metadata.json').read_text()) == metadata


def fit_condition_audits(wire, derived, labels, fit_indices, observers, seed, directory, *, interface,
                         inherited_exposure=None, reused=None, epochs=360, nested_epochs=120,
                         miniature=False, budget=None):
    """Return opaque legal-input candidates and three validation-only pools."""
    if not miniature and (epochs != 360 or nested_epochs != 120 or budget is not None): raise ValueError('Frozen scientific audit budget changed')
    spaces = _validate_inputs(wire, derived, labels, fit_indices, interface)
    if interface == 'E':
        if observers: raise ValueError('Static E has no inherited observers')
    elif set(observers or {}) != set(OBSERVER_ROLES): raise ValueError('All nine genuine observer roles required')
    out = Path(directory); out.mkdir(parents=True, exist_ok=False)
    inherited_exposure, reused = inherited_exposure or {}, reused or {}
    candidates = {b: {} for b in (nested_epochs, epochs)}
    runtime = {'fresh_seconds': 0., 'catchup_seconds': 0., 'reuse_verification_seconds': 0.}
    counts = {'new_five_candidate_roles': 0, 'reused_five_candidate_roles': 0, 'own_catchup_trajectories': 0,
              'new_fresh_mlp_trajectories': 0, 'new_static_candidates': 0, 'P_derived_roles_deduplicated': 11 if interface == 'P' else 0}
    for view in VIEWS:
        for target in AUDIT_ROLES[view]:
            role = view+'/'+target
            ix = np.asarray(fit_indices[target]); val = np.asarray(labels['attacker_validation'][target]) >= 0
            yf, yv = np.asarray(labels['attacker_fit'][target])[ix], np.asarray(labels['attacker_validation'][target])[val]
            classes = CLASSES[target]
            for b in candidates: candidates[b][role] = {}
            for space, views in spaces.items():
                xf, xv = views[view]['attacker_fit'][ix], views[view]['attacker_validation'][val]
                path = out/'fitted'/space/view/target/'fresh'; tick = time.perf_counter()
                reuse = reused.get((view, target)) if space == 'wire' else None
                if reuse is not None:
                    _validate_reuse(reuse, xf, yf, xv, yv, classes, role_seed(seed, view, target), epochs, nested_epochs)
                    result = reuse
                    counts['reused_five_candidate_roles'] += 1
                    runtime['reuse_verification_seconds'] += time.perf_counter()-tick
                else:
                    static = fit_static_auditors(xf, yf, xv, yv, classes, role_seed(seed, view, target), budget=budget)
                    result = fit_extended_auditors(xf, yf, xv, yv, classes, role_seed(seed, view, target), static_candidates=static,
                        epochs=epochs, nested_epochs=nested_epochs, budget=budget)
                    save_extended_audits(result, path)
                    counts['new_five_candidate_roles'] += 1
                    counts['new_fresh_mlp_trajectories'] += 2; counts['new_static_candidates'] += 3
                    runtime['fresh_seconds'] += time.perf_counter()-tick
                for tag, b in (('nested120', nested_epochs), ('nested360', epochs)):
                    for cid, base in result[tag]['candidates'].items():
                        cpath = reuse['paths'][b][cid] if reuse is not None else path/tag/cid
                        ident = space+'__'+cid
                        candidates[b][role][ident] = _wrapped(base, ident, space, view, target, b, cpath, reused=reuse is not None)
            observer_key = view+'__'+target
            if observers and observer_key in observers:
                tick = time.perf_counter(); views = spaces['wire']; path = out/'fitted'/'wire'/view/target/'saved_start'
                result = fit_role_catchup(observers[observer_key], views[view]['attacker_fit'][ix], yf,
                    views[view]['attacker_validation'][val], yv, classes, role_seed(seed, view, target, catchup=True),
                    inherited_exposure=inherited_exposure.get(observer_key), epochs=epochs, nested_epochs=nested_epochs)
                save_extended_audits(result, path); counts['own_catchup_trajectories'] += 1
                runtime['catchup_seconds'] += time.perf_counter()-tick
                for tag, b in (('nested120', nested_epochs), ('nested360', epochs)):
                    candidates[b][role]['catchup'] = _wrapped(result[tag], 'catchup', 'wire', view, target, b, path/tag/'catchup')
                    candidates[b][role]['saved_adversary'] = _wrapped(result['saved'], 'saved_adversary', 'wire', view, target, b, path/'saved')
    widths = {space: (views['A']['attacker_fit'].shape[1], views['B']['attacker_fit'].shape[1]) for space, views in spaces.items()}
    selected, pools, parity, record = {}, {}, [], {}
    for b, roles in candidates.items():
        inherit_singletons(roles, widths)
        selected[b], pools[b], record[b] = {}, {}, {}
        for role, current in roles.items():
            view, target = role.split('/'); valid = np.asarray(labels['attacker_validation'][target]) >= 0
            bundle = {space: views[view]['attacker_validation'][valid] for space, views in spaces.items()}
            for cid, candidate in current.items():
                probability = candidate.predict_proba(bundle)
                assert metrics(np.asarray(labels['attacker_validation'][target])[valid], probability, CLASSES[target]) == candidate.metadata['validation_scores']
                if candidate.metadata['inherited_singleton']:
                    source_view, source_cid = candidate.metadata['source_view'], candidate.metadata['source_candidate_id']
                    source = roles[source_view+'/'+target][source_cid]
                    source_bundle = {space: views[source_view]['attacker_validation'][valid] for space, views in spaces.items()}
                    direct = source.predict_proba(source_bundle)
                    assert np.array_equal(probability, direct)
                    parity.append({'budget': b, 'role': role, 'candidate_id': cid, 'source_view': source_view,
                                   'validation_bitwise_equal': True, 'probability_sha256': _hash_array(probability)})
            selected[b][role], pools[b][role] = select_pools(current)
            for scope, chosen in selected[b][role].items():
                if view == 'AB':
                    eligible = pools[b][role][scope]
                    best = current[chosen].metadata['validation_scores']['log_loss']
                    assert all(best <= current[cid].metadata['validation_scores']['log_loss'] for cid in eligible)
            record[b][role] = {cid: candidate.metadata for cid, candidate in current.items()}
    metadata = {'interface': interface, 'seed': seed, 'audit_roles': AUDIT_ROLES, 'target_schema': CLASSES,
        'budgets': [nested_epochs, epochs], 'selection_rule': 'minimum unweighted attacker-validation log loss, then candidate ID',
        'development_received': False, 'counts': counts, 'runtime': runtime,
        'P_native_equals_wire_deduplicated': interface == 'P', 'E_duplicate_no_second_information_source': interface == 'E',
        'all_sensitive_singleton_candidates_inherited': True, 'inherited_validation_parity': parity,
        'selections': selected, 'selection_pools': pools, 'candidates': record}
    _write(out/'audit_selection.json', metadata)
    return {'candidates': candidates, 'selection': selected, 'selection_pools': pools, 'metadata': metadata,
            'selection_path': str((out/'audit_selection.json').resolve()), 'selection_sha256': _file_hash(out/'audit_selection.json')}
