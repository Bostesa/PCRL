"""Private release audits, legal ancestor routes, and frozen-choice evaluation.

Fitting consumes only the two fitting pools and their role-specific validation
pool. Evaluation accepts already-loaded data after the parent's permit gate,
replays the frozen choices, and cannot fit or select a model. Returned summaries
contain aggregates; identities, conditional probabilities and per-person losses
are written only to the supplied private output directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from .audits import (_features, _hash_array,
                     _token_probs, _weights, expected_token_loss, fit_slate,
                     loss_scores, predict_token_proba, select_losses, validate_ids)
from .config import PRIMARY, SECONDARY, UTILITY


FIT_POOLS = ('downstream_fit', 'attacker_fit')
VALIDATION_POOLS = ('attacker_validation', 'downstream_validation')
FIT_CONTEXT_POOLS = FIT_POOLS + VALIDATION_POOLS
ROLE_SPECS = tuple([('attack', role) for role in PRIMARY + SECONDARY] +
                   [('utility', role) for role in UTILITY])


def _role_key(kind, role):
    return kind + ':' + role


def _safe(role):
    return role.replace(':', '__').replace('/', '__')


def _json(path, value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError('Private output directory must be empty; existing audit artifacts are immutable')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _households(values, n):
    values = np.asarray(values, dtype=object)
    if values.shape != (n,):
        raise ValueError('Household IDs must align with original people')
    for value in values:
        if (value is None or not isinstance(value, (str, int, float, np.integer, np.floating))
                or (isinstance(value, str) and not value.strip())
                or (not isinstance(value, str) and not np.isfinite(value))):
            raise ValueError('Missing or invalid household identifier')
    return _identifier_strings(values)


def _identifier_strings(values):
    # Private NPZ identifiers use strings; detect collisions rather than
    # silently losing numeric/string or integral-float identity distinctions.
    return np.asarray([value if isinstance(value, str) else
                       str(int(value)) if float(value).is_integer() else str(value)
                       for value in values])


def _target_labels(values, n, target):
    y = np.asarray(values)
    n_classes = 9 if target == 'RAC1P' else 2
    if (y.shape != (n,) or y.dtype.kind not in 'biuf' or not np.isfinite(y).all()
            or not np.equal(y, np.floor(y)).all() or (y < -1).any() or (y >= n_classes).any()):
        raise ValueError('Target labels must align and use -1 for missing or the full fixed class schema')
    return y.astype(np.int64, copy=False)


def _validate_pool(data):
    if not isinstance(data, dict):
        raise ValueError('Each pool must provide the declared row-aligned arrays')
    n = len(data['ids'])
    ids = _identifier_strings(validate_ids(data['ids'], n))
    validate_ids(ids, n)
    ha, hb = _features(data['ha']), _features(data['hb'])
    if ha.shape != (n, 4) or hb.shape != (n, 2) or not n:
        raise ValueError('Original service coordinates must be H_A width4 and H_B width2')
    labels = {target: _target_labels(data['labels'][target], n, target)
              for target in {role.split('/')[1] for _, role in ROLE_SPECS}}
    return {'ha': ha, 'hb': hb, 'ids': ids,
            'households': _households(data['households'], n),
            'weights': _weights(data['weights'], n), 'labels': labels}


def _validate_context(ctx, pools, *, fitting):
    if not isinstance(ctx, dict) or 'anchor' not in ctx or 'pools' not in ctx:
        raise ValueError('An explicit anchor context is required')
    if fitting and 'test' in ctx['pools']:
        raise ValueError('Fitting context must not contain sealed evaluation arrays')
    data = {pool: _validate_pool(ctx['pools'][pool]) for pool in pools}
    if fitting:
        seen = set()
        for pool in pools:
            identities = set(data[pool]['ids'])
            if seen.intersection(identities):
                raise ValueError('Original person IDs overlap between fitting/validation pools')
            seen.update(identities)
        fit_households = set(np.concatenate([data[pool]['households'] for pool in FIT_POOLS]))
        for pool in VALIDATION_POOLS:
            if fit_households.intersection(data[pool]['households']):
                raise ValueError('Fitting and validation household IDs overlap')
    return data


def _binary_probabilities(value, n, width, *, name):
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (n, width) or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise ValueError(f'{name} must be finite aligned probabilities with the declared width')
    return q


def _validate_release(release, data, global_offsets=None):
    values = {}
    for pool, rows in data.items():
        spec = release[pool]
        n = len(rows['ids'])
        p = _token_probs(spec['token_probs'], n)
        aux = None if spec.get('aux') is None else _features(spec['aux'])
        if aux is not None and len(aux) != n:
            raise ValueError('Continuous auxiliary coordinates do not align with original people')
        value = {'aux': aux, 'token_probs': p}
        if spec.get('fixed_probabilities') is not None:
            value['fixed_probabilities'] = _binary_probabilities(spec['fixed_probabilities'], n, p.shape[1], name='Fixed decoder')
        offsets = global_offsets[pool] if global_offsets is not None else spec.get('global_offsets')
        if offsets is not None:
            offsets = np.asarray(offsets, dtype=np.float64)
            if offsets.ndim != 2 or offsets.shape[1] < 1:
                raise ValueError('Global-offset probabilities require at least one declared action')
            value['global_offsets'] = _binary_probabilities(offsets, n, offsets.shape[1], name='Global-offset decoder')
        values[pool] = value
    schemas = {(value['token_probs'].shape[1], None if value['aux'] is None else value['aux'].shape[1],
                'fixed_probabilities' in value, None if 'global_offsets' not in value else value['global_offsets'].shape[1])
               for value in values.values()}
    if len(schemas) != 1:
        raise ValueError('Release, decoder and global-offset schemas must agree across pools')
    return values


def _subset_pool(data, mask):
    return {key: ({target: y[mask] for target, y in value.items()} if key == 'labels' else value[mask])
            for key, value in data.items()}


def _subset_release(release, mask):
    return {key: None if value is None else value[mask] for key, value in release.items()}


def _wire(data, release, view, wire):
    if view == 'A':
        h = data['ha']
    elif view == 'B':
        h = data['hb']
    elif view == 'AB':
        h = np.column_stack((data['ha'], data['hb']))
    else:
        raise ValueError('Undeclared recipient view')
    if wire == 'H' or view == 'B':
        return h, np.ones((len(h), 1))
    if wire != 'release':
        raise ValueError('Undeclared candidate wire route')
    if release['aux'] is not None:
        h = np.column_stack((h, release['aux']))
    return h, release['token_probs']


def routed_probabilities(record, data, release):
    """Replay one legal route, returning conditional q and its token law p."""
    route = record['route']
    kind = route['kind']
    if kind == 'model':
        h, p = _wire(data, release, route['source_view'], route['wire'])
        q = predict_token_proba(record['candidate'], h, p.shape[1])
    elif kind == 'fixed_decoder':
        if route['source_view'] != 'A':
            raise ValueError('The released residence decoder is only an A service')
        if 'fixed_probabilities' not in release:
            raise ValueError('Frozen fixed-decoder evaluation requires fixed_probabilities')
        positive = release['fixed_probabilities']
        p = release['token_probs']
        q = np.stack((1-positive, positive), axis=2)
    elif kind == 'global_offset':
        if route['source_view'] != 'A' or route['wire'] != 'H':
            raise ValueError('Public residence recalibration requires H_A')
        if 'global_offsets' not in release:
            raise ValueError('Frozen global-offset evaluation requires release[pool][global_offsets]')
        offsets = release['global_offsets']
        index = route['action_index']
        if offsets.shape[1] != route['n_actions']:
            raise ValueError('Frozen global-offset action schema changed')
        positive = offsets[:, index:index+1]
        p = np.ones((len(positive), 1))
        q = np.stack((1-positive, positive), axis=2)
    else:
        raise ValueError('Unknown candidate kind')
    return q, p


def _routed_loss(record, data, release, target):
    q, p = routed_probabilities(record, data, release)
    return expected_token_loss(q, p, data['labels'][target])


def _registry(value):
    if isinstance(value, (str, Path)):
        value = joblib.load(value)
    if 'registry' in value:
        value = value['registry']
    if value.get('schema') != 1 or 'roles' not in value:
        raise ValueError('Expected a persisted weighted-release audit registry')
    return value


def _signature(data, view, target, val_pool):
    digest = hashlib.sha256()
    for pool in FIT_POOLS + (val_pool,):
        rows = data[pool]
        mask = rows['labels'][target] >= 0
        h, _ = _wire(rows, {}, view, 'H')
        for name, value in [('h', h[mask]), ('ids', rows['ids'][mask]),
                             ('households', rows['households'][mask]),
                             ('weights', rows['weights'][mask]), ('labels', rows['labels'][target][mask])]:
            digest.update((pool + '/' + name + '/' + _hash_array(value)).encode())
    return digest.hexdigest()


def _copy_record(record, *, origin, source):
    return {**record, 'route': dict(record['route']), 'origin': origin, 'inherited_from': source}


def _summarize_role(role, *, reused=False):
    chosen, independent = role['selection'], role['independent_selection']
    return {'selection': chosen, 'independent_selection': independent,
            'validation': role['validation_scores'][chosen],
            'independent_validation': role['validation_scores'][independent],
            'fixed_decoder': role['validation_scores'].get('fixed_decoder'),
            'candidate_count': len(role['candidates']), 'fit_rows': role['fit_rows'],
            'validation_rows': role['validation_rows'], 'reused_B': reused}


def fit_release_audits(name, ctx, release, out_dir, *, h_baseline=None,
                        b_baseline=None, global_offsets=None):
    """Fit and select every declared role; return registry plus public aggregates.

    H/B caches may be prior return values, registries, or their private joblib
    paths. Every unchanged B role is reused after exact original-person/service
    signature checks. Global offsets are per-pool n-by-actions positive-class
    probabilities from the frozen public H_A calibrator; frozen evaluation must
    supply those same ordered actions in release_eval[pool]['global_offsets'].
    """
    data = _validate_context(ctx, FIT_CONTEXT_POOLS, fitting=True)
    release = _validate_release(release, data, global_offsets)
    h_baseline = None if h_baseline is None else _registry(h_baseline)
    b_baseline = None if b_baseline is None else _registry(b_baseline)
    for baseline in (h_baseline, b_baseline):
        if baseline is not None and baseline['anchor'] != ctx['anchor']:
            raise ValueError('Baseline anchor does not match current release')
    if h_baseline is not None and not h_baseline['is_H_release']:
        raise ValueError('Only H-only candidates are legal service ancestors; no J ancestor')
    directory = _directory(out_dir)
    is_h = all(value['aux'] is None and value['token_probs'].shape[1] == 1 for value in release.values())
    registry = {'schema': 1, 'name': str(name), 'anchor': int(ctx['anchor']),
                'is_H_release': is_h, 'roles': {}, 'selection_frozen': True,
                'same_predictions_for_both_weightings': True}
    summary = {}
    ordered = sorted(enumerate(ROLE_SPECS), key=lambda item: ('A', 'B', 'AB').index(item[1][1].split('/')[0]))
    for role_index, (kind, role_name) in ordered:
        view, target = role_name.split('/')
        key = _role_key(kind, role_name)
        val_pool = 'attacker_validation' if kind == 'attack' else 'downstream_validation'
        signature = _signature(data, view, target, val_pool)
        if view == 'B' and b_baseline is not None:
            original = b_baseline['roles'][key]
            if original['service_signature'] != signature:
                raise ValueError('Canonical B original-person/service signature changed')
            cached = {**original, 'reused_from': b_baseline['name']}
            registry['roles'][key] = cached
            summary[key] = _summarize_role(cached, reused=True)
            continue
        val_mask = data[val_pool]['labels'][target] >= 0
        val_data = _subset_pool(data[val_pool], val_mask)
        val_release = _subset_release(release[val_pool], val_mask)
        if not len(val_data['ids']):
            raise ValueError('Every role requires observed validation labels')
        train_h, train_p, train_y, train_w = [], [], [], []
        own_wire = 'H' if view == 'B' or is_h else 'release'
        for pool in FIT_POOLS:
            mask = data[pool]['labels'][target] >= 0
            rows, spec = _subset_pool(data[pool], mask), _subset_release(release[pool], mask)
            h, p = _wire(rows, spec, view, own_wire)
            train_h.append(h); train_p.append(p)
            train_y.append(rows['labels'][target]); train_w.append(rows['weights'])
        hfit, pfit, yfit, wfit = (np.concatenate(parts) for parts in (train_h, train_p, train_y, train_w))
        hv, pv = _wire(val_data, val_release, view, own_wire)
        seed = 2026000 + 10000*int(ctx['anchor']) + 37*role_index
        role_dir = directory/_safe(key)
        fitted = fit_slate(hfit, pfit, yfit, wfit, hv, pv, val_data['labels'][target],
                           val_data['weights'], 9 if target == 'RAC1P' else 2, seed,
                           role_dir/'models', slate='catchup')
        candidates = {cid: {'candidate': candidate, 'model_path': candidate.metadata['directory'],
                             'route': {'kind': 'model', 'source_view': view, 'wire': own_wire},
                             'origin': 'independent', 'source_role': key}
                      for cid, candidate in fitted['candidates'].items()}
        independent_ids = list(candidates)
        if h_baseline is not None:
            baseline_role = h_baseline['roles'][key]
            if baseline_role['service_signature'] != signature:
                raise ValueError('H baseline original-person/service signature changed')
            for cid, record in baseline_role['candidates'].items():
                # H decoder predictions are supplied separately by the shared
                # global-offset interface, never reinterpreted as current Q.
                if record['route']['kind'] != 'model':
                    continue
                if record['route']['wire'] != 'H':
                    raise ValueError('An H ancestor attempted to read auxiliary release data')
                candidates['H__' + cid] = _copy_record(record, origin='H_baseline', source=h_baseline['name']+'/'+cid)
        if view == 'AB' and target in ('SEX', 'RAC1P'):
            for ancestor in ('A', 'B'):
                source_key = 'attack:' + ancestor + '/' + target
                for cid, record in registry['roles'][source_key]['candidates'].items():
                    candidates[f'ancestor_{ancestor}__{cid}'] = _copy_record(
                        record, origin=ancestor+'_ancestor', source=source_key+'/'+cid)
        if kind == 'utility' and view == 'A' and target == 'same_residence':
            if 'fixed_probabilities' in val_release:
                candidates['fixed_decoder'] = {'route': {'kind': 'fixed_decoder', 'source_view': 'A', 'wire': 'release'},
                                                'origin': 'deployment_decoder', 'source_role': key}
            if 'global_offsets' in val_release:
                n_actions = val_release['global_offsets'].shape[1]
                for action in range(n_actions):
                    candidates[f'global_offset_{action:04d}'] = {
                        'route': {'kind': 'global_offset', 'source_view': 'A', 'wire': 'H',
                                  'action_index': action, 'n_actions': n_actions},
                        'origin': 'public_H_recalibration', 'source_role': key}
        losses = {cid: _routed_loss(record, val_data, val_release, target) for cid, record in candidates.items()}
        choices = select_losses(losses, val_data['weights'])
        independent = select_losses({cid: losses[cid] for cid in independent_ids}, val_data['weights'])['selection']
        loss_path = role_dir/'val_losses.npz'
        candidate_ids = sorted(candidates)
        np.savez_compressed(loss_path, ids=val_data['ids'], households=val_data['households'],
                            weights=val_data['weights'], y=val_data['labels'][target],
                            candidate_ids=np.asarray(candidate_ids),
                            losses=np.column_stack([losses[cid] for cid in candidate_ids]))
        stored = {'view': view, 'target': target, 'kind': kind, 'validation_pool': val_pool,
                  'service_signature': signature, 'candidates': candidates,
                  'selection': choices['selection'], 'independent_selection': independent,
                  'validation_scores': choices['scores'], 'fit_rows': len(yfit),
                  'validation_rows': len(val_data['ids']), 'seed': seed,
                  'validation_loss_path': str(loss_path.resolve())}
        registry['roles'][key] = stored
        summary[key] = _summarize_role(stored)
        _json(role_dir/'selection.json', {**{k: v for k, v in stored.items() if k != 'candidates'},
              'candidates': {cid: {k: v for k, v in record.items() if k != 'candidate'} for cid, record in candidates.items()}})
    path = directory/'registry.joblib'
    joblib.dump(registry, path, compress=3)
    _json(directory/'summary.json', summary)
    return {'registry': registry, 'summary': summary, 'registry_path': str(path.resolve())}


def _ess(weights):
    if not len(weights) or weights.sum() <= 0:
        return 0.
    normalized = weights/weights.sum()
    return float(1./np.dot(normalized, normalized))


def _metrics(loss, accuracy, y, weights, n_classes):
    ce = loss_scores(loss, weights)
    normalized = weights/weights.sum()
    classes = []
    for label in range(n_classes):
        mask = y == label
        w = weights[mask]
        if mask.any() and w.sum() > 0:
            class_ce = loss_scores(loss[mask], w)
        else:
            class_ce = {'unweighted': float(loss[mask].mean()) if mask.any() else None,
                        'weighted': None, 'balanced': None}
        classes.append({'class': label, 'support': int(mask.sum()), 'weight_sum': float(w.sum()),
                        'ess': _ess(w), 'ce': class_ce})
    return {'ce': ce, 'accuracy': {'unweighted': float(accuracy.mean()),
                                   'weighted': float(np.dot(normalized, accuracy))},
            'support': len(y), 'weight_sum': float(weights.sum()), 'ess': _ess(weights),
            'per_class': classes}


def evaluate_frozen_audits(registry, ctx_eval, release_eval, out_dir):
    """Replay recorded choices only; the parent enforces evaluation permission.

    Saves selected conditional probabilities and their actual token law, exact
    per-person loss/accuracy, plus independent/fixed-decoder loss diagnostics.
    No fitting or candidate selection occurs, including after missing-label
    masks change. H-routed probabilities retain a deterministic token axis.
    """
    registry = _registry(registry)
    if not registry.get('selection_frozen') or registry['anchor'] != ctx_eval['anchor']:
        raise ValueError('Frozen selection and matching anchor required')
    pools = tuple(ctx_eval['pools'])
    if not pools:
        raise ValueError('Explicit nonempty evaluation pools required')
    data = _validate_context(ctx_eval, pools, fitting=False)
    release = _validate_release(release_eval, data)
    directory = _directory(out_dir)
    summary, artifacts = {}, {}
    for pool in pools:
        pool_dir = directory/pool
        pool_dir.mkdir()
        summary[pool], artifacts[pool] = {}, {}
        for key, role in registry['roles'].items():
            target = role['target']
            mask = data[pool]['labels'][target] >= 0
            rows, spec = _subset_pool(data[pool], mask), _subset_release(release[pool], mask)
            y, weights = rows['labels'][target], rows['weights']
            if not len(y):
                raise ValueError('Frozen evaluation role has no observed labels')
            _weights(weights, len(y))
            selected = role['selection']
            q, p = routed_probabilities(role['candidates'][selected], rows, spec)
            loss = expected_token_loss(q, p, y)
            accuracy = np.sum(p*(q.argmax(2) == y[:, None]), axis=1)
            independent = _routed_loss(role['candidates'][role['independent_selection']], rows, spec, target)
            record = {**_metrics(loss, accuracy, y, weights, q.shape[2]),
                      'selection': selected, 'independent_selection': role['independent_selection'],
                      'independent_ce': loss_scores(independent, weights),
                      'selection_source': 'frozen validation registry; no evaluation selection'}
            arrays = {'ids': rows['ids'], 'households': rows['households'], 'weights': weights,
                      'y': y, 'probabilities': q, 'token_probs': p, 'loss': loss,
                      'accuracy': accuracy, 'independent_loss': independent}
            if 'fixed_decoder' in role['candidates']:
                fixed = _routed_loss(role['candidates']['fixed_decoder'], rows, spec, target)
                arrays['fixed_decoder_loss'] = fixed
                record['fixed_decoder_ce'] = loss_scores(fixed, weights)
            path = pool_dir/(_safe(key)+'.npz')
            np.savez_compressed(path, **arrays)
            summary[pool][key] = record
            artifacts[pool][key] = str(path.resolve())
    _json(directory/'summary.json', summary)
    return {'summary': summary, 'artifacts': artifacts, 'private_directory': str(directory.resolve())}
