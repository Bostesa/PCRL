"""Independent, read-only reconstruction of frozen weighted audit artifacts.

This verifier reloads the separately persisted model files, reconstructs feature
routes and the conditional probability tensor itself, and computes expected
log loss independently. It never calls the fitting, selection, routing or loss
implementations that produced the artifacts. No ACS loader is imported. The
caller supplies already-authorized data and token laws; evaluation permission
remains the responsibility of run.py's frozen-selection gate.

Household subsamples include every observed-label person in each selected
household. Sampled-wire checks draw tokens, never people or households, and
retain original-person weights and household counts in private NPZ artifacts.
"""
from __future__ import annotations

import hashlib
import json
import copy
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .audits import load_candidate


FLOOR = 1e-9  # Frozen audit format version 1, not inferred from evaluation labels.
WIRE_SEED = 2026092301  # Independent of encoder and sampled-tree schedules.
ATOL = 1e-11
RTOL = 1e-10


class ReplayError(ValueError):
    """A frozen artifact does not replay under its declared data/model schema."""


def _require(condition, message):
    if not condition:
        raise ReplayError(message)


def _safe(key):
    return key.replace(':', '__').replace('/', '__')


def _hash_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')


def _output(path):
    if path is None:
        return None
    path = Path(path)
    _require(not path.exists() or not any(path.iterdir()), 'Replay output directory must be empty')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _relocation_roots(original_root, artifact_root):
    _require(original_root is not None and artifact_root is not None,
             'Supply both original_root and artifact_root; no path fallback is allowed')
    original = Path(os.path.normpath(str(original_root)))
    _require(original.is_absolute(), 'Original artifact root must be absolute')
    restored = Path(artifact_root).resolve()
    _require(restored.is_dir(), 'Restored artifact root is missing')
    return original, restored


def _relocated_path(value, original, restored, *, allow_restored=False):
    source = Path(os.path.normpath(str(value)))
    _require(source.is_absolute(), 'Persisted artifact paths must be absolute')
    if allow_restored and source.is_relative_to(restored):
        target = source.resolve()
    else:
        _require(source.is_relative_to(original), 'Persisted path is not contained under original_root')
        target = (restored/source.relative_to(original)).resolve()
    _require(target.is_relative_to(restored), 'Restored path is not contained under artifact_root (symlink escape)')
    _require(target.exists(), 'Required restored artifact is missing; original files are never a fallback')
    return str(target)


def _owned_file(path, root):
    path = Path(path).resolve()
    _require(path.is_relative_to(root), 'Restored file is not contained under artifact_root (symlink escape)')
    _require(path.is_file(), 'Required restored artifact file is missing')
    return path


def relocate_registry(registry, *, original_root, artifact_root):
    """Deep-copy a registry, remapping every consumed path to an owned archive.

    The original registry/model objects are untouched. Each restored model and
    validation-loss path must exist and resolve within artifact_root, including
    through symlinks. Original machine files are never read as a fallback.
    A registry filename may itself be expressed under either declared root.
    """
    original, restored = _relocation_roots(original_root, artifact_root)
    if isinstance(registry, (str, Path)):
        registry = joblib.load(_relocated_path(registry, original, restored, allow_restored=True))
    if 'registry' in registry:
        registry = registry['registry']
    relocated = copy.deepcopy(registry)
    checked_models = set()
    for role in relocated['roles'].values():
        role['validation_loss_path'] = _relocated_path(role['validation_loss_path'], original, restored, allow_restored=True)
        for record in role['candidates'].values():
            if record['route']['kind'] != 'model':
                continue
            record['model_path'] = _relocated_path(record['model_path'], original, restored, allow_restored=True)
            _require(Path(record['model_path']).is_dir(), 'Restored model directory is missing')
            if record['model_path'] not in checked_models:
                model_dir = Path(record['model_path'])
                for filename in ('metadata.json', 'candidate.joblib'):
                    _owned_file(model_dir/filename, restored)
                # Include all files that the model loader or hash manifest may
                # read, not merely the containing directory's symlink target.
                for child in model_dir.iterdir():
                    _require(child.resolve().is_relative_to(restored),
                             'Restored model file is not contained under artifact_root (symlink escape)')
                checked_models.add(record['model_path'])
            candidate = record.get('candidate')
            if candidate is not None:
                for field in ('directory', 'checkpoint_directory'):
                    if field in candidate.metadata:
                        candidate.metadata[field] = _relocated_path(candidate.metadata[field], original, restored, allow_restored=True)
    relocated['replay_relocation'] = {'original_root': str(original), 'artifact_root': str(restored),
                                      'original_fallback_allowed': False}
    return relocated


def _registry(value, *, original_root=None, artifact_root=None):
    if original_root is not None or artifact_root is not None:
        value = relocate_registry(value, original_root=original_root, artifact_root=artifact_root)
    if isinstance(value, (str, Path)):
        value = joblib.load(value)
    if 'registry' in value:
        value = value['registry']
    _require(value.get('schema') == 1 and isinstance(value.get('roles'), dict), 'Unknown registry schema')
    _require(value.get('selection_frozen') is True, 'Registry selection is not frozen')
    _require(value.get('same_predictions_for_both_weightings') is True,
             'One identical selected predictor is required for U and PWGTP')
    for key, role in value['roles'].items():
        _require(key == role['kind'] + ':' + role['view'] + '/' + role['target'], 'Role/class schema mismatch')
        for choice in ('selection', 'independent_selection'):
            _require(role.get(choice) in role['candidates'], 'Frozen selection references a missing candidate')
        for key_name in ('selection_unweighted', 'selection_weighted', 'selection_PWGTP'):
            if key_name in role:
                _require(role[key_name] == role['selection'], 'Weight-specific selection differs from frozen selection')
        if 'selection_by_weighting' in role:
            _require(all(cid == role['selection'] for cid in role['selection_by_weighting'].values()),
                     'Weight-specific selection differs from frozen selection')
    return value


def _chosen_roles(registry, keys):
    keys = list(registry['roles']) if keys is None else list(keys)
    _require(bool(keys) and len(set(keys)) == len(keys) and set(keys) <= set(registry['roles']),
             'Replay role subset must be nonempty, unique and declared')
    return keys


def _identities(values, n, *, unique):
    values = np.asarray(values, dtype=object)
    _require(values.shape == (n,), 'Person/household identity alignment failed')
    strings = []
    for value in values:
        _require(value is not None and isinstance(value, (str, int, float, np.integer, np.floating)),
                 'Invalid person/household identifier')
        if isinstance(value, str):
            _require(bool(value.strip()), 'Missing person/household identifier')
            strings.append(value)
        else:
            _require(np.isfinite(value), 'Nonfinite person/household identifier')
            strings.append(str(int(value)) if float(value).is_integer() else str(value))
    strings = np.asarray(strings)
    _require(not unique or len(np.unique(strings)) == n, 'Duplicate original-person identifier')
    return strings


def _weights(weights, n):
    weights = np.asarray(weights, dtype=np.float64)
    _require(weights.shape == (n,) and np.isfinite(weights).all() and (weights >= 0).all()
             and np.isfinite(weights.sum()) and weights.sum() > 0, 'Invalid original-person weights')
    return weights


def _probabilities(probabilities, *, ndim, shape=None):
    p = np.asarray(probabilities, dtype=np.float64)
    _require(p.ndim == ndim and (shape is None or p.shape == shape) and p.shape[-1] > 0
             and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
             and np.allclose(p.sum(-1), 1., atol=1e-8, rtol=0),
             'Invalid normalized probability/class-order tensor')
    return p


def _rows(data, target):
    raw_y = np.asarray(data['labels'][target])
    n_classes = 9 if target == 'RAC1P' else 2
    n = len(data['ids'])
    _require(raw_y.shape == (n,) and raw_y.dtype.kind in 'biuf' and np.isfinite(raw_y).all()
             and np.equal(raw_y, np.floor(raw_y)).all() and (raw_y >= -1).all()
             and (raw_y < n_classes).all(), 'Invalid target labels or class order')
    ids = _identities(data['ids'], n, unique=True)
    households = _identities(data['households'], n, unique=False)
    weights = _weights(data['weights'], n)
    ha, hb = np.asarray(data['ha'], dtype=np.float64), np.asarray(data['hb'], dtype=np.float64)
    _require(ha.shape == (n, 4) and hb.shape == (n, 2) and np.isfinite(ha).all()
             and np.isfinite(hb).all(), 'Service H coordinate schema/parity failed')
    mask = raw_y >= 0
    _require(mask.any(), 'No observed labels in replay role')
    return {'ids': ids[mask], 'households': households[mask], 'weights': weights[mask],
            'y': raw_y[mask].astype(np.int64), 'ha': ha[mask], 'hb': hb[mask],
            'n_classes': n_classes, 'mask': mask, 'original_n': n}


def _release(spec, rows, offsets=None):
    n, mask = rows['original_n'], rows['mask']
    p = _probabilities(spec['token_probs'], ndim=2)
    _require(len(p) == n, 'Token-law original-person alignment failed')
    result = {'token_probs': p[mask], 'aux': None}
    if spec.get('aux') is not None:
        aux = np.asarray(spec['aux'], dtype=np.float64)
        _require(aux.ndim == 2 and aux.shape[0] == n and aux.shape[1] > 0
                 and np.isfinite(aux).all(), 'Invalid continuous auxiliary coordinates')
        result['aux'] = aux[mask]
    for key in ('fixed_probabilities', 'global_offsets'):
        value = offsets if key == 'global_offsets' and offsets is not None else spec.get(key)
        if value is not None:
            value = np.asarray(value, dtype=np.float64)
            _require(value.ndim == 2 and value.shape[0] == n and value.shape[1] > 0
                     and np.isfinite(value).all() and (value >= 0).all() and (value <= 1).all(),
                     'Invalid aligned fixed-decoder probabilities')
            if key == 'fixed_probabilities':
                _require(value.shape == p.shape, 'Fixed-decoder/token class order mismatch')
            result[key] = value[mask]
    return result


def _subset(rows, release, indices):
    rows = {key: value[indices] if isinstance(value, np.ndarray) and key != 'mask' else value
            for key, value in rows.items()}
    release = {key: value[indices] if value is not None else None for key, value in release.items()}
    return rows, release


def _household_subset(households, maximum):
    if maximum is None:
        return np.arange(len(households))
    _require(isinstance(maximum, int) and maximum > 0, 'Household replay limit must be positive')
    groups = np.unique(households)
    ranked = sorted(groups, key=lambda group: hashlib.sha256(('replay-households-v1|' + group).encode()).digest())
    return np.flatnonzero(np.isin(households, ranked[:maximum]))


def _close(observed, expected, label, *, exact=False):
    a, b = np.asarray(observed), np.asarray(expected)
    okay = a.shape == b.shape
    if okay:
        if exact or a.dtype.kind in 'OUS' or b.dtype.kind in 'OUS':
            okay = np.array_equal(a, b)
        else:
            okay = np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, atol=ATOL, rtol=RTOL)
    _require(okay, label + ' mismatch')


def _scores(loss, weights):
    loss = np.asarray(loss, dtype=np.float64)
    weights = _weights(weights, len(loss))
    _require(loss.shape == weights.shape and np.isfinite(loss).all() and (loss >= 0).all(), 'Invalid person loss')
    u, w = float(loss.mean()), float(np.dot(weights/weights.sum(), loss))
    return {'unweighted': u, 'weighted': w, 'balanced': .5*u + .5*w}


def _compare_scores(saved, rebuilt, label):
    for key, value in rebuilt.items():
        if value is None:
            _require(saved.get(key) is None, label + '/' + key + ' mismatch')
        else:
            _close(saved[key], value, label + '/' + key)


def _loss(q, p, y):
    q = _probabilities(q, ndim=3)
    _require(q.shape[2] >= 2, 'At least two columns in the full class schema are required')
    p = _probabilities(p, ndim=2, shape=q.shape[:2])
    y = np.asarray(y)
    _require(y.shape == (len(p),) and y.dtype.kind in 'iu' and (y >= 0).all()
             and (y < q.shape[2]).all(), 'Label/conditional-probability class order mismatch')
    conditional = -np.log(np.maximum(q[np.arange(len(y))[:, None], np.arange(p.shape[1])[None, :], y[:, None]], FLOOR))
    return (p * conditional).sum(1), conditional


def _route_guard(record, role):
    route = record['route']
    allowed = {'A': {'A'}, 'B': {'B'}, 'AB': {'A', 'B', 'AB'}}[role['view']]
    _require(route.get('source_view') in allowed and route.get('wire') in ('H', 'release'),
             'Candidate has an illegal recipient or ancestor route')
    _require(route['source_view'] != 'B' or route['wire'] == 'H', 'B parity requires unchanged H_B only')
    if record.get('origin') == 'H_baseline':
        _require(route['wire'] == 'H', 'H ancestor attempted to read auxiliary data')
    if route.get('kind') != 'model':
        _require(role['target'] == 'same_residence' and role['kind'] == 'utility'
                 and route['source_view'] == 'A', 'Decoder is incompatible with this target/recipient')


def _load_model(path, cache):
    path = str(Path(path).resolve())
    if path not in cache:
        try:
            model = load_candidate(path)
        except (ValueError, RuntimeError, KeyError) as error:
            raise ReplayError('Persisted model/class schema failed to load: ' + str(error)) from error
        _require(model.metadata.get('class_order') == list(range(model.n_classes)), 'Persisted class order is corrupted')
        _require(model.metadata.get('probability_floor', FLOOR) == FLOOR, 'Unexpected probability-floor policy')
        _require(model.mean.ndim == 1 and model.scale.shape == model.mean.shape
                 and np.isfinite(model.mean).all() and np.isfinite(model.scale).all()
                 and (model.scale > 0).all(), 'Corrupt fitted preprocessing coordinates')
        cache[path] = model
    return cache[path]


def _reconstruct(record, role, rows, release, cache):
    _route_guard(record, role)
    route, n = record['route'], len(rows['y'])
    if route['kind'] == 'fixed_decoder':
        _require('fixed_probabilities' in release, 'Missing frozen fixed decoder')
        positive, p = release['fixed_probabilities'], release['token_probs']
        return np.stack((1-positive, positive), -1), p
    if route['kind'] == 'global_offset':
        _require(route['wire'] == 'H' and 'global_offsets' in release, 'Missing public H offset predictions')
        offsets = release['global_offsets']
        _require(offsets.shape[1] == route['n_actions'] and 0 <= route['action_index'] < route['n_actions'],
                 'Frozen global-offset action order changed')
        positive = offsets[:, route['action_index']:route['action_index']+1]
        return np.stack((1-positive, positive), -1), np.ones((n, 1))
    _require(route['kind'] == 'model', 'Unknown frozen candidate kind')
    model = _load_model(record['model_path'], cache)
    source = route['source_view']
    h = rows['ha'] if source == 'A' else rows['hb'] if source == 'B' else np.column_stack((rows['ha'], rows['hb']))
    p = np.ones((n, 1)) if route['wire'] == 'H' else release['token_probs']
    if route['wire'] == 'release' and release['aux'] is not None:
        h = np.column_stack((h, release['aux']))
    _require(h.shape[1] == len(model.mean) and p.shape[1] == model.n_tokens
             and model.n_classes == rows['n_classes'], 'Persisted feature/token/class order disagrees with route')
    output = np.empty((n, model.n_tokens, model.n_classes), dtype=np.float64)
    for start in range(0, n, 256):
        block = (h[start:start+256] - model.mean) / model.scale
        count = len(block)
        persons = np.repeat(np.arange(count), model.n_tokens)
        tokens = np.tile(np.arange(model.n_tokens), count)
        x = block[persons]
        if model.n_tokens > 1:
            onehot = np.eye(model.n_tokens)[tokens]
            parts = [x, onehot]
            if model.interactions:
                parts.append((x[:, :, None] * onehot[:, None, :]).reshape(len(x), -1))
            x = np.concatenate(parts, axis=1)
        _require(x.shape[1] == model.metadata['input_dim'], 'Persisted encoded feature order changed')
        if model.family == 'mlp':
            with torch.no_grad():
                q = torch.softmax(model.model(torch.as_tensor(x, dtype=torch.float32)), 1).double().numpy()
        elif model.family == 'prior':
            q = np.broadcast_to(model.model, (len(x), model.n_classes)).copy()
        else:
            classes = np.asarray(model.model.classes_)
            _require(classes.ndim == 1 and classes.dtype.kind in 'iu' and len(np.unique(classes)) == len(classes)
                     and (classes >= 0).all() and (classes < model.n_classes).all(), 'Corrupt fitted class columns')
            q = np.zeros((len(x), model.n_classes), dtype=np.float64)
            q[:, classes] = model.model.predict_proba(x)
        _require(np.isfinite(q).all() and (q >= 0).all()
                 and np.allclose(q.sum(1), 1., atol=1e-6, rtol=0), 'Model produced corrupt probabilities')
        q = (1-model.n_classes*FLOOR)*(q/q.sum(1, keepdims=True)) + FLOOR
        output[start:start+count] = q.reshape(count, model.n_tokens, model.n_classes)
    return _probabilities(output, ndim=3), p


def _identity_check(saved, rows):
    for key in ('ids', 'households', 'weights', 'y'):
        _close(saved[key], rows[key], 'Original-person ' + key, exact=True)


def _coverage(rows, indices=None):
    indices = np.arange(len(rows['y'])) if indices is None else indices
    return {'original_people': len(rows['y']), 'original_households': len(np.unique(rows['households'])),
            'replayed_people': len(indices), 'replayed_households': len(np.unique(rows['households'][indices]))}


def _model_hashes(cache):
    # Paths and hashes belong to the private verification manifest.
    return {path: {file.name: _hash_file(file) for file in Path(path).iterdir() if file.is_file()}
            for path in cache}


def verify_validation(registry, ctx, release, *, candidate_scope='all', role_keys=None,
                      max_households=None, global_offsets=None, out_dir=None,
                      original_root=None, artifact_root=None):
    """Verify validation choices and saved losses; optionally sample households.

    'selected' reconstructs deployment, independent and fixed-decoder candidates;
    'all' reconstructs every candidate. In either case the frozen selection is
    checked using ALL rows/candidates in the saved validation-loss matrix. Full
    nominee verification should leave max_households=None and use scope='all'.
    """
    registry = _registry(registry, original_root=original_root, artifact_root=artifact_root)
    _require(registry['anchor'] == ctx['anchor'] and 'test' not in ctx['pools'],
             'Validation replay requires the matching anchor and no evaluation arrays')
    _require(candidate_scope in ('all', 'selected'), 'Unknown replay candidate scope')
    directory, cache = _output(out_dir), {}
    report = {'passed': True, 'stage': 'validation', 'candidate_scope': candidate_scope, 'roles': {}}
    with threadpool_limits(limits=1):
        for key in _chosen_roles(registry, role_keys):
            role = registry['roles'][key]
            pool = role['validation_pool']
            rows = _rows(ctx['pools'][pool], role['target'])
            spec = _release(release[pool], rows, None if global_offsets is None else global_offsets[pool])
            with np.load(role['validation_loss_path'], allow_pickle=False) as saved:
                _identity_check(saved, rows)
                ids = saved['candidate_ids'].astype(str).tolist()
                matrix = saved['losses'].copy()
            _require(len(set(ids)) == len(ids) and set(ids) == set(role['candidates']), 'Validation candidate order/schema mismatch')
            _require(matrix.shape == (len(rows['y']), len(ids)) and np.isfinite(matrix).all()
                     and (matrix >= 0).all(), 'Corrupt saved validation losses')
            scores = {cid: _scores(matrix[:, j], rows['weights']) for j, cid in enumerate(ids)}
            for cid in ids:
                _compare_scores(role['validation_scores'][cid], scores[cid], 'Validation score/' + cid)
            selected = min(ids, key=lambda cid: (scores[cid]['balanced'], cid))
            independent_ids = [cid for cid in ids if role['candidates'][cid].get('origin') == 'independent']
            _require(bool(independent_ids), 'No pure independent candidates in frozen registry')
            independent = min(independent_ids, key=lambda cid: (scores[cid]['balanced'], cid))
            _require(role['selection'] == selected and role['independent_selection'] == independent,
                     'Frozen validation selection does not match the balanced lexical rule')
            picked = set(ids) if candidate_scope == 'all' else {selected, independent} | ({'fixed_decoder'} & set(ids))
            indices = _household_subset(rows['households'], max_households)
            for cid in sorted(picked):
                # Preserve original 256-person forward-pass boundaries even
                # when comparing a household subset (float32 BLAS kernels can
                # change their rounding with a different batch shape).
                q, p = _reconstruct(role['candidates'][cid], role, rows, spec, cache)
                rebuilt, _ = _loss(q[indices], p[indices], rows['y'][indices])
                _close(matrix[indices, ids.index(cid)], rebuilt, 'Reconstructed validation loss/' + cid)
            report['roles'][key] = {**_coverage(rows, indices), 'replayed_candidates': len(picked),
                'saved_candidate_count': len(ids), 'selection': selected,
                'same_selection_U_PWGTP': True, 'selection_verified_on_all_saved_validation_rows': True,
                'reconstruction_scope': 'all observed-label rows' if max_households is None else 'whole-household subset'}
    if directory is not None:
        _json(directory/'verification.json', {**report, 'private_model_hashes': _model_hashes(cache)})
    return report


def sampled_wire_check(q, p, y, weights, ids, households, out_file, *, repetitions=512, seed=WIRE_SEED):
    """Independent categorical draws; original people/households are never resampled.

    Conditional MC standard errors use the known token-loss variance and each
    reporting estimator's person coefficients. A predeclared six-SE + 1e-11
    comparison is a numerical cross-check, not an inferential study endpoint.
    """
    q, p = _probabilities(q, ndim=3), _probabilities(p, ndim=2)
    y = np.asarray(y)
    _require(y.ndim == 1 and y.dtype.kind in 'biuf' and np.isfinite(y).all()
             and np.equal(y, np.floor(y)).all(), 'Wire-check labels must be finite integer class indices')
    y = y.astype(np.int64)
    expected, conditional = _loss(q, p, y)
    n = len(y)
    weights = _weights(weights, n)
    ids, households = _identities(ids, n, unique=True), _identities(households, n, unique=False)
    _require(isinstance(repetitions, int) and repetitions > 0 and isinstance(seed, (int, np.integer)) and seed >= 0,
             'Wire check requires positive repetitions and a fixed nonnegative seed')
    rng = np.random.default_rng(seed)
    cumulative = np.cumsum(p, axis=1); cumulative[:, -1] = 1.
    coefficients = np.vstack((np.full(n, 1/n), weights/weights.sum(), .5/n + .5*weights/weights.sum()))
    variance = np.sum(p * (conditional-expected[:, None])**2, axis=1)
    exact = coefficients @ expected
    risk_draws, accumulated = np.empty((repetitions, 3)), np.zeros(n)
    first_tokens = first_loss = None
    for start in range(0, repetitions, 32):
        count = min(32, repetitions-start)
        uniforms = rng.random((count, n))
        tokens = (uniforms[:, :, None] >= cumulative[None, :, :]).sum(2)
        _require((p[np.arange(n)[None, :], tokens] > 0).all(), 'Wire replay sampled an unsupported token')
        sampled = conditional[np.arange(n)[None, :], tokens]
        if first_tokens is None:
            first_tokens, first_loss = tokens[0].copy(), sampled[0].copy()
        accumulated += sampled.sum(0)
        risk_draws[start:start+count] = sampled @ coefficients.T
    means = risk_draws.mean(0)
    se = np.sqrt((coefficients**2) @ variance / repetitions)
    errors = np.abs(means-exact)
    passed = bool((errors <= 6*se + ATOL).all())
    metrics = {name: {'exact': float(exact[i]), 'sampled_mean': float(means[i]),
                      'absolute_error': float(errors[i]), 'conditional_mc_se': float(se[i]),
                      'six_se_tolerance': float(6*se[i]+ATOL)}
               for i, name in enumerate(('unweighted', 'weighted', 'balanced'))}
    groups, counts = np.unique(households, return_counts=True)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    _require(not out_file.exists(), 'Private wire replay artifact already exists')
    np.savez_compressed(out_file, ids=ids, households=households, household_ids=groups, household_counts=counts,
                        weights=weights, y=y, expected_loss=expected, mean_sampled_loss=accumulated/repetitions,
                        first_tokens=first_tokens, first_loss=first_loss, risk_draws=risk_draws,
                        seed=np.asarray(seed, dtype=np.uint64), repetitions=repetitions)
    return {'passed': passed, 'repetitions': repetitions, 'people': n, 'households': len(groups),
            'metrics': metrics, 'criterion': 'absolute MC error <= 6 conditional MC standard errors + 1e-11',
            'resampling_unit': 'tokens only; original people, weights and household counts preserved'}


def _ess(weights):
    return 0. if not len(weights) or weights.sum() <= 0 else float(1/np.sum((weights/weights.sum())**2))


def _check_metrics(summary, loss, accuracy, rows, n_classes):
    weights, y = rows['weights'], rows['y']
    _compare_scores(summary['ce'], _scores(loss, weights), 'Evaluation CE')
    _compare_scores(summary['accuracy'], {'unweighted': float(accuracy.mean()),
                    'weighted': float(np.dot(weights/weights.sum(), accuracy))}, 'Evaluation accuracy')
    _close(summary['support'], len(y), 'Evaluation support', exact=True)
    _close(summary['weight_sum'], weights.sum(), 'Evaluation weight total')
    _close(summary['ess'], _ess(weights), 'Evaluation person ESS')
    _require(len(summary['per_class']) == n_classes, 'Evaluation full class schema changed')
    for label, saved in enumerate(summary['per_class']):
        mask = y == label; w = weights[mask]
        _require(saved['class'] == label and saved['support'] == int(mask.sum()), 'Per-class order/support mismatch')
        ce = _scores(loss[mask], w) if mask.any() and w.sum() > 0 else {
            'unweighted': float(loss[mask].mean()) if mask.any() else None, 'weighted': None, 'balanced': None}
        _compare_scores(saved['ce'], ce, 'Per-class CE')
        _close(saved['weight_sum'], w.sum(), 'Per-class weight total')
        _close(saved['ess'], _ess(w), 'Per-class ESS')


def verify_evaluation(registry, ctx_eval, release_eval, evaluation_dir, *, out_dir,
                      role_keys=None, wire_repetitions=512, wire_seed=WIRE_SEED,
                      original_root=None, artifact_root=None):
    """Reconstruct saved evaluation predictions and selected diagnostic losses.

    No evaluation-based choice is calculated. Set wire_repetitions=0 for a
    deterministic artifact replay; finalists should retain the fixed MC check.
    Every row is replayed. Use role_keys for a declared material-role sample.
    """
    registry = _registry(registry, original_root=original_root, artifact_root=artifact_root)
    _require(registry['anchor'] == ctx_eval['anchor'], 'Evaluation replay anchor differs from frozen registry')
    _require(isinstance(wire_repetitions, int) and wire_repetitions >= 0, 'Invalid wire repetition count')
    directory, cache = _output(out_dir), {}
    _require(directory is not None, 'Evaluation replay requires a private output directory')
    if original_root is not None or artifact_root is not None:
        original, restored = _relocation_roots(original_root, artifact_root)
        evaluation_dir = _relocated_path(evaluation_dir, original, restored, allow_restored=True)
    evaluation_dir = Path(evaluation_dir)
    summary_path = evaluation_dir/'summary.json'
    if artifact_root is not None:
        summary_path = _owned_file(summary_path, restored)
    summary = json.loads(summary_path.read_text())
    report = {'passed': True, 'stage': 'evaluation', 'pools': {}}
    with threadpool_limits(limits=1):
        for pool, data in ctx_eval['pools'].items():
            report['pools'][pool] = {}
            for key in _chosen_roles(registry, role_keys):
                role = registry['roles'][key]
                rows = _rows(data, role['target']); spec = _release(release_eval[pool], rows)
                saved_summary = summary[pool][key]
                _require(saved_summary['selection'] == role['selection'] and
                         saved_summary['independent_selection'] == role['independent_selection'],
                         'Evaluation selection differs from frozen validation selection')
                array_path = evaluation_dir/pool/(_safe(key)+'.npz')
                if artifact_root is not None:
                    array_path = _owned_file(array_path, restored)
                with np.load(array_path, allow_pickle=False) as saved:
                    _identity_check(saved, rows)
                    saved_q = _probabilities(saved['probabilities'], ndim=3)
                    saved_p = _probabilities(saved['token_probs'], ndim=2)
                    q, p = _reconstruct(role['candidates'][role['selection']], role, rows, spec, cache)
                    _close(saved_q, q, 'Selected conditional probabilities')
                    _close(saved_p, p, 'Selected token law', exact=True)
                    loss, _ = _loss(q, p, rows['y'])
                    accuracy = np.sum(p*(q.argmax(2) == rows['y'][:, None]), axis=1)
                    _close(saved['loss'], loss, 'Selected expected per-person loss')
                    _close(saved['accuracy'], accuracy, 'Selected expected per-person accuracy')
                    _check_metrics(saved_summary, loss, accuracy, rows, q.shape[2])
                    iq, ip = _reconstruct(role['candidates'][role['independent_selection']], role, rows, spec, cache)
                    independent, _ = _loss(iq, ip, rows['y'])
                    _close(saved['independent_loss'], independent, 'Independent expected per-person loss')
                    _compare_scores(saved_summary['independent_ce'], _scores(independent, rows['weights']), 'Independent CE')
                    if 'fixed_decoder' in role['candidates']:
                        fq, fp = _reconstruct(role['candidates']['fixed_decoder'], role, rows, spec, cache)
                        fixed, _ = _loss(fq, fp, rows['y'])
                        _close(saved['fixed_decoder_loss'], fixed, 'Fixed-decoder loss')
                        _compare_scores(saved_summary['fixed_decoder_ce'], _scores(fixed, rows['weights']), 'Fixed-decoder CE')
                row = {**_coverage(rows), 'selection': role['selection'], 'same_selection_U_PWGTP': True,
                       'expected_loss_reconstructed_before_token_averaging': True}
                if wire_repetitions:
                    seed_material = f'{wire_seed}|{registry["anchor"]}|{registry["name"]}|{pool}|{key}'
                    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], 'little')
                    row['wire_check'] = sampled_wire_check(q, p, rows['y'], rows['weights'], rows['ids'], rows['households'],
                        directory/pool/(_safe(key)+'_wire.npz'), repetitions=wire_repetitions, seed=seed)
                    _require(row['wire_check']['passed'], 'Independent sampled-wire check exceeded its fixed MC tolerance')
                report['pools'][pool][key] = row
    _json(directory/'verification.json', {**report, 'private_model_hashes': _model_hashes(cache)})
    return report


def verify_ancestor_parity(h_registry, registry, ctx, h_release, release, *, role_keys=None, out_dir=None,
                           original_root=None, artifact_root=None):
    """Check every H candidate, canonical B choice, and actual A/B ancestor route."""
    baseline, registry = (_registry(value, original_root=original_root, artifact_root=artifact_root)
                          for value in (h_registry, registry))
    _require(baseline.get('is_H_release') is True and baseline['anchor'] == registry['anchor'] == ctx['anchor'],
             'H parity requires a matching H-only baseline anchor')
    directory, cache = _output(out_dir), {}
    report = {'passed': True, 'roles': {}}
    with threadpool_limits(limits=1):
        for key in _chosen_roles(registry, role_keys):
            role, reference = registry['roles'][key], baseline['roles'][key]
            pools = [role['validation_pool']] if role['validation_pool'] in ctx['pools'] else list(ctx['pools'])
            comparisons = []
            for cid, original in reference['candidates'].items():
                if original['route']['kind'] != 'model':
                    continue
                inherited_id = cid if role['view'] == 'B' or registry.get('is_H_release') else 'H__'+cid
                _require(inherited_id in role['candidates'], 'A compatible H ancestor is missing')
                inherited = role['candidates'][inherited_id]
                _require(original['route'] == inherited['route'] and inherited['route']['wire'] == 'H', 'H input-route parity failed')
                _require(Path(original['model_path']).resolve() == Path(inherited['model_path']).resolve(), 'H ancestor model provenance changed')
                comparisons.append((original, reference, inherited, role, True))
            if role['view'] == 'B':
                _require(role['selection'] == reference['selection'] and set(role['candidates']) == set(reference['candidates']),
                         'Canonical B selection/candidate parity failed')
            if role['view'] == 'AB' and role['target'] in ('SEX', 'RAC1P'):
                for source in ('A', 'B'):
                    source_role = registry['roles']['attack:'+source+'/'+role['target']]
                    for cid, original in source_role['candidates'].items():
                        inherited_id = 'ancestor_'+source+'__'+cid
                        _require(inherited_id in role['candidates'], 'An actual singleton ancestor is missing')
                        inherited = role['candidates'][inherited_id]
                        _require(original['route'] == inherited['route'] and
                                 Path(original['model_path']).resolve() == Path(inherited['model_path']).resolve(),
                                 'Actual singleton ancestor route/model provenance changed')
                        comparisons.append((original, source_role, inherited, role, False))
            per_pool = {}
            for pool in pools:
                rows = _rows(ctx['pools'][pool], role['target'])
                original_spec, current_spec = _release(h_release[pool], rows), _release(release[pool], rows)
                for original, source_role, inherited, receiving_role, uses_h in comparisons:
                    q1, p1 = _reconstruct(original, source_role, rows, original_spec if uses_h else current_spec, cache)
                    q2, p2 = _reconstruct(inherited, receiving_role, rows, current_spec, cache)
                    _close(q1, q2, 'H/B/singleton probability parity', exact=True)
                    _close(p1, p2, 'H/B/singleton token-law parity', exact=True)
                per_pool[pool] = {**_coverage(rows), 'candidate_parity_comparisons': len(comparisons)}
            report['roles'][key] = per_pool
    if directory is not None:
        _json(directory/'parity.json', {**report, 'private_model_hashes': _model_hashes(cache)})
    return report


def artifact_disk_inventory(root):
    """Sizes only: inspect no dataset, prediction, score or model contents."""
    groups = {}
    for path in Path(root).rglob('*'):
        if not path.is_file():
            continue
        category = ('registries' if path.name == 'registry.joblib' else
                    'trajectory_checkpoints' if '_trajectory' in str(path.parent) else
                    'private_arrays' if path.suffix == '.npz' else
                    'models' if path.suffix in ('.pt', '.joblib') else 'metadata_and_other')
        record = groups.setdefault(category, {'files': 0, 'bytes': 0})
        record['files'] += 1; record['bytes'] += path.stat().st_size
    return {'categories': groups, 'total_bytes': sum(record['bytes'] for record in groups.values()),
            'content_read': False}
