"""Weighted auditors for a continuous H and an actually released random token.

The risk is E_token[-log q(y | H, token)], never -log E_token[q(y | H,
token)]. Fitting and selection each use 0.5 unweighted + 0.5 PWGTP risk.
No development/test inputs are accepted by the fitting API. All fitted models,
validation-considered MLP weights, and boundary optimizer/RNG states are saved
under the caller's private directory; JSON contains aggregates and hashes only.

HistGradientBoosting's min_samples_leaf counts expanded rows, not weights. We
multiply the nominal minimum by the maximum tokens supported by any person.
Every leaf therefore contains at least the nominal number of distinct people.
This conservative safeguard does NOT ensure that a leaf carries that much
token-probability mass; in a token-specific leaf it can be more restrictive.
Histogram quantile construction also ignores sample weights. The prospectively
amended slate therefore additionally fits both tree leaf sizes on one fixed
sampled token per original person. Those candidates restore ordinary person
counts/binning but carry one-draw Monte Carlo training variation. Validation is
the exact token expectation for every candidate, including sampled trees.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


PROBABILITY_FLOOR = 1e-9
PREDICTION_BATCH_PERSONS = 256
SELECTION_RULE = '0.5 * unweighted expected log loss + 0.5 * PWGTP expected log loss; lexical candidate tie'
AUDIT_CONFIG = {
    'logistic': {'C': 1., 'max_iter': 2000, 'solver': 'lbfgs', 'tol': 1e-4},
    'mlp': {'hidden': [64, 32], 'epochs': 360, 'nested_epochs': 120,
            'batch_size': 256, 'lr': 1e-3, 'weight_decay': 1e-4,
            'validation_interval': 5},
    'histgb': {'max_iter': 150, 'max_leaf_nodes': 15, 'learning_rate': .1,
               'l2_regularization': 1., 'early_stopping': False},
}


def _hash_array(x):
    x = np.ascontiguousarray(x)
    digest = hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode())
    digest.update(x.tobytes())
    return digest.hexdigest()


def _state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(_hash_array(value.detach().cpu().numpy()).encode())
    return digest.hexdigest()


def _write_json(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _features(h):
    try:
        h = np.asarray(h, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError('H must be finite numeric coordinates') from error
    if h.ndim != 2 or h.shape[1] < 1 or not np.isfinite(h).all():
        raise ValueError('H must be a finite matrix with at least one coordinate')
    return h


def _labels(y, n_classes):
    if (isinstance(n_classes, (bool, np.bool_)) or
            not isinstance(n_classes, (int, np.integer)) or n_classes < 2):
        raise ValueError('Declare integer n_classes >= 2')
    y = np.asarray(y)
    if (y.ndim != 1 or y.dtype.kind not in 'biuf' or not np.isfinite(y).all()
            or not np.equal(y, np.floor(y)).all()
            or (y < 0).any() or (y >= n_classes).any()):
        raise ValueError('Labels must be integer indices in the full class schema')
    return y.astype(np.int64, copy=False)


def _weights(weights, n):
    w = np.asarray(weights, dtype=np.float64)
    if (w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any()
            or not np.isfinite(w.sum()) or w.sum() <= 0):
        raise ValueError('Weights must align, be finite nonnegative, and have positive finite total')
    return w


def _token_probs(token_probs, n=None):
    p = np.asarray(token_probs, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] < 1 or (n is not None and len(p) != n)
            or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(1), 1., atol=1e-8, rtol=0)):
        raise ValueError('Token probabilities must be finite nonnegative normalized aligned rows')
    return p


def validate_ids(ids, n, *, name='ids'):
    """Validate original-person identifiers without coercing distinct types."""
    ids = np.asarray(ids, dtype=object)
    if ids.shape != (n,):
        raise ValueError(f'{name} must align with original people')
    keys = []
    for value in ids:
        if (value is None or isinstance(value, (bool, np.bool_))
                or not isinstance(value, (str, int, float, np.integer, np.floating))
                or (isinstance(value, str) and not value.strip())
                or (not isinstance(value, str) and not np.isfinite(value))):
            raise ValueError(f'{name} must contain nonmissing finite scalar identifiers')
        # Numeric aliases such as 1 and 1.0 denote the same original ID.
        # String IDs remain exact ("001" is not silently converted to 1).
        keys.append(('str', value) if isinstance(value, str) else ('numeric', value))
    if len(set(keys)) != n:
        raise ValueError(f'{name} must contain unique original-person identifiers')
    return ids


def validate_inputs(h, token_probs, y, weights, n_classes, *, ids=None, class_order=None):
    """Strict input contract, optionally including externally maintained IDs.

    Labels always index the full declared order 0..n_classes-1. The data layer
    must separately enforce disjoint fitting/validation/development/test IDs.
    """
    h, y = _features(h), _labels(y, n_classes)
    if len(h) != len(y) or not len(y):
        raise ValueError('Aligned nonempty original people required')
    p, w = _token_probs(token_probs, len(y)), _weights(weights, len(y))
    if class_order is not None and list(class_order) != list(range(n_classes)):
        raise ValueError('Class order must be the full declared 0..n_classes-1 schema')
    if ids is not None:
        validate_ids(ids, len(y))
    return h, p, y, w


def balanced_person_weights(weights):
    """Mean-one fitting weights for the one shared U/PWGTP predictor."""
    w = np.asarray(weights, dtype=np.float64)
    w = _weights(w, len(w))
    return .5 + .5 * ((w / w.sum()) * len(w))


def expand_person_tokens(token_probs, person_weights):
    """Exact nonzero person-token support with conserved original weights."""
    p = _token_probs(token_probs)
    w = _weights(person_weights, len(p))
    person, token = np.nonzero(p > 0)
    expanded_weight = w[person] * p[person, token]
    if not np.allclose(np.bincount(person, weights=expanded_weight, minlength=len(p)),
                       w, atol=1e-10, rtol=1e-8):
        raise ValueError('Token expansion failed original-person weight conservation')
    return person, token, expanded_weight


def expected_token_loss(predictions, token_probs, y):
    """Evaluate a full (person, token, class) probability tensor exactly."""
    q = np.asarray(predictions, dtype=np.float64)
    if q.ndim != 3:
        raise ValueError('Predictions must have person, token, full-class axes')
    p = _token_probs(token_probs)
    y = _labels(y, q.shape[2])
    if (q.shape[:2] != p.shape or len(y) != len(p) or not np.isfinite(q).all()
            or (q < 0).any() or (q > 1).any()
            or not np.allclose(q.sum(2), 1., atol=1e-8, rtol=0)):
        raise ValueError('Predictions must be finite normalized full-schema probabilities')
    observed = q[np.arange(len(y))[:, None], np.arange(q.shape[1])[None, :], y[:, None]]
    # A declared floor also makes external fixed predictions with zeros finite.
    return np.sum(p * -np.log(np.maximum(observed, PROBABILITY_FLOOR)), axis=1)


def loss_scores(loss, weights):
    loss = np.asarray(loss, dtype=np.float64)
    w = _weights(weights, len(loss))
    if loss.shape != w.shape or not np.isfinite(loss).all() or (loss < 0).any():
        raise ValueError('Expected losses must be finite nonnegative original-person values')
    unweighted, weighted = float(loss.mean()), float(np.dot(w / w.sum(), loss))
    return {'unweighted': unweighted, 'weighted': weighted,
            'balanced': .5 * unweighted + .5 * weighted}


def select_losses(losses, weights):
    if not losses or any(not isinstance(cid, str) or not cid for cid in losses):
        raise ValueError('At least one explicitly named candidate is required')
    scores = {cid: loss_scores(loss, weights) for cid, loss in losses.items()}
    selected = min(scores, key=lambda cid: (scores[cid]['balanced'], cid))
    return {'selection': selected, 'scores': scores, 'rule': SELECTION_RULE}


def weighted_person_cross_entropy(logits, labels, expanded_weights, n_persons):
    """Exact token-expectation minibatch loss divided by ORIGINAL people."""
    if not isinstance(n_persons, int) or n_persons < 1:
        raise ValueError('Positive original-person minibatch count required')
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    return (loss * expanded_weights).sum() / n_persons


def _encoded(h, token, n_tokens, interactions):
    # A one-token wire is deterministic: omit constant/redundant columns.
    if n_tokens == 1:
        return np.ascontiguousarray(h)
    onehot = np.eye(n_tokens, dtype=h.dtype)[token]
    parts = [h, onehot]
    if interactions:
        parts.append((h[:, :, None] * onehot[:, None, :]).reshape(len(h), -1))
    return np.ascontiguousarray(np.concatenate(parts, axis=1))


def _network(width, n_classes):
    return torch.nn.Sequential(torch.nn.Linear(width, 64), torch.nn.ReLU(),
                               torch.nn.Linear(64, 32), torch.nn.ReLU(),
                               torch.nn.Linear(32, n_classes))


@dataclass
class TokenCandidate:
    family: str
    model: object
    mean: np.ndarray
    scale: np.ndarray
    n_tokens: int
    n_classes: int
    interactions: bool
    metadata: dict

    def _predict_encoded(self, x):
        if self.family == 'prior':
            q = np.broadcast_to(self.model, (len(x), self.n_classes)).copy()
        elif self.family == 'mlp':
            self.model.eval()
            with torch.no_grad():
                q = torch.softmax(self.model(torch.as_tensor(x, dtype=torch.float32)), dim=1).double().numpy()
        else:
            columns = np.asarray(self.model.classes_)
            if (columns.ndim != 1 or columns.dtype.kind not in 'iu'
                    or len(np.unique(columns)) != len(columns)
                    or (columns < 0).any() or (columns >= self.n_classes).any()):
                raise ValueError('Fitted model has corrupted class order')
            q = np.zeros((len(x), self.n_classes), dtype=np.float64)
            q[:, columns] = self.model.predict_proba(x)
        if (not np.isfinite(q).all() or (q < 0).any()
                or not np.allclose(q.sum(1), 1., atol=1e-6, rtol=0)):
            raise FloatingPointError('Nonfinite or corrupted fitted predictions')
        q /= q.sum(1, keepdims=True)
        # A common full-schema uniform mixture gives missing classes a fixed
        # finite nonzero floor without validation-label-dependent smoothing.
        return (1. - self.n_classes * PROBABILITY_FLOOR) * q + PROBABILITY_FLOOR

    def predict_token_proba(self, h, n_tokens=None):
        h = _features(h)
        n_tokens = self.n_tokens if n_tokens is None else n_tokens
        if (n_tokens != self.n_tokens or h.shape[1] != len(self.mean)
                or self.metadata['class_order'] != list(range(self.n_classes))):
            raise ValueError('Prediction token, H, or class schema differs from fitting')
        result = np.empty((len(h), n_tokens, self.n_classes), dtype=np.float64)
        for start in range(0, len(h), PREDICTION_BATCH_PERSONS):
            block = (h[start:start + PREDICTION_BATCH_PERSONS] - self.mean) / self.scale
            person = np.repeat(np.arange(len(block)), n_tokens)
            token = np.tile(np.arange(n_tokens), len(block))
            x = _encoded(block[person], token, n_tokens, self.interactions)
            result[start:start + len(block)] = self._predict_encoded(x).reshape(len(block), n_tokens, self.n_classes)
        return result

    def predict_proba(self, h):
        """Deterministic-token convenience for teachers and continuous arms."""
        if self.n_tokens != 1:
            raise ValueError('Random-token candidates require conditional token predictions')
        return self.predict_token_proba(h, 1)[:, 0, :]

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.metadata['directory'] = str(directory.resolve())
        payload = {'family': self.family, 'mean': self.mean, 'scale': self.scale,
                   'n_tokens': self.n_tokens, 'n_classes': self.n_classes,
                   'interactions': self.interactions}
        if self.family == 'mlp':
            torch.save({'state': self.model.state_dict(),
                        'input_dim': self.metadata['input_dim'],
                        'n_classes': self.n_classes}, directory/'model.pt')
        else:
            payload['model'] = self.model
        joblib.dump(payload, directory/'candidate.joblib', compress=3)
        _write_json(directory/'metadata.json', self.metadata)


def load_candidate(directory):
    """Reload only trusted private checkpoints generated by this module."""
    directory = Path(directory)
    metadata = json.loads((directory/'metadata.json').read_text())
    payload = joblib.load(directory/'candidate.joblib')
    if payload['family'] == 'mlp':
        checkpoint = torch.load(directory/'model.pt', map_location='cpu', weights_only=True)
        model = _network(checkpoint['input_dim'], checkpoint['n_classes'])
        model.load_state_dict(checkpoint['state'])
        payload['model'] = model.eval().requires_grad_(False)
    if metadata['class_order'] != list(range(payload['n_classes'])):
        raise ValueError('Persisted class order differs from the full schema')
    return TokenCandidate(**payload, metadata=metadata)


def predict_token_proba(candidate, h, n_tokens):
    return candidate.predict_token_proba(h, n_tokens)


def expected_loss(candidate, h, token_probs, y):
    """Stream at most 256 original people; retain only per-person losses."""
    h, p, y, _ = validate_inputs(h, token_probs, y, np.ones(len(h)), candidate.n_classes)
    if p.shape[1] != candidate.n_tokens:
        raise ValueError('Token schema differs from fitted candidate')
    out = np.empty(len(y), dtype=np.float64)
    for start in range(0, len(y), PREDICTION_BATCH_PERSONS):
        end = start + PREDICTION_BATCH_PERSONS
        q = predict_token_proba(candidate, h[start:end], p.shape[1])
        out[start:end] = expected_token_loss(q, p[start:end], y[start:end])
    return out


def _common(h, p, y, w, hv, pv, yv, wv, n_classes, seed):
    counts = np.bincount(y, minlength=n_classes)
    return {'n_classes': int(n_classes), 'class_order': list(range(n_classes)),
            'n_tokens': p.shape[1], 'h_dimension': h.shape[1], 'seed': int(seed),
            'fit_rows': len(y), 'validation_rows': len(yv),
            'fit_support': counts.tolist(), 'fit_coverage_complete': bool((counts > 0).all()),
            'fit_hashes': {name: _hash_array(x) for name, x in [('h', h), ('tokens', p), ('y', y), ('weights', w)]},
            'validation_hashes': {name: _hash_array(x) for name, x in [('h', hv), ('tokens', pv), ('y', yv), ('weights', wv)]},
            'probability_floor': PROBABILITY_FLOOR,
            'prediction_regularization': '(1 - n_classes * 1e-9) * full_schema_probability + 1e-9',
            'training_weighting': 'person weight = 0.5 + 0.5 * PWGTP / mean(PWGTP)',
            'selection_rule': SELECTION_RULE,
            'preprocessing': 'unweighted mean/std of continuous H on original fitting people; token onehot unscaled',
            'device': 'cpu', 'threads': 1,
            'fit_expanded_rows': int(np.count_nonzero(p))}


def _evaluate(candidate, hv, pv, yv, wv):
    return loss_scores(expected_loss(candidate, hv, pv, yv), wv)


def _fit_static(family, cid, h, p, y, fit_weight, hv, pv, yv, wv,
                mean, scale, common, directory, parameters, *, leaf=None, sampled_tokens=None):
    start = time.perf_counter()
    interactions = family == 'logistic'
    if sampled_tokens is None:
        person, token, expanded_weight = expand_person_tokens(p, fit_weight)
    else:
        person, token, expanded_weight = np.arange(len(y)), sampled_tokens, fit_weight
    x = _encoded(((h - mean) / scale)[person], token, p.shape[1], interactions)
    meta = {**copy.deepcopy(common), 'family': family, 'candidate_id': cid,
            'input_dim': x.shape[1], 'parameters': dict(parameters),
            'feature_encoding': 'H + token onehot + H*token onehot' if interactions else 'H + token onehot',
            'training_person_exposures': len(y), 'training_expanded_exposures': len(person),
            'expanded_feature_bytes': int(x.nbytes), 'optimizer_steps': 0}
    if leaf is not None:
        max_support = int((p > 0).sum(1).max()) if sampled_tokens is None else 1
        parameters = {**parameters, 'min_samples_leaf': int(leaf * max_support)}
        meta.update(parameters=parameters, nominal_min_samples_leaf=leaf,
                    max_supported_tokens_per_person=max_support,
                    minimum_distinct_persons_per_leaf=min(leaf, len(y)),
                    leaf_support_caveat='Conservative distinct-person lower bound; no token-probability-mass guarantee; can overconstrain token-specific leaves',
                    histogram_binning_caveat='Native histogram quantile construction ignores sample weights; exact-expanded rows influence bins equally regardless of token probability')
        if sampled_tokens is not None:
            meta.update(sampling_seed=20260921 + common['seed'],
                        sampled_token_hash=_hash_array(sampled_tokens),
                        reused_deterministic_exact_tree=False,
                        leaf_support_caveat='One actual sampled token per original person; min_samples_leaf counts original people',
                        histogram_binning_caveat='Histogram bins count original people once; native quantile construction still ignores PWGTP',
                        sampling_limitation='One fixed categorical realization per original person; Monte Carlo training variance remains; validation token expectation is exact')
    if len(np.unique(y)) == 1:
        counts = np.bincount(y, weights=fit_weight, minlength=common['n_classes'])
        model = (counts + 1.) / (counts.sum() + common['n_classes'])
        effective_family = 'prior'
        meta['fallback_reason'] = 'single_fitting_class; all nominal candidates retained'
    else:
        effective_family = family
        if family == 'logistic':
            model = LogisticRegression(**parameters, random_state=common['seed'])
        else:
            model = HistGradientBoostingClassifier(**parameters, random_state=common['seed'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x, y[person], sample_weight=expanded_weight)
        meta['convergence_warnings'] = [str(warning.message) for warning in caught]
        meta['actual_iterations'] = int(np.max(model.n_iter_))
    meta['effective_family'] = effective_family
    candidate = TokenCandidate(effective_family, model, mean.copy(), scale.copy(), p.shape[1],
                               common['n_classes'], interactions, meta)
    meta['validation_scores'] = _evaluate(candidate, hv, pv, yv, wv)
    meta['fit_runtime_seconds'] = time.perf_counter() - start
    candidate.save(directory/cid)
    return candidate


def _fit_mlp(h, p, y, fit_weight, hv, pv, yv, wv, mean, scale, common,
             directory, *, prefix, boundaries, weight_decay, seed):
    started = time.perf_counter()
    config = AUDIT_CONFIG['mlp']
    trajectory_dir = directory/(prefix + '_trajectory')
    trajectory_dir.mkdir()
    standardized = (h - mean) / scale
    width = h.shape[1] if p.shape[1] == 1 else h.shape[1] + p.shape[1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _network(width, common['n_classes']).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=weight_decay)
    rng = np.random.default_rng(seed + 700000)
    initial_hash = _state_hash(model)
    schedule_digest = hashlib.sha256()
    common = {**copy.deepcopy(common), 'family': 'mlp', 'seed': int(seed),
              'input_dim': width, 'feature_encoding': 'H + token onehot',
              'batch_size_persons': config['batch_size'], 'initial_state_hash': initial_hash,
              'schedule_seed': int(seed + 700000),
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'checkpoint_selection': 'balanced validation expected log loss; earliest epoch tie',
              'checkpoint_directory': str(trajectory_dir.resolve()),
              'trajectory_continuation': 'actual last weights and Adam/RNG state; no selected-state rewind'}
    live = TokenCandidate('mlp', model, mean.copy(), scale.copy(), p.shape[1], common['n_classes'], False, common)
    best_state, best_key, best_steps = None, (float('inf'), -1), 0
    steps, person_exposures, expanded_exposures = 0, 0, 0
    curve, candidates = [], {}

    def evaluate(epoch):
        nonlocal best_state, best_key, best_steps
        scores = _evaluate(live, hv, pv, yv, wv)
        state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        state_hash = _state_hash(model)
        row = {'epoch': epoch, 'optimizer_steps': steps, **scores, 'state_hash': state_hash}
        curve.append(row)
        # Persist every validation-considered state, not only its current winner.
        torch.save({'state': state, 'input_dim': width, 'n_classes': common['n_classes'],
                    'epoch': epoch, 'optimizer_steps': steps, 'validation_scores': scores,
                    'state_hash': state_hash}, trajectory_dir/f'epoch_{epoch:04d}.pt')
        key = (scores['balanced'], epoch)
        if key < best_key:
            best_state, best_key, best_steps = state, key, steps

    evaluate(0)
    for epoch in range(1, max(boundaries) + 1):
        order = rng.permutation(len(y))
        schedule_digest.update(order.tobytes())
        for start in range(0, len(y), config['batch_size']):
            rows = order[start:start + config['batch_size']]
            person, token, ew = expand_person_tokens(p[rows], fit_weight[rows])
            x = _encoded(standardized[rows[person]], token, p.shape[1], False)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.as_tensor(x, dtype=torch.float32))
            loss = weighted_person_cross_entropy(logits, torch.as_tensor(y[rows[person]], dtype=torch.long),
                                                 torch.as_tensor(ew, dtype=torch.float32), len(rows))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite weighted original-person MLP objective')
            loss.backward()
            optimizer.step()
            steps += 1
            person_exposures += len(rows)
            expanded_exposures += len(person)
        if epoch % config['validation_interval'] == 0 or epoch in boundaries:
            evaluate(epoch)
        if epoch in boundaries:
            cid = f'{prefix}_{epoch}'
            frozen = copy.deepcopy(model)
            frozen.load_state_dict(best_state)
            frozen.eval().requires_grad_(False)
            meta = {**copy.deepcopy(common), 'candidate_id': cid,
                    'parameters': {**config, 'epochs': epoch, 'weight_decay': weight_decay},
                    'trajectory_epochs': max(boundaries), 'optimizer_steps': steps,
                    'selected_epoch': best_key[1], 'selected_optimizer_steps': best_steps,
                    'training_person_exposures': person_exposures,
                    'training_expanded_exposures': expanded_exposures,
                    'schedule_hash': schedule_digest.hexdigest(),
                    'final_state_hash': _state_hash(model), 'selected_state_hash': _state_hash(frozen),
                    'validation_curve': copy.deepcopy(curve),
                    'fit_runtime_seconds': time.perf_counter() - started}
            candidate = TokenCandidate('mlp', frozen, mean.copy(), scale.copy(), p.shape[1],
                                       common['n_classes'], False, meta)
            meta['validation_scores'] = _evaluate(candidate, hv, pv, yv, wv)
            if meta['validation_scores']['balanced'] != best_key[0]:
                raise AssertionError('Selected weighted MLP checkpoint failed exact replay')
            candidate.save(directory/cid)
            candidates[cid] = candidate
            torch.save({'epoch': epoch, 'optimizer_steps': steps,
                        'model_state': copy.deepcopy(model.state_dict()),
                        'optimizer_state': copy.deepcopy(optimizer.state_dict()),
                        'schedule_rng_state': copy.deepcopy(rng.bit_generator.state),
                        'schedule_hash': schedule_digest.hexdigest(),
                        'state_hash': _state_hash(model), 'selected_state_hash': _state_hash(frozen)},
                       trajectory_dir/f'boundary_{epoch:04d}.pt')
    return candidates


def fit_slate(h_fit, token_probs_fit, y_fit, weights_fit,
              h_val, token_probs_val, y_val, weights_val,
              n_classes, seed, out_dir, *, slate='standard'):
    """Fit the preregistered symmetric audit slate on fitting/validation only.

    ``standard`` trains one 120-epoch MLP; ``catchup`` trains one uninterrupted
    360-epoch trajectory and preserves the nested 120- and 360-epoch selections.
    ``teacher`` is reserved for fit_predictor_slate: three logistic C values
    and two 200-epoch MLP weight-decay values on deterministic continuous input.
    Caller must keep out_dir private. No overwrite/resume is performed.
    """
    if slate not in ('standard', 'catchup', 'teacher'):
        raise ValueError('Unknown fixed candidate slate')
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0 or seed >= 2**32 - 700000):
        raise ValueError('Seed must be a nonnegative bounded integer')
    h, p, y, w = validate_inputs(h_fit, token_probs_fit, y_fit, weights_fit, n_classes)
    hv, pv, yv, wv = validate_inputs(h_val, token_probs_val, y_val, weights_val, n_classes)
    if h.shape[1] != hv.shape[1] or p.shape[1] != pv.shape[1]:
        raise ValueError('Fitting and validation H/token schemas must match')
    if slate == 'teacher' and p.shape[1] != 1:
        raise ValueError('Teacher predictors require deterministic continuous coordinates')
    directory = Path(out_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Private slate directory must be empty; do not overwrite prior candidates')
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    common = _common(h, p, y, w, hv, pv, yv, wv, n_classes, seed)
    fit_weight = balanced_person_weights(w)
    mean, scale = h.mean(0), h.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError('Nonfinite fitting-only continuous normalizer')
    candidates = {}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            cs = (.03, .3, 3.) if slate == 'teacher' else (1.,)
            for c in cs:
                cid = f'logistic_C{c:g}' if slate == 'teacher' else 'logistic'
                candidates[cid] = _fit_static('logistic', cid, h, p, y, fit_weight, hv, pv, yv, wv,
                    mean, scale, common, directory, {**AUDIT_CONFIG['logistic'], 'C': c})
            if slate != 'teacher':
                for leaf in (20, 5):
                    cid = f'hist_gb_{leaf}'
                    candidates[cid] = _fit_static('histgb', cid, h, p, y, fit_weight, hv, pv, yv, wv,
                        mean, scale, common, directory, AUDIT_CONFIG['histgb'], leaf=leaf)
                # Registered common random uniforms are indexed by the supplied
                # original-person order; both leaf sizes see the same draw.
                uniforms = np.random.default_rng(20260921 + int(seed)).random(len(y))
                cumulative = np.cumsum(p, axis=1)
                cumulative[:, -1] = 1.
                sampled_tokens = (uniforms[:, None] >= cumulative).sum(axis=1).astype(np.int64)
                if not (p[np.arange(len(y)), sampled_tokens] > 0).all():
                    raise AssertionError('Categorical schedule sampled an unsupported token')
                np.savez_compressed(directory/'histgb_training_draw.npz', tokens=sampled_tokens,
                                    uniforms=uniforms, original_person_rows=np.arange(len(y)))
                deterministic = bool(((p > 0).sum(1) == 1).all())
                for leaf in (20, 5):
                    cid = f'sampled_hist_gb_{leaf}'
                    if deterministic:
                        exact = candidates[f'hist_gb_{leaf}']
                        candidate = TokenCandidate(exact.family, exact.model, exact.mean.copy(), exact.scale.copy(),
                                                   exact.n_tokens, exact.n_classes, False, copy.deepcopy(exact.metadata))
                        candidate.metadata.update(candidate_id=cid, reused_deterministic_exact_tree=True,
                            reused_from_candidate=f'hist_gb_{leaf}', sampling_seed=20260921 + int(seed),
                            sampled_token_hash=_hash_array(sampled_tokens), fit_runtime_seconds=0.,
                            sampling_limitation='Deterministic support: sampled and exact-expanded training arrays are identical; no token-draw variation')
                        candidate.save(directory/cid)
                        candidates[cid] = candidate
                    else:
                        candidates[cid] = _fit_static('histgb', cid, h, p, y, fit_weight, hv, pv, yv, wv,
                            mean, scale, common, directory, AUDIT_CONFIG['histgb'], leaf=leaf,
                            sampled_tokens=sampled_tokens)
                boundaries = (120, 360) if slate == 'catchup' else (120,)
                candidates.update(_fit_mlp(h, p, y, fit_weight, hv, pv, yv, wv, mean, scale, common,
                    directory, prefix='mlp', boundaries=boundaries, weight_decay=1e-4, seed=int(seed)))
            else:
                for decay in (0., 1e-4):
                    candidates.update(_fit_mlp(h, p, y, fit_weight, hv, pv, yv, wv, mean, scale, common,
                        directory, prefix=f'mlp_wd{decay:g}', boundaries=(200,), weight_decay=decay, seed=int(seed)))
    finally:
        torch.set_num_threads(old_threads)
    scores = {cid: candidate.metadata['validation_scores'] for cid, candidate in candidates.items()}
    chosen = min(scores, key=lambda cid: (scores[cid]['balanced'], cid))
    standard_ids = sorted(cid for cid in candidates if cid != 'mlp_360')
    standard_chosen = min(standard_ids, key=lambda cid: (scores[cid]['balanced'], cid))
    metadata = {**common, 'slate': slate, 'candidate_ids': sorted(candidates),
                'selection': chosen, 'standard_selection': standard_chosen,
                'catchup_selection': chosen, 'validation_scores': scores,
                'fit_runtime_seconds': time.perf_counter() - started,
                'candidates': {cid: candidate.metadata for cid, candidate in candidates.items()},
                'availability_policy': 'same nominal candidate set for every arm; single-class static fits use smoothed prior',
                'same_predictions_for_both_weightings': True}
    _write_json(directory/'slate.json', metadata)
    return {'candidates': candidates, 'selection': chosen, 'standard_selection': standard_chosen,
            'catchup_selection': chosen, 'metadata': metadata}


def fit_predictor_slate(x_fit, y_fit, weights_fit, x_val, y_val, weights_val,
                        n_classes, seed, out_dir, *, slate='teacher'):
    """Deterministic teachers: logistic C=.03,.3,3; MLP WD=0,1e-4, 200 epochs."""
    if slate != 'teacher':
        raise ValueError('Predictor wrapper supports the declared teacher slate only')
    return fit_slate(x_fit, np.ones((len(x_fit), 1)), y_fit, weights_fit,
                     x_val, np.ones((len(x_val), 1)), y_val, weights_val,
                     n_classes, seed, out_dir, slate='teacher')
