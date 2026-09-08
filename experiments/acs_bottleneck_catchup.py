"""Audit and continue a saved adversary in its original mapper coordinates.

The saved 16→64 ReLU→32 ReLU→K network is copied exactly. No input normalizer
is fitted or inherited from an independent auditor. Catch-up uses a new Adam
optimizer on attacker-fitting rows only; validation log loss selects an epoch.
Both returned candidates are frozen and use the historical save/load interface.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time

import numpy as np
import torch

from experiments.acs_transfer_heads import (
    FittedCandidate, InputStandardizer, _features, _hash_array, _labels,
    _state_hash, metrics,
)


CATCHUP_CONFIG = {"input_dim": 16, "hidden": [64, 32], "epochs": 120,
                  "batch_size": 256, "lr": .001, "validation_interval": 5,
                  "weight_decay": 0., "betas": [.9, .999], "eps": 1e-8,
                  "schedule_seed_offset": 700000}


def _check_network(model, n_classes):
    if (not isinstance(model, torch.nn.Sequential) or len(model) != 5
            or not all(isinstance(model[i], torch.nn.Linear) for i in (0, 2, 4))
            or not all(isinstance(model[i], torch.nn.ReLU) for i in (1, 3))):
        raise ValueError("Saved adversary must be Sequential Linear/ReLU/Linear/ReLU/Linear")
    actual = [(model[i].in_features, model[i].out_features) for i in (0, 2, 4)]
    if actual != [(16, 64), (64, 32), (32, n_classes)]:
        raise ValueError("Saved adversary must have the fixed 16→64→32→K architecture")
    if set(model.state_dict()) != {'0.weight', '0.bias', '2.weight', '2.bias', '4.weight', '4.bias'}:
        raise ValueError("Unexpected saved adversary state schema")
    for value in model.state_dict().values():
        if value.device.type != 'cpu' or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("Saved adversary state must be finite float32 CPU tensors")


def _direct_probabilities(model, x):
    """Same batching as FittedCandidate, with no coordinate transformation."""
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            features = torch.as_tensor(x[start:start+4096], dtype=torch.float32)
            chunks.append(torch.softmax(model(features), dim=1).double().numpy())
    return np.concatenate(chunks)


def _identity():
    return InputStandardizer(np.zeros(16, dtype=np.float64), np.ones(16, dtype=np.float64), fitted=False)


def fit_catchup(saved_model, xfit, yfit, xval, yval, n_classes, seed, *,
                epochs=120, inherited_exposure=None):
    """Return a before-catch-up audit and a validation-selected continuation.

    The only variable-budget argument is epochs, permitting a tiny artificial
    regression fixture. Scientific runs use the fixed 120-epoch protocol.
    inherited_exposure is recorded separately, never treated as new fitting
    exposure or used to restore an optimizer. The incoming model is untouched.
    """
    started = time.perf_counter()
    if n_classes not in (2, 9):
        raise ValueError("The fixed protected class schemas have sizes 2 and 9")
    if not isinstance(epochs, (int, np.integer)) or epochs < 0:
        raise ValueError("epochs must be a nonnegative integer")
    _check_network(saved_model, n_classes)
    xf, xv = _features(xfit), _features(xval)
    yf, yv = _labels(yfit, n_classes), _labels(yval, n_classes)
    if (xf.shape != (len(yf), 16) or xv.shape != (len(yv), 16)
            or not len(yf) or not len(yv) or not np.isfinite(xf).all() or not np.isfinite(xv).all()):
        raise ValueError("Require aligned nonempty finite direct-coordinate fitting/validation arrays")
    inherited = copy.deepcopy(inherited_exposure)
    json.dumps(inherited, allow_nan=False)
    initial_hash = _state_hash(saved_model)
    original_modes = [module.training for module in saved_model.modules()]
    original_grad_flags = [value.requires_grad for value in saved_model.parameters()]
    direct_fit, direct_val = _direct_probabilities(saved_model, xf), _direct_probabilities(saved_model, xv)
    saved_copy = copy.deepcopy(saved_model).eval().requires_grad_(False)
    candidate = FittedCandidate('mlp', saved_copy, _identity(), n_classes, {})
    if (not np.array_equal(candidate.predict_proba(xf), direct_fit)
            or not np.array_equal(candidate.predict_proba(xv), direct_val)):
        raise AssertionError("Saved audit does not exactly reproduce original direct-coordinate predictions")
    initial_score = metrics(yv, direct_val, n_classes)
    available_fit_support = np.bincount(yf, minlength=n_classes).tolist()
    fit_hashes = {'x': _hash_array(xf), 'y': _hash_array(yf)}
    validation_hashes = {'x': _hash_array(xv), 'y': _hash_array(yv)}
    common = {
        'family': 'mlp', 'n_classes': int(n_classes), 'class_schema': list(range(n_classes)),
        'input_dim': 16, 'parameters': copy.deepcopy(CATCHUP_CONFIG),
        'preprocessing': 'identity; direct mapper16 coordinates; no mean/scale fitted',
        'preprocessing_fit_rows': 0, 'feature_mean': [0.]*16, 'feature_scale': [1.]*16,
        'provided_attacker_fit_rows': len(yf), 'provided_attacker_fit_support': available_fit_support,
        'validation_rows': len(yv), 'validation_hashes': validation_hashes,
        'source_state_hash': initial_hash, 'inherited_exposure': inherited,
        'inherited_exposure_known': inherited is not None,
        'initial_fidelity_exact': True,
        'original_fit_probability_hash': _hash_array(direct_fit),
        'original_validation_probability_hash': _hash_array(direct_val),
        'device': 'cpu', 'dtype': 'float32 network and direct coordinates',
        'fit_weighted': False,
        'scope': 'empirical audit; prior training exposure and new catch-up exposure remain distinct',
    }
    candidate.metadata = {
        **copy.deepcopy(common), 'audit_kind': 'saved', 'candidate_id': 'saved',
        'fit_rows': 0, 'fit_support': None, 'fit_hashes': None,
        'optimizer_steps': 0, 'training_row_exposures': 0,
        'selected_epoch': 0, 'selected_optimizer_steps': 0,
        'selected_state_hash': initial_hash, 'validation_scores': initial_score,
        'selection': 'unchanged saved adversary; no fitting or validation selection',
    }
    candidate.metadata['parameters']['epochs'] = 0

    trained = copy.deepcopy(saved_model).train().requires_grad_(True)
    optimizer = torch.optim.Adam(trained.parameters(), lr=CATCHUP_CONFIG['lr'],
                                 betas=tuple(CATCHUP_CONFIG['betas']), eps=CATCHUP_CONFIG['eps'],
                                 weight_decay=CATCHUP_CONFIG['weight_decay'])
    initial_optimizer_states = len(optimizer.state)
    if initial_optimizer_states:
        raise AssertionError("Catch-up optimizer must start with no inherited state")
    continued = FittedCandidate('mlp', trained, _identity(), n_classes, {})
    if _state_hash(trained) != initial_hash:
        raise AssertionError("Catch-up did not start at the exact saved adversary")
    x_tensor = torch.as_tensor(xf, dtype=torch.float32)
    y_tensor = torch.as_tensor(yf, dtype=torch.long)
    schedule_seed = int(seed)+CATCHUP_CONFIG['schedule_seed_offset']
    rng = np.random.default_rng(schedule_seed)
    schedule_hash = hashlib.sha256()
    exposures = np.zeros(len(yf), dtype=np.int64)
    curve, steps = [], 0
    best_key, best_state, best_steps = (float('inf'), -1), None, 0

    def evaluate(epoch):
        nonlocal best_key, best_state, best_steps
        scores = metrics(yv, continued.predict_proba(xv), n_classes)
        curve.append({'epoch': epoch, 'optimizer_steps': steps,
                      'validation_log_loss': scores['log_loss'],
                      'validation_auroc': scores['auroc'],
                      'validation_coverage_complete': scores['coverage_complete']})
        key = scores['log_loss'], epoch
        if key < best_key:
            best_key, best_steps = key, steps
            best_state = {k: v.detach().clone() for k, v in trained.state_dict().items()}

    evaluate(0)
    if curve[0]['validation_log_loss'] != initial_score['log_loss']:
        raise AssertionError("Catch-up epoch0 differs from saved audit")
    for epoch in range(1, int(epochs)+1):
        order = rng.permutation(len(yf))
        schedule_hash.update(order.tobytes())
        for start in range(0, len(order), CATCHUP_CONFIG['batch_size']):
            rows = order[start:start+CATCHUP_CONFIG['batch_size']]
            trained.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(trained(x_tensor[rows]), y_tensor[rows])
            if not torch.isfinite(loss):
                error = FloatingPointError('Nonfinite catch-up fitting loss')
                error.fit_accounting = {'optimizer_steps': steps,
                                        'training_row_exposures': int(exposures.sum())}
                raise error
            loss.backward()
            optimizer.step()
            steps += 1
            exposures[rows] += 1
        if epoch % CATCHUP_CONFIG['validation_interval'] == 0 or epoch == epochs:
            evaluate(epoch)
    final_state_hash = _state_hash(trained)
    final_optimizer_steps = [int(state['step'].item()) for state in optimizer.state.values()]
    if any(value != steps for value in final_optimizer_steps):
        raise AssertionError("Adam state step counters do not match actual catch-up updates")
    trained.load_state_dict(best_state)
    trained.eval().requires_grad_(False)
    selected_scores = metrics(yv, continued.predict_proba(xv), n_classes)
    if selected_scores['log_loss'] != best_key[0]:
        raise AssertionError("Selected catch-up checkpoint failed validation replay")
    continued.metadata = {
        **copy.deepcopy(common), 'audit_kind': 'catchup', 'candidate_id': 'catchup',
        'fit_rows': len(yf), 'fit_support': available_fit_support,
        'fit_coverage_complete': bool((np.bincount(yf, minlength=n_classes) > 0).all()),
        'fit_hashes': fit_hashes, 'seed': int(seed), 'schedule_seed': schedule_seed,
        'schedule_hash': schedule_hash.hexdigest(), 'restarts': 1,
        'optimizer': {'name': 'Adam', 'lr': CATCHUP_CONFIG['lr'], 'betas': CATCHUP_CONFIG['betas'],
                      'eps': CATCHUP_CONFIG['eps'], 'weight_decay': CATCHUP_CONFIG['weight_decay'],
                      'restored_state': False, 'initial_state_entries': initial_optimizer_states,
                      'final_state_entries': len(optimizer.state), 'final_state_steps': final_optimizer_steps},
        'initial_state_hash': initial_hash, 'final_state_hash': final_state_hash,
        'selected_state_hash': _state_hash(trained), 'optimizer_steps': steps,
        'training_row_exposures': int(exposures.sum()),
        'row_exposure_min': int(exposures.min()), 'row_exposure_max': int(exposures.max()),
        'selected_epoch': best_key[1], 'selected_optimizer_steps': best_steps,
        'validation_curve': curve, 'validation_scores': selected_scores,
        'selection': 'minimum attacker-validation log loss, then earliest epoch; no evaluation data',
    }
    continued.metadata['parameters']['epochs'] = int(epochs)
    if (_state_hash(saved_model) != initial_hash or _state_hash(saved_copy) != initial_hash
            or [module.training for module in saved_model.modules()] != original_modes
            or [value.requires_grad for value in saved_model.parameters()] != original_grad_flags):
        raise AssertionError("Incoming adversary state/modes or frozen saved audit changed")
    metadata = {
        'config': {**copy.deepcopy(CATCHUP_CONFIG), 'epochs': int(epochs)},
        'source_state_hash': initial_hash, 'source_unchanged': True,
        'initial_fidelity_exact': True, 'identity_coordinates_preserved': True,
        'inherited_exposure': inherited, 'catchup_optimizer_reset': True,
        'candidates': {'saved': candidate.metadata, 'catchup': continued.metadata},
        'fit_runtime_seconds': time.perf_counter()-started,
    }
    return {'saved': candidate, 'catchup': continued, 'metadata': metadata}
