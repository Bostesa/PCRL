"""Tier 0 restore smoke test and numerical portability, run on the cloud host from RESTORED bytes.

Frozen tolerances (declared before any new outcome):
  * mapper execution parity: max |recomputed - stored| <= 1e-5 for A0 and J on every pool;
  * probe reproduction: a restored stored probe re-predicts its stored validation
    probabilities within 1e-6 (max abs) and its stored validation log loss within 1e-6.
Exact serialized identity (sha256 of restored bytes vs the archive manifest) is checked by the
archive verifier; this module checks floating-point EXECUTION parity, a different property.
"""
from __future__ import annotations

import numpy as np

from experiments import acs_spectral_audits as audit
from experiments.run_acs_residual_spectral import load_labels, wires
from experiments.pcrl_nonlinear_rank_v1.inputs import HIST_ROOT
from experiments.pcrl_direct_adversarial_v1 import inputs as dax

from . import extension as ext

PROBE_TOL = 1e-6


def probe_check(seed: int) -> dict:
    reg = dax.Registry.new()
    frame, pools, labels, weights = load_labels(HIST_ROOT, seed)
    h_dir = dax.resolve(f'results/redesign_20260909_acs_fixed_predictions_v1/seed_{seed}/H')
    path = h_dir / 'fitted/utility/B/public_coverage/logistic'
    candidate = audit.load_base(path)
    rel = reg.resolve(dax.fixed_path(seed, 'training/J/releases.npz'))
    w, _ = wires(rel)
    y = labels['downstream_validation']['public_coverage']
    valid = y >= 0
    p = candidate.predict_proba(w['B']['downstream_validation'][valid])
    stored = np.load(dax.resolve(f'results/redesign_20260910_acs_residual_spectral_v1/seed_{seed}/H/predictions.npz'))
    ref = stored['utility/B/public_coverage/None/logistic/validation']
    q = np.clip(p, 1e-12, 1)
    q = q / q.sum(1, keepdims=True)
    ll = float(-np.log(q[np.arange(valid.sum()), y[valid]]).mean())
    meta_ll = candidate.metadata['validation_scores']['log_loss']
    diff = float(np.abs(p - ref).max())
    return {'seed': seed, 'probe': str(path.relative_to(path.parents[7])), 'max_abs_prob_diff': diff,
            'log_loss': ll, 'stored_log_loss': meta_ll, 'log_loss_diff': abs(ll - meta_ll),
            'pass': diff <= PROBE_TOL and abs(ll - meta_ll) <= PROBE_TOL}


def run(seeds=(0, 1, 2)) -> dict:
    out = {'mappers': {}, 'probes': {}}
    for s in seeds:
        _, _, port = ext.portable_state(s, dax.Registry.new())
        out['mappers'][s] = port
        out['probes'][s] = probe_check(s)
    out['pass'] = all(v['within_tolerance'] for p in out['mappers'].values() for v in p.values()) and \
        all(v['pass'] for v in out['probes'].values())
    out['tolerances'] = {'mapper': ext.PORTABLE_TOL, 'probe': PROBE_TOL}
    return out
