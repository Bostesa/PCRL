"""AMENDMENT 1 — cross-architecture tolerance for the historical routed-ancestor score check.

`acs_spectral_audits.build_audits` re-scores every routed H-ancestor candidate and asserts
`actual == c.metadata['validation_scores']` with EXACT dict equality. The check's purpose is
route correctness: a routed ancestor reads the same H columns of the new wire that it was fitted
on, so it must reproduce its recorded scores. Exact float equality additionally requires the same
floating-point execution, which does not hold between the arm64 machine that produced the stored
records and the x86-64 cloud host.

Measured on the cloud host (196 routed candidates, seed 0, `diag_scores.py`, recorded in
RUN_STATUS.md): max |difference| by metric — log_loss 3.9e-9, per-class auroc 1.5e-5,
macro_auroc 7.6e-7; every count, support, accuracy, precision, recall, f1 and prevalence
identical; no None-vs-value flips. Only tie-order-sensitive ranking metrics move.

This module keeps the check and its purpose, and replaces exact equality by:
  * `log_loss`                     : |difference| <= 1e-7   (25x the measured deviation, and four
                                     orders of magnitude below the .001-nat decision scale)
  * ranking metrics (auroc, macro_auroc, observed_macro_auroc, balanced_accuracy,
    observed_balanced_accuracy)    : |difference| <= 1e-3
  * everything else (counts, supports, accuracy, precision, recall, f1, prevalence, schemas,
    booleans, None)                : EXACT equality, as before.
A wrong route is still caught: mis-routed columns move log loss by whole nats (measured 2.5 nats
in the mis-routed diagnostic), thousands of times above the tolerance.

The historical module is NOT edited. `audit.metrics` is wrapped for the duration of this study's
audits so the comparison happens inside a dict subclass, and every deviation is recorded.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

LOG_LOSS_TOL = 1e-7
RANKING_TOL = 1e-3
RANKING_KEYS = {'auroc', 'macro_auroc', 'observed_macro_auroc', 'balanced_accuracy',
                'observed_balanced_accuracy'}
_LOCK = threading.Lock()
_STATE = {'installed': False, 'max': {}, 'comparisons': 0, 'rejected': 0, 'path': None}


def _tol_for(key: str) -> float:
    if key == 'log_loss':
        return LOG_LOSS_TOL
    return RANKING_TOL if key in RANKING_KEYS else 0.0


def _compare(a, b, key='', deviations=None):
    """True if a and b agree within the declared per-key tolerance."""
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or set(a) != set(b):
            return False
        return all(_compare(a[k], b[k], k, deviations) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))) or len(a) != len(b):
            return False
        return all(_compare(x, y, key, deviations) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a is b or a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        d = abs(float(a) - float(b))
        if deviations is not None and d > 0:
            deviations[key] = max(deviations.get(key, 0.0), d)
        return d <= _tol_for(key)
    return a == b


class TolerantScores(dict):
    """A metrics dict whose equality is the amendment's tolerant comparison."""

    def __eq__(self, other):
        deviations = {}
        ok = _compare(dict(self), other, '', deviations)
        with _LOCK:
            _STATE['comparisons'] += 1
            if not ok:
                _STATE['rejected'] += 1
            for k, v in deviations.items():
                _STATE['max'][k] = max(_STATE['max'].get(k, 0.0), v)
        return ok

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = None


def install(record_path=None):
    """Wrap `acs_spectral_audits.metrics` so its results compare tolerantly. Idempotent."""
    from experiments import acs_spectral_audits as audit
    _STATE['path'] = str(record_path) if record_path else _STATE['path']
    if _STATE['installed']:
        return
    original = audit.metrics

    def wrapped(*args, **kwargs):
        return TolerantScores(original(*args, **kwargs))

    wrapped.__wrapped__ = original
    audit.metrics = wrapped
    _STATE['installed'] = True


def record() -> dict:
    with _LOCK:
        rec = {'amendment': 1, 'what': 'tolerant routed-ancestor score comparison (see module docstring)',
               'log_loss_tolerance': LOG_LOSS_TOL, 'ranking_tolerance': RANKING_TOL,
               'ranking_keys': sorted(RANKING_KEYS), 'exact_for': 'everything else',
               'comparisons': _STATE['comparisons'], 'rejected': _STATE['rejected'],
               'max_deviation_by_metric': dict(sorted(_STATE['max'].items(), key=lambda x: -x[1]))}
    if _STATE['path']:
        p = Path(_STATE['path'])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=1))
    return rec
