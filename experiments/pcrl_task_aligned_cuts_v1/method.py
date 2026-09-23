"""Exact fixed-predictor, expected-token coefficients for finite T0 channels.

All weights are normalized over original people.  Returned matrices contain
joint code mass already; the channel objective is ``sum(coeff * Q)``.
This module has no data loader and never fits a predictor.
"""
from __future__ import annotations

import numpy as np

PROBABILITY_FLOOR = 1e-9


def validate_channel(q: np.ndarray, *, n_states: int | None = None,
                     n_tokens: int | None = None, tolerance: float = 1e-8) -> np.ndarray:
    """Return a checked float64 channel; reject zero or malformed rows."""
    q = np.asarray(q, dtype=np.float64)
    if (q.ndim != 2 or min(q.shape) < 1 or
            (n_states is not None and q.shape[0] != n_states) or
            (n_tokens is not None and q.shape[1] != n_tokens) or
            not np.isfinite(q).all() or np.min(q) < 0 or np.max(q) > 1 or
            np.max(np.abs(q.sum(axis=1) - 1)) > tolerance):
        raise ValueError("Q must have finite, nonnegative, normalized token rows")
    return q


def _validated_inputs(t, y, predictions, n_states: int, weights):
    t, y = np.asarray(t), np.asarray(y)
    p = np.asarray(predictions, dtype=np.float64)
    if (not isinstance(n_states, (int, np.integer)) or isinstance(n_states, (bool, np.bool_))
            or n_states < 1):
        raise ValueError("n_states must be a positive integer")
    if (t.ndim != 1 or y.shape != t.shape or len(t) == 0 or
            t.dtype.kind not in "biuf" or y.dtype.kind not in "biuf" or
            not np.isfinite(t).all() or not np.isfinite(y).all() or
            np.any(t != np.floor(t)) or np.any(y != np.floor(y)) or
            np.any(t < 0) or np.any(t >= n_states)):
        raise ValueError("codes and labels must be aligned finite integer indices")
    if (p.ndim != 3 or p.shape[0] != len(t) or p.shape[1] < 1 or p.shape[2] < 2 or
            not np.isfinite(p).all() or np.min(p) < 0 or np.max(p) > 1 or
            np.max(np.abs(p.sum(axis=2) - 1)) > 1e-8 or
            np.any(y < 0) or np.any(y >= p.shape[2])):
        raise ValueError("predictions must be aligned full-schema class probabilities")
    if weights is None:
        w = np.ones(len(t), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
    if (w.shape != t.shape or not np.isfinite(w).all() or np.any(w < 0)
            or not np.isfinite(w.sum()) or w.sum() <= 0):
        raise ValueError("weights must be finite, nonnegative, and positive in total")
    return t.astype(np.int64), y.astype(np.int64), p, w


def observed_token_losses(labels, predictions, *, probability_floor=PROBABILITY_FLOOR):
    """One per-person loss for every token, before channel expectation."""
    p = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(labels)
    if (p.ndim != 3 or p.shape[2] < 2 or y.shape != (len(p),) or
            y.dtype.kind not in "biuf" or not np.isfinite(y).all() or
            np.any(y != np.floor(y)) or np.any(y < 0) or np.any(y >= p.shape[2]) or
            not np.isfinite(p).all() or np.min(p) < 0 or np.max(p) > 1 or
            np.max(np.abs(p.sum(axis=2) - 1)) > 1e-8):
        raise ValueError("full-schema predictions and integer labels required")
    if not (np.isfinite(probability_floor) and 0 < probability_floor < 1):
        raise ValueError("probability_floor must lie in (0,1)")
    observed = p[np.arange(len(y))[:, None], np.arange(p.shape[1])[None, :], y.astype(int)[:, None]]
    return -np.log(np.maximum(observed, probability_floor))


def coefficient_pair(t, y, predictions, n_states: int, weights,
                     *, probability_floor=PROBABILITY_FLOOR):
    """Return separately normalized U and PWGTP coefficient matrices.

    The caller must supply an explicit fixed coefficient-estimation population.
    This routine never masks a missing label or drops an unsupported class.
    ``predictions`` is [person, token, full-schema class].
    """
    t, y, p, w = _validated_inputs(t, y, predictions, n_states, weights)
    losses = observed_token_losses(y, p, probability_floor=probability_floor)
    out_u = np.zeros((n_states, p.shape[1]), dtype=np.float64)
    out_w = np.zeros_like(out_u)
    np.add.at(out_u, t, losses / len(t))
    np.add.at(out_w, t, losses * (w / w.sum())[:, None])
    return {"U": out_u, "W": out_w}


def score_expected(q, t, y, predictions, weights,
                   *, probability_floor=PROBABILITY_FLOOR):
    """Direct person-level replay of fixed predictor loss under one token."""
    q = validate_channel(q)
    t, y, p, w = _validated_inputs(t, y, predictions, len(q), weights)
    if p.shape[1] != q.shape[1]:
        raise ValueError("prediction token alphabet differs from channel")
    losses = observed_token_losses(y, p, probability_floor=probability_floor)
    person = np.sum(q[t] * losses, axis=1)
    return {"U": float(np.mean(person)), "W": float(np.dot(w / w.sum(), person))}


def binary_predictions(probability_of_one):
    """Promote the archived fixed binary action decoder to full class order."""
    p = np.asarray(probability_of_one, dtype=np.float64)
    if p.ndim != 2 or min(p.shape) < 1 or not np.isfinite(p).all() or np.any(p <= 0) or np.any(p >= 1):
        raise ValueError("binary action predictions must lie strictly in (0,1)")
    return np.stack((1 - p, p), axis=2)


def rowwise_unconstrained(cost):
    """Exact deterministic minimum of a fixed linear cost over row simplices."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or min(cost.shape) < 1 or not np.isfinite(cost).all():
        raise ValueError("cost must be a finite state-by-token matrix")
    rows = np.argmin(cost, axis=1)
    q = np.zeros_like(cost)
    q[np.arange(len(rows)), rows] = 1
    return {"Q": q, "objective": float(np.min(cost, axis=1).sum()), "actions": rows}


def common_coverage_law(codes, historical_q, d17):
    """Equal mixture of archived Q, D17 and uniform 17-token coverage.

    This is a *training law* for one common U1 decoder, not a wire release.
    The exact same mixture is used on its training and validation households.
    """
    historical_q = validate_channel(historical_q, n_tokens=17)
    d17 = validate_channel(d17, n_states=len(historical_q), n_tokens=17)
    codes = np.asarray(codes)
    if (codes.ndim != 1 or codes.dtype.kind not in "biuf" or
            not np.isfinite(codes).all() or np.any(codes != np.floor(codes)) or
            np.any(codes < 0) or np.any(codes >= len(historical_q))):
        raise ValueError("frozen T0 codes must be valid integer indices")
    t = codes.astype(np.int64)
    law = (historical_q[t] + d17[t] + 1/17) / 3
    return validate_channel(law, n_states=len(t), n_tokens=17)


def fit_common_decoder(h_fit, t_fit, y_fit, weights_fit,
                       h_validation, t_validation, y_validation, weights_validation,
                       historical_q, d17, *, seed: int, out_dir):
    """Fit the one shared U1 task decoder on training/validation resources.

    All inputs are explicit 2018 development arrays.  A caller must filter and
    record missing-label exclusions before entry; no outer assessment data are
    accepted here.  The inherited predictor slate conserves original-person
    weights during exact token expansion and selects only on validation.
    ``out_dir`` is private and must be empty, matching the inherited API.
    """
    from experiments.pcrl_task_directed_release_v1.audits import fit_slate

    h_fit, h_validation = np.asarray(h_fit), np.asarray(h_validation)
    if (h_fit.ndim != 2 or h_fit.shape[1] != 4 or
            h_validation.ndim != 2 or h_validation.shape[1] != 4 or
            not np.isfinite(h_fit).all() or not np.isfinite(h_validation).all()):
        raise ValueError("U1 decoder receives exactly four H_A coordinates")
    fit_law = common_coverage_law(t_fit, historical_q, d17)
    validation_law = common_coverage_law(t_validation, historical_q, d17)
    selected = fit_slate(h_fit, fit_law, y_fit, weights_fit,
                         h_validation, validation_law, y_validation,
                         weights_validation, 2, seed, out_dir, slate="standard")
    model = selected["candidates"][selected["selection"]]
    return {"decoder": model, "selection": selected["selection"],
            "metadata": selected["metadata"],
            "training_law": "(historical_Q + D17 + uniform_17)/3",
            "frozen_before_channel_fit": True}


def task_cost_from_decoder(decoder, h_coefficient, t_coefficient, y_coefficient,
                           weights_coefficient, n_states=32):
    """Cost matrices from a decoder frozen before channel optimization."""
    h = np.asarray(h_coefficient)
    if h.ndim != 2 or h.shape[1] != 4 or not np.isfinite(h).all():
        raise ValueError("task decoder sees only four H_A coordinates")
    predictions = decoder.predict_token_proba(h, 17)
    if predictions.shape != (len(h), 17, 2):
        raise ValueError("common decoder must return binary full-token predictions")
    return coefficient_pair(t_coefficient, y_coefficient, predictions,
                            n_states, weights_coefficient)
