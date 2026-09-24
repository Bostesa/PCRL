"""Explicit fixed-decoder channel and decoder-update primitives for Branch A.

Caller owns household-disjoint pools, private output directories, immutable
unit IDs, fitted-attack bank refresh, and scientific queue dispatch.  Nothing
in this module reads outer assessment or fits an ACS model at import time.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from experiments.pcrl_task_aligned_cuts_v1 import method
from .reference import solve_calibrated


def coverage_law(codes, current_q, q_ref):
    """Equal current/reference/uniform token law for *training only*."""
    current_q = method.validate_channel(current_q)
    q_ref = method.validate_channel(q_ref, n_states=len(current_q),
                                    n_tokens=current_q.shape[1])
    t = np.asarray(codes)
    if (t.ndim != 1 or t.dtype.kind not in "iu" or not np.isfinite(t).all()
            or np.any(t < 0) or np.any(t >= len(current_q))):
        raise ValueError("codes must be integer state indices")
    n_tokens = current_q.shape[1]
    return method.validate_channel((current_q[t] + q_ref[t] + 1/n_tokens) / 3,
                                   n_states=len(t), n_tokens=n_tokens)


def fit_decoder_round(h_fit, t_fit, y_fit, weights_fit,
                      h_selection, t_selection, y_selection, weights_selection,
                      current_q, q_ref, *, seed: int, out_dir, slate="standard"):
    """Fit one inherited audit-like task slate on exact expected token law.

    The caller must pass distinct eligible training/inner-selection households
    and a fresh private directory. Controls receive this same recipe and budget.
    Previous decoder checkpoints remain separately stored by the caller.
    """
    from experiments.pcrl_task_directed_release_v1.audits import fit_slate

    h_fit = np.asarray(h_fit, dtype=np.float64)
    h_selection = np.asarray(h_selection, dtype=np.float64)
    if (h_fit.ndim != 2 or h_selection.ndim != 2 or h_fit.shape[1] != 4
            or h_selection.shape[1] != 4 or not np.isfinite(h_fit).all()
            or not np.isfinite(h_selection).all()):
        raise ValueError("task decoder receives four unchanged H_A columns")
    fit_law = coverage_law(t_fit, current_q, q_ref)
    selection_law = coverage_law(t_selection, current_q, q_ref)
    return fit_slate(h_fit, fit_law, y_fit, weights_fit,
                     h_selection, selection_law, y_selection, weights_selection,
                     2, seed, out_dir, slate=slate)


def select_frozen_decoder(decoders: Mapping, h_selection, t_selection,
                          y_selection, weights_selection, current_q):
    """Select retained task decoder by exact expected inner-selection loss."""
    if not decoders:
        raise ValueError("at least one frozen decoder required")
    q = method.validate_channel(current_q)
    h = np.asarray(h_selection, dtype=np.float64)
    if h.ndim != 2 or h.shape[1] != 4 or not np.isfinite(h).all():
        raise ValueError("task decoder receives four H_A columns")
    scores = {}
    for cid, decoder in decoders.items():
        if not isinstance(cid, str) or not cid:
            raise ValueError("decoder IDs must be nonempty strings")
        predictions = np.asarray(decoder.predict_token_proba(h, q.shape[1]), dtype=np.float64)
        if predictions.shape != (len(h), q.shape[1], 2):
            raise ValueError("task decoder must return binary full-token probabilities")
        scores[cid] = method.score_expected(q, t_selection, y_selection,
                                             predictions, weights_selection)
    selected = min(scores, key=lambda cid: (
        (scores[cid]["U"] + scores[cid]["W"]) / 2, cid))
    return {"id": selected, "decoder": decoders[selected], "scores": scores,
            "selection_rule": "minimum mean of separately normalized U/W exact token CE, lexical tie"}


def channel_update(task_cost_pair: Mapping, q_ref, bank, delta, *, time_limit=None):
    """One fixed-decoder, fixed-bank affine LP update with D17 witness."""
    if set(task_cost_pair) != {"U", "W"}:
        raise ValueError("task cost pair must have exactly U and W")
    cost_u = np.asarray(task_cost_pair["U"], dtype=np.float64)
    cost_w = np.asarray(task_cost_pair["W"], dtype=np.float64)
    if cost_u.shape != cost_w.shape or not np.isfinite(cost_u).all() or not np.isfinite(cost_w).all():
        raise ValueError("finite matched U/W task costs required")
    result = solve_calibrated(0.5 * (cost_u + cost_w), q_ref, bank, delta,
                              time_limit=time_limit)
    return {**result, "cost_U": cost_u, "cost_W": cost_w,
            "task_weighting": "0.5*U + 0.5*W; each coefficient already includes joint state mass",
            "joint_decoder_channel_global_optimum": False}


def select_and_update(decoders: Mapping, h_selection, t_selection, y_selection,
                      weights_selection, current_q, h_coefficient, t_coefficient,
                      y_coefficient, weights_coefficient, q_ref, bank, delta,
                      *, time_limit=None):
    """Select a retained decoder on inner selection, then rebuild frozen costs.

    This single update is not a joint convex optimum.  A changed bank or
    decoder family invalidates claims of monotone objective decrease.
    """
    chosen = select_frozen_decoder(decoders, h_selection, t_selection,
                                   y_selection, weights_selection, current_q)
    n_states, n_tokens = method.validate_channel(q_ref).shape
    h = np.asarray(h_coefficient, dtype=np.float64)
    if h.ndim != 2 or h.shape[1] != 4 or not np.isfinite(h).all():
        raise ValueError("coefficient H_A must have four unchanged columns")
    predictions = chosen["decoder"].predict_token_proba(h, n_tokens)
    if np.asarray(predictions).shape != (len(h), n_tokens, 2):
        raise ValueError("selected decoder has incompatible token/class schema")
    pair = method.coefficient_pair(t_coefficient, y_coefficient, predictions,
                                   n_states, weights_coefficient)
    updated = channel_update(pair, q_ref, bank, delta, time_limit=time_limit)
    return {"selected_decoder_id": chosen["id"],
            "decoder_selection_scores": chosen["scores"], **updated}
