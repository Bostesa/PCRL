"""Matched 17-token controls for the adaptive-release development study.

The finite-bank routines are the previously validated exact-coefficient
implementations. Coefficients already contain normalized person mass. A cut
means expected sensitive log loss >= its frozen floor. All maps here retain
the same 17-value token wire; no hidden state is sent to an auditor.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from experiments.pcrl_task_aligned_cuts_v1 import controls as finite


N_TOKENS = 17
Bank = finite.Bank


def aggregate_coefficients(private_leaf_ids: np.ndarray,
                           per_person_token_loss: np.ndarray,
                           weights: np.ndarray, *, n_states: int,
                           normalized_weights: bool = False) -> np.ndarray:
    """Group exact per-person token losses into an affine child-state matrix.

    Every original person enters once. The output already includes normalized
    person and state mass, so no later state-mass multiplication is valid.
    The caller keeps private leaf IDs and per-person losses out of public logs.
    """
    leaf = np.asarray(private_leaf_ids)
    losses = np.asarray(per_person_token_loss, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if isinstance(n_states, bool) or not isinstance(n_states, (int, np.integer)) or n_states < 1:
        raise ValueError("positive integer state count required")
    if (leaf.ndim != 1 or leaf.size < 1 or
            not np.issubdtype(leaf.dtype, np.integer) or
            np.any(leaf < 0) or np.any(leaf >= n_states)):
        raise ValueError("invalid private leaf indices")
    if (losses.shape != (len(leaf), N_TOKENS) or
            not np.all(np.isfinite(losses)) or np.any(losses < 0)):
        raise ValueError("expected finite nonnegative per-person token losses")
    if (w.shape != (len(leaf),) or not np.all(np.isfinite(w)) or
            np.any(w < 0) or w.sum() <= 0):
        raise ValueError("weights must have positive finite total")
    if normalized_weights:
        if not np.isclose(w.sum(), 1., rtol=0, atol=1e-10):
            raise ValueError("declared normalized weights do not sum to one")
    else:
        w = w / w.sum()
    out = np.zeros((n_states, N_TOKENS), dtype=np.float64)
    np.add.at(out, leaf, w[:, None] * losses)
    return out


def _channel(value: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] < 1 or q.shape[1] != N_TOKENS:
        raise ValueError("channel must have nonempty state rows and 17 tokens")
    if not np.all(np.isfinite(q)) or np.min(q) < 0 or np.max(q) > 1:
        raise ValueError("nonfinite or out-of-range channel probability")
    if np.max(np.abs(q.sum(axis=1) - 1)) > 1e-10:
        raise ValueError("channel rows must sum to one")
    if deterministic and not np.all((q == 0) | (q == 1)):
        raise ValueError("expected a deterministic channel")
    return q


def _cost(value: np.ndarray) -> np.ndarray:
    c = np.asarray(value, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] < 1 or c.shape[1] != N_TOKENS:
        raise ValueError("cost must have nonempty state rows and 17 tokens")
    if not np.all(np.isfinite(c)):
        raise ValueError("cost must be finite")
    return c


def lift_parent_channel(parent: np.ndarray, parent_of_leaf: np.ndarray) -> np.ndarray:
    """Exact feasible incumbent on a refined partition, without re-encoding."""
    q = _channel(parent)
    indices = np.asarray(parent_of_leaf)
    if (indices.ndim != 1 or indices.size < 1 or
            not np.issubdtype(indices.dtype, np.integer) or
            np.any(indices < 0) or np.any(indices >= len(q))):
        raise ValueError("invalid parent index for child leaves")
    return q[indices].copy()


def make_bank(cost: np.ndarray, cuts: Sequence[Mapping[str, Any]]) -> Bank:
    return finite.make_bank(_cost(cost), cuts)


def replay(q: np.ndarray, bank: Bank, *, deterministic: bool = False,
           tolerance: float = 1e-7) -> dict[str, Any]:
    _channel(q)
    return finite.replay(q, bank, deterministic=deterministic, tolerance=tolerance)


def rowwise_minimizer(cost: np.ndarray) -> np.ndarray:
    return finite.rowwise_minimizer(_cost(cost))


def constant_maps(n_states: int) -> list[np.ndarray]:
    return finite.constant_maps(n_states, N_TOKENS)


def constant_replacement(base: np.ndarray, *, replace_probability: float,
                         token: int) -> np.ndarray:
    """Replace a draw with a visible constant token on the original wire."""
    q = _channel(base)
    p = float(replace_probability)
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("replacement probability outside [0,1]")
    if not isinstance(token, (int, np.integer)) or not 0 <= token < N_TOKENS:
        raise ValueError("constant token outside 17-token alphabet")
    out = (1 - p) * q
    out[:, token] += p
    return _channel(out)


def randomized_response(base: np.ndarray, *, publish_probability: float) -> np.ndarray:
    """Publish base token or an independent uniform 17-token replacement."""
    return _channel(finite.randomized_response(_channel(base), publish_probability))


def solve_deterministic_p1(cost: np.ndarray, cuts: Sequence[Mapping[str, Any]], *,
                           time_limit_seconds: float, relative_gap: float = 1e-6,
                           feasibility_tolerance: float = 1e-7) -> dict[str, Any]:
    return finite.solve_deterministic_p1(
        _cost(cost), cuts, time_limit_seconds=time_limit_seconds,
        relative_gap=relative_gap, feasibility_tolerance=feasibility_tolerance)


def deterministic_coordinate_search(cost: np.ndarray,
                                    cuts: Sequence[Mapping[str, Any]], *,
                                    seconds: float, seed: int,
                                    starts: Sequence[np.ndarray] = ()) -> dict[str, Any]:
    return finite.deterministic_coordinate_search(
        _cost(cost), cuts, seconds=seconds, seed=seed, starts=starts)


def optimize_gradient_bank(cost: np.ndarray, cuts: Sequence[Mapping[str, Any]], *,
                           steps: int, learning_rate: float, penalty: float,
                           seed: int, initial: np.ndarray | None = None,
                           dual_learning_rate: float = 0.0) -> dict[str, Any]:
    return finite.optimize_gradient_bank(
        _cost(cost), cuts, steps=steps, learning_rate=learning_rate,
        penalty=penalty, seed=seed,
        initial=None if initial is None else _channel(initial),
        dual_learning_rate=dual_learning_rate)


def gradient_control_grid(cost: np.ndarray, cuts: Sequence[Mapping[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
    return finite.gradient_control_grid(_cost(cost), cuts, **kwargs)
