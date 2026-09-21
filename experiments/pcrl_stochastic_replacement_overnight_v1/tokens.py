"""Scoring a small finite token release, and fitting attackers against it.

Two invariants this module exists to enforce (`METHOD.md` §5):

1. **Average losses, not probabilities.** For a frozen attacker `f`, person `i` with code `t_i`, the
   expected per-person loss is `sum_z Q[t_i,z] * loss(f(H_i,z), y_i)`. `log` of an averaged
   prediction is a different estimand and is never used.
2. **`K` expanded rows for one person are one person.** Exact weighted expansion gives person `i`
   `K` rows with weights `Q[t_i,z]` summing to that person's original weight (PWGTP multiplies it),
   and carries the household id on every expanded row. Regularization and epoch normalization must
   not treat them as `K` independent people.

A sampled person-token pair is the released view. Enumerating tokens for integration does **not**
grant an attacker all tokens.
"""
from __future__ import annotations

import numpy as np

FLOOR = 1e-12


def expand_exact(h, t, q, *, y=None, serial=None, weight=None):
    """Exact weighted token expansion.

    Returns features with the one-hot token appended, plus per-row weights that **sum to the
    person's original weight**, plus the carried household id and label.
    """
    h = np.asarray(h, np.float64)
    t = np.asarray(t, np.int64)
    q = np.asarray(q, np.float64)
    n, k = len(t), q.shape[1]
    w_person = np.ones(n) if weight is None else np.asarray(weight, np.float64)

    rows = np.repeat(np.arange(n), k)
    toks = np.tile(np.arange(k), n)
    onehot = np.zeros((n * k, k))
    onehot[np.arange(n * k), toks] = 1.0
    x = np.column_stack([h[rows], onehot])
    w = q[t[rows], toks] * w_person[rows]
    out = {'x': x, 'weight': w, 'person': rows, 'token': toks,
           'n_people': int(n), 'n_rows': int(n * k)}
    if y is not None:
        out['y'] = np.asarray(y)[rows]
    if serial is not None:
        out['serial'] = np.asarray(serial)[rows]
    return out


def check_expansion_weights(exp: dict, weight=None, tol: float = 1e-9) -> dict:
    """Per-person expanded weight must equal that person's original weight."""
    n = exp['n_people']
    w_person = np.ones(n) if weight is None else np.asarray(weight, np.float64)
    got = np.bincount(exp['person'], weights=exp['weight'], minlength=n)
    worst = float(np.max(np.abs(got - w_person))) if n else 0.0
    return {'max_abs_weight_error': worst, 'ok': bool(worst <= tol),
            'total_expanded_weight': float(exp['weight'].sum()),
            'total_person_weight': float(w_person.sum()),
            'note': 'K expanded rows are one person; their weights sum to the person weight'}


def expected_loss(predict_proba, h, t, q, y, *, weight=None) -> dict:
    """Expected per-person log loss under the token mechanism.

    `predict_proba(x)` takes `[h, onehot(token)]`. Losses are averaged with `Q[t,z]`; predictions
    are never averaged.
    """
    h = np.asarray(h, np.float64)
    t = np.asarray(t, np.int64)
    q = np.asarray(q, np.float64)
    y = np.asarray(y, np.int64)
    n, k = len(t), q.shape[1]
    per_person = np.zeros(n)
    for z in range(k):
        onehot = np.zeros((n, k))
        onehot[:, z] = 1.0
        p = np.asarray(predict_proba(np.column_stack([h, onehot])), np.float64)
        loss_z = -np.log(np.maximum(p[np.arange(n), y], FLOOR))
        per_person += q[t, z] * loss_z                      # average the LOSS
    w = np.ones(n) if weight is None else np.asarray(weight, np.float64)
    return {'per_person_expected_loss': per_person,
            'mean': float(np.sum(per_person * w) / np.sum(w)),
            'estimand': 'expectation over Q of the loss, not loss of the expectation'}


def loss_of_expected_prediction(predict_proba, h, t, q, y) -> np.ndarray:
    """The WRONG estimand, provided only so a fixture can show the two differ."""
    h = np.asarray(h, np.float64)
    t = np.asarray(t, np.int64)
    q = np.asarray(q, np.float64)
    y = np.asarray(y, np.int64)
    n, k = len(t), q.shape[1]
    acc = None
    for z in range(k):
        onehot = np.zeros((n, k))
        onehot[:, z] = 1.0
        p = np.asarray(predict_proba(np.column_stack([h, onehot])), np.float64)
        acc = q[t, z][:, None] * p if acc is None else acc + q[t, z][:, None] * p
    return -np.log(np.maximum(acc[np.arange(n), y], FLOOR))


def sample_tokens(t, q, rng) -> np.ndarray:
    """One token per person, from a PRIVATE deployment stream (never a public seed + person id)."""
    t = np.asarray(t, np.int64)
    q = np.asarray(q, np.float64)
    u = rng.random(len(t))
    cdf = np.cumsum(q, axis=1)[t]
    return (u[:, None] > cdf).sum(1).clip(0, q.shape[1] - 1).astype(np.int64)


def sampled_wire(h, t, q, rng) -> np.ndarray:
    """The actual released array: `[H_A, onehot(sampled token)]`."""
    z = sample_tokens(t, q, rng)
    k = np.asarray(q).shape[1]
    onehot = np.zeros((len(z), k))
    onehot[np.arange(len(z)), z] = 1.0
    return np.column_stack([np.asarray(h, np.float64), onehot])


def is_sampled_token_block(block, k: int) -> bool:
    """Gate: the token block must be a one-hot indicator, never a probability row."""
    a = np.asarray(block, np.float64)
    if a.ndim != 2 or a.shape[1] != k:
        return False
    return bool(np.all((a == 0.0) | (a == 1.0)) and np.all(a.sum(1) == 1.0))
