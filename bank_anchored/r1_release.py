"""Exact global bank-ID to recipient wire conversion.

The private provider chooses global IDs.  Only ten public bits and their fixed
table-derived task output leave this module for either recipient.  No fine-cell
index, score, action slot, coin, or branch flag is part of the wire.
"""

from __future__ import annotations

import numpy as np

from code_witness.bank import compile_release, decoder_capacity


def _table(value: np.ndarray, classes: int) -> np.ndarray:
    table = np.asarray(value)
    if table.dtype != np.float64 or table.ndim != 2 or table.shape[1] != classes:
        raise ValueError("frozen bank table shape or dtype differs")
    decoder_capacity(table, table.argmax(axis=1).astype(np.int64), 10)
    if len(np.unique(table, axis=0)) != len(table):
        raise ValueError("distinct public IDs must have distinct output vectors")
    return table


def emit_public_pair(code0: np.ndarray, code1: np.ndarray,
                     tables: tuple[np.ndarray, np.ndarray]) -> tuple[dict, dict]:
    """Decode selected canonical global IDs through unchanged W_TSTAR tables."""
    if not isinstance(tables, tuple) or len(tables) != 2:
        raise ValueError("both frozen recipient tables are required")
    ids = (np.asarray(code0), np.asarray(code1))
    if any(value.dtype != np.int64 or value.ndim != 1 for value in ids) or \
            len(ids[0]) != len(ids[1]):
        raise ValueError("aligned int64 global public IDs are required")
    decoded = []
    for value, table, classes in zip(ids, tables, (2, 6)):
        fixed = _table(table, classes)
        one = compile_release(value, fixed, 10)
        decoded.append({"code": one["code"], "prob": one["probabilities"],
                        "decision": one["decision"]})
    return decoded[0], decoded[1]


def release_pair_from_scores(policy, score0: np.ndarray, score1: np.ndarray,
                             cell0: np.ndarray, cell1: np.ndarray,
                             tables: tuple[np.ndarray, np.ndarray],
                             uniforms: np.ndarray | None) -> tuple[dict, dict]:
    """Run a private policy and emit only its two canonical public bank wires.

    `uniforms` is `(n,)` for a joint sampler, `(n,2)` for independent
    recipient coins, and `None` only for a deterministic policy. The caller
    owns the frozen randomness plan; no coin is returned or stored here.
    """
    scores = (np.asarray(score0), np.asarray(score1))
    cells = (np.asarray(cell0), np.asarray(cell1))
    if not hasattr(policy, "sample_public_pair"):
        raise TypeError("selected policy needs a public-pair sampler")
    n = len(scores[0])
    for score, cell, classes in zip(scores, cells, (2, 6)):
        if (score.dtype != np.float64 or score.shape != (n, classes) or
                cell.dtype != np.int64 or cell.shape != (n,) or
                not np.isfinite(score).all() or np.any(score < 0) or
                not np.allclose(score.sum(axis=1), 1, rtol=0, atol=1e-8)):
            raise ValueError("private scores or fine cells differ from frozen route")
    if uniforms is None:
        draws = [None] * n
    else:
        u = np.asarray(uniforms)
        if (u.dtype != np.float64 or u.shape not in ((n,), (n, 2)) or
                not np.isfinite(u).all() or np.any((u < 0) | (u >= 1))):
            raise ValueError("frozen sampling inputs must be one or two uniforms per person")
        draws = [float(v) if u.ndim == 1 else (float(v[0]), float(v[1]))
                 for v in u]
    public = np.empty((n, 2), dtype=np.int64)
    fixed = (_table(tables[0], 2), _table(tables[1], 6))
    for row, draw in enumerate(draws):
        pair = policy.sample_public_pair(scores[0][row], scores[1][row],
                                         int(cells[0][row]), int(cells[1][row]),
                                         draw)
        if (not isinstance(pair, tuple) or len(pair) != 2 or
                any(type(value) not in (int, np.int64) for value in pair)):
            raise ValueError("sampler returned noncanonical global public IDs")
        public[row] = pair
    if (np.any(public < 0) or
            any(np.any(public[:, i] >= len(fixed[i])) for i in (0, 1)) or
            any(not np.array_equal(fixed[i][public[:, i]].argmax(axis=1),
                                   scores[i].argmax(axis=1)) for i in (0, 1))):
        raise ValueError("sampled public ID leaves bank or changes predicted class")
    return emit_public_pair(public[:, 0], public[:, 1], fixed)


def verify_identity_pair(wires: tuple[dict, dict],
                         original_ids: tuple[np.ndarray, np.ndarray],
                         tables: tuple[np.ndarray, np.ndarray]) -> bool:
    """Executable exact B0 checkpoint, including serialization-visible bits."""
    expected = emit_public_pair(*original_ids, tables)
    if not isinstance(wires, tuple) or len(wires) != 2:
        raise ValueError("identity checkpoint needs both recipient wires")
    for actual, target in zip(wires, expected):
        if (not isinstance(actual, dict) or set(actual) != set(target) or
                any(not isinstance(actual[name], np.ndarray) or
                    not np.array_equal(actual[name], target[name]) or
                    actual[name].dtype != target[name].dtype
                    for name in target)):
            raise ValueError("identity release differs from frozen B0 public wire")
    return True
