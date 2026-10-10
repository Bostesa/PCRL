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
