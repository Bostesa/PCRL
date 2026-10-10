"""Independent public-wire checks using restored public decoder tables.

The caller hands in the *entire transmitted wire*, never a projection of it.
The decoder tables are independently reconstructed from a sealed checkpoint,
not calculated by calling the candidate deployment implementation.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


_FORBIDDEN_NAMES = ("latent", "teacher", "soft", "fine", "source_token",
                    "action", "gate", "random", "coin", "expectation")


def _wire(wire: Mapping, code_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping]:
    if not isinstance(wire, Mapping) or set(wire) != {code_name, "probabilities", "decision", "metadata"}:
        raise ValueError("wire has missing or extra transmitted fields")
    for name in wire:
        if any(fragment in name.lower() for fragment in _FORBIDDEN_NAMES):
            raise ValueError("forbidden transmitted field")
    metadata = wire["metadata"]
    if not isinstance(metadata, Mapping) or not metadata:
        raise ValueError("fixed public model/version metadata required")
    for name, value in metadata.items():
        if not isinstance(name, str) or any(fragment in name.lower() for fragment in _FORBIDDEN_NAMES):
            raise ValueError("forbidden metadata field")
        if not isinstance(value, (str, int, float, bool)) or (isinstance(value, float) and not np.isfinite(value)):
            raise ValueError("metadata must be globally fixed scalar values")
    code = np.asarray(wire[code_name])
    prob = np.asarray(wire["probabilities"], dtype=np.float64)
    decision = np.asarray(wire["decision"])
    if (prob.ndim != 2 or not len(prob) or prob.shape[1] < 2
            or not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1))
            or np.max(np.abs(prob.sum(axis=1) - 1)) > 1e-8):
        raise ValueError("public probabilities invalid")
    if (decision.shape != (len(prob),) or not np.issubdtype(decision.dtype, np.integer)
            or not np.array_equal(decision, np.argmax(prob, axis=1))):
        raise ValueError("decision does not follow public probabilities")
    return code, prob, decision, metadata


def verify_d_hard_code_wire(
    wire: Mapping,
    independently_restored_code_head_table: np.ndarray,
    *,
    bit_order: str,
    tolerance: float = 1e-8,
) -> dict:
    """The public ten-bit code alone reconstructs every released task output."""
    code, prob, _, metadata = _wire(wire, "code")
    if (code.shape != (len(prob), 10) or not np.issubdtype(code.dtype, np.integer)
            or not np.isin(code, (0, 1)).all() or bit_order not in ("little", "big")):
        raise ValueError("noncanonical ten-bit hard code")
    powers = 1 << np.arange(10, dtype=np.int64)
    if bit_order == "big":
        powers = powers[::-1]
    ids = code.astype(np.int64) @ powers
    table = np.asarray(independently_restored_code_head_table, dtype=np.float64)
    if (table.shape != (1024, prob.shape[1]) or not np.isfinite(table).all()
            or np.any((table < 0) | (table > 1))
            or np.max(np.abs(table.sum(axis=1) - 1)) > 1e-8):
        raise ValueError("independently restored code-only table invalid")
    discrepancy = float(np.max(np.abs(prob - table[ids])))
    if discrepancy > tolerance:
        raise ValueError("released probability depends on more than public code/table")
    counts = np.bincount(ids, minlength=1024)
    occupied = counts[counts > 0]
    mass = occupied / len(ids)
    return {"rows": len(ids), "occupied_states": len(occupied),
            "maximum_states": 1024, "entropy_bits": float(-np.dot(mass, np.log2(mass))),
            "max_output_difference": discrepancy, "metadata": dict(metadata)}


def verify_a_sampled_wire(
    wire: Mapping,
    globally_canonical_public_bank: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> dict:
    """A global output ID names one probability/decision, regardless of input."""
    code, prob, _, metadata = _wire(wire, "code_id")
    bank = np.asarray(globally_canonical_public_bank, dtype=np.float64)
    if (bank.ndim != 2 or not len(bank) or bank.shape[1] != prob.shape[1]
            or not np.isfinite(bank).all() or np.any((bank < 0) | (bank > 1))
            or np.max(np.abs(bank.sum(axis=1) - 1)) > 1e-8
            or not np.issubdtype(code.dtype, np.integer) or code.shape != (len(prob),)
            or np.any(code < 0) or np.any(code >= len(bank))):
        raise ValueError("invalid globally coded sampled release")
    # Exact equal public vectors must have been merged globally, before lock.
    if np.unique(bank, axis=0).shape[0] != len(bank):
        raise ValueError("duplicate public output identities in bank")
    discrepancy = float(np.max(np.abs(prob - bank[code.astype(np.int64)])))
    if discrepancy > tolerance:
        raise ValueError("sampled wire does not match its globally fixed bank ID")
    return {"rows": len(code), "public_bank_size": len(bank),
            "occupied_outputs": int(np.unique(code).size),
            "max_output_difference": discrepancy, "metadata": dict(metadata)}
