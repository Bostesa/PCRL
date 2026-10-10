"""Group-coherent shuffled and planted real-role attack controls.

These functions only transform caller-provided admitted role arrays. They do
not open role files, fit attackers, or create candidate D releases. The planted
11th bit is a diagnostic public wire, never a claimed 10-bit service output.
"""

from __future__ import annotations

import numpy as np

from ...r02.complete_view import Field, WireBatch


def _group_labels(sex: np.ndarray, unit: np.ndarray):
    y = np.asarray(sex)
    group = np.asarray(unit)
    if (y.ndim != 1 or group.shape != y.shape or not len(y)
            or not np.issubdtype(y.dtype, np.integer)
            or not np.isin(y, (0, 1)).all()):
        raise ValueError("aligned binary SEX and group units required")
    names, inverse, size = np.unique(group, return_inverse=True, return_counts=True)
    labels = np.empty(len(names), dtype=np.uint8)
    for j in range(len(names)):
        values = np.unique(y[inverse == j])
        if len(values) != 1:
            raise ValueError("a duplicate/resampling group has conflicting SEX labels")
        labels[j] = values[0]
    return labels, inverse, size


def shuffled_sex(sex: np.ndarray, unit: np.ndarray, seed: int) -> np.ndarray:
    """Permute group labels within equal-size strata; preserve row prevalence."""
    labels, inverse, size = _group_labels(sex, unit)
    rng = np.random.default_rng(seed)
    shuffled = labels.copy()
    for count in np.unique(size):
        index = np.flatnonzero(size == count)
        shuffled[index] = labels[rng.permutation(index)]
    return shuffled[inverse]


def noisy_public_bit(sex: np.ndarray, unit: np.ndarray, seed: int,
                     replacement_fraction: float = 0.2) -> np.ndarray:
    """Group-coherent plant: replace 20% of group bits with fair random bits."""
    if replacement_fraction != 0.2:
        raise ValueError("plant strength is fixed at 20% replacement")
    labels, inverse, _ = _group_labels(sex, unit)
    rng = np.random.default_rng(seed)
    replace = rng.random(len(labels)) < replacement_fraction
    bits = labels.copy()
    bits[replace] = rng.integers(0, 2, size=int(replace.sum()), dtype=np.uint8)
    return bits[inverse]


def xor_public_bits(sex: np.ndarray, unit: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two marginally fair bits whose XOR equals the noisy protected label."""
    labels, inverse, _ = _group_labels(sex, unit)
    rng = np.random.default_rng(seed)
    first = rng.integers(0, 2, size=len(labels), dtype=np.uint8)
    replace = rng.random(len(labels)) < 0.2
    target = labels.copy()
    target[replace] = rng.integers(0, 2, size=int(replace.sum()), dtype=np.uint8)
    return first[inverse], np.bitwise_xor(first, target)[inverse]


def append_planted_public_bit(wire: WireBatch, bit: np.ndarray,
                              name: str = "planted_bit") -> WireBatch:
    """Add a diagnostic bit and distinct category while keeping q/decision fixed.

    Each augmented category maps to its original code-only q. Its two bit
    variants can therefore have identical probabilities while carrying a
    recoverable category/bit clue. This probes exact-category readers.
    """
    value = np.asarray(bit)
    if (value.shape != np.asarray(wire.row_id).shape
            or not np.issubdtype(value.dtype, np.integer)
            or not np.isin(value, (0, 1)).all()
            or wire.code_fields != ("code_id",)
            or name in wire.fields or "control_code_id" in wire.fields):
        raise ValueError("diagnostic plant requires one valid D code and binary bit")
    base = np.asarray(wire.fields["code_id"].values, dtype=np.int64)
    if np.any(base < 0) or np.any(base >= 1024):
        raise ValueError("base D code is outside the 10-bit alphabet")
    fields = dict(wire.fields)
    fields["control_code_id"] = Field("category", base * 2 + value)
    fields[name] = Field("bits", value.astype(np.uint8))
    return WireBatch(wire.row_id, wire.unit, fields,
                     code_fields=("control_code_id",), public_derived=wire.public_derived)
