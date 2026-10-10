"""Label-free complete views of the actual bank-anchored output service.

The private role artifact may hold global integer IDs, but the public wire is
the corresponding ten little-endian bits plus code-derived probability and
decision. This module never reads roles, labels, source scores, or coins.
"""

from __future__ import annotations

import json
import math
import hashlib
import io
from collections.abc import Mapping

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r02.complete_view import (
    Field, WireBatch, pair_batch, prepare_views,
)


BITS = 10
CLASSES = (2, 6)
ROLE_KEYS = frozenset({"row_id", "unit", "code0", "code1",
                       "prob0", "prob1", "decision0", "decision1"})


def bits_from_ids(ids: np.ndarray, table_size: int) -> np.ndarray:
    """Convert global public table row IDs to their actual ten transmitted bits."""
    value = np.asarray(ids)
    if (value.dtype != np.dtype("int64") or value.ndim != 1 or
            table_size < 1 or table_size > 1 << BITS or
            np.any((value < 0) | (value >= table_size))):
        raise ValueError("global public code ID is invalid for the frozen bank")
    return ((value[:, None].astype(np.uint64) >>
             np.arange(BITS, dtype=np.uint64)[None, :]) & 1).astype(np.uint8)


def ids_from_bits(bits: np.ndarray, table_size: int) -> np.ndarray:
    """Decode the recipient's little-endian bits without ordinal-feature leakage."""
    value = np.asarray(bits)
    if (value.dtype != np.dtype("uint8") or value.ndim != 2 or
            value.shape[1] != BITS or not np.isin(value, (0, 1)).all()):
        raise ValueError("emitted code must be exactly ten hard uint8 bits")
    ids = np.sum(value.astype(np.int64) *
                 (1 << np.arange(BITS, dtype=np.int64))[None, :], axis=1)
    if np.any(ids >= table_size):
        raise ValueError("emitted code is outside the frozen public bank")
    return ids


def materialize_role_from_ids(row_id: np.ndarray, unit: np.ndarray,
                              code0: np.ndarray, code1: np.ndarray,
                              tables: tuple[np.ndarray, np.ndarray]) -> dict:
    """Turn a private R2 code table into exactly the three emitted fields."""
    if len(tables) != 2:
        raise ValueError("both recipient tables are required")
    out = {"row_id": np.asarray(row_id), "unit": np.asarray(unit)}
    for purpose, codes in enumerate((code0, code1)):
        table = _table(tables[purpose], purpose)
        bits = bits_from_ids(codes, len(table))
        ids = ids_from_bits(bits, len(table))
        prob = np.array(table[ids], dtype=np.float64, copy=True)
        out[f"code{purpose}"] = bits
        out[f"prob{purpose}"] = prob
        out[f"decision{purpose}"] = np.asarray(prob.argmax(axis=1), dtype=np.int64)
    _check_role(out, tables)
    return out


def public_wire_npz_sha256(role: Mapping, purpose: int,
                           tables: tuple[np.ndarray, np.ndarray]) -> str:
    """Hash the exact R1-compatible recipient NPZ serialization in memory.

    R1's deployment CLI saves the ordered `code`, `prob`, `decision` arrays
    with `np.savez_compressed`. This digest describes the cached role wire,
    without creating a second per-person output file or opening any label.
    """
    _check_role(role, tables)
    if purpose not in (0, 1):
        raise ValueError("recipient purpose must be zero or one")
    stream = io.BytesIO()
    np.savez_compressed(stream, code=role[f"code{purpose}"],
                        prob=role[f"prob{purpose}"],
                        decision=role[f"decision{purpose}"])
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _table(value: np.ndarray, purpose: int) -> np.ndarray:
    table = np.asarray(value)
    if (purpose not in (0, 1) or table.dtype != np.dtype("float64") or
            table.ndim != 2 or table.shape[1] != CLASSES[purpose] or
            not 1 <= len(table) <= 1 << BITS or
            not np.isfinite(table).all() or np.any(table < 0) or
            not np.allclose(table.sum(axis=1), 1.0, rtol=0, atol=1e-10)):
        raise ValueError("frozen public probability bank is invalid")
    return table


def _check_role(role: Mapping, tables: tuple[np.ndarray, np.ndarray]) -> None:
    if not isinstance(role, Mapping) or set(role) != ROLE_KEYS:
        raise ValueError("private role package must contain exactly the public wire and alignment")
    row, unit = np.asarray(role["row_id"]), np.asarray(role["unit"])
    if (row.dtype != np.dtype("int64") or unit.dtype != np.dtype("int64") or
            row.ndim != 1 or unit.shape != row.shape or len(row) == 0 or
            not np.all(np.diff(row) > 0)):
        raise ValueError("role ordered row/group identity is invalid")
    for purpose in (0, 1):
        table = _table(tables[purpose], purpose)
        code = np.asarray(role[f"code{purpose}"])
        prob = np.asarray(role[f"prob{purpose}"])
        decision = np.asarray(role[f"decision{purpose}"])
        ids = ids_from_bits(code, len(table))
        expected = table[ids]
        if (code.shape[0] != len(row) or prob.dtype != np.dtype("float64") or
                prob.shape != (len(row), CLASSES[purpose]) or
                decision.dtype != np.dtype("int64") or decision.shape != row.shape or
                not np.array_equal(prob, expected) or
                not np.array_equal(decision, expected.argmax(axis=1))):
            raise ValueError("emitted probability/decision differs from frozen code table")
    # One cached release per private duplicate group, including both recipients.
    first: dict[int, bytes] = {}
    for i, group in enumerate(unit):
        payload = b"".join(np.asarray(role[key])[i].tobytes() for key in
                           ("code0", "prob0", "decision0", "code1", "prob1", "decision1"))
        prior = first.setdefault(int(group), payload)
        if prior != payload:
            raise ValueError("duplicate group has different cached public releases")


def _metadata_field(value: object, n: int) -> Field:
    if value is None or type(value) in (str, bool, int):
        return Field("metadata_category", np.full(n, json.dumps(value, sort_keys=True)))
    if type(value) is float and math.isfinite(value):
        return Field("metadata", np.full(n, value, dtype=np.float64))
    raise ValueError("public metadata must be a fixed JSON scalar")


def _recipient(role: Mapping, purpose: int, metadata: Mapping[str, object]) -> WireBatch:
    row, unit = role["row_id"], role["unit"]
    bits = np.asarray(role[f"code{purpose}"])
    prob = np.asarray(role[f"prob{purpose}"])
    decision = np.asarray(role[f"decision{purpose}"])
    fields = {"code": Field("bits", bits),
              "prob": Field("probability", prob),
              "decision": Field("decision", decision)}
    for name in sorted(metadata):
        fields[f"metadata.{name}"] = _metadata_field(metadata[name], len(row))
    return WireBatch(row, unit, fields, ("code",),
                     {"public_log_prob": np.log(np.maximum(prob, 1e-12))})


def _pair(left: WireBatch, right: WireBatch, role: Mapping) -> WireBatch:
    paired = pair_batch(left, right)
    a, b = (np.asarray(role[f"code{purpose}"], dtype=np.float64)
            for purpose in (0, 1))
    cross = (a[:, :, None] * b[:, None, :]).reshape(len(a), BITS * BITS)
    return WireBatch(paired.row_id, paired.unit, paired.fields,
                     paired.code_fields,
                     {**(paired.public_derived or {}), "pair_bit_products": cross})


def prepare_complete_bank_views(
    att: Mapping, inner: Mapping,
    tables: tuple[np.ndarray, np.ndarray],
    *, metadata_att: tuple[Mapping[str, object], Mapping[str, object]],
    metadata_inner: tuple[Mapping[str, object], Mapping[str, object]],
) -> dict:
    """Build fixed ATT-vocabulary local and pair readers from exact public wires.

    `metadata_*` enumerates fields actually delivered by the R1 service or CLI;
    a caller may pass empty mappings only if R1's sealed wire truly has none.
    Neither role's metadata may carry per-person private input.
    """
    if len(tables) != 2 or len(metadata_att) != 2 or len(metadata_inner) != 2:
        raise ValueError("two recipient banks and metadata declarations required")
    _check_role(att, tables)
    _check_role(inner, tables)
    if (not all(isinstance(m, Mapping) for m in (*metadata_att, *metadata_inner)) or
            any(set(metadata_att[i]) != set(metadata_inner[i]) for i in (0, 1))):
        raise ValueError("public metadata field schema differs across roles")
    left_att = _recipient(att, 0, metadata_att[0])
    left_inner = _recipient(inner, 0, metadata_inner[0])
    right_att = _recipient(att, 1, metadata_att[1])
    right_inner = _recipient(inner, 1, metadata_inner[1])
    return {"r1": prepare_views(left_att, left_inner),
            "r2": prepare_views(right_att, right_inner),
            "pair": prepare_views(_pair(left_att, right_att, att),
                                  _pair(left_inner, right_inner, inner))}
