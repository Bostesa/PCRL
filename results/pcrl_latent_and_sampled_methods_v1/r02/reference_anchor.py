"""Exact U0/Ucal public score-wire adapter for the fixed D anchor attacks.

The caller reads the admitted R01 role-specific `references_s{seed}` archive.
Only each recipient's released probability and decision enter the attacker.
No latent features, target labels, SEX, or other recipient-specific source
arrays enter either local view. The pair sees both complete local score wires.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .complete_view import Field, WireBatch, pair_batch, prepare_views

REFERENCE_FIELDS = frozenset({"row_id", "unit", "U0_1", "U0_2",
                              "Ucal_1", "Ucal_2", "d_1", "d_2"})


def wire_batch_anchor(record: Mapping, anchor: str, recipient: int) -> WireBatch:
    """A reference wire has probability and decision, with no code or latent."""
    if set(record) != REFERENCE_FIELDS or anchor not in ("U0", "Ucal") or recipient not in (1, 2):
        raise ValueError("reference role archive or anchor identity differs")
    row = np.asarray(record["row_id"])
    unit = np.asarray(record["unit"])
    q = np.asarray(record[f"{anchor}_{recipient}"])
    decision = np.asarray(record[f"d_{recipient}"])
    classes = (2, 6)[recipient - 1]
    if (row.dtype != np.int64 or unit.dtype != np.int64 or row.ndim != 1
            or unit.shape != row.shape or not len(row)
            or np.any(np.diff(row) <= 0) or np.unique(row).size != len(row)):
        raise ValueError("ordered reference row/group IDs invalid")
    if (q.dtype != np.float64 or q.shape != (len(row), classes)
            or not np.isfinite(q).all() or np.any(q < 0)
            or not np.allclose(q.sum(axis=1), 1, rtol=0, atol=1e-8)):
        raise ValueError("reference probability wire invalid")
    if (decision.dtype != np.int64 or decision.shape != row.shape
            or not np.array_equal(decision, np.argmax(q, axis=1))):
        raise ValueError("reference decision differs from released probability argmax")
    fields = {"probabilities": Field("probability", q),
              "decision": Field("decision", decision)}
    derived = {"log_probabilities": np.log(np.maximum(q, 1e-12))}
    return WireBatch(row, unit, fields, code_fields=(), public_derived=derived)


def prepared_anchor_views(att_refs: Mapping, inner_refs: Mapping,
                          anchor: str) -> dict[str, object]:
    """Freeze ATT encoding for r1/r2/pair on the exact INNER reference wire."""
    a1, a2 = (wire_batch_anchor(att_refs, anchor, recipient) for recipient in (1, 2))
    i1, i2 = (wire_batch_anchor(inner_refs, anchor, recipient) for recipient in (1, 2))
    return {"r1": prepare_views(a1, i1),
            "r2": prepare_views(a2, i2),
            "pair": prepare_views(pair_batch(a1, a2), pair_batch(i1, i2))}
