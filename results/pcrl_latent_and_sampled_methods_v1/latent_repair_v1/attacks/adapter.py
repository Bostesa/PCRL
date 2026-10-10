"""D hard-bit release adapter for R02's shared complete-view attack interface.

Only transmitted code bits, code-only probabilities/decisions, and literal
public metadata enter a view. Exact code IDs and defense-aware transforms are
deterministic functions of those fields. This module reads no private role.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ...r02.complete_view import Field, WireBatch, pair_batch, prepare_views
from .hard_code import AlignedPair, PublicCode


def _metadata_field(value: object, n: int) -> Field:
    if type(value) in (bool, int, float):
        return Field("metadata", np.full(n, float(value), dtype=np.float64))
    return Field("metadata_category", np.full(n, "null" if value is None else value))


def wire_batch(public: PublicCode, unit: np.ndarray) -> WireBatch:
    groups = np.asarray(unit)
    if groups.dtype != np.int64 or groups.shape != public.row_id.shape:
        raise ValueError("admitted int64 resampling units must align with D release rows")
    n = len(public.row_id)
    fields: dict[str, Field] = {
        # The ID is exactly reconstructible from the ten emitted bits. It is
        # categorical, never an ordinal predictor.
        "code_id": Field("category", public.code_id.astype(np.int64)),
        "code_bits": Field("bits", public.code),
        "probabilities": Field("probability", public.probabilities),
        "decision": Field("decision", public.decision),
    }
    for key in sorted(public.metadata):
        fields[f"metadata.{key}"] = _metadata_field(public.metadata[key], n)
    derived = {"log_probabilities": np.log(np.maximum(public.probabilities.astype(np.float64), 1e-12))}
    return WireBatch(public.row_id.astype(np.int64), groups, fields,
                     code_fields=("code_id",), public_derived=derived)


def _pair_wire(first: WireBatch, second: WireBatch, pair: AlignedPair) -> WireBatch:
    joined = pair_batch(first, second)
    # Row-major 10 x 10 products expose complementary bit clues to the
    # defense-aware tabular readers. This is public-only and contains no h/p.
    b1 = pair.first.code.astype(np.float64)
    b2 = pair.second.code.astype(np.float64)
    cross = (b1[:, :, None] * b2[:, None, :]).reshape(len(b1), 100)
    return WireBatch(joined.row_id, joined.unit, joined.fields, joined.code_fields,
                     {**(joined.public_derived or {}), "pair_bit_products": cross})


def prepared_d_views(
    att_row_id: np.ndarray, att_unit: np.ndarray, att_first: Mapping,
    att_second: Mapping, inner_row_id: np.ndarray, inner_unit: np.ndarray,
    inner_first: Mapping, inner_second: Mapping,
) -> dict[str, object]:
    """Build all three ATT/INNER views with one frozen ATT categorical schema.

    SEX labels are deliberately absent. The caller supplies them separately
    only to the locked attacker-fit and screen stages.
    """
    ap = AlignedPair(PublicCode.from_release(att_row_id, att_first),
                     PublicCode.from_release(att_row_id, att_second))
    ip = AlignedPair(PublicCode.from_release(inner_row_id, inner_first),
                     PublicCode.from_release(inner_row_id, inner_second))
    for purpose, release in enumerate((ap.first, ap.second)):
        if release.metadata.get("purpose") != purpose or release.metadata.get("bits") != 10:
            raise ValueError("D public purpose or 10-bit capacity metadata is inconsistent")
    for ai, ii in ((ap.first, ip.first), (ap.second, ip.second)):
        if ai.metadata != ii.metadata:
            raise ValueError("the public D model/version metadata changed across ATT and INNER")
    a1, a2 = wire_batch(ap.first, att_unit), wire_batch(ap.second, att_unit)
    i1, i2 = wire_batch(ip.first, inner_unit), wire_batch(ip.second, inner_unit)
    return {
        "r1": prepare_views(a1, i1),
        "r2": prepare_views(a2, i2),
        "pair": prepare_views(_pair_wire(a1, a2, ap), _pair_wire(i1, i2, ip)),
    }
