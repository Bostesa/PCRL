"""Complete continuous F wire for R02's common three-view attack slate."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ...r02.complete_view import Field, WireBatch, pair_batch, prepare_views


def _metadata_field(value: object, n: int) -> Field:
    if type(value) in (bool, int, float):
        return Field("metadata", np.full(n, float(value), dtype=np.float64))
    if type(value) is str or value is None:
        return Field("metadata_category", np.full(n, "null" if value is None else value))
    raise ValueError("F public metadata must contain scalar wire values")


def wire_batch_f(row_id: np.ndarray, unit: np.ndarray, release: Mapping) -> WireBatch:
    """Encode exactly the transmitted F feature, task output and metadata."""
    row = np.asarray(row_id)
    group = np.asarray(unit)
    if (row.dtype != np.int64 or group.dtype != np.int64 or row.ndim != 1
            or group.shape != row.shape or not len(row)
            or np.unique(row).size != len(row)):
        raise ValueError("F requires ordered, unique int64 row IDs and aligned units")
    n = len(row)
    features = np.asarray(release["features"])
    probabilities = np.asarray(release["probabilities"])
    decision = np.asarray(release["decision"])
    metadata = release["metadata"]
    if (features.shape != (n, 16) or not np.isfinite(features).all()
            or probabilities.ndim != 2 or probabilities.shape[0] != n
            or probabilities.shape[1] not in (2, 6)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6)
            or decision.shape != (n,) or not np.issubdtype(decision.dtype, np.integer)
            or np.any(decision < 0) or np.any(decision >= probabilities.shape[1])
            or not isinstance(metadata, Mapping)
            or metadata.get("schema") != "latent-unchanged-v1"
            or metadata.get("latent_dimensions") != 16
            or metadata.get("purpose") not in (0, 1)
            or probabilities.shape[1] != (2, 6)[metadata["purpose"]]):
        raise ValueError("F public release shape, simplex or purpose invalid")
    fields = {
        "features": Field("numeric", features),
        "probabilities": Field("probability", probabilities),
        "decision": Field("decision", decision),
    }
    for key in sorted(metadata):
        fields[f"metadata.{key}"] = _metadata_field(metadata[key], n)
    derived = {"log_probabilities": np.log(np.maximum(
        probabilities.astype(np.float64), 1e-12))}
    return WireBatch(row, group, fields, code_fields=(), public_derived=derived)


def prepared_f_views(
    att_row_id: np.ndarray, att_unit: np.ndarray, att_first: Mapping,
    att_second: Mapping, inner_row_id: np.ndarray, inner_unit: np.ndarray,
    inner_first: Mapping, inner_second: Mapping,
) -> dict[str, object]:
    """Freeze ATT encoding and prepare F local/pair views for continuous readers."""
    a1 = wire_batch_f(att_row_id, att_unit, att_first)
    a2 = wire_batch_f(att_row_id, att_unit, att_second)
    i1 = wire_batch_f(inner_row_id, inner_unit, inner_first)
    i2 = wire_batch_f(inner_row_id, inner_unit, inner_second)
    for purpose, (att, inner) in enumerate(((att_first, inner_first),
                                             (att_second, inner_second))):
        if (att["metadata"].get("purpose") != purpose
                or inner["metadata"] != att["metadata"]):
            raise ValueError("F task or model/version metadata differs across roles")
    return {"r1": prepare_views(a1, i1),
            "r2": prepare_views(a2, i2),
            "pair": prepare_views(pair_batch(a1, a2), pair_batch(i1, i2))}
