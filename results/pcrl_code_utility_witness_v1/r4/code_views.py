"""Data-free complete-view adapter for ten- and sixteen-bit public codes.

The caller passes actual transmitted releases and admitted row/group order.
This module does not read role files, labels, source latents, or model weights.
The queued worker must additionally replay probabilities against the pinned
public decoder/head; within-wire consistency alone cannot certify that head.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r02.complete_view import (
    Field, WireBatch, pair_batch, prepare_views,
)


_WIRE_COMMON = frozenset({"probabilities", "decision", "metadata"})
_ROLE_KEYS = frozenset({"row_id", "unit", "first", "second"})


def _ids(role: Mapping) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(role, Mapping) or set(role) != _ROLE_KEYS:
        raise ValueError("role release must contain exact public wire and row/group IDs")
    rows, groups = np.asarray(role["row_id"]), np.asarray(role["unit"])
    if (rows.dtype != np.int64 or groups.dtype != np.int64 or rows.ndim != 1
            or groups.shape != rows.shape or not len(rows)
            or np.unique(rows).size != len(rows)):
        raise ValueError("admitted row/group order must be nonempty int64 vectors")
    return rows, groups


def _bits(wire: Mapping, n: int, width: int) -> np.ndarray:
    if not isinstance(wire, Mapping) or len(set(wire).difference(_WIRE_COMMON)) != 1:
        raise ValueError("recipient release must contain exact public wire fields")
    if set(wire) == _WIRE_COMMON | {"code"}:
        code = np.asarray(wire["code"])
        if (code.dtype != np.uint8 or code.shape != (n, width)
                or not np.isin(code, (0, 1)).all()):
            raise ValueError("public code must contain exact-width uint8 hard bits")
        return code.copy()
    if set(wire) == _WIRE_COMMON | {"code_id"}:
        ids = np.asarray(wire["code_id"])
        if (ids.shape != (n,) or ids.dtype.kind not in "iu"
                or np.any(ids < 0) or np.any(ids >= 1 << width)):
            raise ValueError("public code ID must fit the declared global capacity")
        return ((ids.astype(np.uint64)[:, None] >>
                 np.arange(width, dtype=np.uint64)[None, :]) & 1).astype(np.uint8)
    raise ValueError("recipient release must contain exact public wire fields")


def _metadata(value: object, n: int) -> Field:
    if value is None or type(value) in (str, bool, int):
        return Field("metadata_category", np.full(n, str(value)))
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("metadata scalar must be finite")
        return Field("metadata", np.full(n, value, dtype=np.float64))
    arr = np.asarray(value)
    if arr.shape != (n,):
        raise ValueError("per-person metadata must align with public rows")
    if arr.dtype.kind in "US":
        return Field("metadata_category", arr.copy())
    if arr.dtype.kind in "iufb" and np.isfinite(arr).all():
        return Field("metadata", arr.astype(np.float64))
    raise ValueError("metadata needs finite numeric or string public values")


def _recipient(role: Mapping, name: str, purpose: int, width: int) -> tuple[WireBatch, np.ndarray, np.ndarray]:
    rows, units = _ids(role)
    wire = role[name]
    code = _bits(wire, len(rows), width)
    q = np.asarray(wire["probabilities"])
    decision = np.asarray(wire["decision"])
    classes = 2 if purpose == 0 else 6
    if (q.dtype != np.float64 or q.shape != (len(rows), classes)
            or not np.isfinite(q).all() or np.any(q < 0)
            or not np.allclose(q.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
        raise ValueError("public probabilities must be normalized float64 task vectors")
    if (decision.dtype != np.int64 or decision.shape != (len(rows),)
            or not np.array_equal(decision, q.argmax(axis=1))):
        raise ValueError("public decision must be the first-tie argmax of its probability")
    metadata = wire["metadata"]
    if (not isinstance(metadata, Mapping) or not metadata
            or any(not isinstance(key, str) or not key for key in metadata)):
        raise ValueError("public metadata must be a nonempty named mapping")
    declared_bits, declared_purpose = metadata.get("bits"), metadata.get("purpose")
    if (type(declared_bits) not in (int, np.int64, np.int32)
            or type(declared_purpose) not in (int, np.int64, np.int32)
            or declared_bits != width or declared_purpose != purpose):
        raise ValueError("public metadata capacity/purpose differs from wire")
    fields = {"code_bits": Field("bits", code),
              "probabilities": Field("probability", q.copy()),
              "decision": Field("decision", decision.copy())}
    for key in sorted(metadata):
        fields[f"metadata.{key}"] = _metadata(metadata[key], len(rows))
    derived = {"log_probabilities": np.log(np.maximum(q, 1e-12))}
    return WireBatch(rows, units, fields, ("code_bits",), derived), code, q


def _assert_code_only(codes: tuple[np.ndarray, np.ndarray],
                      probabilities: tuple[np.ndarray, np.ndarray]) -> None:
    mapping: dict[bytes, np.ndarray] = {}
    for bits, q in zip(codes, probabilities):
        for code, prob in zip(bits, q):
            key = code.tobytes()
            if key in mapping and not np.array_equal(mapping[key], prob):
                raise ValueError("same public code has different code-only probabilities")
            mapping[key] = prob


def _pair(left: WireBatch, right: WireBatch, code1: np.ndarray, code2: np.ndarray) -> WireBatch:
    joined = pair_batch(left, right)
    cross = (code1[:, :, None].astype(np.float64) *
             code2[:, None, :].astype(np.float64)).reshape(len(code1), -1)
    derived = {**(joined.public_derived or {}), "pair_bit_products": cross}
    return WireBatch(joined.row_id, joined.unit, joined.fields,
                     joined.code_fields, derived)


def prepare_complete_views(att: Mapping, inner: Mapping, *, bits: int) -> dict:
    """Prepare ATT and INNER local/pair views with ATT-only categorical vocabulary.

    A public integer code ID and its exact little-endian hard-bit expansion are
    equivalent encodings. All recipient probability, decision, and metadata
    fields enter the reader views. This function accepts no labels.
    """
    if bits not in (10, 16):
        raise ValueError("predeclared public capacity must be ten or sixteen bits")
    prepared = {}
    for purpose, name in enumerate(("first", "second")):
        ab, ac, aq = _recipient(att, name, purpose, bits)
        ib, ic, iq = _recipient(inner, name, purpose, bits)
        if tuple(ab.fields) != tuple(ib.fields) or any(
                ab.fields[k].kind != ib.fields[k].kind for k in ab.fields):
            raise ValueError("public metadata field schema changed across roles")
        for key in ("purpose", "bits"):
            if not np.array_equal(ab.fields[f"metadata.{key}"].values,
                                  ib.fields[f"metadata.{key}"].values):
                raise ValueError("public metadata constant changed across roles")
        _assert_code_only((ac, ic), (aq, iq))
        prepared[name] = (ab, ib, ac, ic)
    a1, i1, ac1, ic1 = prepared["first"]
    a2, i2, ac2, ic2 = prepared["second"]
    return {"r1": prepare_views(a1, i1),
            "r2": prepare_views(a2, i2),
            "pair": prepare_views(_pair(a1, a2, ac1, ac2),
                                  _pair(i1, i2, ic1, ic2))}
