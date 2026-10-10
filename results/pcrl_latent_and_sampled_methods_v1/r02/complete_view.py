"""Lossless, label-free encoding of actual transmitted fields for fresh attacks.

The caller enumerates the wire contract, including constant and metadata fields.
Categorical vocabularies come from ATTACK_FIT_NEW only. An INNER-only category
gets its own unknown indicator; no row is discarded. Published derived columns
are supplied separately and must be independently replayed from the public wire.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

_SHARED = Path(__file__).resolve().parents[2] / "pcrl_method_portfolio_v1" / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))
from attack_eval import AttackView

KINDS = frozenset({"numeric", "category", "bits", "probability", "decision", "metadata", "metadata_category"})
MAX_CATEGORIES = 4096


@dataclass(frozen=True)
class Field:
    kind: str
    values: object


@dataclass(frozen=True)
class WireBatch:
    row_id: np.ndarray
    unit: np.ndarray
    fields: Mapping[str, Field]
    code_fields: tuple[str, ...] = ()
    public_derived: Mapping[str, np.ndarray] | None = None


@dataclass(frozen=True)
class PreparedViews:
    att: AttackView
    inner: AttackView
    probability_att: np.ndarray
    probability_inner: np.ndarray
    schema: dict


def _ids(batch: WireBatch) -> tuple[np.ndarray, np.ndarray]:
    row = np.asarray(batch.row_id)
    unit = np.asarray(batch.unit)
    if row.dtype != np.int64 or unit.dtype != np.int64 or row.ndim != 1 or unit.shape != row.shape:
        raise ValueError("ordered row/unit IDs must be int64 vectors")
    if not len(row) or np.unique(row).size != len(row):
        raise ValueError("ordered row IDs must be nonempty and unique")
    return row, unit


def _strings(values: object, n: int) -> list[str]:
    a = np.asarray(values)
    if a.shape != (n,) or a.dtype.kind not in "USiu":
        raise ValueError("categorical wire field must be an exact string/integer vector")
    return [str(x) for x in a.tolist()]


def _numeric(field: Field, n: int) -> np.ndarray:
    try:
        a = np.asarray(field.values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("numeric wire field is invalid") from exc
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or a.shape[0] != n or not a.shape[1] or not np.isfinite(a).all():
        raise ValueError("numeric wire field shape or finiteness is invalid")
    if field.kind == "bits" and not np.isin(a, (0, 1)).all():
        raise ValueError("bit wire field must contain only zero and one")
    if field.kind == "probability" and (np.any(a < 0) or
                                         not np.allclose(a.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
        raise ValueError("probability wire field must be a simplex row")
    if field.kind == "decision" and (a.shape[1] != 1 or not np.equal(a, np.floor(a)).all()):
        raise ValueError("decision wire field must be an integer scalar")
    return a


def _validate_schema(att: WireBatch, inner: WireBatch) -> tuple[str, ...]:
    _ids(att)
    _ids(inner)
    names = tuple(att.fields)
    if not names or tuple(inner.fields) != names or any(not isinstance(k, str) or not k for k in names):
        raise ValueError("wire field schema differs or is empty")
    if att.code_fields != inner.code_fields or any(k not in names for k in att.code_fields):
        raise ValueError("code field schema differs")
    if len(set(att.code_fields)) != len(att.code_fields):
        raise ValueError("duplicate code field")
    for k in names:
        if att.fields[k].kind != inner.fields[k].kind or att.fields[k].kind not in KINDS:
            raise ValueError("wire field kind schema differs")
    ad = tuple((att.public_derived or {}))
    if tuple((inner.public_derived or {})) != ad:
        raise ValueError("public derived schema differs")
    return names


def pair_batch(left: WireBatch, right: WireBatch) -> WireBatch:
    """Join ordered recipient views, retaining both wire namespaces exactly."""
    lrow, lunit = _ids(left)
    rrow, runit = _ids(right)
    if not np.array_equal(lrow, rrow) or not np.array_equal(lunit, runit):
        raise ValueError("ordered recipient rows/groups differ")
    fields = {f"r1.{k}": v for k, v in left.fields.items()}
    fields.update({f"r2.{k}": v for k, v in right.fields.items()})
    derived = {f"r1.{k}": v for k, v in (left.public_derived or {}).items()}
    derived.update({f"r2.{k}": v for k, v in (right.public_derived or {}).items()})
    return WireBatch(lrow, lunit, fields,
                     tuple(f"r1.{k}" for k in left.code_fields) +
                     tuple(f"r2.{k}" for k in right.code_fields), derived)


def prepare_views(att: WireBatch, inner: WireBatch) -> PreparedViews:
    """Freeze one ATT vocabulary and form compatible complete ATT/INNER views."""
    names = _validate_schema(att, inner)
    n_att, n_inner = len(att.row_id), len(inner.row_id)
    vocab: dict[str, tuple[str, ...]] = {}
    widths: dict[str, int] = {}
    for name in names:
        field = att.fields[name]
        if field.kind in ("category", "metadata_category"):
            values = _strings(field.values, n_att)
            _strings(inner.fields[name].values, n_inner)
            choices = tuple(sorted(set(values)))
            if len(choices) > MAX_CATEGORIES:
                raise ValueError("categorical wire field exceeds frozen vocabulary cap")
            vocab[name] = choices
            widths[name] = len(choices) + 1
        else:
            a, b = _numeric(field, n_att), _numeric(inner.fields[name], n_inner)
            if a.shape[1] != b.shape[1]:
                raise ValueError("numeric wire field width schema differs")
            widths[name] = a.shape[1]

    def encode(batch: WireBatch) -> tuple[AttackView, np.ndarray]:
        n = len(batch.row_id)
        blocks: list[np.ndarray] = []
        bindings: list[tuple[str, int, int]] = []
        bits: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        metadata: list[np.ndarray] = []
        start = 0
        for name in names:
            field = batch.fields[name]
            if name in vocab:
                values = _strings(field.values, n)
                lookup = {choice: j for j, choice in enumerate(vocab[name])}
                block = np.zeros((n, widths[name]), dtype=np.float64)
                block[np.arange(n), [lookup.get(v, widths[name] - 1) for v in values]] = 1.0
            else:
                block = _numeric(field, n)
            blocks.append(block)
            end = start + widths[name]
            bindings.append((name, start, end))
            start = end
            if field.kind == "bits":
                bits.append(block)
            if field.kind == "probability":
                probabilities.append(block)
            if field.kind in ("metadata", "metadata_category"):
                metadata.append(block)
        x = np.concatenate(blocks, axis=1)
        deriv = []
        for name, values in (batch.public_derived or {}).items():
            block = np.asarray(values, dtype=np.float64)
            if block.ndim == 1:
                block = block[:, None]
            if block.ndim != 2 or block.shape[0] != n or not block.shape[1] or not np.isfinite(block).all():
                raise ValueError(f"public derived field {name} is invalid")
            deriv.append(block)
        codes = [_strings(batch.fields[name].values, n) for name in batch.code_fields]
        cell_key = np.asarray([json.dumps(list(values), separators=(",", ":"))
                               for values in zip(*codes)], dtype=np.str_) if codes else None
        view = AttackView(row_id=batch.row_id, unit=batch.unit, x=x,
                          components=names, wire_components=names,
                          component_columns=tuple(bindings),
                          x_defense_aware=np.concatenate([x, *deriv], axis=1) if deriv else None,
                          cell_key=cell_key,
                          x_metadata=np.concatenate(metadata, axis=1) if metadata else None,
                          x_code_bits=np.concatenate(bits, axis=1) if bits else None)
        return view.validate(), np.concatenate(probabilities, axis=1) if probabilities else np.empty((n, 0))

    att_view, pa = encode(att)
    inner_view, pi = encode(inner)
    schema = {"schema": "pcrl-next-complete-view-v1", "fields": [
        {"name": name, "kind": att.fields[name].kind, "width": widths[name],
         "att_categories": list(vocab[name]) if name in vocab else None}
        for name in names], "code_fields": list(att.code_fields),
        "public_derived": list(att.public_derived or {}),
        "att_view_sha256": att_view.content_hashes(),
        "inner_view_sha256": inner_view.content_hashes()}
    schema["schema_sha256"] = hashlib.sha256(json.dumps(schema, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return PreparedViews(att_view, inner_view, pa, pi, schema)
