"""Independently encode complete fixed A reference ATT and INNER public wires.

This module uses saved release arrays and the public metadata contract. It does
not import the producer's wire adapter or attack view encoder.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np

from .independent_a_q import actual_q_features


WIRE = "SCV1-reference-score-release-v1"
RELEASE_FIELDS = {"row_id", "unit", "prob_r1", "prob_r2", "decision_r1",
                  "decision_r2", "code_r1", "code_r2"}


def _checked_release(release: Mapping, *, finite: bool,
                     bank_sizes: tuple[int, int] | None) -> None:
    if set(release) != RELEASE_FIELDS:
        raise ValueError("fixed reference release member set differs")
    row, unit = release["row_id"], release["unit"]
    if (row.dtype != np.int64 or unit.dtype != np.int64 or row.ndim != 1
            or unit.shape != row.shape or not len(row)
            or np.any(np.diff(row) <= 0)):
        raise ValueError("fixed reference ordered row/group differs")
    for recipient, classes in ((1, 2), (2, 6)):
        q = release[f"prob_r{recipient}"]
        decision = release[f"decision_r{recipient}"]
        code = release[f"code_r{recipient}"]
        if (q.dtype != np.float64 or q.shape != (len(row), classes)
                or not np.isfinite(q).all() or np.any(q < 0)
                or np.max(np.abs(q.sum(axis=1) - 1)) > 1e-8
                or decision.dtype != np.int64 or decision.shape != row.shape
                or not np.array_equal(decision, np.argmax(q, axis=1))
                or code.dtype != np.int64 or code.shape != row.shape):
            raise ValueError("fixed reference task wire differs")
        if finite:
            if bank_sizes is None or np.any((code < 0) | (code >= bank_sizes[recipient - 1])):
                raise ValueError("fixed reference code exceeds public bank")
        elif not np.array_equal(code, np.full(len(row), -1, np.int64)):
            raise ValueError("continuous reference transmitted a code")


def _recipient_fields(release: Mapping, receipt: Mapping, recipient: int,
                      bank_size: int | None) -> dict[str, tuple[str, np.ndarray]]:
    n = len(release["row_id"])
    finite = receipt["interface_kind"] == "finite"
    q = release[f"prob_r{recipient}"]
    decision = release[f"decision_r{recipient}"]
    code = release[f"code_r{recipient}"]
    bits = (bank_size - 1).bit_length() if finite else None
    bank_sha = receipt["bank_sha256_by_recipient"][f"r{recipient}"]
    metadata = {"model_id": receipt["release_id"],
                "source_binding_sha256": receipt["source_binding_sha256"],
                "bank_sha256": bank_sha, "code_length_bits": bits}
    fields = {
        "schema": ("category", np.full(n, WIRE)),
        "recipient": ("numeric", np.full(n, recipient, np.float64)),
        "probability": ("probability", q),
        "decision": ("decision", decision),
        "release_id": ("category", np.full(n, receipt["release_id"])),
    }
    if finite:
        bit_width = max(1, bits)
        fields["code_id"] = ("category", code)
        fields["code_bits"] = ("bits", ((code[:, None] >>
            np.arange(bit_width, dtype=np.int64)) & 1).astype(np.float64))
    fields.update({
        "public_metadata.model_id": ("metadata_category", np.full(n, metadata["model_id"])),
        "public_metadata.source_binding_sha256": (
            "metadata_category", np.full(n, metadata["source_binding_sha256"])),
        "public_metadata.bank_sha256": (
            "metadata_category", np.full(n, bank_sha or "<NULL>")),
        "public_metadata.code_length_bits": (
            "metadata" if bits is not None else "metadata_category",
            np.full(n, bits if bits is not None else "<NULL>")),
        "public_metadata.canonical_json": (
            "metadata_category", np.full(n, json.dumps(metadata, sort_keys=True,
                separators=(",", ":"), allow_nan=False))),
    })
    return fields


def _paired(left: dict, right: dict) -> dict:
    return {**{f"r1.{key}": value for key, value in left.items()},
            **{f"r2.{key}": value for key, value in right.items()}}


def _numeric(value: np.ndarray, n: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or result.shape[0] != n or not np.isfinite(result).all():
        raise ValueError("fixed reference numeric view field differs")
    return result


def _encode_pair(att_fields: dict, inner_fields: dict, *,
                 att_codes: tuple[np.ndarray, np.ndarray],
                 inner_codes: tuple[np.ndarray, np.ndarray],
                 finite: bool, view: str,
                 derived_att: list[np.ndarray], derived_inner: list[np.ndarray]
                 ) -> tuple[dict, dict]:
    if (tuple(att_fields) != tuple(inner_fields) or
            any(att_fields[key][0] != inner_fields[key][0] for key in att_fields)):
        raise ValueError("fixed ATT/INNER wire field schema differs")
    names = tuple(att_fields)
    vocabulary = {}
    for name in names:
        kind, value = att_fields[name]
        if kind in ("category", "metadata_category"):
            vocabulary[name] = tuple(sorted(set(str(x) for x in value.tolist())))
        elif _numeric(value, len(value)).shape[1] != _numeric(
                inner_fields[name][1], len(inner_fields[name][1])).shape[1]:
            raise ValueError("fixed ATT/INNER numeric width differs")

    def one(fields: dict, codes: tuple[np.ndarray, np.ndarray],
            derived: list[np.ndarray]) -> dict:
        n = len(codes[0])
        blocks, metadata, bits, bindings = [], [], [], []
        start = 0
        for name in names:
            kind, raw = fields[name]
            if name in vocabulary:
                choices = vocabulary[name]
                lookup = {value: index for index, value in enumerate(choices)}
                block = np.zeros((n, len(choices) + 1), np.float64)
                block[np.arange(n), [lookup.get(str(value), len(choices))
                    for value in raw.tolist()]] = 1.0
            else:
                block = _numeric(raw, n)
            blocks.append(block)
            bindings.append([name, start, start + block.shape[1]])
            start += block.shape[1]
            if kind in ("metadata", "metadata_category"):
                metadata.append(block)
            if kind == "bits":
                bits.append(_numeric(raw, n))
        x = np.concatenate(blocks, axis=1)
        if finite:
            selected = (codes[0], codes[1]) if view == "pair" else (
                codes[0] if view == "r1" else codes[1],)
            cell = np.asarray([json.dumps([str(int(x[i])) for x in selected],
                                separators=(",", ":")) for i in range(n)], dtype=np.str_)
        else:
            cell = None
        return {"x": x,
                "x_defense_aware": np.concatenate([x, *derived], axis=1),
                "x_metadata": np.concatenate(metadata, axis=1),
                "x_code_bits": np.concatenate(bits, axis=1) if bits else None,
                "cell_key": cell, "component_columns": bindings,
                "components": list(names), "wire_components": list(names),
                "wire_constants": []}

    return (one(att_fields, att_codes, derived_att),
            one(inner_fields, inner_codes, derived_inner))


def build_fixed_views(att: Mapping, inner: Mapping, receipt: Mapping, *,
                      bank_sizes: tuple[int, int] | None,
                      q_table: Mapping | None) -> dict:
    """Rebuild r1, r2 and pair feature arrays from saved public releases."""
    finite = receipt.get("interface_kind") == "finite"
    if receipt.get("interface_kind") not in ("finite", "continuous") or finite != (
            bank_sizes is not None and q_table is not None):
        raise ValueError("fixed reference interface/Q identity differs")
    _checked_release(att, finite=finite, bank_sizes=bank_sizes)
    _checked_release(inner, finite=finite, bank_sizes=bank_sizes)
    role = {}
    for name, release in (("att", att), ("inner", inner)):
        left = _recipient_fields(release, receipt, 1,
                                 None if bank_sizes is None else bank_sizes[0])
        right = _recipient_fields(release, receipt, 2,
                                  None if bank_sizes is None else bank_sizes[1])
        role[name] = {"r1": left, "r2": right, "pair": _paired(left, right)}
    output = {}
    ca = (att["code_r1"], att["code_r2"])
    ci = (inner["code_r1"], inner["code_r2"])
    for view in ("r1", "r2", "pair"):
        derived, zero = {}, {}
        for name, release in (("att", att), ("inner", inner)):
            if finite:
                first = release["code_r1"] if view != "r2" else release["code_r2"]
                second = release["code_r2"] if view == "pair" else None
                value, count = actual_q_features(q_table, view=view,
                                                  code1=first, code2=second)
                derived[name], zero[name] = [value], count
            else:
                recipients = (1, 2) if view == "pair" else ((1,) if view == "r1" else (2,))
                derived[name] = [np.log(np.maximum(release[f"prob_r{i}"], 1e-12))
                                 for i in recipients]
                zero[name] = None
        att_view, inner_view = _encode_pair(
            role["att"][view], role["inner"][view],
            att_codes=ca, inner_codes=ci, finite=finite, view=view,
            derived_att=derived["att"], derived_inner=derived["inner"])
        output[view] = {"att": att_view, "inner": inner_view,
                        "zero_q_support_count": zero}
    return output
