"""Independently encode complete sampled A ATT/INNER attack views.

Every column comes from the actual seven-key recipient wire or a fixed public
Q lookup. The ATT code vocabulary is frozen before INNER encoding.
"""

from __future__ import annotations

import json
from typing import Mapping

import numpy as np

from .independent_a_law import IndependentAPolicy
from .independent_a_q import actual_q_features


WIRE = "SCV1-score-release-v1"


def _metadata(policy: IndependentAPolicy, recipient: int) -> dict:
    bank = policy.arrays[f"bank{recipient}_probability"]
    receipt = policy.receipt
    return {"model_id": receipt["release_id"],
            "source_binding_sha256": receipt["source_binding_sha256"],
            "canonical_bank_sha256": receipt["bank_sha256_by_recipient"][f"r{recipient}"],
            "mechanism_sha256": receipt["mechanism_sha256"],
            "code_length_bits": (len(bank) - 1).bit_length(),
            "fallback": False, "sampling_contract": "cached_one_joint_draw_v1"}


def _recipient_fields(policy: IndependentAPolicy, recipient: int,
                      codes: np.ndarray) -> dict[str, tuple[str, np.ndarray]]:
    bank = policy.arrays[f"bank{recipient}_probability"]
    decision = policy.arrays[f"bank{recipient}_decision"]
    if (codes.dtype != np.int64 or codes.ndim != 1
            or np.any((codes < 0) | (codes >= len(bank)))):
        raise ValueError("A complete-view emitted code outside selected bank")
    n = len(codes)
    bits = max(1, (len(bank) - 1).bit_length())
    bit_values = ((codes[:, None] >> np.arange(bits, dtype=np.int64)) & 1).astype(np.float64)
    metadata = _metadata(policy, recipient)
    flattened = {f"public_metadata.{key}": value for key, value in metadata.items()}
    flattened["public_metadata.canonical_json"] = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    fields = {
        "schema": ("category", np.asarray([WIRE] * n)),
        "recipient": ("numeric", np.full(n, recipient, np.float64)),
        "code_id": ("category", codes),
        "code_bits": ("bits", bit_values),
        "probability": ("probability", bank[codes]),
        "decision": ("decision", decision[codes]),
        "release_id": ("category", np.asarray([policy.receipt["release_id"]] * n)),
    }
    for name in sorted(flattened):
        value = flattened[name]
        if isinstance(value, str):
            fields[name] = ("metadata_category", np.asarray([value] * n))
        else:
            fields[name] = ("metadata", np.full(n, float(value), np.float64))
    return fields


def _pair_fields(left: dict, right: dict) -> dict:
    return {**{f"r1.{name}": value for name, value in left.items()},
            **{f"r2.{name}": value for name, value in right.items()}}


def _as_strings(value: np.ndarray) -> list[str]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError("A categorical wire field is not vector")
    return [str(x) for x in array.tolist()]


def _numeric(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("A complete-view numeric wire field invalid")
    return array


def _encode(att: dict, inner: dict, codes_att: tuple[np.ndarray, np.ndarray],
            codes_inner: tuple[np.ndarray, np.ndarray], *,
            q_att: np.ndarray | None, q_inner: np.ndarray | None) -> tuple[dict, dict, dict]:
    if tuple(att) != tuple(inner):
        raise ValueError("A ATT and INNER field order differs")
    names = tuple(att)
    categories = {}
    for name in names:
        kind, values = att[name]
        if inner[name][0] != kind:
            raise ValueError("A ATT and INNER wire field kinds differ")
        if kind in ("category", "metadata_category"):
            categories[name] = sorted(set(_as_strings(values)))
        elif _numeric(values).shape[1] != _numeric(inner[name][1]).shape[1]:
            raise ValueError("A numeric wire width differs across roles")

    def one(fields: dict, codes: tuple[np.ndarray, np.ndarray], q: np.ndarray | None) -> dict:
        n = len(codes[0])
        blocks, metadata, bits, bindings = [], [], [], []
        start = 0
        for name in names:
            kind, value = fields[name]
            if name in categories:
                vocab = categories[name]
                index = {key: i for i, key in enumerate(vocab)}
                block = np.zeros((n, len(vocab) + 1), np.float64)
                block[np.arange(n), [index.get(v, len(vocab)) for v in
                                      _as_strings(value)]] = 1.0
            else:
                block = _numeric(value)
            blocks.append(block)
            bindings.append([name, start, start + block.shape[1]])
            start += block.shape[1]
            if kind in ("metadata", "metadata_category"):
                metadata.append(block)
            if kind == "bits":
                bits.append(_numeric(value))
        x = np.concatenate(blocks, axis=1)
        if q is not None and (q.shape != (n, 19) or not np.isfinite(q).all()):
            raise ValueError("A public Q-derived features invalid")
        cell = np.asarray([
            json.dumps([str(int(codes[0][i]))] if not names[0].startswith("r1.")
                       else [str(int(codes[0][i])), str(int(codes[1][i]))],
                       separators=(",", ":")) for i in range(n)], dtype=np.str_)
        return {"x": x, "x_defense_aware": None if q is None else np.concatenate((x, q), axis=1),
                "x_metadata": np.concatenate(metadata, axis=1),
                "x_code_bits": np.concatenate(bits, axis=1),
                "cell_key": cell, "component_columns": bindings,
                "components": list(names), "wire_components": list(names),
                "wire_constants": []}

    left = one(att, codes_att, q_att)
    right = one(inner, codes_inner, q_inner)
    if left["component_columns"] != right["component_columns"]:
        raise AssertionError("A ATT/INNER independent component widths differ")
    return left, right, {"categories": categories}


def encode_sampled_views(policy: IndependentAPolicy, *,
                         att_code1: np.ndarray, att_code2: np.ndarray,
                         inner_code1: np.ndarray, inner_code2: np.ndarray,
                         q_table: Mapping | None) -> dict:
    """Return each view's ATT/INNER arrays and zero-Q-support counts."""
    pairs = {"att": (att_code1, att_code2),
             "inner": (inner_code1, inner_code2)}
    wire = {}
    for role, (first, second) in pairs.items():
        left = _recipient_fields(policy, 1, first)
        right = _recipient_fields(policy, 2, second)
        wire[role] = {"r1": left, "r2": right,
                      "pair": _pair_fields(left, right)}
    output = {}
    for view in ("r1", "r2", "pair"):
        qa, qi = None, None
        zero = {"att": None, "inner": None}
        if q_table is not None:
            for role, (first, second) in pairs.items():
                if view == "pair":
                    value, count = actual_q_features(q_table, view="pair",
                                                     code1=first, code2=second)
                else:
                    value, count = actual_q_features(
                        q_table, view=view, code1=first if view == "r1" else second)
                if role == "att":
                    qa = value
                else:
                    qi = value
                zero[role] = count
        ca = pairs["att"]
        ci = pairs["inner"]
        left, right, context = _encode(wire["att"][view], wire["inner"][view],
                                       ca, ci, q_att=qa, q_inner=qi)
        output[view] = {"att": left, "inner": right,
                        "zero_q_support_count": zero,
                        "att_categories": context["categories"]}
    return output


def compare_frozen_view_arrays(encoded: Mapping, saved: Mapping,
                               receipt: dict) -> dict:
    """Compare exact ordered feature bytes with inherited frozen reader custody."""
    differences = {}
    for role in ("att", "inner"):
        expected = encoded[role]
        recorded = saved[role]
        for name in ("x", "x_defense_aware", "x_metadata", "x_code_bits", "cell_key"):
            value = expected[name]
            present = name in recorded
            if value is None:
                if present:
                    raise ValueError("A frozen view has unregistered derived array")
                continue
            if not present or not np.array_equal(value, recorded[name]):
                raise ValueError(f"A frozen {role} {name} differs from complete wire")
            differences[f"{role}.{name}"] = 0.0
    schema = receipt["view_schema"]
    for key in ("components", "wire_components", "component_columns", "wire_constants"):
        if schema[key] != encoded["att"][key]:
            raise ValueError("A frozen complete-view column schema differs")
    return {"ordered_complete_view_arrays_exact": True,
            "arrays_checked": len(differences),
            "zero_q_support_count": encoded["zero_q_support_count"]}
