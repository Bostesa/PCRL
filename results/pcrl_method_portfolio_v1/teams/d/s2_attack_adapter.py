"""Bind Team D's exact release wire fields to S2's complete AttackView.

The caller supplies S2's AttackView class and admitted role record. This module
opens no data, fits nothing, and never receives a protected label. S4 must
compare its declared fields with the deployed release before scoring.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable

import numpy as np

from attack_views import AlignedDViews, view_matrix


VARIABLE_WIRE_FIELDS = ("code_ids", "decoded", "probabilities", "decision")
DA_TRANSFORM_SPEC = {
    "schema": "pcrl-d-public-da-v1",
    "base": "complete variable D attack x in wire component order",
    "per_recipient_append_order": ["natural_log_max_calibrated_probability_1e-12_class_order",
                                   "square_emitted_decoded_coordinate_order"],
    "pair_finite_code_append": "recipient0_bit_low_to_high_outer_recipient1_bit_low_to_high_row_major",
    "source_scope": "transmitted release wire only; no private latent, input, label or other-party input",
}
DA_TRANSFORM_SHA256 = hashlib.sha256(json.dumps(
    DA_TRANSFORM_SPEC, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _metadata_matrix(metadata: dict, n: int) -> np.ndarray:
    numeric = np.tile(np.asarray([
        float(metadata["purpose"]), float(metadata["code_length_bits"]),
        float(metadata["block_count"]), float(metadata["fallback"]),
        float(metadata["stochastic"]),
    ], dtype=np.float32), (n, 1))
    string_parts = []
    for field in ("kind", "calibration", "calibrator_sha256"):
        digest = hashlib.sha256(str(metadata.get(field, "ABSENT")).encode()).digest()
        octets = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
        string_parts.append(np.tile(octets[None, :], (n, 1)))
    return np.concatenate((numeric, *string_parts), axis=1)


def _components(release: dict, prefix: str, n: int) -> list[tuple[str, np.ndarray]]:
    ids = release["code_ids"]
    fields = []
    if ids.shape[1] == 1:
        onehot = np.eye(1024, dtype=np.float32)[ids[:, 0]]
        # Every common reader sees the public bit decomposition in main x.
        code = np.concatenate((onehot, _bit_matrix(release)), axis=1)
        fields.append((f"{prefix}.code_ids", code))
    probability = np.asarray(release["probabilities"], dtype=np.float64)
    fields.extend([
        (f"{prefix}.decoded", np.asarray(release["decoded"], dtype=np.float32)),
        (f"{prefix}.probabilities", probability),
        (f"{prefix}.decision", np.eye(probability.shape[1], dtype=np.float32)[release["decision"]]),
    ])
    return fields


def _bit_matrix(release: dict) -> np.ndarray | None:
    ids = release["code_ids"]
    if ids.shape[1] == 0:
        return None
    return ((ids[:, 0, None] >> np.arange(10, dtype=np.int64)[None, :]) & 1).astype(np.float32)


def as_s2_attack_view(views: AlignedDViews, view: str, attack_view_class: Callable):
    """Create S2 AttackView with one exact column interval per wire field.

    `cell_key` uses categorical local IDs or an unambiguous aligned pair. The
    defense-aware and bit inputs derive only from public transmitted values.
    The caller must still check role hashes against DATA_ACCESS_MANIFEST.
    """
    if view == "recipient0":
        receivers = (("recipient0", views.recipient0),)
    elif view == "recipient1":
        receivers = (("recipient1", views.recipient1),)
    elif view == "pair":
        receivers = (("recipient0", views.recipient0), ("recipient1", views.recipient1))
    else:
        raise ValueError("unknown D view")
    n = len(views.row_id)
    parts = []
    parity_parts = []
    columns = []
    metadata_parts = []
    bit_parts = []
    wire_order = []
    wire_constants = []
    offset = 0
    for prefix, release in receivers:
        fields = _components(release, prefix, n)
        expected_variables = (VARIABLE_WIRE_FIELDS if release["code_ids"].shape[1] == 1
                              else VARIABLE_WIRE_FIELDS[1:])
        if tuple(name.removeprefix(prefix + ".") for name, _ in fields) != expected_variables:
            raise ValueError("D wire field declaration changed")
        if release["code_ids"].shape[1] == 0:
            empty_name = f"{prefix}.code_ids"
            wire_order.append(empty_name)
            wire_constants.append((empty_name, "[]"))
        for name, block in fields:
            width = block.shape[1]
            if block.shape[0] != n or width == 0 or not np.isfinite(block).all():
                raise ValueError("invalid D wire field matrix")
            columns.append((name, offset, offset + width))
            parts.append(block)
            wire_order.append(name)
            if name.endswith(".code_ids"):
                if release["code_ids"].shape[1] == 1:
                    parity_parts.append(block[:, :1024])
            else:
                parity_parts.append(block)
            offset += width
        metadata = release["metadata"]
        metadata_block = _metadata_matrix(metadata, n)
        metadata_parts.append(metadata_block)
        parity_parts.append(metadata_block)
        for field, value in metadata.items():
            name = f"{prefix}.metadata.{field}"
            wire_order.append(name)
            wire_constants.append((name, json.dumps(value, sort_keys=True, separators=(",", ":"))))
        bits = _bit_matrix(release)
        if bits is not None:
            bit_parts.append(bits)
    x = np.concatenate(parts, axis=1)
    if not np.array_equal(np.concatenate(parity_parts, axis=1), view_matrix(views, view)):
        raise ValueError("S2 field binding disagrees with D's complete view")
    x_metadata = np.concatenate(metadata_parts, axis=1)
    x_code_bits = np.concatenate(bit_parts, axis=1) if bit_parts else None
    if view == "pair" and len(bit_parts) == 2:
        cell_key = np.asarray([f"{int(a)}|{int(b)}" for a, b in
                               zip(views.recipient0["code_ids"][:, 0],
                                   views.recipient1["code_ids"][:, 0])])
    elif len(receivers) == 1 and bit_parts:
        cell_key = receivers[0][1]["code_ids"][:, 0].astype(str)
    else:
        cell_key = None
    da_parts = [np.asarray(x, dtype=np.float64)]
    for _, release in receivers:
        q = np.asarray(release["probabilities"], dtype=np.float64)
        decoded = np.asarray(release["decoded"], dtype=np.float64)
        da_parts.extend((np.log(np.maximum(q, 1e-12)), decoded ** 2))
    if view == "pair" and len(bit_parts) == 2:
        cross = bit_parts[0][:, :, None] * bit_parts[1][:, None, :]
        da_parts.append(cross.reshape(n, 100).astype(np.float64))
    defense_aware = np.concatenate(da_parts, axis=1)
    if not np.isfinite(defense_aware).all():
        raise ValueError("public defense-aware transform has a nonfinite value")
    names = tuple(name for name, _, _ in columns)
    result = attack_view_class(
        row_id=views.row_id.copy(), unit=views.unit_id.copy(), x=x,
        components=names, wire_components=tuple(wire_order),
        component_columns=tuple(columns), wire_constants=tuple(wire_constants),
        x_defense_aware=defense_aware, cell_key=cell_key,
        x_metadata=x_metadata, x_code_bits=x_code_bits,
    )
    return result.validate()
