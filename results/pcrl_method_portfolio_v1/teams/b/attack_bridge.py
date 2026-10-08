"""Construct S2 AttackView objects from Team B's actual saved wire releases.

The fixed numeric encoding is chosen without SEX labels or INNER outcomes.
Public linear heads may recompute logits from released z; their parameters do
not reveal a private per-person provider input. Every actual wire scalar has
one named component and a disjoint span of attack columns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import re

from attack_views import CLASSES, aligned_complete_views
from bottleneck import model_fingerprint, schema_fingerprint

_GENERATED_ONEHOT = re.compile(r"^(?:income\.|occupation\.)?decision_onehot\[[0-9]+\]$")


def _encode(records: Sequence[Mapping[str, float | str]]) -> tuple[
    np.ndarray, tuple[str, ...], tuple[tuple[str, int, int], ...], np.ndarray | None,
    tuple[str, ...], tuple[tuple[str, str], ...]
]:
    if not records:
        raise ValueError("Empty Team B attack view")
    # flatten_view's one-hot decision convenience is not a separate wire item.
    keys = tuple(sorted(key for key in records[0] if not _GENERATED_ONEHOT.fullmatch(key)))
    if not keys:
        raise ValueError("No transmitted Team B components")
    for record in records:
        if tuple(sorted(key for key in record if not _GENERATED_ONEHOT.fullmatch(key))) != keys:
            raise ValueError("Team B wire component schema differs across people")
    columns, bindings, metadata_columns, varying, constants = [], [], [], [], []
    start = 0
    for key in keys:
        values = [record[key] for record in records]
        types = {"string" if isinstance(value, str) else "numeric" for value in values}
        if len(types) != 1:
            raise ValueError("Transmitted field changes type across people")
        if "string" in types:
            if len(set(values)) != 1:
                raise ValueError("Person-varying string field needs a prelocked categorical reader")
            constants.append((key, str(values[0])))
            continue
        block = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite transmitted Team B field")
        end = start + block.shape[1]
        columns.append(block)
        bindings.append((key, start, end))
        varying.append(key)
        if key.startswith("metadata.") or ".metadata." in key:
            metadata_columns.extend(range(start, end))
        start = end
    x = np.concatenate(columns, axis=1)
    metadata = x[:, metadata_columns] if metadata_columns else None
    return x, tuple(varying), tuple(bindings), metadata, keys, tuple(constants)


def _public_head_projection(
    rows: Sequence[tuple[str, Mapping[str, Mapping[str, object]]]],
    purpose: str, head: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    weight = np.asarray(head[0], dtype=np.float64)
    bias = np.asarray(head[1], dtype=np.float64)
    classes = CLASSES[purpose]
    if any(views[purpose]["metadata"].get("purpose") != purpose
           or views[purpose]["metadata"].get("calibration_family") != "global_temperature"
           for _, views in rows):
        raise ValueError("Released purpose or calibration family differs from public contract")
    if weight.shape != (classes, 16) or bias.shape != (classes,):
        raise ValueError("Public Team B head parameter dimensions differ")
    latent = np.asarray([views[purpose]["features"] for _, views in rows], dtype=np.float64)
    transmitted = np.asarray([views[purpose]["logits"] for _, views in rows], dtype=np.float64)
    if latent.shape != (len(rows), 16) or transmitted.shape != (len(rows), classes):
        raise ValueError("Public-head replay lacks the actual released z/logits")
    projected = latent @ weight.T + bias
    if not np.allclose(projected, transmitted, rtol=1e-5, atol=1e-5):
        raise ValueError("Released logits do not depend on the public head and released z")
    temperature = np.asarray(
        [views[purpose]["metadata"]["temperature"] for _, views in rows], dtype=np.float64
    )
    if temperature.shape != (len(rows),) or not np.isfinite(temperature).all() or np.any(temperature <= 0):
        raise ValueError("A positive transmitted calibration temperature is required")
    if np.unique(temperature).size != 1:
        raise ValueError("Global temperature must be constant across people")
    scaled = projected / temperature[:, None]
    expected = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)
    probabilities = np.asarray(
        [views[purpose]["probabilities"] for _, views in rows], dtype=np.float64
    )
    decisions = np.asarray([views[purpose]["decision"] for _, views in rows], dtype=np.int64)
    if (probabilities.shape != expected.shape
            or not np.allclose(probabilities, expected, rtol=1e-5, atol=1e-5)
            or not np.array_equal(decisions, np.argmax(expected, axis=1))):
        raise ValueError("Released probability or decision bypasses the public protected head")
    margin = projected - projected.max(axis=1, keepdims=True)
    return np.concatenate((projected, margin), axis=1)


def build_attack_view(
    rows: Sequence[tuple[str, Mapping[str, Mapping[str, object]]]],
    *, view: str, row_id: Sequence[int], unit: Sequence[int],
    public_model,
    expected_source_id: str,
    attack_view_type=None,
):
    """Return S2's validated AttackView for a local or aligned pair view.

    The caller supplies row/group IDs from the admitted role-only export in
    exactly the saved release order. `attack_view_type` allows independent
    synthetic tests; production uses `attack_eval.AttackView`.
    """
    if view not in ("income", "occupation", "pair"):
        raise ValueError("Unknown Team B disclosure view")
    if set(public_model.heads) != set(CLASSES) or not expected_source_id:
        raise ValueError("Both public purpose heads and source identity are required")
    public_model_sha256 = model_fingerprint(public_model)
    public_schema_sha256 = schema_fingerprint(public_model.feature_names)
    raw_rid = np.asarray(row_id)
    raw_groups = np.asarray(unit)
    if raw_rid.dtype != np.int64 or raw_groups.dtype != np.int64:
        raise ValueError("Admitted row and group IDs must be int64 without truncation")
    rid = np.asarray(raw_rid, dtype=np.int64)
    groups = np.asarray(raw_groups, dtype=np.int64)
    if rid.shape != (len(rows),) or groups.shape != rid.shape or np.unique(rid).size != len(rows):
        raise ValueError("Attack row/group alignment differs from saved release")
    if any(person_key != str(int(row)) for (person_key, _), row in zip(rows, rid)):
        raise ValueError("Saved private person key differs from admitted ordered row ID")
    if any(_GENERATED_ONEHOT.fullmatch(field)
           for _, views in rows for purpose in CLASSES for field in views[purpose]):
        raise ValueError("Actual wire field collides with generated decision-onehot feature")
    for _, views in rows:
        for purpose in CLASSES:
            metadata = views[purpose]["metadata"]
            if (metadata["model_sha256"] != public_model_sha256
                    or metadata["schema_sha256"] != public_schema_sha256
                    or metadata["source_id"] != expected_source_id):
                raise ValueError("Released model/schema/source differs from public artifact")
    flattened = aligned_complete_views(rows)[view]
    x, names, bindings, metadata, wire_names, wire_constants = _encode(flattened)
    relevant = ("income", "occupation") if view == "pair" else (view,)
    heads = {
        purpose: (public_model.heads[purpose].weight.detach().cpu().numpy(),
                  public_model.heads[purpose].bias.detach().cpu().numpy())
        for purpose in relevant
    }
    replay = np.concatenate([_public_head_projection(rows, purpose, heads[purpose])
                             for purpose in relevant], axis=1)
    # The ordinary reader already has every wire component. The defense-aware
    # reader additionally gets public-head derived logits/margins, computed
    # solely from the released z and public weights.
    defense_aware = np.concatenate((x, replay), axis=1)
    if attack_view_type is None:
        from attack_eval import AttackView as attack_view_type
    result = attack_view_type(
        row_id=rid, unit=groups, x=x,
        components=names, wire_components=wire_names,
        component_columns=bindings,
        wire_constants=wire_constants,
        x_defense_aware=defense_aware,
        x_metadata=metadata,
    )
    return result.validate()
