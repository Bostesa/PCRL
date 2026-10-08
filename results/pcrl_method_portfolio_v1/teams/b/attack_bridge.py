"""Construct S2 AttackView objects from Team B's actual saved wire releases.

The fixed numeric encoding is chosen without SEX labels or INNER outcomes.
Public linear heads may recompute logits from released z; their parameters do
not reveal a private per-person provider input. Every actual wire scalar has
one named component and a disjoint span of attack columns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json

import numpy as np
import re
import torch
import torch.nn.functional as F

from attack_views import CLASSES, aligned_complete_views
from bottleneck import model_fingerprint, purpose_model_fingerprint, schema_fingerprint

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
    purpose: str, head: tuple[np.ndarray, np.ndarray], calibrator: Mapping[str, object],
) -> np.ndarray:
    weight = np.asarray(head[0], dtype=np.float32)
    bias = np.asarray(head[1], dtype=np.float32)
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
    # The deployment head runs one float32 row at a time. Repeating that
    # operation exactly rejects small planted payloads in transmitted logits.
    with torch.no_grad():
        projected = np.stack([
            F.linear(torch.from_numpy(np.asarray(z, dtype=np.float32)).reshape(1, -1),
                     torch.from_numpy(weight), torch.from_numpy(bias))
            .squeeze(0).numpy() for z in latent
        ]).astype(np.float64)
    if not np.array_equal(projected, transmitted):
        raise ValueError("Released logits do not depend on the public head and released z")
    temperature = np.asarray(
        [views[purpose]["metadata"]["temperature"] for _, views in rows], dtype=np.float64
    )
    if temperature.shape != (len(rows),) or not np.isfinite(temperature).all() or np.any(temperature <= 0):
        raise ValueError("A positive transmitted calibration temperature is required")
    if np.unique(temperature).size != 1:
        raise ValueError("Global temperature must be constant across people")
    if not np.isclose(temperature[0], 1.0 / float(calibrator["alpha"]), rtol=0, atol=1e-12):
        raise ValueError("Released temperature differs from fitted S2 calibrator")
    from common_eval import apply_global_temperature
    # The fitted S2 law first clips raw probabilities to 1e-12. Scaling
    # logits directly would score a different release on saturated rows.
    # ReleaseAdapter calls Torch softmax on one float64 logit row, then S2's
    # exact clipped-log family. SavedRelease preserves its float64 output.
    with torch.no_grad():
        raw = np.stack([
            torch.softmax(torch.from_numpy(np.asarray(row, dtype=np.float32))
                          .to(torch.float64).reshape(1, -1), dim=1)
            .squeeze(0).numpy() for row in transmitted
        ])
    expected = apply_global_temperature(raw, dict(calibrator))
    probabilities = np.asarray(
        [views[purpose]["probabilities"] for _, views in rows], dtype=np.float64
    )
    decisions = np.asarray([views[purpose]["decision"] for _, views in rows], dtype=np.int64)
    if (probabilities.shape != expected.shape
            or not np.array_equal(probabilities, expected)
            or not np.array_equal(decisions, np.argmax(expected, axis=1))):
        raise ValueError("Released probability or decision bypasses the public protected head")
    margin = projected - projected.max(axis=1, keepdims=True)
    return np.concatenate((projected, margin), axis=1)


def build_attack_view(
    rows: Sequence[tuple[str, Mapping[str, Mapping[str, object]]]],
    *, view: str, row_id: Sequence[int], unit: Sequence[int],
    public_model,
    expected_source_id: str,
    calibration_records: Mapping[str, Mapping[str, object]],
    calibration_record_sha256: Mapping[str, str],
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
    if set(calibration_records) != set(CLASSES) or set(calibration_record_sha256) != set(CLASSES):
        raise ValueError("Both fitted calibration records and hashes are required")
    for purpose in CLASSES:
        canonical = json.dumps(calibration_records[purpose], sort_keys=True,
                               separators=(",", ":"), allow_nan=False) + "\n"
        if (calibration_records[purpose].get("task") != purpose
                or hashlib.sha256(canonical.encode()).hexdigest()
                != calibration_record_sha256[purpose]):
            raise ValueError("Fitted S2 calibration record identity differs")
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
                    or metadata["source_id"] != expected_source_id
                    or metadata.get("purpose_model_sha256") != purpose_model_fingerprint(public_model, purpose)
                    or metadata.get("sampler_domain") != "TeamBPurposeDrawV1"
                    or metadata.get("calibration_record_sha256") != calibration_record_sha256[purpose]):
                raise ValueError("Released model/schema/source differs from public artifact")
    flattened = aligned_complete_views(rows)[view]
    x, names, bindings, metadata, wire_names, wire_constants = _encode(flattened)
    relevant = ("income", "occupation") if view == "pair" else (view,)
    heads = {
        purpose: (public_model.heads[purpose].weight.detach().cpu().numpy(),
                  public_model.heads[purpose].bias.detach().cpu().numpy())
        for purpose in relevant
    }
    replay = np.concatenate([_public_head_projection(rows, purpose, heads[purpose],
                                                     calibration_records[purpose])
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
