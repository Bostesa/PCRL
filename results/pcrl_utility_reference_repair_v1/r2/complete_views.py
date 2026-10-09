"""Exact public wires and complete reader matrices for conditional masks.

This adapter accepts role features and restored public packages in memory. It
opens no role file or label, and all per-person matrix columns come from the
two actual recipient wires. Public identities remain literal wire constants.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

from results.pcrl_capacity_matched_feature_selection_v1.m4.views import (
    canonical_union_from_source,
)


REPO = Path(__file__).resolve().parents[3]
for source_dir in (
    REPO / "results/pcrl_method_portfolio_v1/teams/c",
    REPO / "results/pcrl_method_portfolio_v1/shared",
):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
from prototype import AdultGroupedSchema, release_pair  # noqa: E402
from attack_eval import AttackView  # noqa: E402


LOCAL_NUMERIC = ("selected_values", "probabilities", "decisions")
COALITION_NUMERIC = (
    "selected_values", "income_probabilities", "occupation_group_probabilities",
    "income_decisions", "occupation_group_decisions",
)
LOCAL_CONSTANTS = (
    "feature_names", "selected_groups", "mask_id", "mask_sha256",
    "task_package_sha256", "head_id", "calibrator_id", "flags",
)
COALITION_CONSTANTS = (
    "feature_names", "selected_groups", "income_groups",
    "occupation_group_groups", "income_mask_id", "occupation_group_mask_id",
    "income_mask_sha256", "occupation_group_mask_sha256",
    "income_task_package_sha256", "occupation_group_task_package_sha256",
    "income_head_id", "occupation_group_head_id", "income_calibrator_id",
    "occupation_group_calibrator_id", "flags",
)


def release_conditional_pair(
    source_record: Mapping[str, object], public_releases: Mapping[str, object],
) -> dict[str, dict]:
    """Deploy both selected-input packages, exposing their full public IDs."""
    if set(public_releases) != {"income", "occupation_group"}:
        raise ValueError("conditional release needs both purpose packages")
    first, second = (public_releases[name] for name in
                     ("income", "occupation_group"))
    wire = release_pair(
        np.asarray(source_record["X"]), tuple(source_record["feature_names"]),
        first, second,
    )
    coalition = wire["coalition"]
    for name, public in (("income", first), ("occupation_group", second)):
        coalition[f"{name}_mask_sha256"] = public.mask_sha256
        coalition[f"{name}_task_package_sha256"] = public.task_package_sha256
    return wire


def _probability(value: object, rows: int, classes: int, label: str) -> np.ndarray:
    p = np.asarray(value, dtype=np.float64)
    if (p.shape != (rows, classes) or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1))
            or not np.allclose(p.sum(axis=1), 1, rtol=1e-7, atol=1e-8)):
        raise ValueError(f"{label} probabilities are invalid")
    return p


def _decisions(value: object, probability: np.ndarray, label: str) -> np.ndarray:
    d = np.asarray(value)
    if (d.shape != (len(probability),) or not np.issubdtype(d.dtype, np.integer)
            or not np.array_equal(d, probability.argmax(axis=1))):
        raise ValueError(f"{label} decisions differ from probabilities")
    return d.astype(np.int64, copy=False)


def _verify_local(
    name: str, wire: Mapping[str, object], public: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if set(wire) != set(LOCAL_NUMERIC) | set(LOCAL_CONSTANTS):
        raise ValueError(f"{name} wire fields differ from conditional deployment")
    release = public.release
    if (public.purpose != name
            or tuple(wire["selected_groups"]) != public.selected_atoms
            or wire["mask_id"] != public.mask_id
            or wire["mask_sha256"] != public.mask_sha256
            or wire["task_package_sha256"] != public.task_package_sha256):
        raise ValueError(f"{name} mask or package identity differs")
    if (wire["head_id"] != public.head_id
            or wire["calibrator_id"] != public.calibrator_id):
        raise ValueError(f"{name} public head or calibrator identity differs")
    if wire["flags"] != ():
        raise ValueError("conditional flags need an explicit row-wise reader")
    selected = np.asarray(wire["selected_values"])
    classes = 2 if name == "income" else 6
    probability = _probability(wire["probabilities"], len(selected), classes,
                               name)
    decision = _decisions(wire["decisions"], probability, name)
    raw = _probability(release.head.predict_proba(selected.copy()),
                       len(selected), classes, f"{name} public head")
    replay = _probability(release.calibrator.transform(raw.copy()),
                          len(selected), classes, f"{name} public calibrator")
    if not np.array_equal(replay, probability):
        raise ValueError(f"{name} public selected-input head/CAL replay differs")
    return probability, decision, raw


def _matrix(
    name: str, wire: Mapping[str, object], row_id: np.ndarray,
    unit: np.ndarray, raw_values: np.ndarray,
    precal: tuple[np.ndarray, ...],
) -> AttackView:
    numeric = LOCAL_NUMERIC if name != "coalition" else COALITION_NUMERIC
    constants = LOCAL_CONSTANTS if name != "coalition" else COALITION_CONSTANTS
    blocks, spans = [], []
    start = 0
    for field in numeric:
        if field == "selected_values":
            block = raw_values
        elif "decisions" in field:
            classes = 2 if field in ("decisions", "income_decisions") and \
                name != "occupation_group" else 6
            d = np.asarray(wire[field], dtype=np.int64)
            block = np.eye(classes, dtype=np.float64)[d]
        else:
            block = np.asarray(wire[field], dtype=np.float64)
        blocks.append(block)
        end = start + block.shape[1]
        spans.append((field, start, end))
        start = end
    x = np.ascontiguousarray(np.concatenate(blocks, axis=1), dtype=np.float64)
    metadata = tuple((field, json.dumps(wire[field], sort_keys=True,
                                        separators=(",", ":")))
                     for field in constants)
    return AttackView(
        row_id=row_id.copy(), unit=unit.copy(), x=x,
        components=numeric, wire_components=numeric + constants,
        component_columns=tuple(spans), wire_constants=metadata,
        x_defense_aware=np.ascontiguousarray(
            np.concatenate((x, *precal), axis=1), dtype=np.float64),
        x_raw_only=raw_values,
    ).validate()


def build_conditional_attack_views(
    source_record: Mapping[str, object], wire: Mapping[str, Mapping[str, object]],
    public_releases: Mapping[str, object],
) -> tuple[dict[str, AttackView], object]:
    """Validate all three actual wires and expose every transmitted component."""
    if set(wire) != {"income", "occupation_group", "coalition"}:
        raise ValueError("conditional view requires two locals and coalition")
    if set(public_releases) != {"income", "occupation_group"}:
        raise ValueError("conditional view requires two public packages")
    schema = AdultGroupedSchema()
    union = canonical_union_from_source(source_record, wire, schema)
    first, decision_i, raw_i = _verify_local(
        "income", wire["income"], public_releases["income"])
    second, decision_o, raw_o = _verify_local(
        "occupation_group", wire["occupation_group"],
        public_releases["occupation_group"])
    coalition = wire["coalition"]
    if set(coalition) != set(COALITION_NUMERIC) | set(COALITION_CONSTANTS):
        raise ValueError("coalition wire fields differ from conditional deployment")
    expected = {
        "income_mask_id": wire["income"]["mask_id"],
        "occupation_group_mask_id": wire["occupation_group"]["mask_id"],
        "income_mask_sha256": wire["income"]["mask_sha256"],
        "occupation_group_mask_sha256": wire["occupation_group"]["mask_sha256"],
        "income_task_package_sha256": wire["income"]["task_package_sha256"],
        "occupation_group_task_package_sha256":
        wire["occupation_group"]["task_package_sha256"],
        "income_head_id": wire["income"]["head_id"],
        "occupation_group_head_id": wire["occupation_group"]["head_id"],
        "income_calibrator_id": wire["income"]["calibrator_id"],
        "occupation_group_calibrator_id": wire["occupation_group"]["calibrator_id"],
        "income_groups": wire["income"]["selected_groups"],
        "occupation_group_groups": wire["occupation_group"]["selected_groups"],
    }
    if any(coalition[key] != value for key, value in expected.items()):
        raise ValueError("coalition public identities differ from recipient wires")
    if coalition["flags"] != ():
        raise ValueError("coalition flags need an explicit row-wise reader")
    if (not np.array_equal(coalition["income_probabilities"], first)
            or not np.array_equal(coalition["occupation_group_probabilities"], second)
            or not np.array_equal(coalition["income_decisions"], decision_i)
            or not np.array_equal(coalition["occupation_group_decisions"], decision_o)):
        raise ValueError("coalition scores or decisions differ from recipients")
    x = np.asarray(source_record["X"])
    row_id, unit = union.row_id, union.unit
    selected = {
        name: np.ascontiguousarray(
            x[:, schema.column_indices(tuple(wire[name]["selected_groups"]))],
            dtype=np.float64)
        for name in ("income", "occupation_group")
    }
    selected["coalition"] = union.x
    views = {
        "income": _matrix("income", wire["income"], row_id, unit,
                          selected["income"], (raw_i,)),
        "occupation_group": _matrix("occupation_group", wire["occupation_group"],
                                    row_id, unit, selected["occupation_group"],
                                    (raw_o,)),
        "coalition": _matrix("coalition", coalition, row_id, unit,
                             selected["coalition"], (raw_i, raw_o)),
    }
    return views, union
