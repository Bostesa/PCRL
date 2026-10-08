"""Team C complete-view construction for S2's shared attacker/evaluator.

This module sees releases only. It never reads Adult data or any label and
does not choose attacks, reference identities, endpoints or thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from prototype import AdultGroupedSchema


@dataclass(frozen=True)
class AttackView:
    name: str
    raw_matrix: np.ndarray
    raw_names: tuple[str, ...]
    complete_matrix: np.ndarray
    complete_names: tuple[str, ...]
    public_metadata: Mapping[str, object]
    ignore_recipient_aliases: tuple[str, ...] = ()


def _array(value: object, *, name: str, rows: int | None = None) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be a numeric matrix")
    if rows is not None and arr.shape[0] != rows:
        raise ValueError(f"{name} has wrong number of people")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return np.asarray(arr, dtype=float)


def _vector(value: object, *, name: str, rows: int, classes: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != (rows,) or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must contain one integer decision per person")
    if np.any(arr < 0) or np.any(arr >= classes):
        raise ValueError(f"{name} has a decision outside the class vocabulary")
    return arr.astype(int, copy=True)


def _prob(value: object, *, name: str, rows: int, classes: int) -> np.ndarray:
    arr = _array(value, name=name, rows=rows)
    if arr.shape[1] != classes:
        raise ValueError(f"{name} has wrong class count")
    if np.any(arr < 0) or np.any(arr > 1) or not np.allclose(
        arr.sum(axis=1), 1.0, rtol=0, atol=1e-8,
    ):
        raise ValueError(f"{name} is not a probability matrix")
    return arr.copy()


def _features(view: Mapping[str, object], *, name: str) -> tuple[np.ndarray, tuple[str, ...]]:
    raw = _array(view["selected_values"], name=f"{name} raw selected values")
    names = tuple(str(n) for n in view["feature_names"])
    if len(names) != raw.shape[1] or len(names) != len(set(names)):
        raise ValueError(f"{name} feature names do not match selected values")
    if any(not n for n in names):
        raise ValueError(f"{name} has an empty feature name")
    return raw.copy(), names


def _metadata(view: Mapping[str, object], keys: tuple[str, ...]) -> dict[str, object]:
    out = {}
    for key in keys:
        if key not in view:
            raise ValueError(f"missing transmitted metadata {key}")
        value = view[key]
        if isinstance(value, np.ndarray):
            raise ValueError(f"metadata {key} needs an explicit per-person adapter")
        if key == "flags" and value != ():
            raise ValueError("per-person flags require an explicit row-wise attack adapter")
        out[key] = value
    return out


def _validate_group_names(
    schema: AdultGroupedSchema, view: Mapping[str, object],
    names: tuple[str, ...], *, label: str,
) -> tuple[str, ...]:
    if "selected_groups" not in view:
        raise ValueError(f"{label} omits selected group identity")
    groups = tuple(view["selected_groups"])
    normalized = schema.normalize_atoms(groups)
    if groups != normalized:
        raise ValueError(f"{label} groups are not in pinned canonical order")
    expected = tuple(schema.feature_names[i] for i in schema.column_indices(groups))
    if names != expected:
        raise ValueError(f"{label} names are not complete permitted atom blocks")
    if label != "coalition" and view.get("mask_id") != ("+".join(groups) or "none"):
        raise ValueError(f"{label} mask identity disagrees with selected groups")
    return groups


def _matrix(
    name: str, raw: np.ndarray, raw_names: tuple[str, ...],
    probabilities: tuple[tuple[str, np.ndarray], ...],
    decisions: tuple[tuple[str, np.ndarray, int], ...],
    metadata: Mapping[str, object], aliases: tuple[str, ...] = (),
) -> AttackView:
    arrays = [raw]
    names = list(raw_names)
    for prefix, prob in probabilities:
        arrays.append(prob)
        names.extend(f"{prefix}_probability_{j}" for j in range(prob.shape[1]))
    for prefix, decision, classes in decisions:
        onehot = np.eye(classes, dtype=float)[decision]
        arrays.append(onehot)
        names.extend(f"{prefix}_decision_{j}" for j in range(classes))
    # Global public identifiers have no person-level variation, but their
    # identity is retained in the attack contract rather than omitted.
    for key, value in metadata.items():
        arrays.append(np.ones((len(raw), 1), dtype=float))
        names.append(f"public:{key}={value!r}")
    complete = np.concatenate(arrays, axis=1)
    return AttackView(
        name, raw.copy(), raw_names, complete, tuple(names),
        dict(metadata), aliases,
    )


def _check_private_alignment(
    identity: Mapping[str, Mapping[str, object]], *, rows: int,
) -> None:
    if set(identity) != {"income", "occupation_group", "coalition"}:
        raise ValueError("private row alignment requires all three views")
    reference = None
    for name in ("income", "occupation_group", "coalition"):
        pair = identity[name]
        if set(pair) != {"row_id", "unit"}:
            raise ValueError(f"{name} private identity needs row_id and unit")
        row_id = np.asarray(pair["row_id"])
        unit = np.asarray(pair["unit"])
        if (
            row_id.shape != (rows,) or unit.shape != (rows,)
            or not np.issubdtype(row_id.dtype, np.integer)
            or not np.issubdtype(unit.dtype, np.integer)
            or len(np.unique(row_id)) != rows
        ):
            raise ValueError(f"{name} has invalid private row identity")
        values = (row_id.astype(np.int64), unit.astype(np.int64))
        if reference is None:
            reference = values
        elif not all(np.array_equal(a, b) for a, b in zip(reference, values)):
            raise ValueError("recipient and coalition private row order is misaligned")


def attack_views_from_release(
    release: Mapping[str, Mapping[str, object]],
    private_alignment: Mapping[str, Mapping[str, object]],
) -> dict[str, AttackView]:
    """Rebuild and validate complete recipient/coalition matrices.

    Ignore-recipient coalition routes are aliases of the already built local
    views; S2's attack bank may score them without duplicate model fits.
    Public head weights/calibrator tables are supplied separately to S2's
    defense-aware reader, keyed by the returned public identities.
    """
    if set(release) != {"income", "occupation_group", "coalition"}:
        raise ValueError("release must contain exactly two recipients and coalition")
    income, occupation, coalition = (
        release["income"], release["occupation_group"], release["coalition"]
    )
    raw_i, names_i = _features(income, name="income")
    raw_o, names_o = _features(occupation, name="occupation_group")
    raw_c, names_c = _features(coalition, name="coalition")
    schema = AdultGroupedSchema()
    groups_i = _validate_group_names(schema, income, names_i, label="income")
    groups_o = _validate_group_names(schema, occupation, names_o, label="occupation_group")
    groups_c = _validate_group_names(schema, coalition, names_c, label="coalition")
    if set(groups_c) != set(groups_i) | set(groups_o):
        raise ValueError("coalition selected groups do not equal local union")
    rows = len(raw_i)
    if len(raw_o) != rows or len(raw_c) != rows:
        raise ValueError("recipient and coalition rows are not aligned")
    _check_private_alignment(private_alignment, rows=rows)
    merged: dict[str, np.ndarray] = {}
    for raw, names in ((raw_i, names_i), (raw_o, names_o)):
        for j, feature in enumerate(names):
            if feature in merged and not np.array_equal(merged[feature], raw[:, j]):
                raise ValueError("local views disagree on a shared raw feature")
            merged[feature] = raw[:, j]
    if set(names_c) != set(merged) or any(
        not np.array_equal(raw_c[:, j], merged[feature])
        for j, feature in enumerate(names_c)
    ):
        raise ValueError("coalition raw union differs from recipient raw values")

    pi = _prob(income["probabilities"], name="income probabilities", rows=rows, classes=2)
    po = _prob(occupation["probabilities"], name="occupation_group probabilities", rows=rows, classes=6)
    di = _vector(income["decisions"], name="income decisions", rows=rows, classes=2)
    do = _vector(occupation["decisions"], name="occupation_group decisions", rows=rows, classes=6)
    cpi = _prob(coalition["income_probabilities"], name="coalition income probabilities", rows=rows, classes=2)
    cpo = _prob(coalition["occupation_group_probabilities"], name="coalition occupation probabilities", rows=rows, classes=6)
    cdi = _vector(coalition["income_decisions"], name="coalition income decisions", rows=rows, classes=2)
    cdo = _vector(coalition["occupation_group_decisions"], name="coalition occupation decisions", rows=rows, classes=6)
    if not np.array_equal(cpi, pi) or not np.array_equal(cpo, po):
        raise ValueError("coalition probabilities differ from recipient probabilities")
    if not np.array_equal(cdi, di) or not np.array_equal(cdo, do):
        raise ValueError("coalition decisions differ from recipient decisions")

    metadata_i = _metadata(income, ("mask_id", "head_id", "calibrator_id", "flags"))
    metadata_o = _metadata(occupation, ("mask_id", "head_id", "calibrator_id", "flags"))
    metadata_c = _metadata(coalition, (
        "income_mask_id", "occupation_group_mask_id", "income_head_id",
        "occupation_group_head_id", "income_calibrator_id",
        "occupation_group_calibrator_id", "flags",
    ))
    if (
        metadata_c["income_mask_id"] != metadata_i["mask_id"]
        or metadata_c["occupation_group_mask_id"] != metadata_o["mask_id"]
        or metadata_c["income_head_id"] != metadata_i["head_id"]
        or metadata_c["occupation_group_head_id"] != metadata_o["head_id"]
        or metadata_c["income_calibrator_id"] != metadata_i["calibrator_id"]
        or metadata_c["occupation_group_calibrator_id"] != metadata_o["calibrator_id"]
    ):
        raise ValueError("coalition public metadata differs from recipient metadata")
    return {
        "income": _matrix(
            "income", raw_i, names_i, (("income", pi),), (("income", di, 2),), metadata_i,
        ),
        "occupation_group": _matrix(
            "occupation_group", raw_o, names_o, (("occupation_group", po),),
            (("occupation_group", do, 6),), metadata_o,
        ),
        "coalition": _matrix(
            "coalition", raw_c, names_c,
            (("income", pi), ("occupation_group", po)),
            (("income", di, 2), ("occupation_group", do, 6)),
            metadata_c, ("income", "occupation_group"),
        ),
    }
