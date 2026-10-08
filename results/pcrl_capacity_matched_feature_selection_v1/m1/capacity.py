"""Pure source-group and capacity geometry for the focused feature study.

This module accepts an admitted ordered feature-name array; it does not load
Adult rows, task labels, SEX, or prior scientific outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from itertools import combinations, product
from typing import Iterable

import numpy as np


SOURCE_SCHEMA_SHA256 = "7101f2fb37b623a08221c32b33eedf132cf0c4c376bb4042ab9cf16da08ad975"
NUMERIC_SOURCE_FIELDS = (
    "age", "education-num", "capital-gain", "capital-loss", "hours-per-week"
)
CATEGORICAL_SOURCE_FIELDS = (
    "workclass", "education", "marital-status", "relationship", "native-country"
)
ONE_HOT_WIDTHS = {
    "workclass": 8,
    "education": 16,
    "marital-status": 7,
    "relationship": 6,
    "native-country": 41,
}
ATOM_ORDER = (
    "age", "education_fact", "capital-gain", "capital-loss",
    "hours-per-week", "workclass", "marital-status", "relationship",
    "native-country",
)
STANDALONE_NUMERIC_ATOMS = (
    "age", "capital-gain", "capital-loss", "hours-per-week"
)
CATEGORICAL_ATOMS = (
    "education_fact", "workclass", "marital-status", "relationship",
    "native-country"
)
PRIMARY_PROFILE = {
    "numeric_groups": 4,
    "categorical_groups": 4,
    "encoded_columns_max": 77,
}
HISTORICAL_FINAL_DELETIONS = {
    "native-country": ("C-SIMPLE-TASK-SHARED1", "C-SIMPLE-TASK-PAIR1"),
    "relationship": ("C-SIMPLE-REL0", "C-SIMPLE-INDEP1", "C-SIMPLE-SEQ-IO1"),
    "marital-status": ("C-SIMPLE-COMMON1", "C-SIMPLE-INDEP1",
                       "C-SIMPLE-SEQ-IO1", "C-SIMPLE-SEQ-OI1",
                       "C-SIMPLE-TASK-PAIR1"),
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value: object) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def same_wire_bits(first: np.ndarray, second: np.ndarray) -> bool:
    """Compare exact emitted numeric bytes, including signed zero and NaN bits."""
    a, b = np.asarray(first), np.asarray(second)
    return (a.shape == b.shape and a.dtype == b.dtype and
            np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes())


def feature_name_hash(names: Iterable[str]) -> str:
    return sha256("\n".join(tuple(str(name) for name in names)).encode()).hexdigest()


def _ordered_atoms(atoms: Iterable[str]) -> tuple[str, ...]:
    atoms = tuple(atoms)
    if len(atoms) != len(set(atoms)) or set(atoms) - set(ATOM_ORDER):
        raise ValueError("unknown or duplicate source atom")
    return tuple(atom for atom in ATOM_ORDER if atom in atoms)


def source_groups(feature_names: Iterable[str]) -> dict[str, dict]:
    names = tuple(str(name) for name in feature_names)
    if len(names) != 83 or feature_name_hash(names) != SOURCE_SCHEMA_SHA256:
        raise ValueError("ordered source feature schema differs from admitted 83 columns")
    if names[:5] != NUMERIC_SOURCE_FIELDS:
        raise ValueError("numeric source field order changed")
    fields: dict[str, tuple[int, ...]] = {}
    position = 5
    for field in CATEGORICAL_SOURCE_FIELDS:
        width = ONE_HOT_WIDTHS[field]
        block = names[position:position + width]
        if len(block) != width or any(not n.startswith(field + "=") for n in block):
            raise ValueError("categorical one-hot block is partial or reordered")
        if len(set(block)) != width:
            raise ValueError("duplicate one-hot category")
        fields[field] = tuple(range(position, position + width))
        position += width
    if position != 83:
        raise ValueError("unexpected fields after categorical blocks")
    groups: dict[str, dict] = {}
    for atom in ATOM_ORDER:
        if atom == "education_fact":
            columns = (1,) + fields["education"]
            source_fields = ("education-num", "education")
            kind = "categorical_with_duplicate_numeric_code"
        elif atom in STANDALONE_NUMERIC_ATOMS:
            columns = (names.index(atom),)
            source_fields = (atom,)
            kind = "numeric"
        else:
            columns = fields[atom]
            source_fields = (atom,)
            kind = "categorical"
        groups[atom] = {
            "atom": atom,
            "source_fields": list(source_fields),
            "kind": kind,
            "column_indices": list(columns),
            "encoded_columns": len(columns),
        }
    all_indices = [index for group in groups.values() for index in group["column_indices"]]
    if sorted(all_indices) != list(range(83)) or len(set(all_indices)) != 83:
        raise ValueError("source atoms do not partition the admitted columns")
    return groups


def mask_record(groups: dict[str, dict], atoms: Iterable[str]) -> dict:
    selected = _ordered_atoms(atoms)
    columns = sorted(index for atom in selected for index in groups[atom]["column_indices"])
    numeric = sum(atom in STANDALONE_NUMERIC_ATOMS for atom in selected)
    categorical = sum(atom in CATEGORICAL_ATOMS for atom in selected)
    missing = sorted(set(ATOM_ORDER) - set(selected))
    if len(missing) == 1:
        mask_id = "M_DROP_" + missing[0].upper().replace("-", "_")
    else:
        mask_id = "M_" + digest({"atoms": selected})[:16].upper()
    identity = {
        "source_schema_sha256": SOURCE_SCHEMA_SHA256,
        "atoms": list(selected),
        "column_indices": columns,
    }
    history = sorted({arm for atom in missing for arm in HISTORICAL_FINAL_DELETIONS.get(atom, ())})
    return {
        "mask_id": mask_id,
        **identity,
        "numeric_group_count": numeric,
        "categorical_group_count": categorical,
        "source_group_count": len(selected),
        "encoded_columns": len(columns),
        "source_fields": sorted({field for atom in selected for field in groups[atom]["source_fields"]}),
        "capacity_signature": dict(PRIMARY_PROFILE),
        "mask_sha256": digest(identity),
        "historical_exposure": {"fit_proxy_search": True, "final_screen_arms": history},
    }


def feasible_masks(groups: dict[str, dict], profile: dict | None = None) -> list[dict]:
    profile = dict(profile or PRIMARY_PROFILE)
    if set(profile) != {"numeric_groups", "categorical_groups", "encoded_columns_max"}:
        raise ValueError("capacity profile has unexpected fields")
    if profile["numeric_groups"] != len(STANDALONE_NUMERIC_ATOMS):
        raise ValueError("this primary profile retains all standalone numeric groups")
    out = []
    for cats in combinations(CATEGORICAL_ATOMS, profile["categorical_groups"]):
        record = mask_record(groups, STANDALONE_NUMERIC_ATOMS + cats)
        if record["encoded_columns"] <= profile["encoded_columns_max"]:
            record["capacity_signature"] = profile
            out.append(record)
    return sorted(out, key=lambda record: record["mask_id"])


def pair_record(first: dict, second: dict) -> dict:
    if first["source_schema_sha256"] != second["source_schema_sha256"]:
        raise ValueError("recipient masks use different source schemas")
    a, b = set(first["column_indices"]), set(second["column_indices"])
    union_columns = sorted(a | b)
    union_atoms = _ordered_atoms(set(first["atoms"]) | set(second["atoms"]))
    overlap_atoms = _ordered_atoms(set(first["atoms"]) & set(second["atoms"]))
    identity = {
        "source_schema_sha256": SOURCE_SCHEMA_SHA256,
        "union_atoms": list(union_atoms),
        "union_column_indices": union_columns,
    }
    return {
        "pair_id": first["mask_id"] + "__" + second["mask_id"],
        "recipient1_mask_sha256": first["mask_sha256"],
        "recipient2_mask_sha256": second["mask_sha256"],
        "recipient_encoded_columns": [len(a), len(b)],
        "union_atoms": list(union_atoms),
        "union_group_count": len(union_atoms),
        "union_column_indices": union_columns,
        "union_encoded_columns": len(union_columns),
        "overlap_atoms": list(overlap_atoms),
        "overlap_group_count": len(overlap_atoms),
        "overlap_encoded_columns": len(a & b),
        "union_sha256": digest(identity),
    }


def capacity_geometry(feature_names: Iterable[str]) -> dict:
    groups = source_groups(feature_names)
    masks = feasible_masks(groups)
    pairs = [pair_record(a, b) for a, b in product(masks, repeat=2)]
    exact_width_multiplicity: dict[str, int] = {}
    for mask in masks:
        key = str(mask["encoded_columns"])
        exact_width_multiplicity[key] = exact_width_multiplicity.get(key, 0) + 1
    return {
        "schema": "pcrl-focused-capacity-geometry-v1",
        "source_schema_sha256": SOURCE_SCHEMA_SHA256,
        "source_groups": groups,
        "primary_profile": PRIMARY_PROFILE,
        "mask_count": len(masks),
        "pair_count": len(pairs),
        "exact_width_multiplicity": exact_width_multiplicity,
        "masks": masks,
        "pairs": pairs,
    }


def validate_release_input(full_x: np.ndarray, feature_names: Iterable[str], mask: dict,
                           disclosed_x: np.ndarray) -> tuple[int, ...]:
    groups = source_groups(feature_names)
    full = np.asarray(full_x)
    disclosed = np.asarray(disclosed_x)
    if full.ndim != 2 or full.shape[1] != 83 or full.dtype != np.float32:
        raise ValueError("full input must be admitted float32 83-column matrix")
    columns = tuple(mask["column_indices"])
    if mask["source_schema_sha256"] != SOURCE_SCHEMA_SHA256:
        raise ValueError("mask source schema differs")
    if columns != tuple(sorted(set(columns))) or any(i < 0 or i >= 83 for i in columns):
        raise ValueError("mask columns are not a unique source-order subset")
    atoms = _ordered_atoms(mask["atoms"])
    if tuple(mask["atoms"]) != atoms:
        raise ValueError("mask atoms are not in canonical source-group order")
    expected_columns = tuple(sorted(i for atom in atoms for i in groups[atom]["column_indices"]))
    if columns != expected_columns:
        raise ValueError("mask splits, omits, or adds an encoded source group")
    identity = {"source_schema_sha256": SOURCE_SCHEMA_SHA256,
                "atoms": mask["atoms"], "column_indices": list(columns)}
    if mask["mask_sha256"] != digest(identity):
        raise ValueError("mask identity hash mismatch")
    if disclosed.shape != (full.shape[0], len(columns)) or disclosed.dtype != full.dtype:
        raise ValueError("disclosed matrix shape or dtype mismatch")
    if not same_wire_bits(disclosed, full[:, columns]):
        raise ValueError("released raw values differ from selected source columns")
    return columns


def assert_role_alignment(first_row_id: np.ndarray, first_unit: np.ndarray,
                          second_row_id: np.ndarray, second_unit: np.ndarray) -> None:
    """Check private evaluation custody IDs, which are never public wire fields."""
    a, ua = np.asarray(first_row_id), np.asarray(first_unit)
    b, ub = np.asarray(second_row_id), np.asarray(second_unit)
    if a.dtype != np.int64 or b.dtype != np.int64 or ua.dtype != np.int64 or ub.dtype != np.int64:
        raise ValueError("role alignment IDs must be int64")
    if a.ndim != 1 or ua.shape != a.shape or b.shape != a.shape or ub.shape != a.shape:
        raise ValueError("recipient role identity shapes differ")
    if not np.array_equal(a, b) or not np.array_equal(ua, ub):
        raise ValueError("recipient releases have different ordered row/group identities")


def reconstruct_union(first: dict, first_x: np.ndarray, second: dict, second_x: np.ndarray,
                      expected_full_x: np.ndarray | None = None) -> tuple[tuple[int, ...], np.ndarray]:
    a, b = tuple(first["column_indices"]), tuple(second["column_indices"])
    x1, x2 = np.asarray(first_x), np.asarray(second_x)
    if x1.ndim != 2 or x2.ndim != 2 or x1.shape[0] != x2.shape[0]:
        raise ValueError("recipient releases do not align by row")
    if x1.shape[1] != len(a) or x2.shape[1] != len(b) or x1.dtype != x2.dtype:
        raise ValueError("recipient release matrix differs from mask")
    lookup = {index: x1[:, j] for j, index in enumerate(a)}
    for j, index in enumerate(b):
        if index in lookup and not same_wire_bits(lookup[index], x2[:, j]):
            raise ValueError("recipients disagree on an overlapping disclosed source column")
        lookup[index] = x2[:, j]
    columns = tuple(sorted(lookup))
    union = np.stack([lookup[index] for index in columns], axis=1)
    if expected_full_x is not None:
        full = np.asarray(expected_full_x)
        if full.shape[0] != union.shape[0] or not same_wire_bits(union, full[:, columns]):
            raise ValueError("canonical raw union differs from admitted input")
    return columns, union
