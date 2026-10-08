"""Pure, bounded shared mask bank for the focused feature study.

M1 supplies the complete capacity-feasible mask inventory.  This module does
not use labels, search-reader outcomes, or INNER_SELECTION to alter it.  The
current primary profile has exactly five feasible one-categorical-deletion
masks, so every rule receives every mask and all 25 ordered assignments.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence


ATOM_ORDER = (
    "age", "education_fact", "capital-gain", "capital-loss",
    "hours-per-week", "workclass", "marital-status", "relationship",
    "native-country",
)
CATEGORICAL = frozenset((
    "education_fact", "workclass", "marital-status", "relationship",
    "native-country",
))


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def _full_sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


@dataclass(frozen=True)
class MaskRecord:
    mask_id: str
    atoms: tuple[str, ...]
    column_indices: tuple[int, ...]
    numeric_group_count: int
    categorical_group_count: int
    source_group_count: int
    encoded_columns: int
    source_fields: tuple[str, ...]
    source_schema_sha256: str
    capacity_signature: tuple[int, int, int]
    mask_sha256: str
    fit_proxy_exposed: bool
    historical_final_screen_arms: tuple[str, ...]

    @classmethod
    def from_m1(cls, raw: Mapping[str, object]) -> "MaskRecord":
        required = {
            "mask_id", "atoms", "column_indices", "numeric_group_count",
            "categorical_group_count", "source_group_count",
            "encoded_columns", "source_fields", "source_schema_sha256",
            "capacity_signature", "mask_sha256", "historical_exposure",
        }
        if set(raw) != required:
            raise ValueError("M1 mask record keys differ from frozen contract")
        cap = raw["capacity_signature"]
        if not isinstance(cap, Mapping) or set(cap) != {
            "numeric_groups", "categorical_groups", "encoded_columns_max",
        }:
            raise ValueError("capacity signature keys differ from frozen contract")
        history = raw["historical_exposure"]
        if (not isinstance(history, Mapping)
                or set(history) != {"fit_proxy_search", "final_screen_arms"}
                or type(history["fit_proxy_search"]) is not bool
                or not isinstance(history["final_screen_arms"], list)
                or any(not isinstance(arm, str) or not arm
                       for arm in history["final_screen_arms"])):
            raise ValueError("historical exposure record differs from M1 contract")
        mask = cls(
            str(raw["mask_id"]), tuple(raw["atoms"]),
            tuple(raw["column_indices"]), int(raw["numeric_group_count"]),
            int(raw["categorical_group_count"]), int(raw["source_group_count"]),
            int(raw["encoded_columns"]), tuple(raw["source_fields"]),
            str(raw["source_schema_sha256"]),
            (int(cap["numeric_groups"]), int(cap["categorical_groups"]),
             int(cap["encoded_columns_max"])),
            str(raw["mask_sha256"]), history["fit_proxy_search"],
            tuple(history["final_screen_arms"]),
        )
        if (not mask.mask_id or "|" in mask.mask_id
                or "__" in mask.mask_id
                or not _full_sha(mask.source_schema_sha256)
                or not _full_sha(mask.mask_sha256)):
            raise ValueError("mask identity or exposure field is invalid")
        if (mask.source_fields != tuple(sorted(set(mask.source_fields)))
                or mask.historical_final_screen_arms !=
                tuple(sorted(set(mask.historical_final_screen_arms)))):
            raise ValueError("source fields or historical arms are not canonical")
        if (mask.atoms != tuple(atom for atom in ATOM_ORDER if atom in mask.atoms)
                or len(mask.atoms) != len(set(mask.atoms))
                or not set(mask.atoms) <= set(ATOM_ORDER)):
            raise ValueError("mask atoms are not in source order")
        if (mask.column_indices != tuple(sorted(set(mask.column_indices)))
                or any(type(index) is not int or index < 0 or index >= 83
                       for index in mask.column_indices)):
            raise ValueError("mask columns are not unique 83-schema indices")
        if mask.encoded_columns != len(mask.column_indices):
            raise ValueError("mask encoded column count differs from selected columns")
        identity = {
            "source_schema_sha256": mask.source_schema_sha256,
            "atoms": mask.atoms, "column_indices": mask.column_indices,
        }
        if mask.mask_sha256 != _sha(identity):
            raise ValueError("mask SHA differs from M1 canonical identity")
        if (mask.numeric_group_count != sum(a not in CATEGORICAL for a in mask.atoms)
                or mask.categorical_group_count != sum(a in CATEGORICAL for a in mask.atoms)
                or mask.source_group_count != len(mask.atoms)):
            raise ValueError("mask source group counts differ from atoms")
        numeric, categorical, width_cap = mask.capacity_signature
        if (mask.numeric_group_count != numeric
                or mask.categorical_group_count != categorical
                or mask.encoded_columns > width_cap):
            raise ValueError("mask violates recipient capacity signature")
        return mask

    def public_record(self) -> dict:
        return {
            "mask_id": self.mask_id, "atoms": self.atoms,
            "column_indices": self.column_indices,
            "numeric_group_count": self.numeric_group_count,
            "categorical_group_count": self.categorical_group_count,
            "source_group_count": self.source_group_count,
            "encoded_columns": self.encoded_columns,
            "source_fields": self.source_fields,
            "source_schema_sha256": self.source_schema_sha256,
            "capacity_signature": {
                "numeric_groups": self.capacity_signature[0],
                "categorical_groups": self.capacity_signature[1],
                "encoded_columns_max": self.capacity_signature[2],
            },
            "mask_sha256": self.mask_sha256,
            "historical_exposure": {
                "fit_proxy_search": self.fit_proxy_exposed,
                "final_screen_arms": self.historical_final_screen_arms,
            },
        }


@dataclass(frozen=True)
class PairGeometry:
    pair_id: str
    income_mask_id: str
    occupation_mask_id: str
    union_atoms: tuple[str, ...]
    union_columns: tuple[int, ...]
    union_group_count: int
    union_encoded_columns: int
    union_sha256: str


@dataclass(frozen=True)
class SharedBank:
    masks: tuple[MaskRecord, ...]
    pairs: tuple[PairGeometry, ...]
    source_schema_sha256: str
    capacity_signature: tuple[int, int, int]
    bank_sha256: str

    def mask(self, mask_id: str) -> MaskRecord:
        matches = [mask for mask in self.masks if mask.mask_id == mask_id]
        if len(matches) != 1:
            raise KeyError(mask_id)
        return matches[0]


def build_shared_bank(
    feasible_masks: Sequence[Mapping[str, object]], *,
    max_per_purpose: int = 8,
    expected_feasible_count: int = 5,
) -> SharedBank:
    """Admit the entire bounded M1 inventory, with no outcome ranking."""
    if not 1 <= max_per_purpose <= 8:
        raise ValueError("focused bank ceiling must be between one and eight")
    records = tuple(MaskRecord.from_m1(raw) for raw in feasible_masks)
    if len(records) != expected_feasible_count:
        raise ValueError("M1 inventory does not contain the frozen five feasible masks")
    if len(records) > max_per_purpose:
        raise ValueError("capacity inventory exceeds the frozen bank ceiling")
    masks = tuple(sorted(records, key=lambda mask: mask.mask_sha256))
    if (len({mask.mask_id for mask in masks}) != len(masks)
            or len({mask.mask_sha256 for mask in masks}) != len(masks)
            or len({mask.atoms for mask in masks}) != len(masks)):
        raise ValueError("duplicate mask identity or selected group set")
    schemas = {mask.source_schema_sha256 for mask in masks}
    capacities = {mask.capacity_signature for mask in masks}
    if len(schemas) != 1 or len(capacities) != 1:
        raise ValueError("shared bank masks disagree on schema or capacity")
    if capacities != {(4, 4, 77)}:
        raise ValueError("M1 inventory differs from primary 4/4/77 capacity")
    removed = tuple(set(ATOM_ORDER) - set(mask.atoms) for mask in masks)
    if (any(len(missing) != 1 for missing in removed)
            or {next(iter(missing)) for missing in removed} != CATEGORICAL):
        raise ValueError("primary bank requires all five categorical deletions")
    pairs = []
    for income in masks:
        for occupation in masks:
            union_atoms = tuple(atom for atom in ATOM_ORDER
                                if atom in set(income.atoms) | set(occupation.atoms))
            union_columns = tuple(sorted(set(income.column_indices)
                                         | set(occupation.column_indices)))
            union_sha = _sha({
                "source_schema_sha256": income.source_schema_sha256,
                "union_atoms": union_atoms,
                "union_column_indices": union_columns,
            })
            pairs.append(PairGeometry(
                f"{income.mask_id}__{occupation.mask_id}",
                income.mask_id, occupation.mask_id, union_atoms, union_columns,
                len(union_atoms), len(union_columns), union_sha,
            ))
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("shared bank has duplicate pair IDs")
    payload = {
        "schema": "pcrl-focused-shared-bank-v1",
        "masks": [mask.public_record() for mask in masks],
        "pairs": [vars(pair) for pair in pairs],
    }
    return SharedBank(
        masks, tuple(pairs), next(iter(schemas)), next(iter(capacities)),
        _sha(payload),
    )
