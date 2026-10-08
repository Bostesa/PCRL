"""Independent, metadata-only capacity and raw-union arithmetic for M5.

This module imports no selector, scorer, role loader, or Team C implementation.
It accepts a hash-bound ordered source-column partition supplied by admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Mask:
    atoms: tuple[str, ...]
    columns: tuple[int, ...]

    @property
    def encoded_columns(self) -> int:
        return len(self.columns)


def validate_partition(groups: Mapping[str, Sequence[int]], feature_count: int) -> None:
    if type(feature_count) is not int or feature_count < 1 or not groups:
        raise ValueError("positive source feature count and groups required")
    if any(not name or not isinstance(name, str) for name in groups):
        raise ValueError("invalid source group name")
    columns: list[int] = []
    for members in groups.values():
        if not members or any(type(i) is not int for i in members):
            raise ValueError("each source group needs integer columns")
        if len(set(members)) != len(members):
            raise ValueError("duplicate column inside a source group")
        columns.extend(members)
    if len(columns) != feature_count or set(columns) != set(range(feature_count)):
        raise ValueError("groups must partition every ordered source column exactly once")


def enumerate_masks(
    groups: Mapping[str, Sequence[int]],
    numeric_atoms: Sequence[str],
    categorical_atoms: Sequence[str],
    numeric_count: int,
    categorical_count: int,
    max_encoded: int,
) -> tuple[Mask, ...]:
    """Enumerate exact semantic counts under an inclusive encoded-width cap."""
    numeric, categorical = tuple(numeric_atoms), tuple(categorical_atoms)
    if (len(set(numeric)) != len(numeric) or len(set(categorical)) != len(categorical)
            or set(numeric) & set(categorical)
            or set(numeric) | set(categorical) != set(groups)):
        raise ValueError("numeric/categorical atoms must partition source groups")
    if (type(numeric_count) is not int or type(categorical_count) is not int
            or type(max_encoded) is not int or numeric_count < 0
            or categorical_count < 0 or max_encoded < 1):
        raise ValueError("invalid capacity")
    feature_count = sum(len(columns) for columns in groups.values())
    validate_partition(groups, feature_count)
    order = tuple(groups)
    masks: list[Mask] = []
    for nums in combinations(numeric, numeric_count):
        for cats in combinations(categorical, categorical_count):
            chosen = set(nums) | set(cats)
            columns = tuple(sorted(i for name in chosen for i in groups[name]))
            if len(columns) <= max_encoded:
                masks.append(Mask(tuple(name for name in order if name in chosen), columns))
    return tuple(masks)


def pair_union(income: Mask, occupation: Mask) -> tuple[int, ...]:
    """Canonical raw-column union in original source-column order."""
    return tuple(sorted(set(income.columns) | set(occupation.columns)))
