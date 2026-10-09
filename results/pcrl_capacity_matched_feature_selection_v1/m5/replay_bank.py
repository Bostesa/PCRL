"""Reconstruct the frozen five-mask bank from M1 public geometry, independently."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from .capacity_geometry import Mask as RawMask, enumerate_masks, pair_union


ORDER = ("age", "education_fact", "capital-gain", "capital-loss",
         "hours-per-week", "workclass", "marital-status", "relationship",
         "native-country")
CATEGORICAL = ("education_fact", "workclass", "marital-status",
               "relationship", "native-country")


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Mask:
    mask_id: str
    atoms: tuple[str, ...]
    column_indices: tuple[int, ...]
    mask_sha256: str
    capacity_signature: tuple[int, int, int]
    public: dict


@dataclass(frozen=True)
class Pair:
    pair_id: str
    income_mask_id: str
    occupation_mask_id: str
    union_atoms: tuple[str, ...]
    union_columns: tuple[int, ...]
    union_sha256: str


@dataclass(frozen=True)
class Bank:
    masks: tuple[Mask, ...]
    pairs: tuple[Pair, ...]
    bank_sha256: str

    def mask(self, mask_id: str) -> Mask:
        found = [row for row in self.masks if row.mask_id == mask_id]
        if len(found) != 1:
            raise KeyError(mask_id)
        return found[0]


def reconstruct(path: Path) -> Bank:
    spec = json.loads(path.read_text())
    if (spec.get("schema") != "pcrl-focused-capacity-geometry-v1"
            or spec.get("mask_count") != 5 or spec.get("pair_count") != 25
            or spec.get("primary_profile") != {"numeric_groups": 4,
               "categorical_groups": 4, "encoded_columns_max": 77}
            or set(spec.get("source_groups", {})) != set(ORDER)):
        raise PermissionError("M5 public five-mask capacity contract differs")
    groups = {name: tuple(spec["source_groups"][name]["column_indices"])
              for name in ORDER}
    numeric = tuple(name for name in ORDER if name not in CATEGORICAL)
    enumerated = enumerate_masks(groups, numeric, CATEGORICAL, 4, 4, 77)
    if len(enumerated) != 5 or len(spec["masks"]) != 5:
        raise PermissionError("M5 independent mask inventory differs")
    source_sha = spec["source_schema_sha256"]
    masks = []
    for raw in spec["masks"]:
        atoms = tuple(raw["atoms"])
        columns = tuple(raw["column_indices"])
        if ((atoms, columns) not in {(row.atoms, row.columns) for row in enumerated}
                or atoms != tuple(name for name in ORDER if name in atoms)
                or raw["source_schema_sha256"] != source_sha
                or raw["mask_sha256"] != _sha({"source_schema_sha256": source_sha,
                   "atoms": atoms, "column_indices": columns})
                or raw["mask_id"] != "M_DROP_" + next(
                    name for name in CATEGORICAL if name not in atoms).upper().replace("-", "_")
                or raw["capacity_signature"] != spec["primary_profile"]):
            raise PermissionError("M5 mask identity or selected raw columns differ")
        masks.append(Mask(raw["mask_id"], atoms, columns, raw["mask_sha256"],
                          (4, 4, 77), raw))
    if len({row.mask_id for row in masks}) != 5:
        raise PermissionError("M5 mask identity is duplicated")
    masks = sorted(masks, key=lambda row: row.mask_sha256)
    public_pairs = {row["pair_id"]: row for row in spec["pairs"]}
    if len(public_pairs) != 25:
        raise PermissionError("M5 public pair inventory differs")
    pairs = []
    for left, right in product(masks, repeat=2):
        pair_id = f"{left.mask_id}__{right.mask_id}"
        union = pair_union(RawMask(left.atoms, left.column_indices),
                           RawMask(right.atoms, right.column_indices))
        atoms = tuple(name for name in ORDER if name in left.atoms or name in right.atoms)
        digest = _sha({"source_schema_sha256": source_sha, "union_atoms": atoms,
                       "union_column_indices": union})
        public = public_pairs.get(pair_id)
        if (public is None or tuple(public["union_column_indices"]) != union
                or tuple(public["union_atoms"]) != atoms
                or public["union_sha256"] != digest
                or public["recipient1_mask_sha256"] != left.mask_sha256
                or public["recipient2_mask_sha256"] != right.mask_sha256):
            raise PermissionError("M5 public pair union differs from raw source")
        pairs.append(Pair(pair_id, left.mask_id, right.mask_id, atoms, union, digest))
    payload = {"schema": "pcrl-focused-shared-bank-v1",
               "masks": [row.public for row in masks],
               "pairs": [{"pair_id": row.pair_id,
                          "income_mask_id": row.income_mask_id,
                          "occupation_mask_id": row.occupation_mask_id,
                          "union_atoms": row.union_atoms,
                          "union_columns": row.union_columns,
                          "union_group_count": len(row.union_atoms),
                          "union_encoded_columns": len(row.union_columns),
                          "union_sha256": row.union_sha256} for row in pairs]}
    return Bank(tuple(masks), tuple(pairs), _sha(payload))
