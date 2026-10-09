"""Load the previous public mask bank without changing its proposal geometry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from results.pcrl_capacity_matched_feature_selection_v1.m2.bank import (
    SharedBank,
    build_shared_bank,
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_bank(path: str | Path, *, expected_sha256: str) -> SharedBank:
    """Rebuild and cross-check all 25 pairs from sealed public geometry."""
    source = Path(path).resolve(strict=True)
    if _sha_file(source) != expected_sha256:
        raise PermissionError("public bank SHA differs from conditional lock")
    declared = json.loads(source.read_text())
    if (declared.get("schema") != "pcrl-focused-mask-bank-v1"
            or declared.get("status") != "PREFIT_FROZEN_GEOMETRY"):
        raise PermissionError("public bank schema or status differs")
    bank = build_shared_bank(declared["masks"])
    if bank.source_schema_sha256 != declared["source_schema_sha256"]:
        raise PermissionError("public bank source schema differs")
    rows = declared.get("pairs")
    if (not isinstance(rows, list) or len(rows) != 25
            or len({row.get("pair_id") for row in rows}) != 25):
        raise PermissionError("public bank pair inventory differs")
    by_id = {row["pair_id"]: row for row in rows}
    if set(by_id) != {pair.pair_id for pair in bank.pairs}:
        raise PermissionError("public bank pair IDs differ")
    for pair in bank.pairs:
        row = by_id[pair.pair_id]
        first = bank.mask(pair.income_mask_id)
        second = bank.mask(pair.occupation_mask_id)
        expected = {
            "recipient1_mask_sha256": first.mask_sha256,
            "recipient2_mask_sha256": second.mask_sha256,
            "recipient_encoded_columns": [first.encoded_columns,
                                          second.encoded_columns],
            "union_atoms": list(pair.union_atoms),
            "union_column_indices": list(pair.union_columns),
            "union_group_count": pair.union_group_count,
            "union_encoded_columns": pair.union_encoded_columns,
            "union_sha256": pair.union_sha256,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise PermissionError("public bank pair geometry differs")
    return bank

