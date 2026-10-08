"""Deploy a frozen mask pair through the inherited complete Team C wire.

The only per-person input is the admitted 83-column predictor matrix.  Row
and group IDs are private alignment custody and are never transmitted.  This
adapter accepts restored public head/calibrator packages and cannot append a
clean teacher output or use SEX/task labels at deployment.
"""

from __future__ import annotations

import hashlib
import numpy as np

from results.pcrl_method_portfolio_v1.teams.c.prototype import (
    AdultGroupedSchema, release_pair,
)

from bank import PairGeometry, SharedBank


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode() + str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def release_selected_pair(
    X: np.ndarray,
    feature_names: tuple[str, ...],
    row_id: np.ndarray,
    unit: np.ndarray,
    role: str,
    pair: PairGeometry,
    bank: SharedBank,
    income_release: object,
    occupation_release: object,
) -> dict:
    """Return three complete wires and separate private row-alignment data."""
    schema = AdultGroupedSchema()
    schema.validate_feature_names(tuple(feature_names))
    if role not in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        raise ValueError("release role is not admitted for this comparison")
    if pair not in bank.pairs:
        raise ValueError("pair is absent from the frozen shared bank")
    first_mask = bank.mask(pair.income_mask_id)
    second_mask = bank.mask(pair.occupation_mask_id)
    if (first_mask.source_schema_sha256 != bank.source_schema_sha256
            or second_mask.source_schema_sha256 != bank.source_schema_sha256):
        raise ValueError("recipient masks differ from bank source schema")
    feature_sha = hashlib.sha256("\n".join(schema.feature_names).encode()).hexdigest()
    if feature_sha != bank.source_schema_sha256:
        raise ValueError("bank mask source schema differs from pinned Adult features")
    if (tuple(first_mask.column_indices) != schema.column_indices(first_mask.atoms)
            or tuple(second_mask.column_indices) != schema.column_indices(second_mask.atoms)):
        raise ValueError("mask columns split or reorder a source group")
    if (getattr(income_release, "purpose", None) != "income"
            or tuple(getattr(income_release, "selected_atoms", ())) != first_mask.atoms
            or getattr(occupation_release, "purpose", None) != "occupation_group"
            or tuple(getattr(occupation_release, "selected_atoms", ())) != second_mask.atoms):
        raise ValueError("restored purpose release has the wrong selected mask")
    X = np.asarray(X)
    row_id = np.asarray(row_id)
    unit = np.asarray(unit)
    if (X.dtype != np.float32 or X.ndim != 2 or X.shape[1] != 83
            or not np.isfinite(X).all()):
        raise ValueError("release requires finite admitted float32 83-column predictors")
    if (row_id.dtype != np.int64 or unit.dtype != np.int64
            or row_id.shape != (len(X),) or unit.shape != (len(X),)
            or len(np.unique(row_id)) != len(row_id)):
        raise ValueError("private row/group alignment is malformed")

    wire = release_pair(X, tuple(feature_names), income_release,
                        occupation_release)
    expected = {
        "income": X[:, first_mask.column_indices],
        "occupation_group": X[:, second_mask.column_indices],
        "coalition": X[:, pair.union_columns],
    }
    for name, source in expected.items():
        disclosed = np.asarray(wire[name]["selected_values"])
        if (disclosed.shape != source.shape or disclosed.dtype != source.dtype
                or np.ascontiguousarray(disclosed).tobytes() !=
                np.ascontiguousarray(source).tobytes()):
            raise ValueError(f"{name} wire differs from its selected raw source values")
        if "row_id" in wire[name] or "unit" in wire[name]:
            raise ValueError("private role identifiers entered a recipient wire")
    if (tuple(wire["coalition"]["selected_groups"]) != pair.union_atoms
            or tuple(wire["income"]["selected_groups"]) != first_mask.atoms
            or tuple(wire["occupation_group"]["selected_groups"]) != second_mask.atoms):
        raise ValueError("wire public mask identities differ from frozen pair")
    alignment = {
        name: {"row_id": row_id.copy(), "unit": unit.copy()}
        for name in ("income", "occupation_group", "coalition")
    }
    receipt = {
        "schema": "pcrl-focused-release-alignment-v1",
        "role": role, "pair_id": pair.pair_id,
        "bank_sha256": bank.bank_sha256,
        "income_mask_sha256": first_mask.mask_sha256,
        "occupation_mask_sha256": second_mask.mask_sha256,
        "union_sha256": pair.union_sha256,
        "row_count": len(X),
        "ordered_row_id_sha256": _array_sha(row_id),
        "ordered_unit_sha256": _array_sha(unit),
        "source_feature_sha256": feature_sha,
        "income_raw_sha256": _array_sha(expected["income"]),
        "occupation_raw_sha256": _array_sha(expected["occupation_group"]),
        "union_raw_sha256": _array_sha(expected["coalition"]),
        "income_head_id": wire["income"]["head_id"],
        "occupation_head_id": wire["occupation_group"]["head_id"],
        "income_calibrator_id": wire["income"]["calibrator_id"],
        "occupation_calibrator_id": wire["occupation_group"]["calibrator_id"],
    }
    return {"wire": wire, "private_alignment": alignment, "receipt": receipt}
