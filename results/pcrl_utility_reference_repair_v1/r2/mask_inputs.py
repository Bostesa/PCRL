"""Task-only masked matrices for nested FIT selection and final refit.

Only this constructor sees the original 83-column FIT matrix. Returned
per-person matrices contain selected columns and fixed FIT-derived fills.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from results.pcrl_method_portfolio_v1.teams.c.prototype import AdultGroupedSchema
from results.pcrl_utility_reference_repair_v1.r1.fit_splits import (
    inner_split,
    outer_split,
)
from results.pcrl_utility_reference_repair_v1.r2.mask_adapter import (
    fit_public_fill_constants,
    reconstruct_selected,
)


@dataclass(frozen=True)
class MaskInputPlan:
    mask_id: str
    mask_sha256: str
    selected_atoms: tuple[str, ...]
    source_indices: tuple[int, ...]
    selected_fit: np.ndarray
    outer_fit: np.ndarray
    outer_val: np.ndarray
    inner_fit: np.ndarray
    inner_val: np.ndarray
    fill_constants: dict[str, np.ndarray]
    outer_split_receipt: dict
    inner_split_receipt: dict

    def full_matrix(self, stage: str, subset: str) -> np.ndarray:
        """Reconstruct only selected values plus stage-specific public fills."""
        if stage not in ("inner", "outer", "final"):
            raise ValueError("unknown mask training stage")
        row_map = {
            ("inner", "train"): self.inner_fit,
            ("inner", "validation"): self.inner_val,
            ("outer", "fit"): self.outer_fit,
            ("outer", "validation"): self.outer_val,
            ("final", "fit"): np.arange(len(self.selected_fit), dtype=np.int64),
        }
        if (stage, subset) in row_map:
            selected = self.selected_fit[row_map[(stage, subset)]]
        else:
            raise ValueError("mask stage/subset combination is not permitted")
        return reconstruct_selected(selected, self.source_indices,
                                    self.fill_constants[stage])


def build_mask_input_plan(
    fit_x83: np.ndarray,
    fit_income: np.ndarray,
    fit_occupation: np.ndarray,
    fit_unit: np.ndarray,
    mask: object,
    *,
    seed: int,
) -> MaskInputPlan:
    """Apply R1's frozen FIT folds and erase hidden person columns."""
    schema = AdultGroupedSchema()
    source_sha = hashlib.sha256("\n".join(schema.feature_names).encode()).hexdigest()
    columns = tuple(getattr(mask, "column_indices", ()))
    atoms = tuple(getattr(mask, "atoms", ()))
    if (getattr(mask, "source_schema_sha256", None) != source_sha
            or columns != schema.column_indices(atoms)
            or not getattr(mask, "mask_id", None)
            or not getattr(mask, "mask_sha256", None)):
        raise ValueError("mask differs from pinned grouped source schema")
    x = np.asarray(fit_x83)
    if (x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 83
            or len(x) == 0 or not np.isfinite(x).all()):
        raise ValueError("FIT requires finite float32 83-column predictors")
    yi = np.asarray(fit_income)
    yo = np.asarray(fit_occupation)
    unit = np.asarray(fit_unit)
    if (yi.dtype != np.int64 or yo.dtype != np.int64
            or unit.dtype != np.int64 or yi.shape != (len(x),)
            or yo.shape != (len(x),) or unit.shape != (len(x),)):
        raise ValueError("FIT task labels/groups are not aligned int64 arrays")
    outer_fit, outer_val, outer_receipt = outer_split(unit, yi, yo, seed)
    inner_fit_local, inner_val_local, inner_receipt = inner_split(
        unit[outer_fit], yi[outer_fit], yo[outer_fit], seed)
    inner_fit = outer_fit[inner_fit_local]
    inner_val = outer_fit[inner_val_local]
    selected_fit = np.ascontiguousarray(x[:, columns])
    selected_fit.setflags(write=False)
    return MaskInputPlan(
        mask_id=mask.mask_id,
        mask_sha256=mask.mask_sha256,
        selected_atoms=atoms,
        source_indices=columns,
        selected_fit=selected_fit,
        outer_fit=outer_fit,
        outer_val=outer_val,
        inner_fit=inner_fit,
        inner_val=inner_val,
        fill_constants={
            "inner": fit_public_fill_constants(x[inner_fit]),
            "outer": fit_public_fill_constants(x[outer_fit]),
            "final": fit_public_fill_constants(x),
        },
        outer_split_receipt=outer_receipt,
        inner_split_receipt=inner_receipt,
    )


def mask_calibration_matrix(cal_x83: np.ndarray,
                            plan: MaskInputPlan) -> np.ndarray:
    """Open CAL features only after the full-FIT masked head has been trained."""
    cal = np.asarray(cal_x83)
    if (cal.dtype != np.float32 or cal.ndim != 2 or cal.shape[1] != 83
            or len(cal) == 0 or not np.isfinite(cal).all()):
        raise ValueError("calibration requires finite float32 83-column predictors")
    selected = np.ascontiguousarray(cal[:, plan.source_indices])
    return reconstruct_selected(selected, plan.source_indices,
                                plan.fill_constants["final"])
