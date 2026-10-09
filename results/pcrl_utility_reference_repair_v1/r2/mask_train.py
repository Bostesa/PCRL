"""In-memory FIT-only masked recipe stages, before package custody.

The caller must establish R0's seal and gate before giving this module Adult
arrays. This code opens no role archive, SEX member or common comparison row.
CAL is deliberately absent from the training entry point.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from results.pcrl_utility_reference_repair_v1.r1.source_task import (
    fit_source_heads,
    predict_source,
    select_source_head_C,
    train_source_encoder,
)
from results.pcrl_utility_reference_repair_v1.r2.mask_inputs import MaskInputPlan


REPO = Path(__file__).resolve().parents[3]
for source_dir in (
    REPO / "results/pcrl_method_portfolio_v1/teams/c",
    REPO / "results/pcrl_method_portfolio_v1/shared",
):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
from prototype import AdultGroupedSchema  # noqa: E402
from training import HeadSpec, fit_task_head  # noqa: E402
from results.pcrl_method_portfolio_v1.shared import common_eval  # noqa: E402


TASKS = ("income", "occupation_group")


@dataclass(frozen=True)
class MaskFitCoreResult:
    final_kind: str
    mask_id: str
    seed: int
    outer_models: Any
    final_models: Any
    outer_raw_metrics: dict[str, dict]
    selected_C: dict[str, float] | None
    C_search: dict | None
    inner_encoder: Any | None
    outer_split_receipt: dict
    inner_split_receipt: dict


def _metrics(probabilities: tuple[np.ndarray, np.ndarray],
             income: np.ndarray, occupation: np.ndarray) -> dict[str, dict]:
    return {
        "income": common_eval.task_metrics(probabilities[0], income, task="income"),
        "occupation_group": common_eval.task_metrics(
            probabilities[1], occupation, task="occupation"),
    }


def _fit_A_stage(plan: MaskInputPlan, yi: np.ndarray, yo: np.ndarray,
                 unit: np.ndarray, feature_names: tuple[str, ...],
                 seed: int, stage: str):
    if stage == "outer":
        rows = plan.outer_fit
        x = plan.full_matrix("outer", "fit")
    elif stage == "final":
        rows = np.arange(len(yi), dtype=np.int64)
        x = plan.full_matrix("final", "fit")
    else:
        raise ValueError("A training stage differs from frozen outer/final rule")
    output = []
    for task, labels in (("income", yi), ("occupation_group", yo)):
        output.append(fit_task_head(
            x, labels[rows], unit[rows], feature_names,
            plan.selected_atoms, purpose=task,
            model_id=f"UTILITY-REPAIR:A:s{seed}:{task}:{stage}:{plan.mask_id}",
            seed=seed, spec=HeadSpec(),
        ))
    return tuple(output)


def _A_probabilities(heads, x83: np.ndarray,
                     columns: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    selected = x83[:, columns]
    return tuple(head.predict_proba(selected) for head in heads)


def fit_mask_task_only(
    recipe: str,
    plan: MaskInputPlan,
    fit_income: np.ndarray,
    fit_occupation: np.ndarray,
    fit_unit: np.ndarray,
    *,
    seed: int,
    feature_names: tuple[str, ...],
) -> MaskFitCoreResult:
    """Fit one mask×seed with the selected R1 task-only recipe."""
    schema = AdultGroupedSchema()
    schema.validate_feature_names(feature_names)
    yi = np.asarray(fit_income)
    yo = np.asarray(fit_occupation)
    unit = np.asarray(fit_unit)
    if (recipe not in ("A", "B") or type(seed) is not int
            or seed not in (0, 1, 2)
            or yi.dtype != np.int64 or yo.dtype != np.int64
            or unit.dtype != np.int64 or
            yi.shape != (len(plan.selected_fit),) or
            yo.shape != yi.shape or unit.shape != yi.shape
            or plan.selected_atoms != schema.normalize_atoms(plan.selected_atoms)
            or schema.column_indices(plan.selected_atoms) != plan.source_indices):
        raise ValueError("masked training recipe or task/feature alignment differs")
    outer_yi = yi[plan.outer_fit]
    outer_yo = yo[plan.outer_fit]
    if recipe == "A":
        outer = _fit_A_stage(plan, yi, yo, unit, feature_names, seed, "outer")
        outer_prob = _A_probabilities(
            outer, plan.full_matrix("outer", "validation"),
            plan.source_indices)
        final = _fit_A_stage(plan, yi, yo, unit, feature_names, seed, "final")
        inner_encoder = None
        selected_C = None
        c_search = None
    else:
        inner_x = plan.full_matrix("inner", "train")
        inner_model = train_source_encoder(
            inner_x, yi[plan.inner_fit], yo[plan.inner_fit], seed)
        choice = select_source_head_C(
            inner_model, inner_x, yi[plan.inner_fit], yo[plan.inner_fit],
            plan.full_matrix("inner", "validation"),
            yi[plan.inner_val], yo[plan.inner_val],
        )
        selected_C = {task: float(choice[task]["selected_C"]) for task in TASKS}
        outer_x = plan.full_matrix("outer", "fit")
        outer_model = train_source_encoder(outer_x, outer_yi, outer_yo, seed)
        outer_heads = fit_source_heads(
            outer_model, outer_x, outer_yi, outer_yo, selected_C)
        outer = (outer_model, outer_heads)
        outer_prob = predict_source(
            outer_model, outer_heads, plan.full_matrix("outer", "validation"))
        final_x = plan.full_matrix("final", "fit")
        final_model = train_source_encoder(final_x, yi, yo, seed)
        final_heads = fit_source_heads(
            final_model, final_x, yi, yo, selected_C)
        final = (final_model, final_heads)
        inner_encoder = inner_model
        c_search = choice
    return MaskFitCoreResult(
        final_kind=recipe,
        mask_id=plan.mask_id,
        seed=seed,
        outer_models=outer,
        final_models=final,
        outer_raw_metrics=_metrics(
            outer_prob, yi[plan.outer_val], yo[plan.outer_val]),
        selected_C=selected_C,
        C_search=c_search,
        inner_encoder=inner_encoder,
        outer_split_receipt=plan.outer_split_receipt,
        inner_split_receipt=plan.inner_split_receipt,
    )

