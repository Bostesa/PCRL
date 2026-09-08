"""Phase-2 per-class OvR constraint wrapper for BIOS-medium.

NEW MODULE — does NOT touch the existing tabular per-class OvR path used by
Diabetes (``pcrl/training/v2_trainer.py:341+``). The two wrappers serve
different problems:

- Tabular Diabetes: protected attribute itself has K≥6 classes; per-class OvR
  decomposes that single attribute into K binary constraints to defeat the
  K-class averaging pathology (Ravfogel et al. ACL 2023).
- BIOS Phase 2: protected attribute is binary (gender), so per-class OvR on
  gender collapses to a single constraint. The "10 OvR" referred to in the
  spec are the **10 occupation classes**: for each occupation, gender should
  not be linearly predictable from the [CLS] representation *within that
  occupation slice*. This catches subgroup-specific gender leakage that a
  marginal-only constraint can satisfy while individual occupation slices
  leak heavily.

Constraint set returned by :func:`build_phase2_constraints`:
- ``cond_occ_<occupation_name>`` for each occupation k=0..9 — conditional
  R²(z|occ=k, gender) ≤ 0.05.
- ``marginal_gender`` — R²(z, gender) ≤ 0.05 (same as Phase 1).

For each constraint each call to :func:`evaluate_constraints` returns a
**differentiable** scalar value tensor (used by the proxy-Lagrangian loss)
plus its scalar value (used by the dual update). When a per-occupation slice
in the current batch has < ``min_slice_size`` samples or fewer than 2 distinct
gender values, that constraint is undefined (NaN) for the batch. It is omitted from the
differentiable loss and must not drive a dual update or count as a privacy pass.

Caveat for batch=32 + 10 occupations
------------------------------------
With batch=32 and 10 occupations roughly stratified, each per-occupation slice
in a batch averages ~3 samples — too few for stable R². Consider either:
  (a) Batch ≥ 256 for Phase 2.
  (b) An accumulation buffer that aggregates per-occupation [CLS] across
      multiple consecutive batches before computing per-occ R².

This module's ``min_slice_size`` parameter exposes the threshold; the run
script defaults to 16 and warns when fewer than 5/10 occupations clear the
threshold per batch on average.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from pcrl.training.losses import VerificationRegularizer
from pcrl.training.proxy_lagrangian import Constraint

from .bios_dataset import BIOS_TOP10


def build_phase2_constraints(
    *,
    threshold: float = 0.05,
    eta_lambda: float = 0.02,
    lambda_init: float = 1.0,
    lambda_min: float = 5.0,
    lambda_max: float = 100.0,
) -> list[Constraint]:
    """Construct the 11 ``Constraint`` objects (10 per-occupation conditional
    + 1 marginal). All R² constraints, all upper-bound, all sharing the same
    threshold and dual-step parameters.
    """
    constraints: list[Constraint] = []
    for occ_name in BIOS_TOP10:
        constraints.append(Constraint(
            name=f"cond_occ_{occ_name}",
            threshold=threshold,
            direction="<=",
            eta_lambda=eta_lambda,
            lambda_init=lambda_init,
            lambda_max=lambda_max,
            lambda_min=lambda_min,
        ))
    constraints.append(Constraint(
        name="marginal_gender",
        threshold=threshold,
        direction="<=",
        eta_lambda=eta_lambda,
        lambda_init=lambda_init,
        lambda_max=lambda_max,
        lambda_min=lambda_min,
    ))
    return constraints


@dataclass
class Phase2Eval:
    differentiable: dict[str, torch.Tensor]
    scalars: dict[str, float]
    n_active_slices: int
    n_total_slices: int
    support: dict[str, list[int]] = field(default_factory=dict)
    valid_mask: dict[str, bool] = field(default_factory=dict)


def evaluate_constraints(
    z: torch.Tensor,
    gender: torch.Tensor,
    occupation: torch.Tensor,
    verifier: VerificationRegularizer,
    *,
    min_slice_size: int = 16,
) -> Phase2Eval:
    """Evaluate the 11 Phase-2 constraints on a batch.

    Args:
        z: ``(B, repr_dim)`` representation tensor (requires_grad through the
            encoder).
        gender: ``(B,)`` int64 in {0, 1}.
        occupation: ``(B,)`` int64 in [0, 10).
        verifier: ``pcrl.training.losses.VerificationRegularizer`` instance.
        min_slice_size: Minimum number of samples required in a per-occupation
            slice before its R² is computed. Slices below this size return
            undefined scores (NaN).

    Returns:
        Phase2Eval with differentiable per-constraint values, scalar values,
        and active/total slice counts for diagnostics.
    """
    diffs: dict[str, torch.Tensor] = {}
    scalars: dict[str, float] = {}
    support: dict[str, list[int]] = {}
    valid: dict[str, bool] = {}
    n_active = 0

    def evaluate(name: str, features: torch.Tensor, labels: torch.Tensor, min_size: int):
        counts = torch.bincount(labels.long(), minlength=2)
        support[name] = counts.tolist()
        valid[name] = len(labels) >= min_size and bool((counts > 0).all())
        if not valid[name]:
            scalars[name] = float("nan")
            return
        r2 = verifier(features, labels, num_classes=2)
        diffs[name] = r2
        scalars[name] = float(r2.detach().item())

    for k, occ_name in enumerate(BIOS_TOP10):
        mask = occupation == k
        name = f"cond_occ_{occ_name}"
        evaluate(name, z[mask], gender[mask], min_slice_size)
        n_active += int(valid[name])

    evaluate("marginal_gender", z, gender, 2)
    return Phase2Eval(
        differentiable=diffs,
        scalars=scalars,
        n_active_slices=n_active,
        n_total_slices=len(BIOS_TOP10),
        support=support,
        valid_mask=valid,
    )
