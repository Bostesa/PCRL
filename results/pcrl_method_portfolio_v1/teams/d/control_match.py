"""Predeclared task-only compression match for Team D.

The matcher consumes aggregate task utility and reconstruction distortion from
CALIBRATION_HELDOUT. It never accepts SEX labels or recovery scores. It is a
control-selection procedure, not the scientific strongest-control selection.
Only S2's common evaluator may produce the final endpoint measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import re

import numpy as np


BETAS = (0.0, 0.25, 1.0)
TASK_TOLERANCE = np.array((0.002, 0.002, 0.001), dtype=np.float64)
DISTORTION_REL_TOLERANCE = 0.10
DISTORTION_DENOMINATOR_FLOOR = 1e-6
METRICS = ("accuracy", "nll", "brier_sum")


@dataclass(frozen=True)
class MatchObservation:
    """Two tasks x three metrics, plus one standardized-latent MSE per purpose."""

    utility: np.ndarray
    distortion: np.ndarray

    def __post_init__(self) -> None:
        utility = np.asarray(self.utility, dtype=np.float64)
        distortion = np.asarray(self.distortion, dtype=np.float64)
        if utility.shape != (2, 3) or distortion.shape != (2,):
            raise ValueError("utility must be (2,3) and distortion must be (2,)")
        if not np.isfinite(utility).all() or not np.isfinite(distortion).all():
            raise ValueError("match observations must be finite")
        if np.any(distortion < 0) or np.any(utility[:, 0] < 0) or np.any(utility[:, 0] > 1):
            raise ValueError("invalid distortion or accuracy")
        if np.any(utility[:, 1:] < 0):
            raise ValueError("NLL and Brier must be nonnegative")
        object.__setattr__(self, "utility", utility.copy())
        object.__setattr__(self, "distortion", distortion.copy())


def utility_observation(
    labels: tuple[np.ndarray, np.ndarray],
    calibrated_probabilities: tuple[np.ndarray, np.ndarray],
    standardized_latents: tuple[np.ndarray, np.ndarray],
    decoded_features: tuple[np.ndarray, np.ndarray],
) -> MatchObservation:
    """Compute fixed CAL/INNER match inputs from a complete released code view.

    Probabilities must already have received S2's common held-out calibration.
    Brier is the *sum* over classes. Distortion is mean squared coordinate error
    in the same purpose-standardized latent space for all arms. No SEX is read.
    """
    if not all(len(x) == 2 for x in (labels, calibrated_probabilities,
                                     standardized_latents, decoded_features)):
        raise ValueError("exactly two paired purposes are required")
    utility = np.empty((2, 3), dtype=np.float64)
    distortion = np.empty(2, dtype=np.float64)
    for purpose in (0, 1):
        y = np.asarray(labels[purpose])
        p = np.asarray(calibrated_probabilities[purpose], dtype=np.float64)
        h = np.asarray(standardized_latents[purpose], dtype=np.float64)
        z = np.asarray(decoded_features[purpose], dtype=np.float64)
        if (y.ndim != 1 or p.ndim != 2 or h.ndim != 2 or z.shape != h.shape
                or len(y) != len(p) or len(y) != len(h) or len(y) == 0
                or not np.issubdtype(y.dtype, np.integer)
                or np.any(y < 0) or np.any(y >= p.shape[1])
                or not all(np.isfinite(a).all() for a in (p, h, z))
                or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0)):
            raise ValueError("invalid paired task or release arrays")
        clipped = np.clip(p, 1e-12, 1.0)
        onehot = np.eye(p.shape[1], dtype=np.float64)[y]
        utility[purpose] = (
            float(np.mean(p.argmax(1) == y)),
            float(np.mean(-np.log(clipped[np.arange(len(y)), y]))),
            float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        )
        distortion[purpose] = float(np.mean((h - z) ** 2))
    return MatchObservation(utility, distortion)


def gaps(candidate: MatchObservation, control: MatchObservation) -> dict:
    utility_abs = np.abs(control.utility - candidate.utility)
    denominator = np.maximum(candidate.distortion, DISTORTION_DENOMINATOR_FLOOR)
    distortion_relative = np.abs(control.distortion - candidate.distortion) / denominator
    scaled_utility = utility_abs / TASK_TOLERANCE[None, :]
    return {
        "utility_abs": utility_abs.tolist(),
        "utility_scaled_max": float(np.max(scaled_utility)),
        "distortion_relative": distortion_relative.tolist(),
        "eligible": bool(np.all(scaled_utility <= 1.0)
                         and np.all(distortion_relative <= DISTORTION_REL_TOLERANCE)),
    }


def select_match(
    candidate: MatchObservation,
    task_only_grid: Mapping[float, MatchObservation],
    *,
    role: str,
    candidate_fit_id: str,
    candidate_calibration_id: str,
    control_fit_ids: Mapping[float, str],
    control_calibration_ids: Mapping[float, str],
    row_order_sha256: str,
    metric_convention_id: str,
) -> dict:
    """Choose on CAL only; no closest miss and no sensitive-recovery input."""
    if role != "CALIBRATION_HELDOUT":
        raise ValueError("control matching may use CALIBRATION_HELDOUT only")
    if (set(task_only_grid) != set(BETAS) or set(control_fit_ids) != set(BETAS)
            or set(control_calibration_ids) != set(BETAS)):
        raise ValueError("all three predeclared task-only grid fits and calibrations are required")
    ids = [candidate_fit_id, candidate_calibration_id, metric_convention_id,
           *control_fit_ids.values(), *control_calibration_ids.values()]
    if any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("all fit/calibration/convention IDs must be nonempty")
    if not isinstance(row_order_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", row_order_sha256) is None:
        raise ValueError("a full admitted CAL row/order SHA-256 is required")
    rows = []
    for beta in BETAS:
        rec = {"beta": beta, **gaps(candidate, task_only_grid[beta])}
        rows.append(rec)
    eligible = [row for row in rows if row["eligible"]]
    chosen = min(eligible, key=lambda row: (row["utility_scaled_max"], row["beta"])) if eligible else None
    return {
        "status": "MATCHED" if chosen else "NO_MATCH",
        "chosen_beta": None if chosen is None else chosen["beta"],
        "candidate_fit_id": candidate_fit_id,
        "candidate_calibration_id": candidate_calibration_id,
        "selected_control_fit_id": None if chosen is None else control_fit_ids[chosen["beta"]],
        "selected_control_calibration_id": None if chosen is None else control_calibration_ids[chosen["beta"]],
        "calibration_row_order_sha256": row_order_sha256,
        "metric_convention_id": metric_convention_id,
        "all_grid_ids": [{"beta": b, "fit_id": control_fit_ids[b],
                          "calibration_id": control_calibration_ids[b]} for b in BETAS],
        "calibration_role_reused_for_temperature_and_matching": True,
        "all_grid_rows": rows,
        "rule": "eligible iff every per-task accuracy/NLL/Brier absolute gap <= (0.002,0.002,0.001) and each purpose standardized-latent MSE relative gap <=0.10; relative denominator=max(candidate MSE,1e-6); among eligible minimize max scaled utility gap, ties smaller beta",
    }


def recheck_inner(candidate: MatchObservation, chosen: MatchObservation, *, role: str,
                  frozen_receipt: dict, observed_control_fit_id: str,
                  observed_control_calibration_id: str, candidate_fit_id: str,
                  candidate_calibration_id: str, row_order_sha256: str,
                  metric_convention_id: str) -> dict:
    """Report a frozen CAL-selected control's INNER gaps without reselection."""
    if role != "INNER_SELECTION":
        raise ValueError("match recheck may use INNER_SELECTION only")
    if frozen_receipt.get("status") != "MATCHED":
        raise ValueError("there is no frozen CAL match to recheck")
    required = {
        "selected_control_fit_id": observed_control_fit_id,
        "selected_control_calibration_id": observed_control_calibration_id,
        "candidate_fit_id": candidate_fit_id,
        "candidate_calibration_id": candidate_calibration_id,
        "metric_convention_id": metric_convention_id,
    }
    if any(frozen_receipt.get(key) != value for key, value in required.items()):
        raise ValueError("INNER check does not use the frozen CAL-selected fit/calibration identities")
    if not isinstance(row_order_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", row_order_sha256) is None:
        raise ValueError("a full admitted INNER row/order SHA-256 is required")
    out = gaps(candidate, chosen)
    return {"role": role, "row_order_sha256": row_order_sha256,
            "frozen_calibration_row_order_sha256": frozen_receipt["calibration_row_order_sha256"],
            "match_persists": out["eligible"], **out,
            "attribution": "POSSIBLE" if out["eligible"] else "UNRESOLVED_NO_INNER_MATCH"}
