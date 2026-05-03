"""Per-occupation true-positive-rate gap (BIOS standard fairness metric).

The canonical BIOS-medium fairness metric (De-Arteaga et al. 2019,
arXiv:1901.09451). For each occupation :math:`y`:

.. math::
    \\mathrm{TPR\\text{-}gap}_y = \\Pr(\\hat{y}=y \\mid y, \\mathrm{male})
                             - \\Pr(\\hat{y}=y \\mid y, \\mathrm{female})

Aggregates: RMS over occupations, max absolute gap. Belrose et al. 2023
(LEACE) Table 3 reports both for BIOS post-erasure baselines.

This is the **primary** Round-2 metric (per the user 2026-05-03). R² is
diagnostic only.
"""
from __future__ import annotations

import numpy as np


def tpr_gaps_per_occupation(
    predictions: np.ndarray,
    true_occupations: np.ndarray,
    genders: np.ndarray,
    *,
    n_occupations: int = 10,
    male_label: int = 0,
    female_label: int = 1,
) -> np.ndarray:
    """Returns a length-``n_occupations`` array of TPR gaps (male - female).

    A gap ``> 0`` means the model is more likely to recover the correct
    occupation when the bio is male-coded; gap ``< 0`` means the opposite.
    Slices with zero samples for either gender contribute 0 (no signal).
    """
    predictions = np.asarray(predictions)
    true_occupations = np.asarray(true_occupations)
    genders = np.asarray(genders)
    correct = (predictions == true_occupations)
    gaps = np.zeros(n_occupations, dtype=np.float64)
    for k in range(n_occupations):
        true_k = (true_occupations == k)
        m_mask = true_k & (genders == male_label)
        f_mask = true_k & (genders == female_label)
        n_m = int(m_mask.sum())
        n_f = int(f_mask.sum())
        if n_m == 0 or n_f == 0:
            continue  # leave gap = 0
        tpr_m = float(correct[m_mask].sum()) / n_m
        tpr_f = float(correct[f_mask].sum()) / n_f
        gaps[k] = tpr_m - tpr_f
    return gaps


def tpr_gap_summary(gaps: np.ndarray) -> dict:
    """Return RMS and max-abs aggregates plus per-occupation values."""
    g = np.asarray(gaps, dtype=np.float64)
    return {
        "rms_gap": float(np.sqrt((g ** 2).mean())),
        "max_abs_gap": float(np.abs(g).max()),
        "per_occupation": g.tolist(),
    }


def theil_adjusted_r2(r2: float, n: int, d: int) -> float:
    """Theil's adjusted R² (1961). Has expectation 0 under independence
    at any ``(n, d)``, unlike the raw R² which has bias ``≈ d / (n-1)``.

    .. math::
        \\bar{R}^2 = 1 - (1 - R^2) \\frac{n - 1}{n - d - 1}

    Returns 0.0 if ``n - d - 1 ≤ 0`` (estimator undefined).
    """
    if n - d - 1 <= 0:
        return 0.0
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - d - 1))
