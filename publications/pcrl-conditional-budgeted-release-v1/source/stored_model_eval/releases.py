"""Fit-free release generation for the pilot's documented noise arms.

durable-guarantees' post-hoc noise channel on the frozen Round-4 encoder released
X = H + default_rng(s).normal(0, sigma_abs, size=H.shape), drawn once over the full matrix per seed s
(honest_reaudit.py:78; run_tpr_failing59.py:103-108). One release per row per seed: a recipient sees one
draw (release_count = "one"). Repeated independent releases are a separate access model (A3) and are only
valid if the interface issues fresh noise per query; they are generated with a per-query seed.

No parameter is fitted here: sigma is a documented constant, so regenerating these releases is a frozen
transformation, not a defense refit.
"""
from __future__ import annotations

import numpy as np

# Absolute sigmas documented for the Adult Round-4 seed-0 cell (tpr59_scores/adult_noise_s*,
# tpr_ext_scores/hr_noise_s4/s8). sigma = 0 is the untreated reference.
DOCUMENTED_ADULT_SIGMAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def gaussian_release(H: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """One release of every row, matching the historical convention (one draw over the full matrix)."""
    H = np.asarray(H, dtype=np.float64)
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0:
        return H.copy()
    return H + np.random.default_rng(seed).normal(0.0, sigma, size=H.shape)


def repeated_releases(H: np.ndarray, sigma: float, seed: int, n_queries: int, fresh_per_query: bool) -> np.ndarray:
    """n_queries releases per row, shape (n_queries, n, d).

    fresh_per_query=True models an interface that redraws noise on each query; False models a persistent
    per-row token (every query returns the identical release), so averaging gains nothing.
    """
    if n_queries < 1:
        raise ValueError("n_queries must be >= 1")
    if not fresh_per_query:
        r = gaussian_release(H, sigma, seed)
        return np.repeat(r[None], n_queries, axis=0)
    return np.stack([gaussian_release(H, sigma, seed * 1_000_003 + q) for q in range(n_queries)])
