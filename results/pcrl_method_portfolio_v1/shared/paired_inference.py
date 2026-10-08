"""Development-only paired group and stochastic-draw inference.

This module consumes an in-memory endpoint callback and opens no data. The
callback must return the same ordered, prospectively registered margin vector
for selected rows of the actual draw0 release. S4 replays from private saved
arrays using an independent implementation. Seven other predeclared stochastic
draws are descriptive sensitivity checks, not added to the primary interval:
group bootstrapping draw0 already resamples realized per-person channel noise.
"""

from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np


SCHEMA = "pcrl-portfolio-paired-inference-v1"
REPLICATES = 2048
INNER_DRAWS = 8
STUDENTIZATION_FLOOR = 1e-12


def paired_group_inference(unit, margin_names: tuple[str, ...],
                           margin_at: Callable[[np.ndarray, int], np.ndarray],
                           private_seed: bytes, *, replicates: int = REPLICATES) -> dict:
    """Two-sided simultaneous 95% bounds over one registered margin family.

    Each margin is oriented so nonnegative is favourable. Across seeds, arms,
    references and readers the caller's callback must use the *same* sampled
    people and reader bank. For stochastic A, this uses the actual draw0 only.
    """
    groups = np.asarray(unit, dtype=np.int64)
    if groups.ndim != 1 or groups.size == 0 or np.unique(groups).size != 2234:
        raise ValueError("INNER_SELECTION group identity/count mismatch")
    if len(private_seed) != 32 or replicates != REPLICATES:
        raise ValueError("inference seed/replicate contract mismatch")
    if not margin_names or len(set(margin_names)) != len(margin_names):
        raise ValueError("registered margin names missing or repeated")
    ordered = np.unique(groups)
    positions = {int(g): np.flatnonzero(groups == g) for g in ordered}
    rng = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(private_seed[:16], "big")))

    def evaluate(idx: np.ndarray, draw: int) -> np.ndarray:
        m = np.asarray(margin_at(idx, draw), dtype=np.float64)
        if m.shape != (len(margin_names),) or not np.all(np.isfinite(m)):
            raise ValueError("margin callback returned invalid schema/value")
        return m

    point = evaluate(np.arange(len(groups), dtype=np.int64), 0)
    boot = np.empty((replicates, len(margin_names)), dtype=np.float64)
    for b in range(replicates):
        sampled = rng.choice(ordered, size=len(ordered), replace=True)
        idx = np.concatenate([positions[int(g)] for g in sampled])
        boot[b] = evaluate(idx, 0)
    se = boot.std(axis=0, ddof=1)
    # Machine summation of a constant margin may produce ~1e-17 sample SD.
    # Mark a clause NUMERICALLY_CONSTANT only when *every* replicate agrees
    # with its point to the frozen 1e-12 tolerance; this is not a proof of
    # structural equality, although the computed interval has zero width.
    zero = np.max(np.abs(boot - point), axis=0) <= STUDENTIZATION_FLOOR
    se[zero] = 0.0
    student_scale = np.maximum(se, STUDENTIZATION_FLOOR)
    scaled = np.zeros_like(boot)
    scaled[:, ~zero] = np.abs((boot[:, ~zero] - point[~zero]) / student_scale[~zero])
    max_t = scaled.max(axis=1)
    critical = float(np.quantile(max_t, 0.95, method="higher"))
    width = critical * np.where(zero, 0.0, student_scale)
    lower = point - width
    upper = point + width
    clauses = []
    for j, name in enumerate(margin_names):
        status = "PASS" if lower[j] >= 0 else "VIOLATION" if upper[j] < 0 else "UNRESOLVED"
        clauses.append({"name": name, "point_margin": float(point[j]), "se": float(se[j]),
                        "lower": float(lower[j]), "upper": float(upper[j]), "status": status,
                        "numerical_se_status": "NUMERICALLY_CONSTANT" if zero[j] else "VARIABLE"})
    return {"schema": SCHEMA, "bootstrap_replicates": replicates, "inner_groups": len(ordered),
            "mechanism_draws_in_primary_interval": 1,
            "critical_family": "two-sided-maximum-absolute-studentized",
            "critical_95": critical, "studentization_floor": STUDENTIZATION_FLOOR,
            "numpy_version": np.__version__, "private_seed_sha256": hashlib.sha256(private_seed).hexdigest(),
            "draw0_is_point": True, "clauses": clauses}


def eight_draw_sensitivity(row_count: int, margin_names: tuple[str, ...],
                           margin_at: Callable[[np.ndarray, int], np.ndarray]) -> dict:
    """Descriptive eight-draw spread with frozen draw0 readers and no selection."""
    idx = np.arange(row_count, dtype=np.int64)
    values = np.stack([np.asarray(margin_at(idx, draw), dtype=np.float64)
                       for draw in range(INNER_DRAWS)])
    if values.shape != (INNER_DRAWS, len(margin_names)) or not np.all(np.isfinite(values)):
        raise ValueError("eight-draw sensitivity margin schema mismatch")
    return {"schema": "pcrl-portfolio-eight-draw-sensitivity-v1", "draws": INNER_DRAWS,
            "reader_fit_draw": 0, "selected_draw": None, "margin_names": list(margin_names),
            "per_draw_margins": values.tolist(), "minimum": values.min(axis=0).tolist(),
            "maximum": values.max(axis=0).tolist(), "mean": values.mean(axis=0).tolist(),
            "interpretation": "descriptive one-shot mechanism variance; not a primary interval or population guarantee"}
