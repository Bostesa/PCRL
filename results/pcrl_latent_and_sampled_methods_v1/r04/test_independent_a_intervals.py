"""Data-free A family mean coupling and fixed-K missing-source checks."""

from __future__ import annotations

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r04.audit_a_family_s1 import independent_ids
from results.pcrl_latent_and_sampled_methods_v1.r04.independent_a_intervals import (
    _bounds, _compose)


def test_means_resample_coupled_draw_indices_but_fixed_slots_stay_fixed():
    base = {(seed, draw): np.full(765, seed * 10 + draw, np.float64)
            for seed in (0, 1) for draw in (0, 1, 2)}
    result = _compose(base, {0: (2, 2, 0), 1: (1, 1, 2)})
    assert np.allclose([result[765 * i] for i in range(9)],
                       [0, 1, 2, 10, 11, 12, 4 / 3, 34 / 3, 19 / 3],
                       rtol=0, atol=1e-14)


def test_missing_arm_preserves_6885_denominator_and_incomplete_slots():
    names = independent_ids()
    point = np.full(len(names), np.nan, np.float64)
    point[0] = .2
    boot = np.linspace(.1, .3, 2048, dtype=np.float64)[:, None]
    clauses, family, critical = _bounds(names, point, boot)
    assert len(clauses) == 6885
    assert clauses[0]["status"] == "BOUND_ONLY"
    assert clauses[1]["status"] == "INCOMPLETE_MISSING_ARM"
    assert family == "two-sided-fixed-K-Bonferroni-t-approximation"
    assert critical > 4
