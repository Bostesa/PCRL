"""Tiny public-bank ordering regression for fixed diagnostic replay."""

import numpy as np

from .replay_a_fixed_attacks_v2 import _effective_bank


def test_diagnostic_uses_global_canonical_ids_but_tstar_keeps_old_ids():
    old = np.array([[0.9, 0.1], [0.6, 0.4], [0.1, 0.9]], np.float64)
    canonical = _effective_bank("A_CLASS_ONLY", old)
    np.testing.assert_array_equal(canonical, old[[1, 0, 2]])
    np.testing.assert_array_equal(_effective_bank("A_IDENTITY", old), canonical)
    np.testing.assert_array_equal(_effective_bank("T_STAR", old), old)
