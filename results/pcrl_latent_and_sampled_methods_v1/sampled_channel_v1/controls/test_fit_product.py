"""Tiny data-free fit check for a separately optimized central product control."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "impl"))
from bank import FixedBank
from channel import GateWeights
from fit import FitRecipe, TaskOnlyData, ProtectedData

from factor_controls import FactorGate, ProductPolicy
from fit_product import fit_product


class ProductFitTest(unittest.TestCase):
    def test_two_starts_fit_actual_product_law_with_exact_caps(self):
        s1 = np.array([.9, .1], dtype=np.float64)
        s2 = np.array([.8, .2], dtype=np.float64)
        b1 = FixedBank.from_source(np.array([[.9, .1], [.6, .4]]),
                                   np.arange(2), np.zeros(2, dtype=np.int64))
        b2 = FixedBank.from_source(np.array([[.8, .2], [.7, .3]]),
                                   np.arange(2), np.zeros(2, dtype=np.int64))
        choices1 = (tuple(map(int, b1.old_to_global)),)
        choices2 = (tuple(map(int, b2.old_to_global)),)
        factor = FactorGate(np.zeros((8, 16)), np.zeros(16),
                            np.zeros((16, 4)), np.zeros(4))
        policy = ProductPolicy(b1, b2, choices1, choices2,
                               GateWeights.zeros(input_width=8), factor, factor)
        task = TaskOnlyData(np.tile(s1, (4, 1)), np.tile(s2, (4, 1)),
                            np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64),
                            np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64))
        protected = ProtectedData(task, np.array([0, 1, 0, 1], dtype=np.int64))
        recipe = FitRecipe(updates=2, checkpoint_every=1, cpu_seconds=30,
                           seed=19, dual_every=1)
        caps = dict(r1_nll=1.0, r1_brier=1.0, r2_nll=1.0, r2_brier=1.0)
        result = fit_product(protected, policy, caps=caps, recipe=recipe)
        self.assertTrue(result["complete"])
        self.assertTrue(result["success"])
        self.assertEqual(len(result["starts"]), 2)
        self.assertEqual(result["policy"].factor1.parameter_count, 212)
        self.assertAlmostEqual(sum(p for _, _, p in result["policy"].sparse_law(
            s1, s2, 0, 0)), 1)


if __name__ == "__main__":
    unittest.main()
