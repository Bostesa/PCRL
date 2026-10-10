"""Tiny own-input local fit and paired-composition checks."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "impl"))
from bank import FixedBank
from channel import GateWeights
from fit import FitRecipe

from fit_local import LocalFactorData, fit_local_pair


class LocalFitTest(unittest.TestCase):
    def test_each_worker_gets_only_own_score_and_label(self):
        p1 = np.array([[.9, .1], [.6, .4]], dtype=np.float64)
        p2 = np.array([[.8, .2], [.7, .3]], dtype=np.float64)
        b1 = FixedBank.from_source(p1, np.arange(2), np.zeros(2, dtype=np.int64))
        b2 = FixedBank.from_source(p2, np.arange(2), np.zeros(2, dtype=np.int64))
        a1 = (tuple(map(int, b1.old_to_global)),)
        a2 = (tuple(map(int, b2.old_to_global)),)
        sex = np.array([0, 1, 0, 1], dtype=np.int64)
        d1 = LocalFactorData(np.tile(p1[0], (4, 1)), np.zeros(4, dtype=np.int64),
                             np.zeros(4, dtype=np.int64), sex, b1, a1,
                             unit_ids=("a", "b", "c", "d"))
        d2 = LocalFactorData(np.tile(p2[0], (4, 1)), np.zeros(4, dtype=np.int64),
                             np.zeros(4, dtype=np.int64), sex, b2, a2,
                             unit_ids=("a", "b", "c", "d"))
        self.assertFalse(hasattr(d1, "other_score"))
        recipe = FitRecipe(updates=2, checkpoint_every=1, cpu_seconds=30,
                           seed=23, dual_every=1)
        result = fit_local_pair(d1, d2, gate=GateWeights.zeros(input_width=8),
                                caps1=(1.0, 1.0), caps2=(1.0, 1.0), recipe=recipe)
        self.assertTrue(result["complete"])
        self.assertTrue(result["success"])
        self.assertEqual(len(result["recipient1"]["starts"]), 2)
        self.assertEqual(len(result["recipient2"]["starts"]), 2)
        self.assertEqual(result["policy"].factor1.parameter_count, 148)
        self.assertAlmostEqual(sum(p for _, _, p in result["policy"].sparse_law(
            p1[0], p2[0], 0, 0)), 1)


if __name__ == "__main__":
    unittest.main()
