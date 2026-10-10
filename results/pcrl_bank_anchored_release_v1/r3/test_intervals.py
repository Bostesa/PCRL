"""Data-free paired-group expected-utility inference checks."""

import unittest

import numpy as np

from bank_anchored.r3_intervals import expected_utility_intervals


def record(nll_excess: float):
    n = 4
    out = {"row_id": np.array([1, 2, 3, 4], dtype=np.int64),
           "unit": np.array([1, 1, 2, 3], dtype=np.int64)}
    for task in ("income", "occupation"):
        for metric in ("accuracy", "nll", "brier"):
            base = .7 if metric == "accuracy" else .5
            for name in ("B0", "U0", "Ucal"):
                out[f"{name}_{task}_{metric}"] = np.full(n, base)
            out[f"candidate_{task}_{metric}"] = np.full(
                n, base + (nll_excess if metric == "nll" else 0))
    return out


class UtilityIntervalTests(unittest.TestCase):
    def test_fixed_family_keeps_absent_arms_null_and_zero_cost_passes(self):
        # Break caught: silently reducing K to the one observed arm.
        results = expected_utility_intervals(
            {("B0", 0): record(0), ("B0", 1): record(0)},
            expected_groups=3, bootstrap_draws=32)
        self.assertEqual(len(results), 360)
        self.assertEqual(sum(v["point"] is not None for v in results.values()), 36)
        self.assertEqual(results["s0:utility:B0:income:U0:nll"]["status"], "PASS")
        self.assertEqual(results["mean:utility:B0:occupation:Ucal:brier"]["point"], 0)
        self.assertEqual(results["s0:utility:J:income:U0:nll"]["status"],
                         "NOT_ESTIMABLE_DEPENDENCY")

    def test_constant_cost_above_cap_is_interval_violation(self):
        # Break caught: classifying a demonstrated excess as unresolved.
        results = expected_utility_intervals(
            {("T", 0): record(.012), ("T", 1): record(.012)},
            expected_groups=3, bootstrap_draws=32)
        cell = results["mean:utility:T:income:U0:nll"]
        self.assertAlmostEqual(cell["point"], .012)
        self.assertEqual(cell["status"], "VIOLATION")

    def test_row_group_mismatch_between_seeds_is_rejected(self):
        # Break caught: treating different people as paired evaluations.
        bad = record(0)
        bad["unit"] = np.array([1, 2, 2, 3], dtype=np.int64)
        with self.assertRaises(ValueError):
            expected_utility_intervals({("B0", 0): record(0), ("B0", 1): bad},
                                       expected_groups=3, bootstrap_draws=32)


if __name__ == "__main__":
    unittest.main()
