"""Data-free fixed-family checks for the versioned recovery correction."""

from __future__ import annotations

import unittest

import numpy as np

from bank_anchored import r5_recovery_intervals_v2 as recovery


class ConditionalNullFamilyTest(unittest.TestCase):
    def test_243_s21_slots_are_null_and_k_remains_fixed(self) -> None:
        values = {}
        for arm in recovery.ESTIMABLE_ARMS:
            for seed in (0, 1):
                for draw in range(3 if arm in recovery.STOCHASTIC else 1):
                    for view in recovery.VIEWS:
                        values[(arm, seed, draw, view)] = np.full(
                            recovery.BOOTSTRAPS + 1, .5, dtype=np.float64)
        resampled = np.zeros((recovery.BOOTSTRAPS, 2, 3), dtype=np.int64)
        clauses = recovery.intervals(values, resampled)
        self.assertEqual(len(clauses), 1215)
        self.assertEqual(recovery.K, 1575)
        self.assertEqual(sum(v["status"] == "CONDITIONAL_NULL"
                             for v in clauses.values()), 243)
        self.assertEqual(sum(v["status"] == "BOUND_ONLY"
                             for v in clauses.values()), 972)
        for name, value in clauses.items():
            if ":S21:" in name:
                self.assertIsNone(value["point"])
                self.assertIsNone(value["lower"])
                self.assertIsNone(value["upper"])
            else:
                self.assertEqual(value["point"], 0.0)

    def test_null_slots_cannot_supply_a_directed_comparison(self) -> None:
        name = "mean:recovery:J:S21:pair:auc"
        clauses = {name: {"point": None, "lower": None, "upper": None,
                          "status": "CONDITIONAL_NULL"}}
        with self.assertRaises(ValueError):
            recovery.directed_criterion(clauses, "J", "S21", "pair",
                                        direction=1, cap=.020, kind="benefit")


if __name__ == "__main__":
    unittest.main()
