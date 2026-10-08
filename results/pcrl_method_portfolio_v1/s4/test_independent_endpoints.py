"""Synthetic checks for S4's independent endpoint replay math."""

import unittest

import numpy as np

from independent_endpoints import ArmRows, SeedRows, TaskRows, _RankedReader, replay_family


class IndependentEndpointTests(unittest.TestCase):
    def test_weighted_auc_ties_equal_explicit_repeated_rows(self):
        score = np.array([0.2, 0.2, 0.8, 0.9])
        sex = np.array([0, 1, 1, 0])
        weight = np.array([1.0, 2.0, 1.0, 3.0])
        reader = _RankedReader(score, sex)
        self.assertAlmostEqual(reader.auc(weight), 1.0 / 6.0)

    def test_identical_candidate_control_and_references_have_exact_margins(self):
        n = 2234
        row = np.arange(n, dtype=np.int64)
        income = TaskRows(np.tile([0.5, 0.5], (n, 1)), np.zeros(n, dtype=np.int64))
        occupation = TaskRows(np.full((n, 6), 1.0 / 6.0), np.zeros(n, dtype=np.int64))
        tasks = {"income": income, "occupation": occupation}
        reader = np.full(n, 0.5)
        arm = ArmRows(tasks, {view: {"NULL": reader} for view in ("r1", "r2", "pair")})
        seed = SeedRows(row, row, {"income": row % 2, "occupation": row % 6}, row % 2,
                        {ref: tasks for ref in ("U0", "Ucal")}, arm, {"CTRL": arm})
        report = replay_family({0: seed}, ("CTRL",), bytes(range(32)))
        self.assertEqual(len(report["clauses"]), 15)
        margins = {item["name"]: item for item in report["clauses"]}
        self.assertEqual(margins["income/U0/ACC"]["point_margin"], 0.010)
        self.assertEqual(margins["occupation/Ucal/BRIER"]["point_margin"], 0.005)
        self.assertEqual(margins["CTRL/R1_LOCAL"]["point_margin"], 0.010)
        self.assertEqual(margins["CTRL/PAIR_BENEFIT"]["point_margin"], -0.020)
        self.assertEqual(margins["CTRL/PAIR_BENEFIT"]["status"], "VIOLATION")


if __name__ == "__main__":
    unittest.main()
